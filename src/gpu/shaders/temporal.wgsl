// Quaternionic phase difference + temporal bandpass. The heart of the method.
//
// Treat the monogenic triple as a quaternion q = I + i*R1 + j*R2. The product
// q_t * conj(q_{t-1}) rotates last frame's coefficient onto this frame's; the
// angle of that rotation is the *phase advance* between frames — i.e. motion,
// measured per pixel per scale, without ever computing optical flow.
//
//   w  = I_t*I_p + R1_t*R1_p + R2_t*R2_p        (scalar part)
//   vx = I_p*R1_t - I_t*R1_p                    (i part)
//   vy = I_p*R2_t - I_t*R2_p                    (j part)
//   dphi = atan2(|(vx,vy)|, w) * (vx,vy)/|(vx,vy)|
//
// (The k part encodes orientation change between frames and is dropped,
// following the paper.) Because we accumulate *differences*, phase is already
// unwrapped — no mod-2pi headaches.
//
// Numerics note: state lives in rgba16float. A raw accumulated phase would
// drift to values where fp16 spacing (~0.06 at phi≈100) swamps the ~0.01 rad
// pulse signal. The leak term keeps the accumulator zero-centered (it is a
// one-pole DC blocker at ~0.05 Hz, far below the heartbeat band), so every
// stored quantity stays small and fp16-exact.
//
// The bandpass is the difference of two one-pole lowpasses (cutoffs set from
// the UI frequency band). Cheap, stable, and per-pixel state is just 6 floats.

struct U {
  a1: f32,   // lowpass coefficient for the *high* cutoff (faster pole)
  a2: f32,   // lowpass coefficient for the *low* cutoff (slower pole)
  leak: f32, // accumulator leak, ~exp(-2*pi*0.05*dt)
  _pad: f32,
}

@group(0) @binding(0) var coeff: texture_2d<f32>;      // current (I, R1, R2, A)
@group(0) @binding(1) var prevCoeff: texture_2d<f32>;  // previous frame's triple
@group(0) @binding(2) var stateIn: texture_2d<f32>;    // (phi.x, phi.y, lp1.x, lp1.y)
@group(0) @binding(3) var state2In: texture_2d<f32>;   // (lp2.x, lp2.y, -, -)
@group(0) @binding(4) var stateOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var state2Out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var filtered: texture_storage_2d<rgba16float, write>; // (bp.x, bp.y, A^2, 0)
@group(0) @binding(7) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(filtered);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy);

  let qc = textureLoad(coeff, p, 0);
  let qp = textureLoad(prevCoeff, p, 0);

  let w  = dot(qc.xyz, qp.xyz);
  let vx = qp.x * qc.y - qc.x * qp.y;
  let vy = qp.x * qc.z - qc.x * qp.z;
  let vlen = sqrt(vx * vx + vy * vy);

  var dphi = vec2f(0.0);
  // Guard: where amplitude is ~0 (flat regions, first frame) the phase is
  // undefined noise — contribute nothing rather than garbage.
  if (vlen > 1e-6 && qc.w > 1e-4 && qp.w > 1e-4) {
    dphi = vec2f(vx, vy) * (atan2(vlen, w) / vlen);
  }

  let s1 = textureLoad(stateIn, p, 0);
  let s2 = textureLoad(state2In, p, 0);

  let phi = s1.xy * u.leak + dphi;
  let lp1 = mix(s1.zw, phi, u.a1);
  let lp2 = mix(s2.xy, phi, u.a2);
  let bp = lp1 - lp2;

  textureStore(stateOut, p, vec4f(phi, lp1));
  textureStore(state2Out, p, vec4f(lp2, 0.0, 0.0));
  textureStore(filtered, p, vec4f(bp, qc.w * qc.w, 0.0));
}
