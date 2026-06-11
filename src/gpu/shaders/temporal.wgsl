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
// The temporal bandpass is a second-order RBJ biquad (transposed Direct Form
// II) applied per pixel to the leaky phase accumulator. A biquad has a steep
// ±12 dB/octave skirt, so the heartbeat band is isolated far more cleanly than
// the earlier difference-of-one-poles (±6 dB/octave, broad and leaky) — less
// of the surrounding spectrum, including residual sensor noise, survives into
// the magnified motion. Per-pixel state is 6 floats: the accumulator (2) plus
// the biquad's two delay lines (z1, z2) for each of the two phase channels.
//
// Coefficients (b0, a1, a2, with b1 = 0 and b2 = -b0) are computed on the CPU
// from the slider's center frequency and a fixed Q; see pipeline.ts.

struct U {
  b0: f32,   // biquad numerator gain (b2 = -b0, b1 = 0)
  a1: f32,   // biquad denominator coeff (normalized)
  a2: f32,   // biquad denominator coeff (normalized)
  leak: f32, // accumulator leak, ~exp(-2*pi*0.05*dt)
}

@group(0) @binding(0) var coeff: texture_2d<f32>;      // current (I, R1, R2, A)
@group(0) @binding(1) var prevCoeff: texture_2d<f32>;  // previous frame's triple
@group(0) @binding(2) var stateIn: texture_2d<f32>;    // (phi.x, phi.y, z1.x, z1.y)
@group(0) @binding(3) var state2In: texture_2d<f32>;   // (z2.x, z2.y, -, -)
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

  // Leaky position accumulator keeps phi small and fp16-exact.
  let phi = s1.xy * u.leak + dphi;

  // Biquad bandpass, transposed Direct Form II, both phase channels at once.
  // With b1 = 0 and b2 = -b0:
  //   y  = b0*phi + z1
  //   z1 = -a1*y + z2
  //   z2 = -b0*phi - a2*y
  let z1 = s1.zw;
  let z2 = s2.xy;
  let bp = u.b0 * phi + z1;
  let z1n = -u.a1 * bp + z2;
  let z2n = -u.b0 * phi - u.a2 * bp;

  textureStore(stateOut, p, vec4f(phi, z1n));
  textureStore(state2Out, p, vec4f(z2n, 0.0, 0.0));
  textureStore(filtered, p, vec4f(bp, qc.w * qc.w, 0.0));
}
