// Phase-shift reconstruction: advance each coefficient's phase by alpha times
// the bandpassed phase, along the locally measured orientation.
//
//   I_out = cos(phi_m) * I - sin(phi_m) * (o.x * R1 + o.y * R2)
//
// This is exactly "rotate the monogenic quaternion further along the rotation
// it was already making" — sub-pixel motion in the chosen temporal band gets
// multiplied by alpha, everything else is untouched.
//
// Two more guards live here:
//  - phi_m is clamped: a single hot pixel or blink can't fling a coefficient
//    around the phase circle and leave a smearing artifact.
//  - a per-level gain (set on the CPU) attenuates the finest level, where
//    sensor noise dominates and real pulse motion contributes least.

struct U {
  alpha: f32,    // effective amplification for this level (slider x level gain)
  clampPhi: f32, // max |magnified phase|, radians
  _pad: vec2f,
}

@group(0) @binding(0) var coeff: texture_2d<f32>;   // (I, R1, R2, A)
@group(0) @binding(1) var phase: texture_2d<f32>;   // blurred (bp.x, bp.y, -, -)
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> u: U;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let p = vec2i(id.xy);

  let q = textureLoad(coeff, p, 0);
  let bp = textureLoad(phase, p, 0).xy * u.alpha;
  let m = length(bp);

  var outv = q.x;
  if (m > 1e-6) {
    let phiM = min(m, u.clampPhi);
    let o = bp / m;
    outv = cos(phiM) * q.x - sin(phiM) * (o.x * q.y + o.y * q.z);
  }
  textureStore(dst, p, vec4f(outv, 0.0, 0.0, 1.0));
}
