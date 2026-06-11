// Amplitude-weighted spatial smoothing of the bandpassed phase — the single
// most important noise-control step in the pipeline.
//
// Phase is only meaningful where the coefficient has amplitude; in flat image
// regions it is pure sensor noise. A plain blur would average that noise into
// real edges. Weighting by amplitude^2 (blur(phi*A^2)/blur(A^2), per the
// paper) means strong structure dominates and noise-floor pixels borrow their
// phase from real neighbors instead of contaminating them.
//
// Separable 9-tap Gaussian (sigma = 2): blur_h premultiplies and filters
// horizontally, blur_v filters vertically and divides through.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

const G = array<f32, 9>(
  0.0276, 0.0663, 0.1238, 0.1802, 0.2042, 0.1802, 0.1238, 0.0663, 0.0276
);

fn load(p: vec2i, size: vec2i) -> vec4f {
  return textureLoad(src, clamp(p, vec2i(0), size - 1), 0);
}

@compute @workgroup_size(8, 8)
fn blur_h(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(dst));
  if (i32(id.x) >= size.x || i32(id.y) >= size.y) { return; }
  let p = vec2i(id.xy);
  var acc = vec3f(0.0);
  for (var k = -4; k <= 4; k++) {
    let s = load(p + vec2i(k, 0), size); // (bp.x, bp.y, A^2, -)
    acc += G[k + 4] * vec3f(s.xy * s.z, s.z);
  }
  textureStore(dst, p, vec4f(acc, 0.0)); // (num.x, num.y, den, -)
}

@compute @workgroup_size(8, 8)
fn blur_v(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(dst));
  if (i32(id.x) >= size.x || i32(id.y) >= size.y) { return; }
  let p = vec2i(id.xy);
  var acc = vec3f(0.0);
  for (var k = -4; k <= 4; k++) {
    acc += G[k + 4] * load(p + vec2i(0, k), size).xyz;
  }
  let bp = acc.xy / max(acc.z, 1e-7);
  textureStore(dst, p, vec4f(bp, 0.0, 0.0));
}
