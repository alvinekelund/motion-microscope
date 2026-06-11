// Approximate Riesz transform of one Laplacian level.
//
// The Riesz transform is the 2D generalization of the Hilbert transform: it
// turns the real bandpassed signal I into a *monogenic* triple (I, R1, R2)
// from which local amplitude, orientation and phase can be read directly.
// Wadhwa et al. 2014 show the 3-tap derivative pair [0.5, 0, -0.5] is an
// adequate approximation on Laplacian levels — which is what makes the whole
// pipeline cheap enough to run per-frame.
@group(0) @binding(0) var lap: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

fn lum(p: vec2i, size: vec2i) -> f32 {
  return textureLoad(lap, clamp(p, vec2i(0), size - 1), 0).x;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = vec2i(textureDimensions(dst));
  if (i32(id.x) >= size.x || i32(id.y) >= size.y) { return; }
  let p = vec2i(id.xy);
  let i0 = textureLoad(lap, p, 0).x;
  let r1 = 0.5 * (lum(p + vec2i(-1, 0), size) - lum(p + vec2i(1, 0), size));
  let r2 = 0.5 * (lum(p + vec2i(0, -1), size) - lum(p + vec2i(0, 1), size));
  let a = sqrt(i0 * i0 + r1 * r1 + r2 * r2);
  textureStore(dst, p, vec4f(i0, r1, r2, a));
}
