// Pyramid collapse: out_k = magnified_laplacian_k + upsample(out_{k+1}),
// seeded with the (unmagnified) lowpass residual at the coarsest level.
@group(0) @binding(0) var lap: texture_2d<f32>;
@group(0) @binding(1) var lower: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let up = textureSampleLevel(lower, samp, uv, 0.0).x;
  let v = textureLoad(lap, vec2i(id.xy), 0).x + up;
  textureStore(dst, vec2i(id.xy), vec4f(v, 0.0, 0.0, 1.0));
}
