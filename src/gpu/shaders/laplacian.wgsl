// Laplacian level = current Gaussian level - bilinear upsample of the next
// coarser level. Each level is a spatial bandpass; phase in a bandpassed
// signal is what lets us read sub-pixel motion.
@group(0) @binding(0) var full: texture_2d<f32>;
@group(0) @binding(1) var low: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(size);
  let up = textureSampleLevel(low, samp, uv, 0.0).x;
  let v = textureLoad(full, vec2i(id.xy), 0).x - up;
  textureStore(dst, vec2i(id.xy), vec4f(v, 0.0, 0.0, 1.0));
}
