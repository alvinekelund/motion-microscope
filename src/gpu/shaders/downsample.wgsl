// Lowpass + decimate by 2 with a separable 5-tap binomial kernel, done as a
// single 5x5 gather at the output resolution. This builds the Gaussian
// pyramid chain that the Laplacian levels are derived from.
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

const K = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let osize = textureDimensions(dst);
  if (id.x >= osize.x || id.y >= osize.y) { return; }
  let isize = vec2i(textureDimensions(src));
  let base = vec2i(id.xy) * 2;
  var acc = 0.0;
  for (var j = -2; j <= 2; j++) {
    for (var i = -2; i <= 2; i++) {
      let p = clamp(base + vec2i(i, j), vec2i(0), isize - 1);
      acc += K[i + 2] * K[j + 2] * textureLoad(src, p, 0).x;
    }
  }
  textureStore(dst, vec2i(id.xy), vec4f(acc, 0.0, 0.0, 1.0));
}
