// RGB -> YCoCg split.
//
// Only luma (Y) enters the magnification pipeline; chroma (Co, Cg) is carried
// through untouched and recombined at the end. Webcam chroma is by far the
// noisiest part of the signal (Bayer interpolation + low-light gain), so
// keeping it out of the amplifier is the cheapest denoising step we have.
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let size = textureDimensions(dst);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let c = textureLoad(src, vec2i(id.xy), 0).rgb;
  let y  =  0.25 * c.r + 0.5 * c.g + 0.25 * c.b;
  let co =  0.5  * c.r             - 0.5  * c.b;
  let cg = -0.25 * c.r + 0.5 * c.g - 0.25 * c.b;
  textureStore(dst, vec2i(id.xy), vec4f(y, co, cg, 1.0));
}
