// Final composite: magnified luma + original chroma -> RGB, with a draggable
// split line for the original-vs-magnified comparison. The view is mirrored
// here (selfie convention) so the processing pipeline stays in sensor space.

struct U {
  split: f32,     // 0..1, in *displayed* (mirrored) coordinates
  lineWidth: f32, // split line half-width in uv units
  _pad: vec2f,
}

@group(0) @binding(0) var ycocg: texture_2d<f32>;   // original (Y, Co, Cg)
@group(0) @binding(1) var lumaMag: texture_2d<f32>; // magnified Y
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: U;

struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
  // Fullscreen triangle.
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var out: VOut;
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
  let suv = vec2f(1.0 - in.uv.x, in.uv.y); // mirror for display
  let orig = textureSampleLevel(ycocg, samp, suv, 0.0).xyz;
  let yMag = textureSampleLevel(lumaMag, samp, suv, 0.0).x;

  // Left of the split shows the untouched feed.
  let y = select(yMag, orig.x, in.uv.x < u.split);

  let r = y + orig.y - orig.z;
  let g = y + orig.z;
  let b = y - orig.y - orig.z;
  var rgb = clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));

  if (abs(in.uv.x - u.split) < u.lineWidth) {
    rgb = mix(rgb, vec3f(1.0, 0.30, 0.37), 0.9);
  }
  return vec4f(rgb, 1.0);
}
