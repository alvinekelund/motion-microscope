/**
 * Global translational optical flow between two luma frames, via the
 * Lucas-Kanade brightness-constancy least-squares solution over the whole
 * image:
 *
 *   minimize Σ (gx·dx + gy·dy + gt)²   ⇒   [Σgx²  Σgxgy][dx]   [Σgx·gt]
 *                                          [Σgxgy Σgy² ][dy] = -[Σgy·gt]
 *
 * This is exactly the bulk movement the magnifier makes visible — the head and
 * shoulders rising with a breath — measured directly, rather than inferred
 * from a brightness centroid over a flickering skin mask. Per-frame motion at
 * 30 fps is far below a pixel, which is precisely the regime where the linear
 * brightness-constancy assumption holds, so the estimate is clean.
 *
 * Returns the sub-pixel shift (dx, dy) taking `prev` to `cur`. Falls back to
 * zero when the image is too flat to constrain a solution (det ≈ 0 — the
 * aperture problem).
 */
export function globalTranslation(
  prev: Float64Array,
  cur: Float64Array,
  w: number,
  h: number
): { dx: number; dy: number } {
  let sxx = 0, sxy = 0, syy = 0, sxt = 0, syt = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = (cur[i + 1] - cur[i - 1]) * 0.5;
      const gy = (cur[i + w] - cur[i - w]) * 0.5;
      const gt = cur[i] - prev[i];
      sxx += gx * gx;
      sxy += gx * gy;
      syy += gy * gy;
      sxt += gx * gt;
      syt += gy * gt;
    }
  }
  const det = sxx * syy - sxy * sxy;
  if (Math.abs(det) < 1e-6) return { dx: 0, dy: 0 };
  // [dx; dy] = -A^{-1} [sxt; syt]
  const dx = (sxy * syt - syy * sxt) / det;
  const dy = (sxy * sxt - sxx * syt) / det;
  return { dx, dy };
}
