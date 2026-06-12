/**
 * Global translational optical flow between two luma frames, via the
 * Lucas-Kanade brightness-constancy least-squares solution:
 *
 *   minimize Σ wᵢ (gx·dx + gy·dy + gt)²   ⇒   [Σw·gx²  Σw·gxgy][dx]   [Σw·gx·gt]
 *                                             [Σw·gxgy Σw·gy² ][dy] = -[Σw·gy·gt]
 *
 * The optional per-pixel weight `w` is what makes this usable on a real
 * webcam. A naive uniform fit assumes the *whole frame* translates together —
 * but the background is static, and since it's usually most of the pixels it
 * drags the estimate toward zero and averages the person's motion away.
 * Weighting by where motion actually is (a foreground/activity map) excludes
 * the static background from both sides of the normal equations, so the fit
 * locks onto the thing that's moving.
 *
 * Per-frame motion at 30 fps is far below a pixel — exactly the regime where
 * the linear brightness-constancy assumption holds — so the sub-pixel estimate
 * is clean. Returns the shift (dx, dy) taking `prev` to `cur`, or zero when
 * there's too little weighted gradient to constrain a solution.
 */
export function globalTranslation(
  prev: Float64Array,
  cur: Float64Array,
  w: number,
  h: number,
  weight?: Float64Array
): { dx: number; dy: number } {
  let sxx = 0, sxy = 0, syy = 0, sxt = 0, syt = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const wi = weight ? weight[i] : 1;
      if (wi <= 0) continue;
      const gx = (cur[i + 1] - cur[i - 1]) * 0.5;
      const gy = (cur[i + w] - cur[i - w]) * 0.5;
      const gt = cur[i] - prev[i];
      sxx += wi * gx * gx;
      sxy += wi * gx * gy;
      syy += wi * gy * gy;
      sxt += wi * gx * gt;
      syt += wi * gy * gt;
    }
  }
  const det = sxx * syy - sxy * sxy;
  if (Math.abs(det) < 1e-9 || sxx + syy < 1e-9) return { dx: 0, dy: 0 };
  // [dx; dy] = -A^{-1} [sxt; syt]
  const dx = (sxy * syt - syy * sxt) / det;
  const dy = (sxy * sxt - sxx * syt) / det;
  return { dx, dy };
}
