/**
 * Cheap per-pixel skin classifier (Kovac et al. uniform-daylight rule, in
 * RGB). Used to mask the vitals probe so that background, hair and clothing
 * don't dilute the rPPG signal — the pulse lives only in skin pixels.
 *
 * Deliberately permissive, and always paired with a whole-region fallback at
 * the call site: if too few pixels pass (unusual lighting, darker or cooler
 * skin tones the daylight rule under-selects), the probe reverts to averaging
 * everything rather than reporting from a near-empty mask. The mask is an
 * accuracy boost when it fires and a no-op when it doesn't — never a way to
 * exclude a face it failed to recognize.
 *
 * @param r,g,b channel values in 0..255
 */
export function isSkin(r: number, g: number, b: number): boolean {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  return (
    r > 95 &&
    g > 40 &&
    b > 20 &&
    mx - mn > 15 && // chromatic, not gray
    r - g > 15 && // warm
    r > g &&
    r > b
  );
}
