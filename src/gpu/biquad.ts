/**
 * RBJ bandpass biquad with constant 0 dB peak gain (Robert Bristow-Johnson's
 * Audio EQ Cookbook). Returns the normalized coefficients consumed by
 * temporal.wgsl, where b1 = 0 and b2 = -b0 are implied.
 *
 * @param f0 center frequency, Hz
 * @param fs sample rate, Hz
 * @param Q  quality factor — higher = narrower band, sharper skirts
 */
export function bandpassCoeffs(f0: number, fs: number, Q: number): { b0: number; a1: number; a2: number } {
  const w0 = (2 * Math.PI * f0) / fs;
  const cosw = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * Q);
  const a0 = 1 + alpha;
  return {
    b0: alpha / a0,
    a1: (-2 * cosw) / a0,
    a2: (1 - alpha) / a0,
  };
}
