import { describe, expect, it } from "vitest";
import { fft } from "../src/pulse/fft";

describe("fft", () => {
  it("transforms an impulse to a flat spectrum", () => {
    const n = 64;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    re[0] = 1;
    fft(re, im);
    for (let k = 0; k < n; k++) {
      expect(re[k]).toBeCloseTo(1, 10);
      expect(im[k]).toBeCloseTo(0, 10);
    }
  });

  it("localizes a pure sinusoid in the correct bin", () => {
    const n = 256;
    const bin = 19;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < n; i++) re[i] = Math.cos((2 * Math.PI * bin * i) / n);
    fft(re, im);
    const mags = Array.from({ length: n }, (_, k) => Math.hypot(re[k], im[k]));
    expect(mags[bin]).toBeCloseTo(n / 2, 6);
    // Energy everywhere else is negligible.
    const rest = mags.filter((_, k) => k !== bin && k !== n - bin);
    expect(Math.max(...rest)).toBeLessThan(1e-8);
  });

  it("rejects non-power-of-two input", () => {
    expect(() => fft(new Float64Array(100), new Float64Array(100))).toThrow();
  });
});
