import { describe, expect, it } from "vitest";
import { bandpassCoeffs } from "../src/gpu/biquad";

/**
 * Drive a sinusoid at `f` through the SAME transposed-Direct-Form-II
 * recurrence the WGSL shader runs (b1 = 0, b2 = -b0) and return the
 * steady-state output amplitude. This is the on-GPU filter, in JS.
 */
function steadyStateGain(f0: number, fs: number, Q: number, f: number): number {
  const { b0, a1, a2 } = bandpassCoeffs(f0, fs, Q);
  let z1 = 0;
  let z2 = 0;
  let amp = 0;
  const n = Math.round(fs * 40); // long enough to settle
  for (let i = 0; i < n; i++) {
    const x = Math.sin((2 * Math.PI * f * i) / fs);
    const y = b0 * x + z1;
    z1 = -a1 * y + z2;
    z2 = -b0 * x - a2 * y;
    if (i > n - fs * 4) amp = Math.max(amp, Math.abs(y)); // peak over last cycles
  }
  return amp;
}

describe("bandpass biquad", () => {
  const fs = 30;
  const f0 = 1.2; // heartbeat
  const Q = 2.5;

  it("passes the center frequency at ~unity gain", () => {
    const g = steadyStateGain(f0, fs, Q, f0);
    expect(g).toBeGreaterThan(0.9);
    expect(g).toBeLessThan(1.1);
  });

  it("attenuates an octave above and below the center", () => {
    const center = steadyStateGain(f0, fs, Q, f0);
    const below = steadyStateGain(f0, fs, Q, f0 / 2);
    const above = steadyStateGain(f0, fs, Q, f0 * 2);
    expect(below / center).toBeLessThan(0.5);
    expect(above / center).toBeLessThan(0.5);
  });

  it("strongly rejects DC and very low drift", () => {
    const center = steadyStateGain(f0, fs, Q, f0);
    expect(steadyStateGain(f0, fs, Q, 0.05) / center).toBeLessThan(0.1);
  });

  it("is sharper than the old difference-of-one-poles at half-octave offset", () => {
    // Old filter: difference of one-pole lowpasses at f0*1.5 and f0/1.5.
    const onePole = (fc: number) => 1 - Math.exp((-2 * Math.PI * fc) / fs);
    const a1c = onePole(f0 * 1.5);
    const a2c = onePole(f0 / 1.5);
    const oldGain = (f: number): number => {
      let lp1 = 0, lp2 = 0, amp = 0;
      const n = fs * 40;
      for (let i = 0; i < n; i++) {
        const x = Math.sin((2 * Math.PI * f * i) / fs);
        lp1 += a1c * (x - lp1);
        lp2 += a2c * (x - lp2);
        if (i > n - fs * 4) amp = Math.max(amp, Math.abs(lp1 - lp2));
      }
      return amp;
    };
    const offset = f0 * Math.SQRT2; // half octave above
    const biquadRel = steadyStateGain(f0, fs, Q, offset) / steadyStateGain(f0, fs, Q, f0);
    const oldRel = oldGain(offset) / oldGain(f0);
    expect(biquadRel).toBeLessThan(oldRel); // biquad rolls off faster
  });
});
