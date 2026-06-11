import { describe, expect, it } from "vitest";
import { PulseTracker } from "../src/pulse/bpm";

/** Feed a synthetic rPPG trace: pulse sinusoid + drift + noise at 30 fps. */
function syntheticTrace(tracker: PulseTracker, bpm: number, seconds: number): void {
  const fs = 30;
  const f = bpm / 60;
  let seed = 42;
  const rand = () => {
    // Deterministic LCG so the test never flakes.
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 2 ** 32 - 0.5;
  };
  for (let i = 0; i < seconds * fs; i++) {
    const t = i / fs;
    const pulse = 0.8 * Math.sin(2 * Math.PI * f * t);
    const drift = 0.05 * t; // auto-exposure ramp
    const noise = 0.3 * rand();
    tracker.addSample(120 + pulse + drift + noise, i * (1000 / fs));
  }
}

describe("PulseTracker", () => {
  it("reports nothing before enough samples arrive", () => {
    const tracker = new PulseTracker();
    syntheticTrace(tracker, 72, 3);
    expect(tracker.estimate().bpm).toBeNull();
  });

  it("recovers 72 bpm from a noisy drifting trace", () => {
    const tracker = new PulseTracker();
    syntheticTrace(tracker, 72, 17);
    const est = tracker.estimate();
    expect(est.bpm).not.toBeNull();
    expect(Math.abs(est.bpm! - 72)).toBeLessThanOrEqual(2);
    expect(est.confidence).toBeGreaterThan(0.25);
  });

  it("recovers a fast heart rate (150 bpm)", () => {
    const tracker = new PulseTracker();
    syntheticTrace(tracker, 150, 17);
    const est = tracker.estimate();
    expect(est.bpm).not.toBeNull();
    expect(Math.abs(est.bpm! - 150)).toBeLessThanOrEqual(3);
  });

  it("stays silent on pure noise", () => {
    const tracker = new PulseTracker();
    let seed = 7;
    for (let i = 0; i < 510; i++) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      tracker.addSample(120 + (seed / 2 ** 32 - 0.5) * 2, i * 33.3);
    }
    // White noise has no prominent band peak; smoothed bpm should never lock.
    expect(tracker.estimate().confidence).toBeLessThan(0.6);
  });
});
