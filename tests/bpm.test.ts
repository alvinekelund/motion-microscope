import { describe, expect, it } from "vitest";
import { RhythmTracker, bandForCenterFreq } from "../src/pulse/bpm";

const fs = 30;
let seed = 42;
function rand(): number {
  // Deterministic LCG so the tests never flake.
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 2 ** 32 - 0.5;
}

/** Pulse: a chrominance oscillation strongest in green, no motion. */
function pulseTrace(tracker: RhythmTracker, bpm: number, seconds: number): void {
  const f = bpm / 60;
  for (let i = 0; i < seconds * fs; i++) {
    const t = i / fs;
    const p = Math.sin(2 * Math.PI * f * t);
    const drift = 0.04 * t;
    tracker.addSample(
      150 + drift + 0.2 * p + 0.3 * rand(),
      140 + drift + 1.0 * p + 0.3 * rand(),
      120 + drift - 0.1 * p + 0.3 * rand(),
      16 + 0.05 * rand(), // centroid steady (no motion)
      16 + 0.05 * rand(),
      i * (1000 / fs)
    );
  }
}

/** Motion: the brightness centroid translates sinusoidally; colour steady. */
function motionTrace(tracker: RhythmTracker, hz: number, seconds: number, amp = 3): void {
  for (let i = 0; i < seconds * fs; i++) {
    const t = i / fs;
    const d = amp * Math.sin(2 * Math.PI * hz * t);
    tracker.addSample(
      150 + 0.3 * rand(),
      140 + 0.3 * rand(),
      120 + 0.3 * rand(),
      16 + d + 0.1 * rand(), // vertical breathing motion lands on cx here, fine
      16 + d + 0.1 * rand(),
      i * (1000 / fs)
    );
  }
}

describe("bandForCenterFreq", () => {
  it("routes slider position to the right readout", () => {
    expect(bandForCenterFreq(0.3).kind).toBe("breathing");
    expect(bandForCenterFreq(1.2).kind).toBe("pulse");
    expect(bandForCenterFreq(8).kind).toBe("vibration");
  });
});

describe("RhythmTracker — pulse band", () => {
  const band = bandForCenterFreq(1.2);

  it("reports nothing before enough samples arrive", () => {
    seed = 42;
    const tracker = new RhythmTracker();
    pulseTrace(tracker, 72, 3);
    const est = tracker.estimate(band);
    expect(est.value).toBeNull();
    expect(est.unit).toBe("BPM");
  });

  it("recovers 72 BPM from a noisy drifting trace via POS", () => {
    seed = 42;
    const tracker = new RhythmTracker();
    pulseTrace(tracker, 72, 30);
    const est = tracker.estimate(band);
    expect(est.value).not.toBeNull();
    expect(Math.abs(est.value! - 72)).toBeLessThanOrEqual(2);
    expect(est.confidence).toBeGreaterThan(0.25);
  });

  it("recovers a fast heart rate (150 BPM)", () => {
    seed = 42;
    const tracker = new RhythmTracker();
    pulseTrace(tracker, 150, 30);
    const est = tracker.estimate(band);
    expect(Math.abs(est.value! - 150)).toBeLessThanOrEqual(3);
  });

  it("exposes freqHz consistent with the BPM (drives the lock auto-tune)", () => {
    seed = 42;
    const tracker = new RhythmTracker();
    pulseTrace(tracker, 72, 30);
    const est = tracker.estimate(band);
    expect(est.freqHz).not.toBeNull();
    expect(est.freqHz! * 60).toBeCloseTo(est.value!, 0);
    expect(est.freqHz!).toBeGreaterThan(1.0); // ~1.2 Hz
    expect(est.freqHz!).toBeLessThan(1.4);
  });
});

describe("RhythmTracker — breathing band (motion)", () => {
  it("recovers a 15 breaths/min rhythm from centroid motion", () => {
    seed = 7;
    const tracker = new RhythmTracker();
    motionTrace(tracker, 15 / 60, 30); // 0.25 Hz
    const est = tracker.estimate(bandForCenterFreq(0.3));
    expect(est.unit).toBe("breaths/min");
    expect(est.value).not.toBeNull();
    expect(Math.abs(est.value! - 15)).toBeLessThanOrEqual(2);
  });

  it("locks on even when colour is dead flat (the whole point)", () => {
    seed = 11;
    const tracker = new RhythmTracker();
    // No colour variation at all — only motion. Old mean-luma path would fail.
    for (let i = 0; i < 30 * fs; i++) {
      const d = 3 * Math.sin(2 * Math.PI * (12 / 60) * (i / fs));
      tracker.addSample(150, 140, 120, 16 + d, 16 + d, i * (1000 / fs));
    }
    const est = tracker.estimate(bandForCenterFreq(0.3));
    expect(Math.abs(est.value! - 12)).toBeLessThanOrEqual(1);
    expect(est.confidence).toBeGreaterThan(0.4);
  });
});

describe("RhythmTracker — vibration band", () => {
  it("reports a raw frequency in Hz", () => {
    seed = 99;
    const tracker = new RhythmTracker();
    motionTrace(tracker, 8, 12); // 8 Hz tremor
    const est = tracker.estimate(bandForCenterFreq(8));
    expect(est.unit).toBe("Hz");
    expect(est.value).not.toBeNull();
    expect(Math.abs(est.value! - 8)).toBeLessThanOrEqual(0.5);
  });
});

describe("RhythmTracker — noise rejection", () => {
  it("stays low-confidence on pure noise", () => {
    seed = 7;
    const tracker = new RhythmTracker();
    for (let i = 0; i < 900; i++) {
      tracker.addSample(150 + rand() * 2, 140 + rand() * 2, 120 + rand() * 2, 16 + rand(), 16 + rand(), i * (1000 / fs));
    }
    expect(tracker.estimate(bandForCenterFreq(1.2)).confidence).toBeLessThan(0.6);
  });
});
