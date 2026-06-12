import { describe, expect, it } from "vitest";
import { VitalsProbe } from "../src/pulse/probe";
import { bandForCenterFreq } from "../src/pulse/bpm";

const W = 96;
const H = 72;
const FPS = 30;

function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32 - 0.5;
  };
}

interface SceneOpts {
  breathHz: number;
  breathScale: number; // fractional expand/contract amplitude (non-rigid!)
  breathShiftPx: number; // small rigid vertical component
  pulseHz: number;
  pulseAmp: number;
  noise: number;
  rng: () => number;
}

/**
 * A frame of synthetic webcam scene where the skin-toned foreground
 * *expands and contracts* with breathing (radially, about its centre) plus a
 * small vertical rise — i.e. genuinely non-rigid motion. A single global
 * translation cancels the outward halves of the expansion; this is the case
 * that exposes the old detector and that the per-block one must handle.
 */
function renderFrame(tSec: number, o: SceneOpts): Uint8ClampedArray {
  const buf = new Uint8ClampedArray(W * H * 4);
  const scale = 1 + o.breathScale * Math.sin(2 * Math.PI * o.breathHz * tSec);
  const yShift = o.breathShiftPx * Math.sin(2 * Math.PI * o.breathHz * tSec);
  const pulse = o.pulseAmp * Math.sin(2 * Math.PI * o.pulseHz * tSec);
  const cx = W * 0.5;
  const cy = H * 0.5;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const sx = (x - cx) / scale + cx; // source coords scale with the breath
      const sy = (y - cy - yShift) / scale + cy;
      const inFg = sx > W * 0.25 && sx < W * 0.75 && sy > H * 0.2 && sy < H * 0.8;
      let r: number, g: number, b: number;
      if (inFg) {
        const tex = 18 * Math.sin(0.6 * sx) + 18 * Math.sin(0.7 * sy);
        r = 200 + 0.4 * tex;
        g = 150 + 0.4 * tex + pulse;
        b = 120 + 0.4 * tex;
      } else {
        const tex = 22 * Math.sin(0.5 * x) + 22 * Math.sin(0.45 * y); // static bg
        r = 55 + tex;
        g = 80 + tex;
        b = 150 + tex;
      }
      const p = (y * W + x) * 4;
      buf[p] = r + o.noise * o.rng();
      buf[p + 1] = g + o.noise * o.rng();
      buf[p + 2] = b + o.noise * o.rng();
      buf[p + 3] = 255;
    }
  }
  return buf;
}

function runScene(seconds: number, o: SceneOpts): VitalsProbe {
  const probe = new VitalsProbe(W, H);
  const frames = seconds * FPS;
  for (let i = 0; i < frames; i++) {
    probe.process(renderFrame(i / FPS, o), (i * 1000) / FPS);
  }
  return probe;
}

const STILL = { breathHz: 0, breathScale: 0, breathShiftPx: 0, pulseHz: 0, pulseAmp: 0, noise: 3 };

describe("VitalsProbe — end-to-end frame → number", () => {
  it("recovers breathing (15/min) from non-rigid expansion over a textured background", () => {
    const probe = runScene(32, { ...STILL, breathHz: 15 / 60, breathScale: 0.04, breathShiftPx: 0.5, rng: lcg(11) });
    const est = probe.estimate(bandForCenterFreq(0.3));
    expect(est.unit).toBe("breaths/min");
    expect(est.value).not.toBeNull();
    expect(Math.abs(est.value! - 15)).toBeLessThanOrEqual(2);
    expect(est.confidence).toBeGreaterThan(0.25);
  });

  it("recovers a faster breathing rate (24/min)", () => {
    const probe = runScene(32, { ...STILL, breathHz: 24 / 60, breathScale: 0.04, breathShiftPx: 0.5, rng: lcg(7) });
    const est = probe.estimate(bandForCenterFreq(0.3));
    expect(Math.abs(est.value! - 24)).toBeLessThanOrEqual(2.5);
  });

  it("recovers breathing from pure expansion (no rigid component at all)", () => {
    // The case a global translation cancels to nothing.
    const probe = runScene(32, { ...STILL, breathHz: 18 / 60, breathScale: 0.05, breathShiftPx: 0, rng: lcg(21) });
    const est = probe.estimate(bandForCenterFreq(0.3));
    expect(est.value).not.toBeNull();
    expect(Math.abs(est.value! - 18)).toBeLessThanOrEqual(2.5);
  });

  it("recovers pulse (72 BPM) from green modulation with the subject still", () => {
    const probe = runScene(32, { ...STILL, pulseHz: 72 / 60, pulseAmp: 4, rng: lcg(3) });
    const est = probe.estimate(bandForCenterFreq(1.2));
    expect(est.unit).toBe("BPM");
    expect(est.value).not.toBeNull();
    expect(Math.abs(est.value! - 72)).toBeLessThanOrEqual(3);
  });

  it("stays silent (no false breath) on a completely still, noisy scene", () => {
    const probe = runScene(32, { ...STILL, noise: 5, rng: lcg(99) });
    const est = probe.estimate(bandForCenterFreq(0.3));
    expect(est.confidence).toBeLessThan(0.25);
  });
});
