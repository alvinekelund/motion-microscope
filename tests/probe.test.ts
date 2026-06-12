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
  breathAmpPx: number; // vertical translation amplitude, probe pixels
  pulseHz: number;
  pulseAmp: number; // green modulation amplitude (0..255)
  noise: number; // per-channel per-frame noise amplitude
  rng: () => number;
}

/**
 * One frame of a synthetic webcam scene: a textured, skin-toned foreground
 * block that translates vertically with breathing, over a textured *static*
 * non-skin background, plus sensor noise. The static textured background is
 * the whole point — it's what a naive global-flow estimate drowns in.
 */
function renderFrame(tSec: number, o: SceneOpts): Uint8ClampedArray {
  const buf = new Uint8ClampedArray(W * H * 4);
  const yShift = o.breathAmpPx * Math.sin(2 * Math.PI * o.breathHz * tSec);
  const pulse = o.pulseAmp * Math.sin(2 * Math.PI * o.pulseHz * tSec);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const fy = y - yShift; // foreground texture/edges move with the breath
      const inFg = x > W * 0.25 && x < W * 0.75 && fy > H * 0.2 && fy < H * 0.8;
      let r: number, g: number, b: number;
      if (inFg) {
        const tex = 18 * Math.sin(0.6 * x) + 18 * Math.sin(0.7 * fy);
        r = 200 + 0.4 * tex;
        g = 150 + 0.4 * tex + pulse;
        b = 120 + 0.4 * tex;
      } else {
        const tex = 22 * Math.sin(0.5 * x) + 22 * Math.sin(0.45 * y); // static
        r = 55 + tex;
        g = 80 + tex;
        b = 150 + tex; // blue-ish → not skin, but textured
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

describe("VitalsProbe — end-to-end frame → number", () => {
  it("recovers breathing (15/min) from a block moving over a textured static background", () => {
    const probe = runScene(32, {
      breathHz: 15 / 60,
      breathAmpPx: 1.0,
      pulseHz: 0,
      pulseAmp: 0,
      noise: 3,
      rng: lcg(11),
    });
    const est = probe.estimate(bandForCenterFreq(0.3));
    expect(est.unit).toBe("breaths/min");
    expect(est.value).not.toBeNull();
    expect(Math.abs(est.value! - 15)).toBeLessThanOrEqual(2);
    expect(est.confidence).toBeGreaterThan(0.25);
  });

  it("recovers a faster breathing rate (24/min)", () => {
    const probe = runScene(32, {
      breathHz: 24 / 60,
      breathAmpPx: 1.0,
      pulseHz: 0,
      pulseAmp: 0,
      noise: 3,
      rng: lcg(7),
    });
    const est = probe.estimate(bandForCenterFreq(0.3));
    expect(Math.abs(est.value! - 24)).toBeLessThanOrEqual(2.5);
  });

  it("recovers pulse (72 BPM) from green modulation with the subject still", () => {
    const probe = runScene(32, {
      breathHz: 0,
      breathAmpPx: 0,
      pulseHz: 72 / 60,
      pulseAmp: 4,
      noise: 3,
      rng: lcg(3),
    });
    const est = probe.estimate(bandForCenterFreq(1.2));
    expect(est.unit).toBe("BPM");
    expect(est.value).not.toBeNull();
    expect(Math.abs(est.value! - 72)).toBeLessThanOrEqual(3);
  });
});
