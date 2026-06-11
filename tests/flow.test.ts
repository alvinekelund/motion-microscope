import { describe, expect, it } from "vitest";
import { globalTranslation } from "../src/pulse/flow";

const W = 48;
const H = 36;

/** Separable cross-hatch texture: independent gradients in x and y (no aperture). */
function texture(shiftX: number, shiftY: number): Float64Array {
  const img = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const sx = x - shiftX;
      const sy = y - shiftY;
      img[y * W + x] = 128 + 40 * Math.sin(0.3 * sx) + 35 * Math.sin(0.37 * sy);
    }
  }
  return img;
}

describe("globalTranslation", () => {
  it("recovers a known sub-pixel shift", () => {
    const prev = texture(0, 0);
    const cur = texture(0.4, -0.25);
    const { dx, dy } = globalTranslation(prev, cur, W, H);
    expect(Math.abs(dx - 0.4)).toBeLessThan(0.08);
    expect(Math.abs(dy + 0.25)).toBeLessThan(0.08);
  });

  it("reports near-zero for identical frames", () => {
    const a = texture(0, 0);
    const { dx, dy } = globalTranslation(a, a.slice(), W, H);
    expect(Math.abs(dx)).toBeLessThan(1e-6);
    expect(Math.abs(dy)).toBeLessThan(1e-6);
  });

  it("recovers a tiny per-frame breathing-scale shift", () => {
    // ~0.05 px/frame is the order of real breathing motion at this probe size.
    const { dx, dy } = globalTranslation(texture(0, 0), texture(0, 0.05), W, H);
    expect(Math.abs(dx)).toBeLessThan(0.02);
    expect(Math.abs(dy - 0.05)).toBeLessThan(0.02);
  });

  it("does not blow up on a flat (gradient-free) image", () => {
    const flat = new Float64Array(W * H).fill(100);
    const { dx, dy } = globalTranslation(flat, flat.slice(), W, H);
    expect(dx).toBe(0);
    expect(dy).toBe(0);
  });
});
