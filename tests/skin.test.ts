import { describe, expect, it } from "vitest";
import { isSkin } from "../src/pulse/skin";

describe("isSkin", () => {
  it("accepts a range of warm skin tones", () => {
    expect(isSkin(255, 224, 196)).toBe(true); // light
    expect(isSkin(198, 134, 96)).toBe(true); // medium
    expect(isSkin(141, 85, 56)).toBe(true); // deep but warm
  });

  it("rejects neutral gray (background, walls)", () => {
    expect(isSkin(120, 120, 120)).toBe(false);
    expect(isSkin(200, 200, 200)).toBe(false);
  });

  it("rejects cool / non-skin colours", () => {
    expect(isSkin(50, 80, 200)).toBe(false); // blue
    expect(isSkin(40, 160, 60)).toBe(false); // green
    expect(isSkin(20, 18, 15)).toBe(false); // near-black hair
  });

  it("rejects colours where red doesn't dominate", () => {
    expect(isSkin(120, 140, 100)).toBe(false);
  });
});
