import { RhythmTracker, type RhythmBand, type RhythmEstimate } from "./bpm";
import { isSkin } from "./skin";
import { globalTranslation } from "./flow";

/**
 * Turns a stream of small raw-camera frames into vital-sign estimates. Pure
 * and DOM-free: it takes pixel buffers + timestamps, so the whole frame→number
 * pipeline can be driven by a synthetic video in a test (see probe.test.ts) —
 * which is the only honest way to know breathing/pulse actually lock, since a
 * throttled background browser tab can't run the real-time path.
 *
 * Two observables, from the same frames:
 *  - **colour** — mean RGB over skin pixels → POS pulse (handled by the tracker)
 *  - **motion** — the per-frame global translation (dx, dy), foreground-weighted
 *    so the static background doesn't dilute it, fed as *velocity* (not an
 *    accumulated position, which would random-walk and bury the breathing peak)
 */
export class VitalsProbe {
  private tracker = new RhythmTracker();
  private prevLuma: Float64Array | null = null;
  private activity: Float64Array; // per-pixel EMA of |temporal change| → foreground map

  constructor(private readonly w: number, private readonly h: number) {
    this.activity = new Float64Array(w * h);
  }

  reset(): void {
    this.tracker.reset();
    this.prevLuma = null;
    this.activity.fill(0);
  }

  /** Feed one frame. `rgba` is width*height*4 bytes (canvas getImageData order). */
  process(rgba: Uint8ClampedArray, tMs: number): void {
    const { w, h } = this;
    const n = w * h;
    const luma = new Float64Array(n);

    // Skin-masked mean colour for the pulse path, plus a luma frame for flow.
    let sr = 0, sg = 0, sb = 0, sN = 0;
    let fr = 0, fg = 0, fb = 0;
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      const r = rgba[p], g = rgba[p + 1], b = rgba[p + 2];
      luma[i] = 0.299 * r + 0.587 * g + 0.114 * b;
      fr += r; fg += g; fb += b;
      if (isSkin(r, g, b)) { sr += r; sg += g; sb += b; sN++; }
    }
    const useSkin = sN >= n * 0.05;
    const cN = useSkin ? sN : n;
    const cr = (useSkin ? sr : fr) / cN;
    const cg = (useSkin ? sg : fg) / cN;
    const cb = (useSkin ? sb : fb) / cN;

    // Foreground-weighted global flow (velocity). Update the activity map from
    // this frame's temporal change, then weight the flow fit by it so only the
    // moving region (not the static background) drives the estimate.
    let dx = 0, dy = 0;
    if (this.prevLuma) {
      const act = this.activity;
      for (let i = 0; i < n; i++) {
        act[i] = 0.9 * act[i] + 0.1 * Math.abs(luma[i] - this.prevLuma[i]);
      }
      ({ dx, dy } = globalTranslation(this.prevLuma, luma, w, h, act));
    }
    this.prevLuma = luma;

    this.tracker.addSample(cr, cg, cb, dx, dy, tMs);
  }

  estimate(band: RhythmBand): RhythmEstimate {
    return this.tracker.estimate(band);
  }
}
