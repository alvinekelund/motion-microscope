import {
  RhythmTracker,
  bandHighpass,
  magnitudeSpectrum,
  FFT_SIZE,
  type RhythmBand,
  type RhythmEstimate,
} from "./bpm";
import { isSkin } from "./skin";

const WINDOW = 900; // ~30 s at 30 fps
const GX = 5; // motion-grid columns
const GY = 4; // motion-grid rows

/**
 * Turns a stream of small raw-camera frames into vital-sign estimates. Pure
 * and DOM-free, so the whole frame→number pipeline can be driven by a
 * synthetic video in a test (probe.test.ts) — the only honest way to know the
 * estimates lock, since a throttled browser tab can't run the real-time path.
 *
 * Pulse (colour, rPPG/POS) is handled by RhythmTracker. Breathing (motion) is
 * handled *here*, and deliberately not as one global shift. Breathing is
 * non-rigid — the chest expands outward while the shoulders rise — so a single
 * global translation averages the outward halves against each other and
 * cancels exactly the motion that's most visible. Instead the frame is split
 * into a grid; each block gets its own local optical-flow velocity; and the
 * breathing rate is the frequency where the blocks' power spectra *add up*.
 * Local motion accumulates instead of cancelling, the way the magnifier
 * reveals it. Each block's contribution is weighted by how much it actually
 * moved, so the static background contributes noise floor, not false peaks.
 */
export class VitalsProbe {
  private tracker = new RhythmTracker();
  private prevLuma: Float64Array | null = null;
  private activity: Float64Array; // per-pixel EMA of |temporal change|
  private blockOf: Int32Array; // pixel index → block index
  private vx: number[][]; // per-block x-velocity history
  private vy: number[][]; // per-block y-velocity history
  private times: number[] = [];

  constructor(private readonly w: number, private readonly h: number) {
    this.activity = new Float64Array(w * h);
    this.blockOf = new Int32Array(w * h);
    for (let y = 0; y < h; y++) {
      const by = Math.min(GY - 1, Math.floor((y / h) * GY));
      for (let x = 0; x < w; x++) {
        const bx = Math.min(GX - 1, Math.floor((x / w) * GX));
        this.blockOf[y * w + x] = by * GX + bx;
      }
    }
    this.vx = Array.from({ length: GX * GY }, () => []);
    this.vy = Array.from({ length: GX * GY }, () => []);
  }

  reset(): void {
    this.tracker.reset();
    this.prevLuma = null;
    this.activity.fill(0);
    this.times = [];
    for (let b = 0; b < GX * GY; b++) {
      this.vx[b] = [];
      this.vy[b] = [];
    }
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
    this.tracker.addSample(
      (useSkin ? sr : fr) / cN,
      (useSkin ? sg : fg) / cN,
      (useSkin ? sb : fb) / cN,
      0, 0, tMs
    );

    // Per-block local optical flow, weighted by a per-pixel activity map so
    // the static background can't anchor each block's fit at zero.
    const B = GX * GY;
    const sxx = new Float64Array(B), sxy = new Float64Array(B), syy = new Float64Array(B);
    const sxt = new Float64Array(B), syt = new Float64Array(B);
    if (this.prevLuma) {
      const prev = this.prevLuma;
      const act = this.activity;
      for (let i = 0; i < n; i++) act[i] = 0.9 * act[i] + 0.1 * Math.abs(luma[i] - prev[i]);
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const i = y * w + x;
          const wi = act[i];
          if (wi <= 0) continue;
          const gx = (luma[i + 1] - luma[i - 1]) * 0.5;
          const gy = (luma[i + w] - luma[i - w]) * 0.5;
          const gt = luma[i] - prev[i];
          const b = this.blockOf[i];
          sxx[b] += wi * gx * gx;
          sxy[b] += wi * gx * gy;
          syy[b] += wi * gy * gy;
          sxt[b] += wi * gx * gt;
          syt[b] += wi * gy * gt;
        }
      }
    }
    for (let b = 0; b < B; b++) {
      const det = sxx[b] * syy[b] - sxy[b] * sxy[b];
      let dx = 0, dy = 0;
      if (Math.abs(det) > 1e-9) {
        dx = (sxy[b] * syt[b] - syy[b] * sxt[b]) / det;
        dy = (sxy[b] * sxt[b] - sxx[b] * syt[b]) / det;
      }
      this.vx[b].push(dx);
      this.vy[b].push(dy);
      if (this.vx[b].length > WINDOW) { this.vx[b].shift(); this.vy[b].shift(); }
    }
    this.times.push(tMs);
    if (this.times.length > WINDOW) this.times.shift();
    this.prevLuma = luma;
  }

  estimate(band: RhythmBand): RhythmEstimate {
    if (band.kind === "pulse") return this.tracker.estimate(band);
    return this.estimateMotion(band);
  }

  private sampleRate(): number {
    const t = this.times;
    if (t.length < 2) return 30;
    const span = (t[t.length - 1] - t[0]) / 1000;
    return span > 0 ? (t.length - 1) / span : 30;
  }

  /** Breathing/vibration from the per-block velocity field. */
  private estimateMotion(band: RhythmBand): RhythmEstimate {
    const unit = band.kind === "breathing" ? "breaths/min" : "Hz";
    const label = band.kind === "breathing" ? "BREATHS" : "FREQUENCY";
    const waveSeconds = band.kind === "breathing" ? 24 : 7;
    const fs = this.sampleRate();
    const n = this.times.length;

    // Each block's velocity, high-passed, with its motion energy as a weight.
    const B = GX * GY;
    const series: { hp: number[]; energy: number }[] = [];
    for (let b = 0; b < B; b++) {
      for (const raw of [this.vx[b], this.vy[b]]) {
        const hp = bandHighpass(raw, fs, band.lo);
        let e = 0;
        for (const v of hp) e += v * v;
        series.push({ hp, energy: e });
      }
    }
    const totalEnergy = series.reduce((s, x) => s + x.energy, 0) || 1e-9;

    // Display wave: the single most active block-axis, so the chart shows a
    // real breathing trace rather than a global average.
    const best = series.reduce((a, x) => (x.energy > a.energy ? x : a), series[0]);
    const wave = best.hp.slice(-Math.round(waveSeconds * fs));
    const blank: RhythmEstimate = { value: null, unit, label, freqHz: null, confidence: 0, wave, waveSeconds };

    const minSamples = Math.min(WINDOW * 0.5, Math.max(150, Math.round((3 * fs) / Math.max(band.lo, 0.2))));
    if (n < minSamples) return blank;

    // Sum block spectra weighted by motion energy: periodic breathing adds up
    // across blocks; static-region noise (low energy) is suppressed.
    const spec = new Float64Array(FFT_SIZE / 2);
    for (const s of series) {
      if (s.energy <= 0) continue;
      const m = magnitudeSpectrum(s.hp, n);
      const wgt = s.energy / totalEnergy;
      for (let k = 0; k < spec.length; k++) spec[k] += wgt * m[k];
    }

    const binHz = fs / FFT_SIZE;
    const lo = Math.max(1, Math.ceil(band.lo / binHz));
    const hi = Math.min(FFT_SIZE / 2 - 1, Math.floor(Math.min(band.hi, 0.45 * fs) / binHz));
    if (hi <= lo + 2) return blank;

    let peak = lo, peakMag = 0, bandSum = 0;
    for (let k = lo; k <= hi; k++) {
      bandSum += spec[k];
      if (spec[k] > peakMag) { peakMag = spec[k]; peak = k; }
    }
    const bandMean = bandSum / (hi - lo + 1);
    const confidence = bandMean > 0 ? (peakMag - bandMean) / (peakMag + bandMean) : 0;

    const y0 = spec[peak - 1], y2 = spec[peak + 1];
    const denom = y0 - 2 * peakMag + y2;
    const delta = denom !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / denom)) : 0;
    const freq = (peak + delta) * binHz;
    const raw = band.kind === "vibration" ? freq : freq * 60;

    if (confidence > 0.25) {
      this.smoothed = this.smoothed === null ? raw : 0.85 * this.smoothed + 0.15 * raw;
    }
    const sm = this.smoothed;
    const value = sm === null ? null : band.kind === "vibration" ? Math.round(sm * 10) / 10 : Math.round(sm);
    const freqHz = sm === null ? null : band.kind === "vibration" ? sm : sm / 60;
    return { value, unit, label, freqHz, confidence, wave, waveSeconds };
  }

  private smoothed: number | null = null;
}
