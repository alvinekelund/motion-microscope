import { fft } from "./fft";

export type BandKind = "breathing" | "pulse" | "vibration";

export interface RhythmBand {
  kind: BandKind;
  /** Frequency search range, Hz. */
  lo: number;
  hi: number;
}

export interface RhythmEstimate {
  /** Rate in the band's display unit, or null while warming up / low confidence. */
  value: number | null;
  /** Display unit for `value`. */
  unit: "breaths/min" | "BPM" | "Hz";
  /** Big label shown under the number. */
  label: string;
  /** Peak prominence in the search band, 0..1. >0.25 is trustworthy. */
  confidence: number;
  /** Recent signal for the sparkline (newest last). */
  wave: number[];
}

const WINDOW = 900; // samples kept (~30 s at 30 fps) — enough for slow breathing
const FFT_SIZE = 2048; // zero-padded: ~0.015 Hz bin spacing at 30 fps

/**
 * Pick the measurement band from the slider's center frequency. The readout
 * tracks whatever the visitor is looking at: slow chest motion reads out in
 * breaths/min, the cardiac band in BPM, and anything faster as a raw
 * frequency in Hz.
 */
export function bandForCenterFreq(f: number): RhythmBand {
  if (f < 0.5) return { kind: "breathing", lo: 0.1, hi: 0.6 };
  if (f < 4.0) return { kind: "pulse", lo: 0.7, hi: 3.0 };
  return { kind: "vibration", lo: 3.0, hi: 14.0 };
}

/**
 * Reads a periodic vital sign from the spatial-mean RGB of the face region,
 * over a sliding ~30 s window. The measurement band follows the UI's
 * frequency slider, so the number beside the video always describes the same
 * motion the magnifier is amplifying.
 *
 * Two source signals are derived from the same RGB history:
 *
 *  - **pulse** uses POS (plane-orthogonal-to-skin, Wang et al. 2017): the
 *    per-channel temporal means are divided out and the result is projected
 *    onto two axes orthogonal to the skin-tone vector, then combined as
 *    h = S1 + (σ(S1)/σ(S2))·S2. This cancels the brightness flicker that
 *    raw-green rPPG mistakes for signal, and is markedly more accurate under
 *    ordinary webcam lighting.
 *  - **breathing / vibration** are motion-driven, so they read the mean
 *    luminance instead — periodic shading change as the subject moves.
 *
 * Either way this path never touches the magnified output: the displayed
 * number and the visible pulsing are independent measurements that
 * corroborate each other.
 */
export class RhythmTracker {
  private r: number[] = [];
  private g: number[] = [];
  private b: number[] = [];
  private times: number[] = [];
  private smoothed = new Map<BandKind, number>();

  addSample(r: number, g: number, b: number, tMs: number): void {
    this.r.push(r);
    this.g.push(g);
    this.b.push(b);
    this.times.push(tMs);
    if (this.r.length > WINDOW) {
      this.r.shift();
      this.g.shift();
      this.b.shift();
      this.times.shift();
    }
  }

  reset(): void {
    this.r = [];
    this.g = [];
    this.b = [];
    this.times = [];
    this.smoothed.clear();
  }

  estimate(band: RhythmBand): RhythmEstimate {
    const unit = band.kind === "breathing" ? "breaths/min" : band.kind === "pulse" ? "BPM" : "Hz";
    const label = band.kind === "breathing" ? "BREATHS" : band.kind === "pulse" ? "BPM" : "FREQUENCY";
    const blank = (wave: number[]): RhythmEstimate => ({ value: null, unit, label, confidence: 0, wave });

    const n = this.r.length;
    const signal = band.kind === "pulse" ? this.posSignal() : this.lumaSignal();

    // Warm-up: need a few cycles of the slowest frequency in the band.
    const fs = this.sampleRate();
    const minSamples = Math.min(WINDOW * 0.6, Math.max(150, Math.round((4 * fs) / band.lo)));
    if (n < minSamples) return blank(signal.slice(-160));

    const re = new Float64Array(FFT_SIZE);
    const im = new Float64Array(FFT_SIZE);
    for (let i = 0; i < n; i++) {
      const w = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
      re[i] = signal[i] * w;
    }
    fft(re, im);

    const binHz = fs / FFT_SIZE;
    const lo = Math.max(1, Math.ceil(band.lo / binHz));
    const hi = Math.min(FFT_SIZE / 2 - 1, Math.floor(Math.min(band.hi, 0.45 * fs) / binHz));
    if (hi <= lo + 2) return blank(signal.slice(-160));

    const mag = (k: number) => Math.hypot(re[k], im[k]);
    let peak = lo;
    let peakMag = 0;
    let bandSum = 0;
    for (let k = lo; k <= hi; k++) {
      const m = mag(k);
      bandSum += m;
      if (m > peakMag) {
        peakMag = m;
        peak = k;
      }
    }
    const bandMean = bandSum / (hi - lo + 1);
    const confidence = bandMean > 0 ? (peakMag - bandMean) / (peakMag + bandMean) : 0;

    // Parabolic interpolation for sub-bin frequency.
    const y0 = mag(peak - 1);
    const y2 = mag(peak + 1);
    const denom = y0 - 2 * peakMag + y2;
    const delta = denom !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / denom)) : 0;
    const freq = (peak + delta) * binHz;
    const raw = band.kind === "vibration" ? freq : freq * 60;

    if (confidence > 0.25) {
      const prev = this.smoothed.get(band.kind);
      this.smoothed.set(band.kind, prev === undefined ? raw : 0.85 * prev + 0.15 * raw);
    }
    const sm = this.smoothed.get(band.kind);
    const value = sm === undefined ? null : band.kind === "vibration" ? Math.round(sm * 10) / 10 : Math.round(sm);
    return { value, unit, label, confidence, wave: signal.slice(-160) };
  }

  private sampleRate(): number {
    const t = this.times;
    if (t.length < 2) return 30;
    const span = (t[t.length - 1] - t[0]) / 1000;
    return span > 0 ? (t.length - 1) / span : 30;
  }

  /** POS chrominance pulse signal over the current window (Wang et al. 2017). */
  private posSignal(): number[] {
    const n = this.r.length;
    if (n === 0) return [];
    const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / n;
    const mr = mean(this.r) || 1;
    const mg = mean(this.g) || 1;
    const mb = mean(this.b) || 1;

    const s1 = new Array<number>(n);
    const s2 = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      const rn = this.r[i] / mr;
      const gn = this.g[i] / mg;
      const bn = this.b[i] / mb;
      s1[i] = gn - bn;
      s2[i] = -2 * rn + gn + bn;
    }
    const std = (a: number[]) => {
      const m = mean(a);
      return Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / n) || 1e-9;
    };
    const alpha = std(s1) / std(s2);
    const h = s1.map((v, i) => v + alpha * s2[i]);
    const hm = mean(h);
    return h.map((v) => v - hm);
  }

  /** Linearly detrended mean luminance — the motion-driven source. */
  private lumaSignal(): number[] {
    const n = this.r.length;
    if (n === 0) return [];
    const luma = new Array<number>(n);
    for (let i = 0; i < n; i++) luma[i] = 0.299 * this.r[i] + 0.587 * this.g[i] + 0.114 * this.b[i];
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) {
      sx += i;
      sy += luma[i];
      sxx += i * i;
      sxy += i * luma[i];
    }
    const denom = n * sxx - sx * sx;
    const slope = denom !== 0 ? (n * sxy - sx * sy) / denom : 0;
    const intercept = (sy - slope * sx) / n;
    return luma.map((v, i) => v - (intercept + slope * i));
  }
}
