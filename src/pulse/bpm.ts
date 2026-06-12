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
  /** Smoothed detected frequency in Hz, or null — drives the "lock" auto-tune. */
  freqHz: number | null;
  /** Peak prominence in the search band, 0..1. >0.25 is trustworthy. */
  confidence: number;
  /** Recent signal for the waveform display (newest last). */
  wave: number[];
  /** How many seconds `wave` spans — band-aware, sized to show several cycles. */
  waveSeconds: number;
}

const WINDOW = 900; // samples kept (~30 s at 30 fps) — enough for slow breathing
export const FFT_SIZE = 2048; // zero-padded: ~0.015 Hz bin spacing at 30 fps

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
 * Reads a periodic vital sign from the face region over a sliding ~30 s
 * window. The measurement band follows the UI's frequency slider, so the
 * number beside the video always describes the same motion the magnifier is
 * amplifying. Two kinds of observable are tracked, because pulse and
 * breathing are physically different signals:
 *
 *  - **pulse** is a *colour* change — blood volume modulates skin reflectance
 *    by a fraction of a percent, with no visible movement. It uses POS
 *    (plane-orthogonal-to-skin, Wang et al. 2017): per-channel temporal means
 *    are divided out and the result is projected onto two axes orthogonal to
 *    the skin-tone vector, h = S1 + (σ(S1)/σ(S2))·S2. This cancels the
 *    brightness flicker that raw-green rPPG mistakes for signal.
 *
 *  - **breathing / vibration** are *motion* — the head and shoulders visibly
 *    translate. Crucially, the mean brightness of a fixed crop barely changes
 *    when content merely slides across it, so mean luminance is the wrong
 *    observable (it is weakest exactly when the motion is largest). Instead we
 *    track the brightness *centroid* of the region, which moves linearly with
 *    displacement, on both axes, and combine their power spectra so the
 *    direction of movement doesn't matter.
 *
 * Either way this path never touches the magnified output: the displayed
 * number and the visible pulsing are independent measurements.
 */
export class RhythmTracker {
  private r: number[] = [];
  private g: number[] = [];
  private b: number[] = [];
  private cx: number[] = [];
  private cy: number[] = [];
  private times: number[] = [];
  private smoothed = new Map<BandKind, number>();

  /**
   * @param r,g,b   spatial-mean colour of the face region (0..255)
   * @param cx,cy   brightness centroid of the region, in pixels — the motion
   *                observable for the breathing/vibration bands
   */
  addSample(r: number, g: number, b: number, cx: number, cy: number, tMs: number): void {
    this.r.push(r);
    this.g.push(g);
    this.b.push(b);
    this.cx.push(cx);
    this.cy.push(cy);
    this.times.push(tMs);
    if (this.r.length > WINDOW) {
      this.r.shift();
      this.g.shift();
      this.b.shift();
      this.cx.shift();
      this.cy.shift();
      this.times.shift();
    }
  }

  reset(): void {
    this.r = [];
    this.g = [];
    this.b = [];
    this.cx = [];
    this.cy = [];
    this.times = [];
    this.smoothed.clear();
  }

  estimate(band: RhythmBand): RhythmEstimate {
    const unit = band.kind === "breathing" ? "breaths/min" : band.kind === "pulse" ? "BPM" : "Hz";
    const label = band.kind === "breathing" ? "BREATHS" : band.kind === "pulse" ? "BPM" : "FREQUENCY";

    const n = this.r.length;
    const fs = this.sampleRate();
    // Source signal(s): pulse is one colour signal; motion sums both centroid
    // axes. Each is high-passed at the band's low edge — subtracting a
    // moving-average baseline ~1/band.lo seconds wide removes slow lighting
    // drift (including the non-linear auto-exposure ramps a single linear
    // detrend leaves behind) without touching the signal in-band.
    const rawSources = band.kind === "pulse" ? [this.posSignal()] : [this.cx.slice(), this.cy.slice()];
    const sources = rawSources.map((s) => bandHighpass(s, fs, band.lo));

    // Waveform for display: span several cycles of the band's typical rate
    // (a breath is ~4-5 s, so a fixed short window showed less than one wave),
    // and pick the *dominant* source — for vertical breathing the y-axis
    // carries the signal while x is noise, so always showing source 0 was wrong.
    const waveSeconds = band.kind === "breathing" ? 24 : 7;
    const rms = (a: number[]) => a.reduce((s, v) => s + v * v, 0);
    const display = sources.length === 1
      ? sources[0]
      : rms(sources[0]) >= rms(sources[1]) ? sources[0] : sources[1];
    const wave = display.slice(-Math.round(waveSeconds * fs));
    const blank: RhythmEstimate = { value: null, unit, label, freqHz: null, confidence: 0, wave, waveSeconds };
    if (n === 0) return blank;

    // Warm-up: enough samples for ~3 cycles of a *typical* rate in the band.
    // Gating on the absolute band floor (e.g. 0.1 Hz) would stall the readout
    // for half a minute; real breathing sits nearer 0.2 Hz, so warm up to that.
    const typicalLo = Math.max(band.lo, band.kind === "breathing" ? 0.2 : band.lo);
    const minSamples = Math.min(WINDOW * 0.5, Math.max(150, Math.round((3 * fs) / typicalLo)));
    if (n < minSamples) return blank;

    // Combined magnitude spectrum across all source signals.
    const spec = new Float64Array(FFT_SIZE / 2);
    for (const s of sources) {
      const m = magnitudeSpectrum(s, n);
      for (let k = 0; k < spec.length; k++) spec[k] += m[k];
    }

    const binHz = fs / FFT_SIZE;
    const lo = Math.max(1, Math.ceil(band.lo / binHz));
    const hi = Math.min(FFT_SIZE / 2 - 1, Math.floor(Math.min(band.hi, 0.45 * fs) / binHz));
    if (hi <= lo + 2) return blank;

    let peak = lo;
    let peakMag = 0;
    let bandSum = 0;
    for (let k = lo; k <= hi; k++) {
      bandSum += spec[k];
      if (spec[k] > peakMag) {
        peakMag = spec[k];
        peak = k;
      }
    }
    const bandMean = bandSum / (hi - lo + 1);
    const confidence = bandMean > 0 ? (peakMag - bandMean) / (peakMag + bandMean) : 0;

    // Parabolic interpolation for sub-bin frequency.
    const y0 = spec[peak - 1];
    const y2 = spec[peak + 1];
    const denom = y0 - 2 * peakMag + y2;
    const delta = denom !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / denom)) : 0;
    const freq = (peak + delta) * binHz;
    const raw = band.kind === "vibration" ? freq : freq * 60;

    if (confidence > 0.25) {
      const prev = this.smoothed.get(band.kind);
      this.smoothed.set(band.kind, prev === undefined ? raw : 0.85 * prev + 0.15 * raw);
    }
    const sm = this.smoothed.get(band.kind);
    const value =
      sm === undefined ? null : band.kind === "vibration" ? Math.round(sm * 10) / 10 : Math.round(sm);
    const freqHz = sm === undefined ? null : band.kind === "vibration" ? sm : sm / 60;
    return { value, unit, label, freqHz, confidence, wave, waveSeconds };
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
}

/** Hamming-windowed, zero-padded magnitude spectrum (half, length FFT_SIZE/2). */
export function magnitudeSpectrum(signal: number[], n: number): Float64Array {
  const re = new Float64Array(FFT_SIZE);
  const im = new Float64Array(FFT_SIZE);
  for (let i = 0; i < n; i++) {
    const w = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
    re[i] = signal[i] * w;
  }
  fft(re, im);
  const out = new Float64Array(FFT_SIZE / 2);
  for (let k = 0; k < out.length; k++) out[k] = Math.hypot(re[k], im[k]);
  return out;
}

/**
 * Band-aware high-pass: subtract a centered moving-average baseline whose
 * width is ~1/fLo seconds. This flattens any drift slower than the band's low
 * edge — auto-exposure ramps, someone dimming a lamp, slow head reposition —
 * including non-linear drift a single linear fit can't follow, while leaving
 * everything in-band untouched. Prefix sums keep it O(n).
 */
export function bandHighpass(values: number[], fs: number, fLo: number): number[] {
  const n = values.length;
  if (n === 0) return [];
  const W = Math.max(3, Math.round(fs / Math.max(fLo, 1e-3)));
  const half = W >> 1;
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + values[i];
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - half);
    const b = Math.min(n, i + half + 1);
    out[i] = values[i] - (pre[b] - pre[a]) / (b - a);
  }
  return out;
}
