import { fft } from "./fft";

export interface PulseEstimate {
  /** Smoothed beats per minute, or null while warming up / low confidence. */
  bpm: number | null;
  /** Peak prominence in the heart-rate band, 0..1-ish (>0.25 is trustworthy). */
  confidence: number;
  /** Recent detrended signal for the sparkline (newest last). */
  wave: number[];
}

const BAND_LO = 0.7; // Hz, 42 bpm
const BAND_HI = 3.0; // Hz, 180 bpm
const WINDOW = 512; // samples kept (~17 s at 30 fps)
const MIN_SAMPLES = 240; // ~8 s before we report anything
const FFT_SIZE = 1024; // zero-padded for finer bin spacing

/**
 * Remote photoplethysmography (rPPG), the classic way: blood volume changes
 * modulate skin reflectance, strongest in green. We track the mean green
 * level of the face region over time and find the dominant frequency in the
 * 0.7-3 Hz band.
 *
 * This deliberately does NOT read the magnified output — the BPM shown next
 * to the magnified video is measured independently from the raw feed, so the
 * two corroborate each other instead of one dressing up the other.
 */
export class PulseTracker {
  private values: number[] = [];
  private times: number[] = [];
  private smoothedBpm: number | null = null;

  addSample(meanGreen: number, tMs: number): void {
    this.values.push(meanGreen);
    this.times.push(tMs);
    if (this.values.length > WINDOW) {
      this.values.shift();
      this.times.shift();
    }
  }

  reset(): void {
    this.values = [];
    this.times = [];
    this.smoothedBpm = null;
  }

  estimate(): PulseEstimate {
    const n = this.values.length;
    if (n < MIN_SAMPLES) {
      return { bpm: null, confidence: 0, wave: this.detrended().slice(-160) };
    }

    const fs = this.sampleRate();
    const x = this.detrended();

    // Hamming window, zero-padded FFT.
    const re = new Float64Array(FFT_SIZE);
    const im = new Float64Array(FFT_SIZE);
    for (let i = 0; i < n; i++) {
      const w = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
      re[i] = x[i] * w;
    }
    fft(re, im);

    const binHz = fs / FFT_SIZE;
    const lo = Math.max(1, Math.ceil(BAND_LO / binHz));
    const hi = Math.min(FFT_SIZE / 2 - 1, Math.floor(BAND_HI / binHz));
    if (hi <= lo + 2) return { bpm: null, confidence: 0, wave: x.slice(-160) };

    let peak = lo;
    let peakMag = 0;
    let bandSum = 0;
    for (let k = lo; k <= hi; k++) {
      const m = Math.hypot(re[k], im[k]);
      bandSum += m;
      if (m > peakMag) {
        peakMag = m;
        peak = k;
      }
    }
    const bandMean = bandSum / (hi - lo + 1);
    const confidence = bandMean > 0 ? (peakMag - bandMean) / (peakMag + bandMean) : 0;

    // Parabolic interpolation around the peak bin for sub-bin frequency.
    const mag = (k: number) => Math.hypot(re[k], im[k]);
    const y0 = mag(peak - 1);
    const y1 = peakMag;
    const y2 = mag(peak + 1);
    const denom = y0 - 2 * y1 + y2;
    const delta = denom !== 0 ? Math.max(-0.5, Math.min(0.5, (0.5 * (y0 - y2)) / denom)) : 0;
    const freq = (peak + delta) * binHz;
    const bpm = freq * 60;

    if (confidence > 0.25) {
      this.smoothedBpm =
        this.smoothedBpm === null ? bpm : 0.85 * this.smoothedBpm + 0.15 * bpm;
    }
    return {
      bpm: this.smoothedBpm !== null ? Math.round(this.smoothedBpm) : null,
      confidence,
      wave: x.slice(-160),
    };
  }

  private sampleRate(): number {
    const t = this.times;
    if (t.length < 2) return 30;
    const span = (t[t.length - 1] - t[0]) / 1000;
    return span > 0 ? (t.length - 1) / span : 30;
  }

  /** Remove mean + linear trend (lighting drift, auto-exposure ramps). */
  private detrended(): number[] {
    const n = this.values.length;
    if (n === 0) return [];
    let sx = 0;
    let sy = 0;
    let sxx = 0;
    let sxy = 0;
    for (let i = 0; i < n; i++) {
      sx += i;
      sy += this.values[i];
      sxx += i * i;
      sxy += i * this.values[i];
    }
    const denom = n * sxx - sx * sx;
    const slope = denom !== 0 ? (n * sxy - sx * sy) / denom : 0;
    const intercept = (sy - slope * sx) / n;
    return this.values.map((v, i) => v - (intercept + slope * i));
  }
}
