import type { MagParams } from "./gpu/pipeline";
import type { RhythmEstimate } from "./pulse/bpm";

// Slider spans only the two bands that actually show on a face: breathing
// (~0.2 Hz) through the cardiac range (up to ~210 bpm).
const FREQ_MIN = 0.2;
const FREQ_MAX = 3.5;

/** Map slider position 0..1000 onto a log frequency scale. */
function sliderToFreq(v: number): number {
  return FREQ_MIN * Math.pow(FREQ_MAX / FREQ_MIN, v / 1000);
}
function freqToSlider(f: number): number {
  return Math.round((1000 * Math.log(f / FREQ_MIN)) / Math.log(FREQ_MAX / FREQ_MIN));
}

export class UI {
  readonly params: MagParams = {
    alpha: 15,
    centerFreq: 1.2,
    denoise: 0.6,
    split: 0.0,
  };

  private el = {
    panel: document.getElementById("panel") as HTMLElement,
    startScreen: document.getElementById("start-screen") as HTMLElement,
    startBtn: document.getElementById("start-btn") as HTMLButtonElement,
    startError: document.getElementById("start-error") as HTMLElement,
    stage: document.getElementById("stage") as HTMLElement,
    canvas: document.getElementById("gpu-canvas") as HTMLCanvasElement,
    splitHandle: document.getElementById("split-handle") as HTMLElement,
    alpha: document.getElementById("alpha") as HTMLInputElement,
    alphaOut: document.getElementById("alpha-out") as HTMLElement,
    freq: document.getElementById("freq") as HTMLInputElement,
    freqOut: document.getElementById("freq-out") as HTMLElement,
    denoise: document.getElementById("denoise") as HTMLInputElement,
    denoiseOut: document.getElementById("denoise-out") as HTMLElement,
    presets: Array.from(document.querySelectorAll<HTMLButtonElement>(".presets button")),
    bpmValue: document.getElementById("bpm-value") as HTMLElement,
    bpmLabel: document.getElementById("bpm-label") as HTMLElement,
    bpmHint: document.getElementById("bpm-hint") as HTMLElement,
    wave: document.getElementById("wave") as HTMLCanvasElement,
    waveSpan: document.getElementById("wave-span") as HTMLElement,
    beatRing: document.getElementById("beat-ring") as HTMLElement,
    beatPulse: document.getElementById("beat-pulse") as HTMLElement,
    lock: document.getElementById("lock") as HTMLInputElement,
    explainToggle: document.getElementById("explain-toggle") as HTMLButtonElement,
    explain: document.getElementById("explain") as HTMLElement,
    perf: document.getElementById("perf") as HTMLElement,
  };

  // Beat ring state — a continuous oscillator at the detected rate.
  private beatHz = 0;
  private beatConf = 0;
  private beatPhase = 0;
  private beatLast = 0;

  // Waveform display state.
  private waveScale = 1e-6; // smoothed autoscale so the trace breathes, not jumps

  constructor() {
    this.el.alpha.addEventListener("input", () => {
      this.params.alpha = Number(this.el.alpha.value);
      this.el.alphaOut.textContent = `×${this.params.alpha}`;
    });

    this.el.freq.value = String(freqToSlider(this.params.centerFreq));
    this.el.freq.addEventListener("input", () => {
      this.setFreq(sliderToFreq(Number(this.el.freq.value)));
    });
    for (const btn of this.el.presets) {
      btn.addEventListener("click", () => {
        const f = Number(btn.dataset.freq);
        this.el.freq.value = String(freqToSlider(f));
        this.setFreq(f);
      });
    }

    this.el.denoise.addEventListener("input", () => {
      this.params.denoise = Number(this.el.denoise.value) / 100;
      this.el.denoiseOut.textContent = this.params.denoise.toFixed(2);
    });

    // When locking is on, the magnifier is driven by the detected rate, so
    // the manual frequency controls are disabled to make that explicit.
    this.el.lock.addEventListener("change", () => {
      const on = this.el.lock.checked;
      this.el.freq.disabled = on;
      for (const btn of this.el.presets) btn.disabled = on;
    });

    this.el.explainToggle.addEventListener("click", () => {
      this.el.explain.hidden = !this.el.explain.hidden;
    });

    this.bindSplitDrag();
  }

  private setFreq(f: number): void {
    this.params.centerFreq = f;
    this.el.freqOut.textContent = `${f.toFixed(f < 1 ? 2 : 1)} Hz`;
    for (const btn of this.el.presets) {
      btn.classList.toggle("active", Math.abs(Number(btn.dataset.freq) - f) < 0.01);
    }
  }

  onStart(handler: () => void): void {
    this.el.startBtn.addEventListener("click", handler);
  }

  showStartError(message: string): void {
    this.el.startError.textContent = message;
    this.el.startError.hidden = false;
  }

  enterLiveMode(): void {
    this.el.startScreen.style.display = "none";
    this.el.panel.hidden = false;
    this.el.splitHandle.style.display = "block";
    this.positionSplitHandle();
    requestAnimationFrame(this.animateBeat);
  }

  get canvas(): HTMLCanvasElement {
    return this.el.canvas;
  }

  /**
   * The ring around the big number swells once per detected beat/breath — a
   * continuous oscillator, independent of the 1 Hz vitals tick so the motion
   * is smooth. It's *always* alive while the tool runs — a calm idle pulse
   * when there's no lock yet (so you can see it's working) that grows into a
   * strong pulse at your actual rate once a confident lock lands.
   */
  private animateBeat = (now: number): void => {
    const dt = this.beatLast ? Math.min((now - this.beatLast) / 1000, 0.1) : 0;
    this.beatLast = now;
    const locked = this.beatHz > 0 && this.beatConf > 0.3;
    // Idle ~0.25 Hz heartbeat-like cadence until we lock onto the real rate.
    const hz = locked ? this.beatHz : 0.4;
    this.beatPhase += 2 * Math.PI * hz * dt;
    const s = 0.5 + 0.5 * Math.sin(this.beatPhase); // 0..1, peaks once per cycle
    const grow = locked ? 0.16 : 0.07;
    const baseOpacity = locked ? 0.35 : 0.16;
    const swing = locked ? 0.55 : 0.22;
    this.el.beatPulse.style.opacity = (baseOpacity + swing * s).toFixed(3);
    this.el.beatPulse.style.transform = `scale(${(1 + grow * s).toFixed(3)})`;
    this.el.beatRing.classList.toggle("locked", locked);
    requestAnimationFrame(this.animateBeat);
  };

  private bindSplitDrag(): void {
    const handle = this.el.splitHandle;
    const onMove = (clientX: number) => {
      const rect = this.el.canvas.getBoundingClientRect();
      this.params.split = Math.min(0.95, Math.max(0, (clientX - rect.left) / rect.width));
      this.positionSplitHandle();
    };
    handle.addEventListener("pointerdown", (e) => {
      handle.setPointerCapture(e.pointerId);
      const move = (ev: PointerEvent) => onMove(ev.clientX);
      const up = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
    window.addEventListener("resize", () => this.positionSplitHandle());
  }

  positionSplitHandle(): void {
    const rect = this.el.canvas.getBoundingClientRect();
    const stageRect = this.el.stage.getBoundingClientRect();
    const x = rect.left - stageRect.left + this.params.split * rect.width;
    this.el.splitHandle.style.left = `${x}px`;
  }

  updateVitals(est: RhythmEstimate): void {
    this.el.bpmLabel.textContent = est.unit;
    if (est.value !== null && est.confidence > 0.2) {
      this.el.bpmValue.textContent = String(est.value);
      this.el.bpmHint.textContent = `signal confidence ${(est.confidence * 100).toFixed(0)}%`;
    } else {
      this.el.bpmValue.textContent = "—";
      this.el.bpmHint.textContent =
        est.wave.length < 60 ? "measuring… hold still" : "locking on… stay still, face the light";
    }
    this.el.waveSpan.textContent = `${est.waveSeconds}s window`;
    this.drawWave(est.wave);

    // Feed the beat ring: pulse at the detected rate when confident.
    this.beatHz = est.freqHz ?? 0;
    this.beatConf = est.confidence;

    // Auto-tune: steer the magnifier to the rate we just measured. Safe
    // because the detector never reads the magnified output — no feedback.
    if (this.el.lock.checked && est.freqHz !== null && est.confidence > 0.3) {
      const f = Math.min(FREQ_MAX, Math.max(FREQ_MIN, est.freqHz));
      this.el.freq.value = String(freqToSlider(f));
      this.setFreq(f);
    }
  }

  updatePerf(fps: number): void {
    this.el.perf.textContent = `${fps.toFixed(0)} fps`;
  }

  /**
   * Big, readable trace: spans several full cycles (the window length comes
   * from the band), auto-scaled to fill the height with a smoothed gain so
   * the curve doesn't jump, with a midline and a subtle grid.
   */
  private drawWave(wave: number[]): void {
    const ctx = this.el.wave.getContext("2d");
    if (!ctx) return;
    const w = this.el.wave.width;
    const h = this.el.wave.height;
    ctx.clearRect(0, 0, w, h);

    // Grid: midline + quarter lines.
    ctx.strokeStyle = "#1b232c";
    ctx.lineWidth = 1;
    for (const fy of [0.25, 0.5, 0.75]) {
      ctx.beginPath();
      ctx.moveTo(0, h * fy + 0.5);
      ctx.lineTo(w, h * fy + 0.5);
      ctx.stroke();
    }

    if (wave.length < 16) return;

    // Light moving-average smoothing for display only — the estimator sees
    // the raw signal; this just makes the trace read like a breath/pulse
    // instead of a noisy velocity record.
    const K = 5;
    const sm = new Array<number>(wave.length);
    for (let i = 0; i < wave.length; i++) {
      let acc = 0;
      let cnt = 0;
      for (let k = -K; k <= K; k++) {
        const j = i + k;
        if (j >= 0 && j < wave.length) { acc += wave[j]; cnt++; }
      }
      sm[i] = acc / cnt;
    }
    wave = sm;

    // Smoothed autoscale: track the 98th-percentile amplitude so one spike
    // doesn't flatten the rest, easing the gain between frames.
    const sorted = wave.map(Math.abs).sort((a, b) => a - b);
    const p98 = sorted[Math.floor(sorted.length * 0.98)] || 1e-6;
    this.waveScale = 0.7 * this.waveScale + 0.3 * Math.max(p98, 1e-6);
    const gain = (h * 0.42) / this.waveScale;

    ctx.strokeStyle = "#ff4d5e";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < wave.length; i++) {
      const x = (i / (wave.length - 1)) * w;
      const y = h / 2 - Math.max(-h * 0.46, Math.min(h * 0.46, wave[i] * gain));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}
