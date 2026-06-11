import type { MagParams } from "./gpu/pipeline";
import type { PulseEstimate } from "./pulse/bpm";

const FREQ_MIN = 0.2;
const FREQ_MAX = 12;

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
    bpmHint: document.getElementById("bpm-hint") as HTMLElement,
    spark: document.getElementById("pulse-spark") as HTMLCanvasElement,
    perf: document.getElementById("perf") as HTMLElement,
  };

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
  }

  get canvas(): HTMLCanvasElement {
    return this.el.canvas;
  }

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

  updatePulse(est: PulseEstimate): void {
    if (est.bpm !== null && est.confidence > 0.2) {
      this.el.bpmValue.textContent = String(est.bpm);
      this.el.bpmHint.textContent = `signal confidence ${(est.confidence * 100).toFixed(0)}%`;
    } else {
      this.el.bpmValue.textContent = "—";
      this.el.bpmHint.textContent =
        est.wave.length < 160 ? "measuring… hold still" : "weak signal — more light helps";
    }
    this.drawSparkline(est.wave);
  }

  updatePerf(fps: number): void {
    this.el.perf.textContent = `${fps.toFixed(0)} fps`;
  }

  private drawSparkline(wave: number[]): void {
    const ctx = this.el.spark.getContext("2d");
    if (!ctx) return;
    const { width: w, height: h } = this.el.spark;
    ctx.clearRect(0, 0, w, h);
    if (wave.length < 8) return;
    let max = 1e-6;
    for (const v of wave) max = Math.max(max, Math.abs(v));
    ctx.strokeStyle = "#ff4d5e";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    wave.forEach((v, i) => {
      const x = (i / (wave.length - 1)) * w;
      const y = h / 2 - (v / max) * (h / 2 - 4);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}
