import { initWebGPU } from "./gpu/context";
import { MotionMicroscope } from "./gpu/pipeline";
import { startCamera } from "./camera";
import { RhythmTracker, bandForCenterFreq } from "./pulse/bpm";
import { isSkin } from "./pulse/skin";
import { globalTranslation } from "./pulse/flow";
import { UI } from "./ui";

const ui = new UI();

ui.onStart(async () => {
  try {
    await run();
  } catch (err) {
    ui.showStartError(err instanceof Error ? err.message : String(err));
  }
});

async function run(): Promise<void> {
  const gpu = await initWebGPU(ui.canvas);
  const video = await startCamera();

  ui.canvas.width = video.videoWidth;
  ui.canvas.height = video.videoHeight;

  const scope = await MotionMicroscope.create(gpu, video);
  ui.enterLiveMode();

  // --- independent vitals path (never reads the magnified output) ---------
  // Sample the *raw* feed at a tiny resolution. Two observables come out of
  // the same probe: mean colour over skin pixels drives the POS pulse
  // estimate, and the global optical-flow translation of the frame drives the
  // breathing (motion) estimate — the bulk movement the magnifier makes
  // visible, measured directly rather than via a noisy centroid.
  const probe = document.createElement("canvas");
  const PW = 64;
  const PH = 48;
  const SX = 2, SY = 2, SW = PW - 4, SH = PH - 4; // scan region (2px border trimmed)
  probe.width = PW;
  probe.height = PH;
  const probeCtx = probe.getContext("2d", { willReadFrequently: true })!;
  const tracker = new RhythmTracker();

  let prevLuma: Float64Array | null = null;
  let posX = 0; // accumulated flow — the band high-pass removes its drift
  let posY = 0;

  const samplePulse = (tMs: number) => {
    probeCtx.drawImage(video, 0, 0, PW, PH);
    const region = probeCtx.getImageData(SX, SY, SW, SH).data;

    // One pass: skin-masked colour (for pulse) plus a luma frame (for flow).
    const luma = new Float64Array(SW * SH);
    let sr = 0, sg = 0, sb = 0, sN = 0;
    let fr = 0, fg = 0, fb = 0;
    for (let p = 0, j = 0; j < SW * SH; j++, p += 4) {
      const pr = region[p], pg = region[p + 1], pb = region[p + 2];
      luma[j] = 0.299 * pr + 0.587 * pg + 0.114 * pb;
      fr += pr; fg += pg; fb += pb;
      if (isSkin(pr, pg, pb)) { sr += pr; sg += pg; sb += pb; sN++; }
    }

    // Colour: prefer the skin mask, fall back to the whole region if too little
    // skin is found so we never average a near-empty mask.
    const total = SW * SH;
    const useSkin = sN >= total * 0.05;
    const n = useSkin ? sN : total;
    const r = useSkin ? sr : fr;
    const g = useSkin ? sg : fg;
    const b = useSkin ? sb : fb;

    // Motion: accumulate the per-frame global translation into a position.
    if (prevLuma) {
      const { dx, dy } = globalTranslation(prevLuma, luma, SW, SH);
      posX += dx;
      posY += dy;
    }
    prevLuma = luma;

    tracker.addSample(r / n, g / n, b / n, posX, posY, tMs);
  };
  setInterval(
    () => ui.updateVitals(tracker.estimate(bandForCenterFreq(ui.params.centerFreq))),
    1000
  );

  // --- frame loop ----------------------------------------------------------
  let last = performance.now();
  let fpsAcc = 0;
  let fpsCount = 0;

  const tick = (now: number) => {
    const dt = now - last;
    last = now;

    scope.frame(dt, ui.params);
    samplePulse(now);

    fpsAcc += dt;
    if (++fpsCount === 30) {
      ui.updatePerf(30000 / fpsAcc);
      fpsAcc = 0;
      fpsCount = 0;
    }
    scheduleNext();
  };

  // Drive off video frames when supported so we process each camera frame
  // exactly once; fall back to rAF.
  const scheduleNext = () => {
    if ("requestVideoFrameCallback" in video) {
      video.requestVideoFrameCallback((now) => tick(now));
    } else {
      requestAnimationFrame(tick);
    }
  };
  scheduleNext();
}
