import { initWebGPU } from "./gpu/context";
import { MotionMicroscope } from "./gpu/pipeline";
import { startCamera } from "./camera";
import { RhythmTracker, bandForCenterFreq } from "./pulse/bpm";
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
  // Sample a small center crop of the *raw* feed: mean colour drives the POS
  // pulse estimate, the brightness centroid drives the breathing/vibration
  // (motion) estimate.
  const probe = document.createElement("canvas");
  const PW = 64;
  const PH = 48;
  const RX = 16, RY = 8, RW = 32, RH = 32; // center region of interest
  probe.width = PW;
  probe.height = PH;
  const probeCtx = probe.getContext("2d", { willReadFrequently: true })!;
  const tracker = new RhythmTracker();

  const samplePulse = (tMs: number) => {
    probeCtx.drawImage(video, 0, 0, PW, PH);
    const region = probeCtx.getImageData(RX, RY, RW, RH).data;
    let r = 0, g = 0, b = 0;
    let wsum = 0, wx = 0, wy = 0;
    for (let p = 0, y = 0; y < RH; y++) {
      for (let x = 0; x < RW; x++, p += 4) {
        const pr = region[p], pg = region[p + 1], pb = region[p + 2];
        r += pr;
        g += pg;
        b += pb;
        // Luminance-weighted centroid: shifts as the subject translates.
        const l = 0.299 * pr + 0.587 * pg + 0.114 * pb;
        wsum += l;
        wx += l * x;
        wy += l * y;
      }
    }
    const px = RW * RH;
    const cx = wsum > 0 ? wx / wsum : RW / 2;
    const cy = wsum > 0 ? wy / wsum : RH / 2;
    tracker.addSample(r / px, g / px, b / px, cx, cy, tMs);
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
