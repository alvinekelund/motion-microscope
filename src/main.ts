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

  // --- independent rPPG path for the BPM counter --------------------------
  // Mean green over the center crop of a tiny downscale of the *raw* feed.
  const probe = document.createElement("canvas");
  probe.width = 64;
  probe.height = 48;
  const probeCtx = probe.getContext("2d", { willReadFrequently: true })!;
  const tracker = new RhythmTracker();

  const samplePulse = (tMs: number) => {
    probeCtx.drawImage(video, 0, 0, probe.width, probe.height);
    const region = probeCtx.getImageData(16, 8, 32, 32).data;
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < region.length; i += 4) {
      r += region[i];
      g += region[i + 1];
      b += region[i + 2];
    }
    const px = region.length / 4;
    tracker.addSample(r / px, g / px, b / px, tMs);
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
