import { initWebGPU } from "./gpu/context";
import { MotionMicroscope } from "./gpu/pipeline";
import { startCamera } from "./camera";
import { bandForCenterFreq } from "./pulse/bpm";
import { VitalsProbe } from "./pulse/probe";
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
  // Downscale the raw feed into the VitalsProbe each frame. The probe extracts
  // skin colour (→ POS pulse) and foreground-weighted optical-flow velocity
  // (→ breathing) from the same pixels. This exact class is driven by a
  // synthetic-video test that proves both signals lock (see probe.test.ts).
  const PW = 96;
  const PH = 72;
  const probeCanvas = document.createElement("canvas");
  probeCanvas.width = PW;
  probeCanvas.height = PH;
  const probeCtx = probeCanvas.getContext("2d", { willReadFrequently: true })!;
  const probe = new VitalsProbe(PW, PH);

  // Sampling is driven by a worker-timer rather than the render loop:
  // dedicated-worker timers aren't subject to hidden-tab throttling, so the
  // measurement keeps its 30 Hz cadence even if the visitor briefly switches
  // tabs (the render loop, which IS compositor-gated, just pauses). Frames
  // that haven't advanced are skipped so duplicates never enter the spectrum.
  let lastVideoTime = -1;
  const samplePulse = () => {
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;
    probeCtx.drawImage(video, 0, 0, PW, PH);
    probe.process(probeCtx.getImageData(0, 0, PW, PH).data, performance.now());
  };
  const samplerWorker = new Worker(
    URL.createObjectURL(
      new Blob(["setInterval(() => postMessage(0), 33);"], { type: "application/javascript" })
    )
  );
  samplerWorker.onmessage = samplePulse;
  setInterval(
    () => ui.updateVitals(probe.estimate(bandForCenterFreq(ui.params.centerFreq))),
    1000
  );

  // Diagnostic hook: lets a debugger inspect the live probe and pull real
  // estimates for both bands regardless of what the UI is showing.
  (window as unknown as Record<string, unknown>).__mm = {
    probe,
    est: (f: number) => probe.estimate(bandForCenterFreq(f)),
  };

  // --- frame loop ----------------------------------------------------------
  let last = performance.now();
  let fpsAcc = 0;
  let fpsCount = 0;

  const tick = (now: number) => {
    const dt = now - last;
    last = now;

    scope.frame(dt, ui.params);

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
