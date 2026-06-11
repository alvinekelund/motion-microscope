import { initWebGPU } from "./gpu/context";
import { MotionMicroscope } from "./gpu/pipeline";
import { startCamera } from "./camera";
import { RhythmTracker, bandForCenterFreq } from "./pulse/bpm";
import { isSkin } from "./pulse/skin";
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
  // Sample the *raw* feed at a tiny resolution and scan nearly the whole
  // frame for skin: averaging colour over every skin pixel (not a fixed
  // center crop) lifts the rPPG signal-to-noise and finds the face wherever
  // it sits. Mean skin colour drives the POS pulse estimate; the skin
  // brightness centroid drives the breathing/vibration (motion) estimate.
  const probe = document.createElement("canvas");
  const PW = 64;
  const PH = 48;
  const SX = 2, SY = 2, SW = PW - 4, SH = PH - 4; // scan region (2px border trimmed)
  probe.width = PW;
  probe.height = PH;
  const probeCtx = probe.getContext("2d", { willReadFrequently: true })!;
  const tracker = new RhythmTracker();

  const samplePulse = (tMs: number) => {
    probeCtx.drawImage(video, 0, 0, PW, PH);
    const region = probeCtx.getImageData(SX, SY, SW, SH).data;

    // Accumulate skin-masked and whole-region stats in one pass; pick the
    // skin mask only if enough of it is present, else fall back so we never
    // measure from a near-empty mask.
    let sr = 0, sg = 0, sb = 0, sN = 0, sW = 0, sWx = 0, sWy = 0; // skin
    let fr = 0, fg = 0, fb = 0, fW = 0, fWx = 0, fWy = 0; //          full
    for (let p = 0, y = 0; y < SH; y++) {
      for (let x = 0; x < SW; x++, p += 4) {
        const pr = region[p], pg = region[p + 1], pb = region[p + 2];
        const l = 0.299 * pr + 0.587 * pg + 0.114 * pb; // luminance for centroid
        fr += pr; fg += pg; fb += pb;
        fW += l; fWx += l * x; fWy += l * y;
        if (isSkin(pr, pg, pb)) {
          sr += pr; sg += pg; sb += pb; sN++;
          sW += l; sWx += l * x; sWy += l * y;
        }
      }
    }

    const px = SW * SH;
    const useSkin = sN >= px * 0.05 && sW > 0;
    const n = useSkin ? sN : px;
    const r = useSkin ? sr : fr;
    const g = useSkin ? sg : fg;
    const b = useSkin ? sb : fb;
    const W = useSkin ? sW : fW;
    const Wx = useSkin ? sWx : fWx;
    const Wy = useSkin ? sWy : fWy;
    const cx = W > 0 ? Wx / W : SW / 2;
    const cy = W > 0 ? Wy / W : SH / 2;
    tracker.addSample(r / n, g / n, b / n, cx, cy, tMs);
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
