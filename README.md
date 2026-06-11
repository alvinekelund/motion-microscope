# Motion Microscope

**See your own heartbeat.** Live phase-based motion magnification, written
entirely in WebGPU compute shaders. Allow the webcam, sit still, and watch
your face visibly pulse with each heartbeat — BPM counter ticking alongside,
measured independently from the raw feed.

> All processing is local. No frames, no data, nothing leaves your machine.

## Run it

```bash
npm install
npm run dev      # open http://localhost:5173 in Chrome/Edge (or Safari 26+)
```

Sit still, face the camera in decent light. Drag the red divider to compare
the untouched feed against the magnified one. Sweep the frequency slider from
breathing (0.3 Hz) through heartbeat (~1.2 Hz) up to tremor and vibration.

## How it works

This is the Riesz-pyramid variant of phase-based Eulerian video magnification
([Wadhwa, Rubinstein, Durand, Freeman — SIGGRAPH 2014][riesz]), chosen because
its Riesz transform approximation is three taps wide, which makes the whole
method cheap enough to run per-frame in real time.

The core idea: in a spatially bandpassed image, *local phase* encodes
sub-pixel position. Measure how phase advances over time at every pixel and
scale, isolate the temporal frequency band you care about, multiply it, and
push the coefficients further along the rotation they were already making.
Motion far below one pixel — skin displacing as a pulse wave passes — becomes
visible, without ever computing optical flow.

Per frame, ten compute passes per pyramid level, all WGSL:

```
webcam ─ YCoCg split ─ luma ─► Laplacian pyramid (per level k):
                  │              Riesz transform  (I, R1, R2) = monogenic signal
                  │              q_t · conj(q_{t-1})  ─► per-pixel phase advance
                  │              leaky accumulate + IIR bandpass (the freq slider)
                  │              amplitude²-weighted Gaussian blur
                  │              phase-shift coefficients by α·φ̂  ─► magnified level
                  │            collapse pyramid ─► magnified luma
                  └─ chroma ──────────────────────► recombine ─► canvas
```

The quaternionic step is the elegant part. The monogenic triple is treated as
a quaternion `q = I + i·R1 + j·R2`; the product `q_t · q̄_{t-1}` is the
rotation taking last frame's coefficient onto this frame's, and its angle —
decomposed along the local orientation — is exactly the phase advance caused
by motion. Because consecutive-frame *differences* are accumulated, phase
arrives pre-unwrapped: no mod-2π discontinuities to patch.

## The invisible 50%: not amplifying garbage

A naive implementation of the paper amplifies webcam sensor noise into a
boiling mess. Most of the engineering here is in what you *don't* see:

- **Chroma never enters the amplifier.** Frames are split to YCoCg and only
  luma is processed. Webcam chroma (Bayer interpolation + low-light gain) is
  the noisiest channel, and it is carried through untouched.
- **Amplitude-weighted phase smoothing.** Phase is meaningless where the
  coefficient amplitude is near zero — in flat regions it is pure noise. The
  bandpassed phase is blurred weighted by amplitude² (`blur(φ·A²)/blur(A²)`),
  so structure dominates and flat pixels borrow phase from real neighbors
  instead of contaminating them.
- **The finest pyramid level is gated.** It holds the most sensor noise and
  the least pulse signal; the noise-gate slider fades it out.
- **Magnified phase is clamped** to ±π, so a blink or hot pixel can't fling a
  coefficient around the phase circle and leave smearing artifacts.
- **Amplitude guards** zero the phase update wherever either frame's
  coefficient sits at the noise floor.
- **fp16-safe state.** Filter state lives in `rgba16float`. A raw phase
  accumulator would drift to magnitudes where half-float spacing (~0.06 at
  φ≈100) swamps a ~0.01 rad pulse signal, so the accumulator has a slow leak —
  a one-pole DC blocker at ~0.05 Hz, two decades below the heartbeat — that
  keeps every stored value small and fp16-exact indefinitely.

## The BPM counter is deliberately independent

The number is classic remote photoplethysmography on the **raw** feed: mean
green level over the face region → detrend (kills auto-exposure ramps) →
Hamming window → zero-padded FFT → strongest peak in 0.7–3 Hz with parabolic
sub-bin interpolation and a prominence gate. It never reads the magnified
output — so the visible pulsing and the ticking number are two separate
measurements corroborating each other, not one effect dressing up the other.

The signal-processing path is plain TypeScript with unit tests
(`npm test`): FFT correctness, recovery of 72 and 150 bpm from noisy drifting
synthetic traces, and silence on pure noise.

## Performance

At 640×480 the frame costs ~40 small compute dispatches across a 4-level
pyramid (~0.55 Mpx of rgba16float work) — well under a millisecond on any
recent GPU, so the pipeline idles at the camera's 30 fps. Frame-to-frame
state is held in ping-pong texture pairs indexed by frame parity; nothing is
read and written in the same pass, and no pixel data ever crosses back to the
CPU except the 64×48 rPPG probe.

## Browser support

WebGPU: Chrome/Edge 113+, Safari 26+, Firefox 141+ (Windows; other platforms
in progress). The page tells you if your browser can't run it.

[riesz]: https://people.csail.mit.edu/nwadhwa/riesz-pyramid/
