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
                  │              leaky accumulate + biquad bandpass (the freq slider)
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
- **A steep temporal bandpass.** The per-pixel filter is a second-order RBJ
  biquad (±12 dB/octave) rather than a difference of one-pole lowpasses
  (±6 dB/octave). The steeper skirt is itself noise control: a broad filter
  lets the spectrum on either side of the heartbeat — much of it sensor
  noise — leak into the magnified motion. The biquad runs as a transposed
  Direct Form II recurrence in WGSL; its coefficients are unit-tested in JS
  (same recurrence) for unity passband gain, octave attenuation, and DC
  rejection.
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

## The vitals counter is deliberately independent

The number beside the video is measured from the **raw** feed and tracks
whichever band the slider is on — breaths/min, BPM, or a raw Hz — over a ~30 s
sliding window. It is computed from a different observable depending on what
kind of signal that band is, because pulse and breathing are not the same
physics:

- **Pulse** is a *colour* change — blood volume modulates skin reflectance by
  a fraction of a percent, with no visible movement. So the cardiac band uses
  **POS** (plane-orthogonal-to-skin, Wang et al. 2017): per-channel temporal
  means are divided out and the residual is projected onto axes orthogonal to
  the skin-tone vector. This rejects the ambient-brightness flicker that
  naïve green-channel rPPG mistakes for a heartbeat.
- **Breathing and vibration** are *motion* — the head and shoulders visibly
  translate. The subtle but important point: the mean brightness of a fixed
  crop is nearly invariant when content just slides across it, so mean
  luminance is weakest exactly when the motion is largest. Instead the
  breathing/vibration bands track the **brightness centroid** of the region,
  which moves linearly with displacement, on both axes, and sum their power
  spectra so the direction of movement doesn't matter.

Both observables are sampled only over **skin pixels**, scanned across nearly
the whole tiny probe frame rather than a fixed center crop: a cheap per-pixel
classifier masks the probe so background, hair and clothing don't dilute the
signal, and averaging over every skin pixel both lifts the signal-to-noise and
finds a face that isn't centered (a corner face the old crop scored at zero
skin pixels the full scan picks up cleanly). Measured contamination is real —
an unmasked crop over a blue background pulls the mean colour from
`[200,150,120]` to `[136,118,136]`, halving the pulse amplitude. The mask is
paired with a whole-region fallback, so a skin tone or lighting it
under-selects degrades to the old behaviour rather than reporting from an
empty mask.

The masked signals are then detrended, Hamming-windowed, zero-padded FFT'd,
and reduced to the dominant in-band peak with parabolic sub-bin interpolation
behind a prominence gate. None of it reads the magnified output — the visible
pulsing and the ticking number are independent measurements corroborating each
other, not one effect dressing up the other.

The signal path is plain TypeScript with unit tests (`npm test`): FFT
correctness, POS recovery of 72 and 150 BPM from noisy drifting traces,
breathing recovery from centroid motion **with colour held perfectly flat**
(the case the old luminance approach could not see), vibration in Hz, and
silence on pure noise.

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
