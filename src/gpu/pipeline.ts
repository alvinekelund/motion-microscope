import lumaWgsl from "./shaders/luma.wgsl?raw";
import downsampleWgsl from "./shaders/downsample.wgsl?raw";
import laplacianWgsl from "./shaders/laplacian.wgsl?raw";
import rieszWgsl from "./shaders/riesz.wgsl?raw";
import temporalWgsl from "./shaders/temporal.wgsl?raw";
import blurWgsl from "./shaders/blur.wgsl?raw";
import magnifyWgsl from "./shaders/magnify.wgsl?raw";
import collapseWgsl from "./shaders/collapse.wgsl?raw";
import renderWgsl from "./shaders/render.wgsl?raw";
import type { GpuContext } from "./context";

export interface MagParams {
  /** Phase amplification factor (the slider). */
  alpha: number;
  /** Center of the temporal passband, Hz. */
  centerFreq: number;
  /** 0..1 — attenuates the finest (noisiest) pyramid level. */
  denoise: number;
  /** Original/magnified comparison split, 0..1 of display width. */
  split: number;
}

interface Size {
  w: number;
  h: number;
}

const WG = 8;

/**
 * Riesz-pyramid phase-based motion magnification, fully on the GPU.
 *
 * Per frame: YCoCg split -> Gaussian/Laplacian pyramid -> Riesz transform per
 * level -> quaternionic phase difference + IIR temporal bandpass ->
 * amplitude-weighted spatial blur -> phase-shift magnification -> pyramid
 * collapse -> recombine with original chroma and present.
 *
 * Frame-to-frame state (previous coefficients, filter state) is held in
 * ping-pong texture pairs indexed by frame parity, so no texture is ever read
 * and written in the same pass.
 */
export class MotionMicroscope {
  readonly levels: number;
  private frameIndex = 0;
  private smoothedDt = 1 / 30;

  private constructor(
    private readonly gpu: GpuContext,
    private readonly video: HTMLVideoElement,
    private readonly dims: Size[],
    private readonly pipelines: Record<string, GPUComputePipeline>,
    private readonly renderPipeline: GPURenderPipeline,
    private readonly tex: ReturnType<typeof MotionMicroscope.createTextures>,
    private readonly uniforms: {
      temporal: GPUBuffer;
      magnify: GPUBuffer[];
      render: GPUBuffer;
    },
    private readonly bindGroups: ReturnType<typeof MotionMicroscope.createBindGroups>
  ) {
    this.levels = dims.length - 1;
  }

  static async create(gpu: GpuContext, video: HTMLVideoElement): Promise<MotionMicroscope> {
    const w = video.videoWidth;
    const h = video.videoHeight;

    // Gaussian chain: full res down to a coarse residual.
    const dims: Size[] = [{ w, h }];
    while (Math.min(dims[dims.length - 1].w, dims[dims.length - 1].h) >= 48) {
      const d = dims[dims.length - 1];
      dims.push({ w: Math.ceil(d.w / 2), h: Math.ceil(d.h / 2) });
    }

    const { device } = gpu;
    const make = (code: string, entryPoint = "main") =>
      device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint },
      });

    const blurModule = device.createShaderModule({ code: blurWgsl });
    const pipelines = {
      luma: make(lumaWgsl),
      downsample: make(downsampleWgsl),
      laplacian: make(laplacianWgsl),
      riesz: make(rieszWgsl),
      temporal: make(temporalWgsl),
      blurH: device.createComputePipeline({
        layout: "auto",
        compute: { module: blurModule, entryPoint: "blur_h" },
      }),
      blurV: device.createComputePipeline({
        layout: "auto",
        compute: { module: blurModule, entryPoint: "blur_v" },
      }),
      magnify: make(magnifyWgsl),
      collapse: make(collapseWgsl),
    };

    const renderModule = device.createShaderModule({ code: renderWgsl });
    const renderPipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: renderModule, entryPoint: "vs" },
      fragment: {
        module: renderModule,
        entryPoint: "fs",
        targets: [{ format: gpu.format }],
      },
      primitive: { topology: "triangle-list" },
    });

    const tex = MotionMicroscope.createTextures(device, dims);
    const levels = dims.length - 1;

    const uniforms = {
      temporal: device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      magnify: Array.from({ length: levels }, () =>
        device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
      ),
      render: device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    };

    const bindGroups = MotionMicroscope.createBindGroups(
      device, dims, pipelines, renderPipeline, tex, uniforms
    );

    return new MotionMicroscope(gpu, video, dims, pipelines, renderPipeline, tex, uniforms, bindGroups);
  }

  private static createTextures(device: GPUDevice, dims: Size[]) {
    const levels = dims.length - 1;
    const storageTex = (d: Size) =>
      device.createTexture({
        size: [d.w, d.h],
        format: "rgba16float",
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
    const perLevel = (f: (d: Size) => GPUTexture) => dims.slice(0, levels).map(f);
    const perLevelPair = (f: (d: Size) => GPUTexture) =>
      dims.slice(0, levels).map((d) => [f(d), f(d)] as const);

    return {
      video: device.createTexture({
        size: [dims[0].w, dims[0].h],
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT,
      }),
      ycocg: storageTex(dims[0]),
      // gauss[0] aliases ycocg (.x carries luma); coarser levels are own textures.
      gauss: dims.slice(1).map(storageTex),
      lap: perLevel(storageTex),
      coeff: perLevelPair(storageTex),
      state1: perLevelPair(storageTex),
      state2: perLevelPair(storageTex),
      filtered: perLevel(storageTex),
      blurTmp: perLevel(storageTex),
      blurred: perLevel(storageTex),
      lapMag: perLevel(storageTex),
      recon: perLevel(storageTex),
      sampler: device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
      }),
    };
  }

  private static createBindGroups(
    device: GPUDevice,
    dims: Size[],
    pipelines: Record<string, GPUComputePipeline>,
    renderPipeline: GPURenderPipeline,
    tex: ReturnType<typeof MotionMicroscope.createTextures>,
    uniforms: { temporal: GPUBuffer; magnify: GPUBuffer[]; render: GPUBuffer }
  ) {
    const levels = dims.length - 1;
    const v = (t: GPUTexture) => t.createView();
    const bg = (
      pipeline: GPUComputePipeline | GPURenderPipeline,
      entries: (GPUTextureView | GPUSampler | { buffer: GPUBuffer })[]
    ) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map((resource, binding) => ({ binding, resource })),
      });

    // gaussTex(i): texture holding Gaussian level i.
    const gaussTex = (i: number) => (i === 0 ? tex.ycocg : tex.gauss[i - 1]);

    return {
      luma: bg(pipelines.luma, [v(tex.video), v(tex.ycocg)]),
      downsample: Array.from({ length: levels }, (_, i) =>
        bg(pipelines.downsample, [v(gaussTex(i)), v(gaussTex(i + 1))])
      ),
      laplacian: Array.from({ length: levels }, (_, k) =>
        bg(pipelines.laplacian, [v(gaussTex(k)), v(gaussTex(k + 1)), tex.sampler, v(tex.lap[k])])
      ),
      riesz: Array.from({ length: levels }, (_, k) =>
        [0, 1].map((par) => bg(pipelines.riesz, [v(tex.lap[k]), v(tex.coeff[k][par])]))
      ),
      temporal: Array.from({ length: levels }, (_, k) =>
        [0, 1].map((par) =>
          bg(pipelines.temporal, [
            v(tex.coeff[k][par]),
            v(tex.coeff[k][1 - par]),
            v(tex.state1[k][1 - par]),
            v(tex.state2[k][1 - par]),
            v(tex.state1[k][par]),
            v(tex.state2[k][par]),
            v(tex.filtered[k]),
            { buffer: uniforms.temporal },
          ])
        )
      ),
      blurH: Array.from({ length: levels }, (_, k) =>
        bg(pipelines.blurH, [v(tex.filtered[k]), v(tex.blurTmp[k])])
      ),
      blurV: Array.from({ length: levels }, (_, k) =>
        bg(pipelines.blurV, [v(tex.blurTmp[k]), v(tex.blurred[k])])
      ),
      magnify: Array.from({ length: levels }, (_, k) =>
        [0, 1].map((par) =>
          bg(pipelines.magnify, [
            v(tex.coeff[k][par]),
            v(tex.blurred[k]),
            v(tex.lapMag[k]),
            { buffer: uniforms.magnify[k] },
          ])
        )
      ),
      collapse: Array.from({ length: levels }, (_, k) =>
        bg(pipelines.collapse, [
          v(tex.lapMag[k]),
          v(k === levels - 1 ? gaussTex(levels) : tex.recon[k + 1]),
          tex.sampler,
          v(tex.recon[k]),
        ])
      ),
      render: bg(renderPipeline, [v(tex.ycocg), v(tex.recon[0]), tex.sampler, { buffer: uniforms.render }]),
    };
  }

  /** Encode and submit one frame. dtMs = time since previous frame. */
  frame(dtMs: number, params: MagParams): void {
    const { device, context } = this.gpu;
    const levels = this.levels;
    const par = this.frameIndex & 1;

    // --- uniforms ---------------------------------------------------------
    this.smoothedDt = 0.9 * this.smoothedDt + 0.1 * Math.min(dtMs / 1000, 0.1);
    const dt = this.smoothedDt;
    // Half-octave passband around the chosen center, capped below Nyquist.
    const fHi = Math.min(params.centerFreq * 1.5, 0.45 / dt);
    const fLo = Math.min(params.centerFreq / 1.5, fHi * 0.66);
    const onePole = (f: number) => 1 - Math.exp(-2 * Math.PI * f * dt);
    const leak = Math.exp(-2 * Math.PI * 0.05 * dt);
    device.queue.writeBuffer(
      this.uniforms.temporal, 0,
      new Float32Array([onePole(fHi), onePole(fLo), leak, 0])
    );
    for (let k = 0; k < levels; k++) {
      // Finest level carries the most sensor noise and the least pulse
      // signal; the noise-gate slider fades it out.
      const levelGain = k === 0 ? 1 - params.denoise : 1;
      device.queue.writeBuffer(
        this.uniforms.magnify[k], 0,
        new Float32Array([params.alpha * levelGain, Math.PI, 0, 0])
      );
    }
    device.queue.writeBuffer(
      this.uniforms.render, 0,
      new Float32Array([params.split, 1.2 / this.dims[0].w, 0, 0])
    );

    device.queue.copyExternalImageToTexture(
      { source: this.video },
      { texture: this.tex.video },
      [this.dims[0].w, this.dims[0].h]
    );

    // --- encode -----------------------------------------------------------
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const dispatch = (
      pipeline: GPUComputePipeline,
      group: GPUBindGroup,
      d: Size
    ) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(d.w / WG), Math.ceil(d.h / WG));
    };
    const bgs = this.bindGroups;
    const p = this.pipelines;

    dispatch(p.luma, bgs.luma, this.dims[0]);
    for (let i = 0; i < levels; i++) dispatch(p.downsample, bgs.downsample[i], this.dims[i + 1]);
    for (let k = 0; k < levels; k++) dispatch(p.laplacian, bgs.laplacian[k], this.dims[k]);
    for (let k = 0; k < levels; k++) dispatch(p.riesz, bgs.riesz[k][par], this.dims[k]);
    for (let k = 0; k < levels; k++) dispatch(p.temporal, bgs.temporal[k][par], this.dims[k]);
    for (let k = 0; k < levels; k++) dispatch(p.blurH, bgs.blurH[k], this.dims[k]);
    for (let k = 0; k < levels; k++) dispatch(p.blurV, bgs.blurV[k], this.dims[k]);
    for (let k = 0; k < levels; k++) dispatch(p.magnify, bgs.magnify[k][par], this.dims[k]);
    for (let k = levels - 1; k >= 0; k--) dispatch(p.collapse, bgs.collapse[k], this.dims[k]);
    pass.end();

    const rp = enc.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    rp.setPipeline(this.renderPipeline);
    rp.setBindGroup(0, this.bindGroups.render);
    rp.draw(3);
    rp.end();

    device.queue.submit([enc.finish()]);
    this.frameIndex++;
  }
}
