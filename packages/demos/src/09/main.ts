/**
 * Демо 09 - Разбор и транскодирование KTX2
 *
 * Генерирует синтетический KTX2-файл в памяти (несжатый RGBA8, 256×256,
 * 8 мип-уровней, каждый - отдельный сплошной цвет). Разбирает его с помощью
 * KTX2Parser, транскодирует каждый мип-уровень через TranscodePipeline
 * (путь без сжатия), загружает на GPU и отрисовывает полноэкранный квад.
 *
 * Проверяет: разбор заголовка KTX2, извлечение смещений байт мипов,
 * путь no-op TranscodePipeline для несжатых данных, выравнивание строк,
 * потоковую загрузку lodMinClamp.
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import type { IRenderPass, RenderPassInitContext, FrameContext } from "@webgpu-streaming/gpu-types";
import { parseKTX2 } from "@webgpu-streaming/texture-streaming";
import { TranscodePipeline } from "@webgpu-streaming/texture-streaming";
import { FormatRouter } from "@webgpu-streaming/texture-streaming";
import { createOverlay, FpsTracker } from "../shared/overlay.js";
import { generateTestKtx2 } from "./generateTestKtx2.js";

const overlay = createOverlay("09 - KTX2 Parse + Transcode");
const fps = new FpsTracker();
overlay.set("Status", "Initializing…");

const BASE_SIZE = 256;
const FIRST_MIP = 5; // начинаем с трёх самых грубых

const VERT_WGSL = /* wgsl */ `
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn main(@builtin(vertex_index) vi: u32) -> VSOut {
  var p = array<vec2<f32>,6>(vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(-1,1),vec2(1,-1),vec2(1,1));
  var u = array<vec2<f32>,6>(vec2(0,1),vec2(1,1),vec2(0,0),vec2(0,0),vec2(1,1),vec2(1,0));
  var o: VSOut; o.pos=vec4(p[vi],0,1); o.uv=u[vi]; return o;
}`;

const FRAG_WGSL = /* wgsl */ `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, uv);
}`;

class KTX2Pass implements IRenderPass {
  readonly name = "ktx2-pass";

  private _device: GPUDevice | null = null;
  private _texture: GPUTexture | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _sampler: GPUSampler | null = null;
  private _transcode: TranscodePipeline | null = null;

  residentMin = FIRST_MIP;
  lodMinClamp  = FIRST_MIP;
  totalMips    = 0;

  // Timing
  lastTranscodeMs = 0;
  lastUploadMs    = 0;

  private _ktx2Bytes: ArrayBuffer | null = null;
  private _parsed: ReturnType<typeof parseKTX2> | null = null;

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, presentationFormat } = ctx;
    this._device = device;

    // Генерируем синтетический KTX2.
    const t0 = performance.now();
    this._ktx2Bytes = generateTestKtx2(BASE_SIZE, BASE_SIZE);
    const genMs = (performance.now() - t0).toFixed(1);

    // Разбираем его.
    this._parsed = parseKTX2(this._ktx2Bytes);
    this.totalMips = this._parsed.levelCount;

    const format = new FormatRouter(device.features).selectFormat("color");

    overlay.set("File size", `${(this._ktx2Bytes.byteLength / 1024).toFixed(1)} KB`);
    overlay.set("Detected format", this._parsed.isSrgb ? "RGBA8 sRGB" : "RGBA8 linear");
    overlay.set("Mip count", this.totalMips);
    overlay.set("Dimensions", `${this._parsed.pixelWidth}×${this._parsed.pixelHeight}`);
    overlay.set("Supercompression", this._parsed.supercompressionScheme === 0 ? "none" : String(this._parsed.supercompressionScheme));
    overlay.set("GPU format", format);
    overlay.set("Generate time", `${genMs} ms`);

    this._transcode = new TranscodePipeline(2);

    device.pushErrorScope("out-of-memory");
    this._texture = device.createTexture({
      label: "ktx2-texture",
      size: [BASE_SIZE, BASE_SIZE],
      mipLevelCount: this.totalMips,
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[KTX2Pass] OOM:", e); });

    // Транскодируем и загружаем начальные мипы (от FIRST_MIP до totalMips-1).
    for (let level = FIRST_MIP; level < this.totalMips; level++) {
      await this._transcodeAndUpload(level);
    }

    this._pipeline = device.createRenderPipeline({
      label: "ktx2-pipeline",
      layout: "auto",
      vertex: { module: device.createShaderModule({ code: VERT_WGSL }), entryPoint: "main" },
      fragment: { module: device.createShaderModule({ code: FRAG_WGSL }), entryPoint: "main", targets: [{ format: presentationFormat }] },
      primitive: { topology: "triangle-list" },
    });

    this._rebuildSamplerAndBG();
  }

  private async _transcodeAndUpload(level: number): Promise<void> {
    if (!this._ktx2Bytes || !this._parsed || !this._texture || !this._device || !this._transcode) return;

    const t0 = performance.now();
    const result = await this._transcode.transcode({
      ktx2Bytes: this._ktx2Bytes,
      parsed: this._parsed,
      targetFormat: "rgba8unorm",
      mipLevel: level,
    });
    this.lastTranscodeMs = performance.now() - t0;

    const t1 = performance.now();
    this._device.queue.writeTexture(
      { texture: this._texture, mipLevel: level },
      result.data,
      { bytesPerRow: result.bytesPerRow, rowsPerImage: result.height },
      [result.width, result.height]
    );
    this.lastUploadMs = performance.now() - t1;
  }

  private _rebuildSamplerAndBG(): void {
    if (!this._device || !this._pipeline || !this._texture) return;
    this._sampler = this._device.createSampler({
      magFilter: "linear", minFilter: "linear", mipmapFilter: "linear",
      lodMinClamp: this.lodMinClamp, lodMaxClamp: this.totalMips - 1,
    });
    this._bindGroup = this._device.createBindGroup({
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._texture.createView() },
        { binding: 1, resource: this._sampler },
      ],
    });
  }

  async loadNextMip(): Promise<boolean> {
    if (this.residentMin === 0) return false;
    const nextLevel = this.residentMin - 1;
    await this._transcodeAndUpload(nextLevel);
    this.residentMin = nextLevel;
    this.lodMinClamp = nextLevel;
    this._rebuildSamplerAndBG();
    return true;
  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._bindGroup) return;
    const pass = ctx.encoder.beginRenderPass({
      colorAttachments: [{ view: ctx.colorAttachment, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.draw(6);
    pass.end();
  }

  onResize(_w: number, _h: number): void {}

  destroy(): void {
    this._texture?.destroy();
    this._transcode?.destroy();
    this._texture = null; this._pipeline = null; this._bindGroup = null;
    this._device = null; this._ktx2Bytes = null; this._parsed = null;
  }
}

// ---- Подключение ------------------------------------------------------------------------------------------------------------------------------

const el = document.getElementById("canvas") as HTMLElement & {
  clearColor: GPUColorDict;
  addRenderPass: (p: IRenderPass) => Promise<void>;
};
const btnNext = document.getElementById("btn-next") as HTMLButtonElement;
const btnAuto = document.getElementById("btn-auto") as HTMLButtonElement;

const COLOR_NAMES = ["белый","фиолетовый","синий","голубой","зелёный","жёлтый","оранжевый","красный"];

el.addEventListener("webgpu-error", (e) => overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`));

el.addEventListener("webgpu-ready", async (e: Event) => {
  void (e as CustomEvent<WebGPUReadyDetail>).detail;
  el.clearColor = { r: 0.08, g: 0.08, b: 0.08, a: 1.0 };

  const pass = new KTX2Pass();
  await el.addRenderPass(pass);

  overlay.set("Status", "Running");

  function updateOverlay() {
    overlay.set("lodMinClamp", pass.lodMinClamp.toFixed(1));
    overlay.set("Resident mips", `${pass.residentMin}–${pass.totalMips - 1}`);
    overlay.set("Showing color", COLOR_NAMES[pass.residentMin] ?? "?");
    overlay.set("Transcode time", `${pass.lastTranscodeMs.toFixed(2)} ms`);
    overlay.set("Upload time", `${pass.lastUploadMs.toFixed(2)} ms`);
  }
  updateOverlay();

  btnNext.addEventListener("click", async () => {
    btnNext.disabled = true;
    await pass.loadNextMip();
    updateOverlay();
    btnNext.disabled = false;
  });

  let running = false;
  btnAuto.addEventListener("click", async () => {
    if (running) return;
    running = true;
    btnAuto.textContent = "⏸ Загрузка…";
    while (await pass.loadNextMip()) {
      updateOverlay();
      await new Promise((r) => setTimeout(r, 400));
    }
    updateOverlay();
    btnAuto.textContent = "✓ Готово";
    running = false;
  });

  function tick() {
    fps.tick();
    overlay.set("FPS", fps.fps);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
