/**
 * Демо 05 - Кольцевой буфер стейджинга
 *
 * Аналогичная визуализация с Демо 04 (потоковая загрузка мипов с lodMinClamp),
 * но загрузка выполняется через StagingRingBuffer вместо queue.writeTexture.
 *
 * Показывает: активный слот кольца, состояния буферов (mapped/pending/in-flight),
 * байт записано за кадр, время загрузки каждого мипа.
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import type { IRenderPass, RenderPassInitContext, FrameContext } from "@webgpu-streaming/gpu-types";
import { StagingRingBuffer } from "@webgpu-streaming/texture-streaming";
import { createOverlay, FpsTracker } from "../shared/overlay.js";

const overlay = createOverlay("05 - Staging Ring Buffer");
const fps = new FpsTracker();
overlay.set("Status", "Initializing…");

// ---- Общие константы из демо 04 ------------------------------------------------------------------------------------------------

const BASE_SIZE = 256;
const MIP_COUNT = 8;
const FIRST_UPLOADED_MIP = 5;

const MIP_COLORS: [number, number, number, number][] = [
  [1.0, 1.0, 1.0, 1.0], // 0 - белый
  [0.6, 0.0, 0.8, 1.0], // 1 - фиолетовый
  [0.1, 0.1, 1.0, 1.0], // 2 - синий
  [0.0, 0.8, 0.8, 1.0], // 3 - голубой
  [0.1, 0.8, 0.1, 1.0], // 4 - зелёный
  [1.0, 1.0, 0.0, 1.0], // 5 - жёлтый
  [1.0, 0.5, 0.0, 1.0], // 6 - оранжевый
  [1.0, 0.1, 0.1, 1.0], // 7 - красный
];

function solidMipData(level: number): Uint8Array<ArrayBuffer> {
  const size = Math.max(1, BASE_SIZE >> level);
  const [r, g, b, a] = MIP_COLORS[level]!;
  const data = new Uint8Array(new ArrayBuffer(size * size * 4));
  for (let i = 0; i < size * size; i++) {
    data[i * 4]     = Math.round(r * 255);
    data[i * 4 + 1] = Math.round(g * 255);
    data[i * 4 + 2] = Math.round(b * 255);
    data[i * 4 + 3] = Math.round(a * 255);
  }
  return data;
}

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

// ---- Рендер-проход --------------------------------------------------------------------------------------------------------------------------

class StagingRingPass implements IRenderPass {
  readonly name = "staging-ring-pass";

  private _device: GPUDevice | null = null;
  private _texture: GPUTexture | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _sampler: GPUSampler | null = null;
  private _ring: StagingRingBuffer | null = null;

  residentMin = FIRST_UPLOADED_MIP;
  lodMinClamp  = FIRST_UPLOADED_MIP;

  /** Устанавливается в номер мип-уровня для постановки загрузки в очередь в следующем вызове prepareFrame. */
  pendingUpload: number | null = null;
  lastUploadMs = 0;
  lastUploadBytes = 0;

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, presentationFormat } = ctx;
    this._device = device;

    // 4 слота кольца × 1 МБ каждый (мип 0 для 256² RGBA8 = 256 КБ, с запасом).
    this._ring = new StagingRingBuffer(device, 4, 1 * 1024 * 1024);

    device.pushErrorScope("out-of-memory");
    this._texture = device.createTexture({
      label: "ring-mip-tex",
      size: [BASE_SIZE, BASE_SIZE],
      mipLevelCount: MIP_COUNT,
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[StagingRingPass] OOM:", e); });

    // Загружаем начальные мипы через кольцо.
    // Поскольку слоты кольца изначально отображены, загрузка выполняется синхронно.
    for (let level = FIRST_UPLOADED_MIP; level < MIP_COUNT; level++) {
      this._ringUploadMip(null, level); // null encoder = прямой путь (только при инициализации)
    }

    this._pipeline = device.createRenderPipeline({
      label: "ring-pipeline",
      layout: "auto",
      vertex:   { module: device.createShaderModule({ code: VERT_WGSL }), entryPoint: "main" },
      fragment: { module: device.createShaderModule({ code: FRAG_WGSL }), entryPoint: "main", targets: [{ format: presentationFormat }] },
      primitive: { topology: "triangle-list" },
    });

    this._rebuildSamplerAndBG();
  }

  /**
   * Загружает один мип-уровень через staging-кольцо.
   * Если encoder передан, записывает copyBufferToTexture в него.
   * Если encoder равен null (путь инициализации), создаёт временный encoder и отправляет.
   */
  private _ringUploadMip(encoder: GPUCommandEncoder | null, level: number): void {
    if (!this._ring || !this._texture || !this._device) return;

    const mipSize = Math.max(1, BASE_SIZE >> level);
    const srcData = solidMipData(level);
    const bytesPerRow = Math.max(256, Math.ceil((mipSize * 4) / 256) * 256);
    const totalBytes = bytesPerRow * mipSize;

    const slot = this._ring.acquire();
    const dst = new Uint8Array(slot.arrayBuffer, 0, totalBytes);

    // Записываем дополненные строки в отображённый буфер.
    const t0 = performance.now();
    for (let row = 0; row < mipSize; row++) {
      dst.set(srcData.subarray(row * mipSize * 4, (row + 1) * mipSize * 4), row * bytesPerRow);
    }
    this.lastUploadMs = performance.now() - t0;
    this.lastUploadBytes = totalBytes;

    const enc = encoder ?? this._device.createCommandEncoder({ label: "init-upload" });
    this._ring.recordCopy(enc, this._texture, level, [mipSize, mipSize], bytesPerRow, totalBytes);
    this._ring.submit();

    if (!encoder) {
      this._device.queue.submit([enc.finish()]);
    }
  }

  private _rebuildSamplerAndBG(): void {
    if (!this._device || !this._pipeline || !this._texture) return;
    this._sampler = this._device.createSampler({
      magFilter: "linear", minFilter: "linear", mipmapFilter: "linear",
      lodMinClamp: this.lodMinClamp, lodMaxClamp: MIP_COUNT - 1,
    });
    this._bindGroup = this._device.createBindGroup({
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._texture.createView() },
        { binding: 1, resource: this._sampler },
      ],
    });
  }

  streamNext(): boolean {
    if (this.residentMin === 0) return false;
    this.pendingUpload = this.residentMin - 1;
    return true;
  }

  execute(ctx: FrameContext): void {
    // If there's a pending upload, do it now via the ring into this frame's encoder.
    if (this.pendingUpload !== null) {
      const level = this.pendingUpload;
      this.pendingUpload = null;
      this._ringUploadMip(ctx.encoder, level);
      this.residentMin = level;
      this.lodMinClamp = level;
      this._rebuildSamplerAndBG();
    }

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
    this._ring?.destroy();
    this._texture = null;
    this._ring = null;
    this._pipeline = null;
    this._bindGroup = null;
    this._device = null;
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

el.addEventListener("webgpu-error", (e) => {
  overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`);
});

el.addEventListener("webgpu-ready", async (e) => {
  const { device } = (e as CustomEvent<WebGPUReadyDetail>).detail;
  void device;

  el.clearColor = { r: 0.08, g: 0.08, b: 0.08, a: 1.0 };
  const pass = new StagingRingPass();
  await el.addRenderPass(pass);
  overlay.set("Status", "Running");

  function updateOverlay() {
    overlay.set("lodMinClamp", pass.lodMinClamp.toFixed(1));
    overlay.set("Resident mips", `${pass.residentMin}–${MIP_COUNT - 1}`);
    overlay.set("Current color", COLOR_NAMES[pass.residentMin] ?? "?");
    overlay.set("Ring slot", pass["_ring"]?.currentIndex ?? "-");
    overlay.set("Ring states", (pass["_ring"]?.getStates() ?? []).join(" "));
    overlay.set("Last upload", `${pass.lastUploadMs.toFixed(2)} ms`);
    overlay.set("Bytes/frame", pass.lastUploadBytes.toLocaleString());
  }

  btnNext.addEventListener("click", () => {
    pass.streamNext();
    setTimeout(updateOverlay, 50); // даём следующему кадру завершиться
  });

  let autoTimer: ReturnType<typeof setInterval> | null = null;
  btnAuto.addEventListener("click", () => {
    if (autoTimer) {
      clearInterval(autoTimer); autoTimer = null;
      btnAuto.textContent = "⏵ Авто-поток";
    } else {
      btnAuto.textContent = "⏸ Стоп";
      autoTimer = setInterval(() => {
        const more = pass.streamNext();
        setTimeout(updateOverlay, 50);
        if (!more) { clearInterval(autoTimer!); autoTimer = null; btnAuto.textContent = "⏵ Авто-поток"; }
      }, 500);
    }
  });

  function tick() {
    fps.tick();
    overlay.set("FPS", fps.fps);
    updateOverlay();
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
