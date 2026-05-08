/**
 * Демо 04 - Покадровая загрузка мипов + валидация lodMinClamp
 *
 * КЛЮЧЕВОЙ ЭКСПЕРИМЕНТ: проверяет, что GPUSampler.lodMinClamp корректно
 * запрещает выборку мип-уровней, которые ещё не были загружены.
 *
 * Параметры:
 *   - Текстура 256×256, mipLevelCount: 8 (уровни 0–7)
 *   - Изначально загружены только мипы 5–7 (8×8, 4×4, 2×2)
 *   - Начальное значение lodMinClamp равно 5.0
 *   - Каждое нажатие «Следующий мип» загружает следующий более чёткий мип и снижает clamp
 *
 * Цветовая кодировка мипов (уникальный сплошной цвет для каждого уровня):
 *   7 = красный, 6 = оранжевый, 5 = жёлтый, 4 = зелёный,
 *   3 = голубой, 2 = синий, 1 = фиолетовый, 0 = белый
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import type { IRenderPass, RenderPassInitContext, FrameContext } from "@webgpu-streaming/gpu-types";
import { createOverlay, FpsTracker } from "../shared/overlay.js";

const overlay = createOverlay("04 - Mip Upload / lodMinClamp");
const fps = new FpsTracker();
overlay.set("Status", "Initializing…");

// ---- Константы ----------------------------------------------------------------------------------------------------------------------------------

const BASE_SIZE   = 256;
const MIP_COUNT   = 8; // уровни 0–7
const FIRST_UPLOADED_MIP = 5; // начинаем с трёх самых грубых: 5, 6, 7

/** Цвета RGBA для каждого мип-уровня (линейные). */
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

/** Генерирует пиксельные данные мип-уровня с одним сплошным цветом. */
function solidMip(level: number): Uint8Array<ArrayBuffer> {
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

// ---- WGSL ------------------------------------------------------------------------------------------------------------------------------------------

const VERT_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}
@vertex
fn main(@builtin(vertex_index) vi: u32) -> VSOut {
  var pos = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
    vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
  );
  var uv = array<vec2<f32>, 6>(
    vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
    vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(1.0, 0.0),
  );
  var o: VSOut;
  o.pos = vec4<f32>(pos[vi], 0.0, 1.0);
  o.uv  = uv[vi];
  return o;
}
`;

const FRAG_WGSL = /* wgsl */ `
@group(0) @binding(0) var tex:  texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, uv);
}
`;

// ---- Рендер-проход потоковой загрузки мипов ------------------------------------------------------------------------

class MipStreamPass implements IRenderPass {
  readonly name = "mip-stream-pass";

  private _device: GPUDevice | null = null;
  private _texture: GPUTexture | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _sampler: GPUSampler | null = null;
  private _presentationFormat: GPUTextureFormat = "bgra8unorm";

  /** Самый детальный мип-уровень, загруженный в данный момент (меньше = выше разрешение). */
  residentMin = FIRST_UPLOADED_MIP;
  /** Текущее значение lodMinClamp - совпадает с residentMin. */
  lodMinClamp = FIRST_UPLOADED_MIP;

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    this._device = ctx.device;
    this._presentationFormat = ctx.presentationFormat;
    const device = ctx.device;

    device.pushErrorScope("out-of-memory");
    this._texture = device.createTexture({
      label: "mip-stream-tex",
      size: [BASE_SIZE, BASE_SIZE],
      mipLevelCount: MIP_COUNT,
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[MipStreamPass] Нехватка памяти:", e); });

    // Загружаем начальные мипы (5, 6, 7).
    for (let level = FIRST_UPLOADED_MIP; level < MIP_COUNT; level++) {
      this._uploadMip(level);
    }

    this._buildPipeline();
    this._rebuildSamplerAndBindGroup();
  }

  private _uploadMip(level: number): void {
    if (!this._texture || !this._device) return;
    const size = Math.max(1, BASE_SIZE >> level);
    const data = solidMip(level);
    // bytesPerRow должен быть кратен 256.
    const bytesPerRow = Math.max(256, Math.ceil((size * 4) / 256) * 256);
    if (size * 4 === bytesPerRow) {
      this._device.queue.writeTexture(
        { texture: this._texture, mipLevel: level },
        data,
        { bytesPerRow, rowsPerImage: size },
        [size, size]
      );
    } else {
      const padded = new Uint8Array(new ArrayBuffer(bytesPerRow * size));
      for (let row = 0; row < size; row++) {
        padded.set(data.subarray(row * size * 4, (row + 1) * size * 4), row * bytesPerRow);
      }
      this._device.queue.writeTexture(
        { texture: this._texture, mipLevel: level },
        padded,
        { bytesPerRow, rowsPerImage: size },
        [size, size]
      );
    }
  }

  private _buildPipeline(): void {
    if (!this._device || !this._texture) return;
    this._pipeline = this._device.createRenderPipeline({
      label: "mip-stream-pipeline",
      layout: "auto",
      vertex: {
        module: this._device.createShaderModule({ label: "ms-vert", code: VERT_WGSL }),
        entryPoint: "main",
      },
      fragment: {
        module: this._device.createShaderModule({ label: "ms-frag", code: FRAG_WGSL }),
        entryPoint: "main",
        targets: [{ format: this._presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  private _rebuildSamplerAndBindGroup(): void {
    if (!this._device || !this._pipeline || !this._texture) return;
    this._sampler = this._device.createSampler({
      label: `sampler-lod${this.lodMinClamp}`,
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      lodMinClamp: this.lodMinClamp,
      lodMaxClamp: MIP_COUNT - 1,
    });
    this._bindGroup = this._device.createBindGroup({
      label: "mip-stream-bg",
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._texture.createView() },
        { binding: 1, resource: this._sampler },
      ],
    });
  }

  /**
   * Загружает следующий более детальный мип-уровень.
   * Возвращает false, если уже достигнуто полное разрешение.
   */
  streamNextMip(): boolean {
    if (this.residentMin === 0) return false;
    this.residentMin--;
    this._uploadMip(this.residentMin);
    this.lodMinClamp = this.residentMin;
    this._rebuildSamplerAndBindGroup();
    return true;
  }

  /** Вытесняет более детальные мипы (увеличивает lodMinClamp без удаления - только зажим). */
  streamPrevMip(): boolean {
    if (this.residentMin >= MIP_COUNT - 1) return false;
    this.residentMin++;
    this.lodMinClamp = this.residentMin;
    this._rebuildSamplerAndBindGroup();
    return true;
  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._bindGroup) return;
    const pass = ctx.encoder.beginRenderPass({
      label: "mip-stream-rp",
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
    this._texture = null;
    this._pipeline = null;
    this._bindGroup = null;
    this._sampler = null;
    this._device = null;
  }
}

// ---- Подключение ------------------------------------------------------------------------------------------------------------------------------

const el = document.getElementById("canvas") as HTMLElement & {
  clearColor: GPUColorDict;
  addRenderPass: (pass: IRenderPass) => Promise<void>;
};

const btnNext = document.getElementById("btn-next") as HTMLButtonElement;
const btnPrev = document.getElementById("btn-prev") as HTMLButtonElement;
const btnAuto = document.getElementById("btn-auto") as HTMLButtonElement;

function colorName(level: number): string {
  return ["белый","фиолетовый","синий","голубой","зелёный","жёлтый","оранжевый","красный"][level] ?? "?";
}

el.addEventListener("webgpu-error", (e) => {
  overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`);
});

el.addEventListener("webgpu-ready", async () => {
  el.clearColor = { r: 0.08, g: 0.08, b: 0.08, a: 1.0 };

  const pass = new MipStreamPass();
  await el.addRenderPass(pass);

  function updateOverlay() {
    overlay.set("lodMinClamp", pass.lodMinClamp.toFixed(1));
    overlay.set("Resident mip range", `${pass.residentMin}–${MIP_COUNT - 1}`);
    overlay.set("Current color", colorName(pass.residentMin));
    overlay.set("Full res?", pass.residentMin === 0 ? "YES ✓" : "no");
  }
  updateOverlay();
  overlay.set("Status", "Running");

  btnNext.addEventListener("click", () => {
    pass.streamNextMip();
    updateOverlay();
  });
  btnPrev.addEventListener("click", () => {
    pass.streamPrevMip();
    updateOverlay();
  });

  let autoTimer: ReturnType<typeof setInterval> | null = null;
  btnAuto.addEventListener("click", () => {
    if (autoTimer) {
      clearInterval(autoTimer);
      autoTimer = null;
      btnAuto.textContent = "⏵ Авто-поток";
    } else {
      btnAuto.textContent = "⏸ Стоп";
      autoTimer = setInterval(() => {
        const more = pass.streamNextMip();
        updateOverlay();
        if (!more) {
          clearInterval(autoTimer!);
          autoTimer = null;
          btnAuto.textContent = "⏵ Авто-поток";
        }
      }, 500);
    }
  });

  function tick() {
    fps.tick();
    overlay.set("FPS", fps.fps);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
