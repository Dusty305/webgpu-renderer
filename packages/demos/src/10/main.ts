/**
 * Демо 10 - Потоковая загрузка мипов с управлением бюджетом
 *
 * Регистрирует 30 синтетических KTX2-текстур (512×512, уровень 0) в
 * TextureStreamingManager. Отрисовывает сетку 6×5 инстансных квадов.
 * Каждая текстура начинается с самого грубого мипа (белый) и постепенно
 * загружается до своего уникального цвета по мере выделения бюджета.
 *
 * Управление:
 *   Слайдер бюджета      - общий лимит памяти GPU-текстур (8 – 512 МБ)
 *   Слайдер лимита кадра - макс. байт загрузки за кадр (1 – 32 МБ)
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import type { IRenderPass, RenderPassInitContext, FrameContext, ResourceRegistry } from "@webgpu-streaming/gpu-types";
import {
  TextureStreamingManager,
  MATERIAL_BIND_GROUP_KEY,
  parseKTX2,
} from "@webgpu-streaming/texture-streaming";
import { createOverlay, FpsTracker } from "../shared/overlay.js";
import { generateColorKtx2, hsvToRgb } from "./generateColorKtx2.js";

const COLS         = 6;
const ROWS         = 5;
const NUM_TEXTURES = COLS * ROWS;   // 30
const TEX_SIZE     = 512;

// ---- WGSL --------------------------------------------------------------------------------------------------------------------------------------------

const VERT_WGSL = /* wgsl */ `
struct GridUniforms { cols: u32, rows: u32, _p0: u32, _p1: u32 }
@group(1) @binding(0) var<uniform> grid: GridUniforms;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) @interpolate(flat) matId: u32,
}

@vertex fn vs_main(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VSOut {
  var lp = array<vec2<f32>,6>(
    vec2(0.,0.), vec2(1.,0.), vec2(0.,1.),
    vec2(0.,1.), vec2(1.,0.), vec2(1.,1.),
  );
  let cols_f = f32(grid.cols);
  let rows_f = f32(grid.rows);
  let col_f  = f32(ii % grid.cols);
  let row_f  = f32(ii / grid.cols);
  let qw = 2.0 / cols_f;
  let qh = 2.0 / rows_f;
  let m  = 0.004;
  let p  = lp[vi];
  let x  = -1.0 + col_f * qw + m + p.x * (qw - 2.0*m);
  let y  =  1.0 - row_f * qh - m - p.y * (qh - 2.0*m);
  var out: VSOut;
  out.pos   = vec4<f32>(x, y, 0.0, 1.0);
  out.uv    = p;
  out.matId = ii;
  return out;
}`;

const FRAG_WGSL = /* wgsl */ `
struct MaterialEntry { tierIndex: u32, layerIndex: u32, residentMip: u32, _pad: u32 }

@group(0) @binding(0) var tier0Tex:  texture_2d_array<f32>;
@group(0) @binding(1) var tier1Tex:  texture_2d_array<f32>;
@group(0) @binding(2) var tier2Tex:  texture_2d_array<f32>;
@group(0) @binding(3) var tier0Samp: sampler;
@group(0) @binding(4) var tier1Samp: sampler;
@group(0) @binding(5) var tier2Samp: sampler;
@group(0) @binding(6) var<storage, read> materials: array<MaterialEntry>;

@fragment fn fs_main(
  @location(0) uv: vec2<f32>,
  @location(1) @interpolate(flat) matId: u32,
) -> @location(0) vec4<f32> {
  let mat   = materials[matId];
  let layer = i32(mat.layerIndex);
  // textureSampleLevel может вызываться в неравномерном потоке управления.
  // Явно запрашиваем самый детальный резидентный мип, чтобы
  // неинициализированные грубые мипы никогда не использовались.
  let lod   = f32(mat.residentMip);
  var color: vec4<f32>;
  if mat.tierIndex == 0u {
    color = textureSampleLevel(tier0Tex, tier0Samp, uv, layer, lod);
  } else if mat.tierIndex == 1u {
    color = textureSampleLevel(tier1Tex, tier1Samp, uv, layer, lod);
  } else {
    color = textureSampleLevel(tier2Tex, tier2Samp, uv, layer, lod);
  }
  return color;
}`;

// ---- Рендер-проход --------------------------------------------------------------------------------------------------------------------------

class StreamingGridPass implements IRenderPass {
  readonly name = "streaming-grid";

  private _registry: ResourceRegistry | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _gridBindGroup: GPUBindGroup | null = null;
  private _gridBuf: GPUBuffer | null = null;

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, presentationFormat } = ctx;
    this._registry = ctx.registry;

    const vertMod = device.createShaderModule({ label: "sg-vert", code: VERT_WGSL });
    const fragMod = device.createShaderModule({ label: "sg-frag", code: FRAG_WGSL });

    // Группа 0: макет bind group материала - должен точно совпадать с BindGroupManager.createLayout(),
    // чтобы bind group, зарегистрированная в ResourceRegistry, была совместима.
    // Использование layout:"auto" создаёт анонимные макеты, отклоняющие внешние bind groups.
    const matBgl = device.createBindGroupLayout({
      label: "sg-mat-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      ],
    });

    // Группа 1: форм-переменные сетки (буфер форм только для вершинного шейдера)
    const gridBgl = device.createBindGroupLayout({
      label: "sg-grid-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    });

    const pipelineLayout = device.createPipelineLayout({
      label: "sg-pipeline-layout",
      bindGroupLayouts: [matBgl, gridBgl],
    });

    device.pushErrorScope("validation");
    this._pipeline = device.createRenderPipeline({
      label: "streaming-grid-pipeline",
      layout: pipelineLayout,
      vertex: { module: vertMod, entryPoint: "vs_main" },
      fragment: { module: fragMod, entryPoint: "fs_main", targets: [{ format: presentationFormat }] },
      primitive: { topology: "triangle-list" },
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[StreamingGridPass] pipeline error:", e); });

    device.pushErrorScope("out-of-memory");
    this._gridBuf = device.createBuffer({
      label: "grid-uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[StreamingGridPass] OOM grid buffer:", e); });

    device.queue.writeBuffer(this._gridBuf, 0, new Uint32Array([COLS, ROWS, 0, 0]));

    // Используем gridBgl (не pipeline.getBindGroupLayout), чтобы явно задать объект макета.
    this._gridBindGroup = device.createBindGroup({
      label: "grid-bg",
      layout: gridBgl,
      entries: [{ binding: 0, resource: { buffer: this._gridBuf } }],
    });
  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._gridBindGroup) return;
    const matBg = this._registry?.request<GPUBindGroup>(MATERIAL_BIND_GROUP_KEY);
    if (!matBg) return;

    const pass = ctx.encoder.beginRenderPass({
      colorAttachments: [{ view: ctx.colorAttachment, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, matBg);
    pass.setBindGroup(1, this._gridBindGroup);
    pass.draw(6, NUM_TEXTURES);
    pass.end();
  }

  onResize(_w: number, _h: number): void {}

  destroy(): void {
    this._gridBuf?.destroy();
    this._gridBuf = null;
    this._pipeline = null;
    this._gridBindGroup = null;
    this._registry = null;
  }
}

// ---- Подключение ------------------------------------------------------------------------------------------------------------------------------

const overlay = createOverlay("10 - Budget Streaming");
const fps     = new FpsTracker();
overlay.set("Status", "Initializing…");

const el = document.getElementById("canvas") as HTMLElement & {
  clearColor: GPUColorDict;
  addRenderPass: (p: IRenderPass) => Promise<void>;
  pluginHost: { registerResourceManager: (m: unknown) => void } | null;
  registry: unknown;
};
const budgetSlider = document.getElementById("budget-slider") as HTMLInputElement;
const budgetLabel  = document.getElementById("budget-label") as HTMLSpanElement;
const frameSlider  = document.getElementById("frame-slider") as HTMLInputElement;
const frameLabel   = document.getElementById("frame-label") as HTMLSpanElement;

el.addEventListener("webgpu-error", (e) =>
  overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`)
);

el.addEventListener("webgpu-ready", async (e: Event) => {
  const detail = (e as CustomEvent<WebGPUReadyDetail>).detail;
  el.clearColor = { r: 0.08, g: 0.08, b: 0.08, a: 1.0 };

  const DEFAULT_BUDGET      = parseInt(budgetSlider.value, 10) * 1024 * 1024;
  const DEFAULT_FRAME_CAP   = parseInt(frameSlider.value, 10) * 1024 * 1024;

  const streamingManager = new TextureStreamingManager({
    budgetBytes:       DEFAULT_BUDGET,
    frameUploadBudget: DEFAULT_FRAME_CAP,
  });

  // Инициализируем вручную, чтобы вызвать registerTexture до первого кадра.
  await streamingManager.initialize({
    device:   detail.device,
    registry: el.registry as import("@webgpu-streaming/gpu-types").ResourceRegistry,
  });

  // Регистрируем менеджер в PluginHost, чтобы prepareFrame() вызывался каждый кадр.
  el.pluginHost?.registerResourceManager(streamingManager);

  // Генерируем и регистрируем 30 KTX2-текстур.
  const bsphere = new Float32Array([0, 0, -5, 0.5]); // все на одной глубине
  for (let i = 0; i < NUM_TEXTURES; i++) {
    const hue = i / NUM_TEXTURES;
    const [r, g, b] = hsvToRgb(hue, 1, 1);
    const ktx2 = generateColorKtx2(TEX_SIZE, r, g, b);
    const parsed = parseKTX2(ktx2);
    streamingManager.registerTexture(`tex-${i}`, parsed, ktx2, i, bsphere);
  }

  // Добавляем рендер-проход.
  const gridPass = new StreamingGridPass();
  await el.addRenderPass(gridPass);

  // Привязка слайдеров.
  budgetSlider.addEventListener("input", () => {
    const mb = parseInt(budgetSlider.value, 10);
    budgetLabel.textContent = `${mb} MB`;
    if (streamingManager.budgetTracker) {
      streamingManager.budgetTracker.budget = mb * 1024 * 1024;
    }
  });
  frameSlider.addEventListener("input", () => {
    const mb = parseInt(frameSlider.value, 10);
    frameLabel.textContent = `${mb} MB/frame`;
    streamingManager.frameUploadBudget = mb * 1024 * 1024;
  });

  overlay.set("Status", "Стриминг…");
  overlay.set("Текстур", NUM_TEXTURES);
  overlay.set("Размер текстуры", `${TEX_SIZE}×${TEX_SIZE}`);

  function tick() {
    fps.tick();
    overlay.set("FPS", fps.fps);

    const bt = streamingManager.budgetTracker;
    if (bt) {
      const usedMB  = (bt.totalUsed / (1024 * 1024)).toFixed(1);
      const budgMB  = (bt.budget    / (1024 * 1024)).toFixed(0);
      const pct     = (bt.utilization * 100).toFixed(1);
      overlay.set("Budget", `${usedMB} / ${budgMB} MB (${pct}%)`);
    }

    overlay.set("Загрузок/кадр", streamingManager.uploadsLastFrame);
    overlay.set("Вытеснений/кадр", streamingManager.evictionsLastFrame);

    // Считаем текстуры на самом детальном мипе (полностью загружены).
    let fully = 0;
    for (const entry of streamingManager.entries.values()) {
      if (entry.residentMip === 0) fully++;
    }
    overlay.set("Полностью загружено", `${fully} / ${NUM_TEXTURES}`);

    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
