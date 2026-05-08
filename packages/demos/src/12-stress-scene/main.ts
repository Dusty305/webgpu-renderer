/**
 * Демо 12 - Генератор сцены для стресс-тестирования
 *
 * Генерирует большую сетку квадратов с процедурными текстурами с помощью
 * StressSceneGenerator, регистрирует их в TextureStreamingManager
 * (безлимитный бюджет - режим Config A) и рендерит сцену с
 * орбитальной камерой. Цель: проверить, что генератор сцен создаёт
 * ожидаемую нагрузку на память GPU перед запуском многоконфигурационных бенчмарков.
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import { CameraController } from "@webgpu-streaming/core";
import {
  TextureStreamingManager,
  MATERIAL_BIND_GROUP_KEY,
} from "@webgpu-streaming/texture-streaming";
import type { IRenderPass, RenderPassInitContext, FrameContext, ResourceRegistry } from "@webgpu-streaming/gpu-types";
import { MATERIAL_ENTRY_WGSL } from "@webgpu-streaming/gpu-types";
import {
  SCENE_PRESETS,
  generateStressScene,
  computeMaxLayersPerTier,
} from "../shared/StressSceneGenerator.js";
import type { ScenePreset, StressObject } from "../shared/StressSceneGenerator.js";
import { FpsTracker } from "../shared/overlay.js";

// ---- Ссылки на DOM --------------------------------------------------------------------------------------------------------------------------

const canvasEl  = document.getElementById("canvas")          as HTMLElement;
const genBtn    = document.getElementById("gen-btn")         as HTMLButtonElement;
const presetSel = document.getElementById("preset-select")   as HTMLSelectElement;
const warnEl    = document.getElementById("warn")            as HTMLSpanElement;
const ovStatus  = document.getElementById("ov-status")       as HTMLSpanElement;
const ovObjects = document.getElementById("ov-objects")      as HTMLSpanElement;
const ovExpected= document.getElementById("ov-expected")     as HTMLSpanElement;
const ovTracked = document.getElementById("ov-tracked")      as HTMLSpanElement;
const ovUploads = document.getElementById("ov-uploads")      as HTMLSpanElement;
const ovFps     = document.getElementById("ov-fps")          as HTMLSpanElement;
const progLabel = document.getElementById("progress-label")  as HTMLSpanElement;
const progBar   = document.getElementById("progress-bar")    as HTMLDivElement;

function mb(bytes: number): string { return (bytes / 1024 / 1024).toFixed(1) + " МБ"; }

// ---- WGSL ------------------------------------------------------------------------------------------------------------------------------------------

const VERT_WGSL = /* wgsl */`
struct GridUni { cols: u32, rows: u32 }
@group(1) @binding(0) var<uniform> grid: GridUni;

struct VSOut {
  @builtin(position)              pos:        vec4<f32>,
  @location(0)                    uv:         vec2<f32>,
  @location(1) @interpolate(flat) materialId: u32,
}

@vertex
fn main(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> VSOut {
  let col = inst % grid.cols;
  let row = inst / grid.cols;
  let cw  = 2.0 / f32(grid.cols);
  let ch  = 2.0 / f32(grid.rows);
  let pad = 0.015;
  let x0  = -1.0 + f32(col) * cw + pad;
  let y0  = -1.0 + f32(row) * ch + pad;
  let x1  = x0 + cw - pad * 2.0;
  let y1  = y0 + ch - pad * 2.0;
  var pos = array<vec2<f32>,6>(
    vec2(x0,y0), vec2(x1,y0), vec2(x0,y1),
    vec2(x0,y1), vec2(x1,y0), vec2(x1,y1));
  var uvs = array<vec2<f32>,6>(
    vec2(0,0), vec2(1,0), vec2(0,1),
    vec2(0,1), vec2(1,0), vec2(1,1));
  var o: VSOut;
  o.pos        = vec4<f32>(pos[vi], 0.0, 1.0);
  o.uv         = uvs[vi];
  o.materialId = inst;
  return o;
}
`;

const FRAG_WGSL = /* wgsl */`
${MATERIAL_ENTRY_WGSL}
@group(0) @binding(0) var tier0: texture_2d_array<f32>;
@group(0) @binding(1) var tier1: texture_2d_array<f32>;
@group(0) @binding(2) var tier2: texture_2d_array<f32>;
@group(0) @binding(3) var samp0: sampler;
@group(0) @binding(4) var samp1: sampler;
@group(0) @binding(5) var samp2: sampler;
@group(0) @binding(6) var<storage, read> materials: array<MaterialEntry>;

@fragment
fn main(
  @location(0)                    uv:         vec2<f32>,
  @location(1) @interpolate(flat) materialId: u32,
) -> @location(0) vec4<f32> {
  let m   = materials[materialId];
  let lod = f32(m.residentMip);   // сэмплируем лучший mip, который действительно загружен
  if m.tierIndex == 0u {
    return textureSampleLevel(tier0, samp0, uv, m.layerIndex, lod);
  } else if m.tierIndex == 1u {
    return textureSampleLevel(tier1, samp1, uv, m.layerIndex, lod);
  } else {
    return textureSampleLevel(tier2, samp2, uv, m.layerIndex, lod);
  }
}
`;

// ---- Render pass ------------------------------------------------------------------------------------------------------------------------------

class StressGridPass implements IRenderPass {
  readonly name = "stress-grid-pass";

  private _pipeline:    GPURenderPipeline | null = null;
  private _gridUniBuf:  GPUBuffer | null         = null;
  private _gridBgl:     GPUBindGroupLayout | null = null;
  private _gridBg:      GPUBindGroup | null       = null;
  private _matBgl:      GPUBindGroupLayout | null = null;
  private _objectCount  = 0;
  private _gridCols     = 1;
  private _gridRows     = 1;
  private _registry:    ResourceRegistry | null = null;

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, presentationFormat, registry } = ctx;
    this._registry = registry;

    // Схема bind group для материалов - должна точно соответствовать BindGroupManager.
    this._matBgl = device.createBindGroupLayout({
      label: "stress-mat-bgl",
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

    // Схема uniform-буфера сетки (группа 1).
    this._gridBgl = device.createBindGroupLayout({
      label: "stress-grid-bgl",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    });

    const layout = device.createPipelineLayout({
      label: "stress-pipeline-layout",
      bindGroupLayouts: [this._matBgl, this._gridBgl],
    });

    device.pushErrorScope("validation");
    this._pipeline = device.createRenderPipeline({
      label: "stress-pipeline",
      layout,
      vertex:   { module: device.createShaderModule({ code: VERT_WGSL }), entryPoint: "main" },
      fragment: { module: device.createShaderModule({ code: FRAG_WGSL }), entryPoint: "main",
                  targets: [{ format: presentationFormat }] },
      primitive: { topology: "triangle-list" },
    });
    void device.popErrorScope().then((e) => {
      if (e) console.error("[StressGridPass] Ошибка пайплайна:", e);
    });
  }

  /**
   * Настраивает размеры сетки и создаёт uniform-буфер.
   * Должна быть вызвана после initialize() и до первого кадра.
   */
  setup(device: GPUDevice, cols: number, rows: number, objectCount: number): void {
    this._gridCols   = cols;
    this._gridRows   = rows;
    this._objectCount = objectCount;

    this._gridUniBuf?.destroy();
    this._gridUniBuf = device.createBuffer({
      label: "stress-grid-uni",
      size:  8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this._gridUniBuf, 0, new Uint32Array([cols, rows]));

    this._gridBg = device.createBindGroup({
      label: "stress-grid-bg",
      layout: this._gridBgl!,
      entries: [{ binding: 0, resource: { buffer: this._gridUniBuf } }],
    });
  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._gridBg || !this._registry || this._objectCount === 0) return;
    const matBg = this._registry.request<GPUBindGroup>(MATERIAL_BIND_GROUP_KEY);
    if (!matBg) return;

    const pass = ctx.encoder.beginRenderPass({
      colorAttachments: [{ view: ctx.colorAttachment, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, matBg);
    pass.setBindGroup(1, this._gridBg);
    pass.draw(6, this._objectCount);
    pass.end();
  }

  onResize(_w: number, _h: number): void {}

  destroy(): void {
    this._gridUniBuf?.destroy();
    this._gridUniBuf  = null;
    this._pipeline    = null;
    this._gridBg      = null;
    this._gridBgl     = null;
    this._matBgl      = null;
    this._registry    = null;
  }
}

// ---- Подключение ------------------------------------------------------------------------------------------------------------------------------

const el = canvasEl as HTMLElement & {
  setCameraController: (c: unknown) => void;
  addResourceManager:  (m: unknown) => Promise<void>;
  addRenderPass:       (p: unknown) => Promise<void>;
  canvasElement:       HTMLCanvasElement;
  registry:            ResourceRegistry;
  device:              GPUDevice | null;
};

const fps = new FpsTracker();
let streaming: TextureStreamingManager | null = null;
let renderPass: StressGridPass | null = null;
let cameraCtrl: CameraController | null = null;

// Предупреждения для больших пресетов
const LARGE_PRESETS = new Set(["medium", "large", "thesis"]);
const LARGE_WARN = "⚠ Может потребовать 500 МБ–2 ГБ памяти GPU. При нехватке памяти будет сообщено об ошибке.";

presetSel.addEventListener("change", () => {
  warnEl.textContent = LARGE_PRESETS.has(presetSel.value) ? LARGE_WARN : "";
});

el.addEventListener("webgpu-error", (e) => {
  ovStatus.textContent = `ОШИБКА: ${(e as CustomEvent<string>).detail}`;
});

el.addEventListener("webgpu-ready", async (e: Event) => {
  const { device } = (e as CustomEvent<WebGPUReadyDetail>).detail;

  // Подключаем орбитальную камеру.
  cameraCtrl = new CameraController(el.canvasElement);
  el.setCameraController(cameraCtrl);

  genBtn.disabled = false;
  ovStatus.textContent = "Готово - выберите пресет и нажмите «Сгенерировать»";

  genBtn.addEventListener("click", async () => {
    genBtn.disabled   = true;
    presetSel.disabled = true;
    ovStatus.textContent = "Генерация текстур…";

    const preset: ScenePreset = SCENE_PRESETS[presetSel.value as keyof typeof SCENE_PRESETS];

    ovObjects.textContent  = String(preset.objectCount);
    ovExpected.textContent = mb(preset.expectedTotalBytes);

    // Генерация текстур с отображением прогресса.
    let objects: StressObject[];
    try {
      objects = await generateStressScene(preset, (done, total, label) => {
        progLabel.textContent = `Генерация: ${label}`;
        progBar.style.width   = `${(done / total) * 100}%`;
      });
    } catch (err) {
      ovStatus.textContent = `Ошибка генерации: ${err}`;
      genBtn.disabled = false;
      presetSel.disabled = false;
      return;
    }

    progLabel.textContent = "Регистрация в менеджере потоковой передачи…";
    progBar.style.width   = "100%";

    // Уничтожаем предыдущий менеджер потоковой передачи при повторной генерации.
    streaming?.destroy();

    const maxLayers = computeMaxLayersPerTier(preset);
    streaming = new TextureStreamingManager({
      budgetBytes:       512 * 1024 * 1024 * 1024, // безлимитный (Config A)
      frameUploadBudget: 512 * 1024 * 1024 * 1024,
      maxLayersPerTier:  maxLayers,
      forceFullQuality:  true,                      // загружать все mip-уровни независимо от расстояния до камеры
    });

    await el.addResourceManager(streaming);

    for (const obj of objects) {
      streaming.registerTexture(
        obj.id, obj.parsed, obj.ktx2Bytes, obj.materialId, obj.boundingSphere
      );
    }

    // Создаём или заменяем render pass.
    if (!renderPass) {
      renderPass = new StressGridPass();
      await el.addRenderPass(renderPass);
    }
    const gridRows = Math.ceil(preset.objectCount / preset.gridCols);
    renderPass.setup(device, preset.gridCols, gridRows, preset.objectCount);

    // Регулируем расстояние камеры в зависимости от пресета.
    cameraCtrl?.setRadius(preset.cameraDistance);

    // Блокируем управление - повторная генерация привела бы к двойной регистрации в PluginHost; для повтора нужна перезагрузка.
    presetSel.disabled = true;
    progLabel.textContent = "Готово - перезагрузите страницу для смены пресета";

    ovStatus.textContent = "Выполняется (Config A - безлимитный бюджет)";

    // Цикл обновления статистики.
    function tick() {
      fps.tick();
      ovFps.textContent     = fps.fps.toFixed(1);
      const bt = streaming?.budgetTracker;
      ovTracked.textContent = bt ? mb(bt.totalUsed) : "-";
      ovUploads.textContent = String(streaming?.uploadsLastFrame ?? 0);
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
});
