/**
 * Демо 12 - Стенд для бенчмарков (редакция фазы 4.3)
 *
 * Сцена: процедурная стресс-тестовая сцена (пресет StressSceneGenerator "small" -
 *   25 × 512px текстур ≈ 33 МБ RGBA8 полная цепочка мипов).
 *
 * Конфиги:
 *   A - Naive:        безлимитный бюджет, forceFullQuality. Ждёт пока каждая
 *                     текстура достигнет мипа 0, затем записывает 120 стабильных кадров.
 *   B - Compressed:   идентичная загрузка как у A; пиковая память делится на 4 для
 *                     проекции стоимости BC7. Помечено как расчётное.
 *   C - Mip stream:   бюджет 4 МБ, без ограничения на кадр. LRU управляет загрузками.
 *   D - Array mgr:    бюджет 12 МБ, без ограничения на кадр.
 *   E - Full system:  бюджет 8 МБ + ограничение 1 МБ на кадр.
 *
 * Фаза загрузки (только A/B):
 *   Запускает prepareFrame пока fullyLoadedPct = 100 ИЛИ сработает эвристика OOM
 *   ИЛИ не будет достигнут жёсткий таймаут 60 секунд. loadStopReason записывает причину.
 *
 * Фаза измерений (все конфиги):
 *   A/B: 120 кадров после завершения загрузки (установившееся состояние, 0 загрузок/кадр).
 *   C/D/E: 300 кадров (потоковая передача идёт всё время).
 *
 * Фаза качества (все конфиги):
 *   После измерений - рендер трёх фиксированных ракурсов и вычисление PSNR + SSIM
 *   относительно эталонных захватов конфига A.
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import {
  TextureStreamingManager,
  MATERIAL_BIND_GROUP_KEY,
} from "@webgpu-streaming/texture-streaming";
import type { ResourceRegistry } from "@webgpu-streaming/gpu-types";
import type { FrameContext } from "@webgpu-streaming/gpu-types";
import { MATERIAL_ENTRY_WGSL } from "@webgpu-streaming/gpu-types";
import {
  generateStressScene,
  SCENE_PRESETS,
  computeMaxLayersPerTier,
} from "../shared/StressSceneGenerator.js";
import type { StressObject } from "../shared/StressSceneGenerator.js";
import {
  captureFrame,
  computePSNR,
  computeSSIM,
  MEASUREMENT_POSES,
} from "../shared/QualityMetrics.js";

// ---- Константы ----------------------------------------------------------------------------------------------------------------------------------

const PRESET          = SCENE_PRESETS.small; // 25 × 512px ≈ 33 MB
const MEASURE_FRAMES  = 120;                 // кадры установившегося состояния для A/B
const STREAM_FRAMES   = 300;                 // кадры для конфигов потоковой передачи C/D/E
const LOAD_TIMEOUT_MS = 60_000;              // жёсткий таймаут для фазы загрузки
const GRID_COLS           = PRESET.gridCols;
const GRID_ROWS           = Math.ceil(PRESET.objectCount / GRID_COLS);
const POSE_WARMUP_FRAMES  = 20; // кадры для адаптации потоковой передачи к камере каждого ракурса

// ---- WGSL для захвата качества ------------------------------------------------------------------------------------------------------------
// Полностью повторяет шейдер стресс-сетки из демо 12-stress-scene, чтобы
// захваченное изображение было идентично тому, что рендерит живое демо.

const QCAP_VERT_WGSL = /* wgsl */`
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

function makeQCapFragWgsl(): string {
  return /* wgsl */`
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
  let lod = f32(m.residentMip);
  if m.tierIndex == 0u {
    return textureSampleLevel(tier0, samp0, uv, m.layerIndex, lod);
  } else if m.tierIndex == 1u {
    return textureSampleLevel(tier1, samp1, uv, m.layerIndex, lod);
  } else {
    return textureSampleLevel(tier2, samp2, uv, m.layerIndex, lod);
  }
}
`;
}

// ---- Вспомогательные функции --------------------------------------------------------------------------------------------------------------------------------------

/** Расстояние от камеры до ближайшей точки поверхности ограничивающей сферы. */
function sphereDistance(sphere: Float32Array, cam: Float32Array): number {
  const dx = (sphere[0] ?? 0) - (cam[0] ?? 0);
  const dy = (sphere[1] ?? 0) - (cam[1] ?? 0);
  const dz = (sphere[2] ?? 0) - (cam[2] ?? 0);
  return Math.max(0, Math.sqrt(dx*dx + dy*dy + dz*dz) - (sphere[3] ?? 0));
}

/**
 * Желаемый уровень мипа для текстуры на заданном расстоянии от камеры.
 * Повторяет формулу из TextureStreamingManager / mip-math.ts.
 */
function desiredMipLocal(dist: number, texWidth: number, fovY: number, screenHeight: number): number {
  if (dist <= 0 || texWidth <= 0 || screenHeight <= 0) return 0;
  const projPx = screenHeight / (2 * dist * Math.tan(fovY / 2));
  if (projPx <= 0) return 0;
  return Math.max(0, Math.floor(Math.log2(texWidth / projPx)));
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.min(Math.ceil(sorted.length * p) - 1, sorted.length - 1);
  return sorted[idx] ?? 0;
}
function median(arr: number[]): number { return percentile(arr, 0.5); }

function makeFakeCtx(device: GPUDevice, viewportWidth: number, viewportHeight: number): FrameContext {
  return {
    device,
    encoder:     device.createCommandEncoder(),
    camera: {
      viewMatrix:           new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      projectionMatrix:     new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      viewProjectionMatrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      position:  new Float32Array([0, 0, 5]),  // камера на расстоянии 5 единиц от сетки
      fovY:      Math.PI / 4,
      near:      0.1,
      far:       1000,
      viewportWidth,
      viewportHeight,
    },
    scene:      { nodes: [] },
    frameIndex: 0,
    deltaTime:  1 / 60,
    colorAttachment: null as unknown as GPUTextureView,
    depthAttachment: null as unknown as GPUTextureView,
  };
}

// ---- Схема результатов --------------------------------------------------------------------------------------------------------------------------

export interface QualityMeasurement {
  pose:    string;
  psnrDb:  number;
  ssim:    number;
}

export interface BenchmarkResult {
  config:                  string;
  description:             string;
  peakGPUMemoryMB:         number;
  peakGPUMemoryNote:       string;
  loadTimeMs:              number;
  loadCompleted:           boolean;
  /** "complete" | "oom" | "timeout" для A/B; "streaming" для C/D/E */
  loadStopReason:          "complete" | "oom" | "timeout" | "streaming";
  frameMedianMs:           number;
  frameP95Ms:              number;
  frameP99Ms:              number;
  fullyLoadedPct:          number;
  textureCount:            number;
  totalTextureBytesBudget: number;
  compressionFormat:       string;
  measurementFrames:       number;
  qualityMeasurements:     QualityMeasurement[];
  /** Время (мс) от начала измерений до сходимости всех видимых текстур. */
  convergenceMs:           number | null;
  /** Индекс кадра в фазе измерений, когда сходимость была достигнута впервые. */
  convergenceFrames:       number | null;
  /** Количество кадров измерений, в которых хотя бы одна видимая текстура изменила уровень мипа. */
  popInDurationFrames:     number;
  /** Количество текстур, видимых во время измерения качества. */
  visibleTextureCount:     number;
}

// ---- Таблица конфигов ----------------------------------------------------------------------------------------------------------------------------

interface ConfigDef {
  id:                string;
  label:             string;
  budgetBytes:       number;
  frameUploadBudget: number;
  forceFullQuality:  boolean;
  isStreaming:       boolean;
}

const CONFIGS: ConfigDef[] = [
  {
    id: "A",
    label: "Naive - unlimited budget, full quality, no streaming",
    budgetBytes:       512 * 1024 * 1024 * 1024,
    frameUploadBudget: 512 * 1024 * 1024 * 1024,
    forceFullQuality:  true,
    isStreaming:       false,
  },
  {
    id: "B",
    label: "Compressed - BC7 projected (÷4 memory, same upload path as A)",
    budgetBytes:       512 * 1024 * 1024 * 1024,
    frameUploadBudget: 512 * 1024 * 1024 * 1024,
    forceFullQuality:  true,
    isStreaming:       false,
  },
  {
    id: "C",
    label: "Mip streaming - 4 MB budget, no per-frame cap (heavy LRU)",
    budgetBytes:        4 * 1024 * 1024,
    frameUploadBudget: 512 * 1024 * 1024,
    forceFullQuality:  false,
    isStreaming:       true,
  },
  {
    id: "D",
    label: "Array manager - 12 MB budget, no per-frame cap (moderate LRU)",
    budgetBytes:       12 * 1024 * 1024,
    frameUploadBudget: 512 * 1024 * 1024,
    forceFullQuality:  false,
    isStreaming:       true,
  },
  {
    id: "E",
    label: "Full system - 8 MB budget + 1 MB/frame cap (slow stream + LRU)",
    budgetBytes:        8 * 1024 * 1024,
    frameUploadBudget:  1 * 1024 * 1024,
    forceFullQuality:  false,
    isStreaming:       true,
  },
];

// ---- Пайплайн захвата качества ----------------------------------------------------------------------------------------------------

interface QualCapPipeline {
  pipeline:   GPURenderPipeline;
  matBgl:     GPUBindGroupLayout;
  gridBgl:    GPUBindGroupLayout;
  gridUniBuf: GPUBuffer;
  gridBg:     GPUBindGroup;
  destroy:    () => void;
}

/**
 * Создаёт пайплайн для захватов метрик качества.
 * Должен вызываться после инициализации первого менеджера потоковой передачи,
 * чтобы можно было переиспользовать layout matBgl для совместимости группы привязки.
 *
 * @param device       - Активный GPUDevice.
 * @param streamingBgl - Layout из streaming.bindGroupLayout (группа 0).
 */
function createQualCapPipeline(
  device: GPUDevice,
  streamingBgl: GPUBindGroupLayout,
): QualCapPipeline {
  // Переиспользуем точный layout менеджера потоковой передачи, чтобы группа привязки,
  // созданная BindGroupManager, была совместима с этим пайплайном.
  const matBgl = streamingBgl;

  const gridBgl = device.createBindGroupLayout({
    label: "qcap-grid-bgl",
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
  });

  const layout = device.createPipelineLayout({
    label: "qcap-pipeline-layout",
    bindGroupLayouts: [matBgl, gridBgl],
  });

  const pipeline = device.createRenderPipeline({
    label: "qcap-pipeline",
    layout,
    vertex: {
      module:     device.createShaderModule({ code: QCAP_VERT_WGSL }),
      entryPoint: "main",
    },
    fragment: {
      module:     device.createShaderModule({ code: makeQCapFragWgsl() }),
      entryPoint: "main",
      targets: [{ format: "rgba8unorm" }],
    },
    primitive: { topology: "triangle-list" },
  });

  const gridUniBuf = device.createBuffer({
    label: "qcap-grid-uni",
    size:  8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gridUniBuf, 0, new Uint32Array([GRID_COLS, GRID_ROWS]));

  const gridBg = device.createBindGroup({
    label:   "qcap-grid-bg",
    layout:  gridBgl,
    entries: [{ binding: 0, resource: { buffer: gridUniBuf } }],
  });

  return {
    pipeline, matBgl, gridBgl, gridUniBuf, gridBg,
    destroy() { gridUniBuf.destroy(); },
  };
}

/**
 * Захватывает один кадр сравнения качества для каждого MEASUREMENT_POSE.
 *
 * Перед захватом каждого ракурса менеджер потоковой передачи прогревается
 * в течение POSE_WARMUP_FRAMES кадров с позицией камеры данного ракурса. Это гарантирует,
 * что состояние резидентности мипов отражает решения потоковой передачи,
 * подходящие для каждой точки зрения (крупный план = желательны более детальные мипы;
 * общий план = более грубые), что даёт действительно различные значения PSNR/SSIM между ракурсами.
 *
 * @param device        - Активный GPUDevice.
 * @param registry      - ResourceRegistry, содержащий группу привязки материалов.
 * @param qcap          - Заранее собранный пайплайн захвата качества.
 * @param objectCount   - Количество инстанцированных объектов для отрисовки.
 * @param refFrames     - Эталонные пиксели конфига A (null при захвате самого A).
 * @param outRefFrames  - Если не null, заполняется захваченными пикселями для каждого ракурса.
 * @param streaming     - Живой менеджер потоковой передачи (используется для прогревочных кадров).
 * @param ctx           - Контекст кадра, изменяемый для каждого ракурса во время прогрева.
 * @returns Измерения PSNR (дБ) и SSIM для каждого ракурса.
 */
async function captureQuality(
  device:       GPUDevice,
  registry:     ResourceRegistry,
  qcap:         QualCapPipeline,
  objectCount:  number,
  refFrames:    ReadonlyMap<string, Uint8Array> | null,
  outRefFrames: Map<string, Uint8Array> | null,
  streaming:    TextureStreamingManager,
  ctx:          FrameContext,
  captureW:     number,
  captureH:     number,
): Promise<QualityMeasurement[]> {
  const results: QualityMeasurement[] = [];

  for (const pose of MEASUREMENT_POSES) {
    // Направляем менеджер потоковой передачи на камеру данного ракурса,
    // чтобы он загружал или вытеснял мипы, подходящие для этой точки зрения.
    ctx.camera.position.set(pose.position);
    for (let i = 0; i < POSE_WARMUP_FRAMES; i++) {
      (ctx as Record<string, unknown>)["encoder"]    = device.createCommandEncoder();
      (ctx as Record<string, unknown>)["frameIndex"] = (ctx.frameIndex as number) + 1;
      streaming.prepareFrame(ctx);
      device.queue.submit([(ctx.encoder as GPUCommandEncoder).finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    // Повторно получаем группу привязки: прогрев мог пересобрать её после вытеснений/загрузок.
    const matBg = registry.request<GPUBindGroup>(MATERIAL_BIND_GROUP_KEY);
    if (!matBg) {
      results.push({ pose: pose.name, psnrDb: 0, ssim: 0 });
      continue;
    }

    // Захватываем состояние потоковой передачи во внеэкранную текстуру RGBA8.
    const pixels = await captureFrame(device, captureW, captureH, (encoder, colorView) => {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view:       colorView,
          clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 },
          loadOp:     "clear",
          storeOp:    "store",
        }],
      });
      pass.setPipeline(qcap.pipeline);
      pass.setBindGroup(0, matBg);
      pass.setBindGroup(1, qcap.gridBg);
      pass.draw(6, objectCount);
      pass.end();
    });

    // Сохраняем для последующего сравнения (только для конфига A).
    outRefFrames?.set(pose.name, pixels);

    const ref = refFrames?.get(pose.name);
    if (!ref) {
      // Конфиг A: самосравнение → идеальный результат.
      // Используем 100 вместо Infinity, чтобы значение было JSON-сериализуемым
      // (JSON.stringify(Infinity) === "null"). 100 дБ - условный маркер "идентично";
      // ни одна реальная сцена никогда не превышает ~60 дБ.
      results.push({ pose: pose.name, psnrDb: 100, ssim: 1.0 });
    } else {
      const raw = computePSNR(ref, pixels);
      results.push({
        pose:   pose.name,
        psnrDb: isFinite(raw) ? raw : 100, // ограничиваем Infinity (идентичные кадры)
        ssim:   computeSSIM(ref, pixels, captureW, captureH),
      });
    }
  }

  return results;
}

// ---- Запуск ----------------------------------------------------------------------------------------------------------------------------------------

async function runConfig(
  cfg:          ConfigDef,
  device:       GPUDevice,
  registry:     ResourceRegistry,
  objects:      StressObject[],
  hasBC:        boolean,
  refFrames:    ReadonlyMap<string, Uint8Array> | null,
  outRefFrames: Map<string, Uint8Array> | null,
  captureW:     number,
  captureH:     number,
): Promise<BenchmarkResult> {
  // FrameContext по контракту интерфейса readonly, но бенчмарк мутирует
  // encoder, frameIndex и camera.position на месте - тот же паттерн,
  // используемый во всём стенде бенчмарков фазы 4.x.

  const maxLayers = computeMaxLayersPerTier(PRESET);
  const streaming = new TextureStreamingManager({
    budgetBytes:       cfg.budgetBytes,
    frameUploadBudget: cfg.frameUploadBudget,
    maxLayersPerTier:  maxLayers,
    forceFullQuality:  cfg.forceFullQuality,
  });
  await streaming.initialize({ device, registry });

  // Собираем пайплайн захвата качества, используя собственный layout группы привязки
  // менеджера потоковой передачи для гарантии совместимости.
  const bgl = streaming.bindGroupLayout;
  const qcap = bgl ? createQualCapPipeline(device, bgl) : null;

  // Регистрируем все текстуры.
  const t0 = performance.now();
  for (const obj of objects) {
    streaming.registerTexture(
      obj.id, obj.parsed, obj.ktx2Bytes, obj.materialId, obj.boundingSphere
    );
  }

  const ctx = makeFakeCtx(device, captureW, captureH);
  let frameIndex = 0;

  // ---- Фаза загрузки (только конфиги A/B) --------------------------------------------------------------------------------
  let loadStopReason: "complete" | "oom" | "timeout" | "streaming" = "streaming";
  let loadTimeMs   = -1;
  let loadFrameCount = 0; // кадры, выполненные в ходе фазы загрузки

  if (!cfg.isStreaming) {
    let stuckFrames   = 0;
    let lastTotalUsed = -1;

    while (true) {
      (ctx as Record<string, unknown>)["encoder"]    = device.createCommandEncoder();
      (ctx as Record<string, unknown>)["frameIndex"] = frameIndex++;
      streaming.prepareFrame(ctx);
      device.queue.submit([(ctx.encoder as GPUCommandEncoder).finish()]);
      loadFrameCount++;

      const allLoaded = [...streaming.entries.values()].every(e => e.residentMip === 0);
      if (allLoaded) {
        loadStopReason = "complete";
        loadTimeMs     = performance.now() - t0;
        break;
      }

      if (performance.now() - t0 >= LOAD_TIMEOUT_MS) {
        loadStopReason = "timeout";
        break;
      }

      // Эвристика OOM: загрузки прекратились, но сцена не загружена полностью.
      const currentUsed = streaming.budgetTracker?.totalUsed ?? 0;
      if (streaming.uploadsLastFrame === 0 && currentUsed === lastTotalUsed) {
        if (++stuckFrames >= 5) { loadStopReason = "oom"; break; }
      } else {
        stuckFrames = 0;
      }
      lastTotalUsed = currentUsed;

      // Уступаем управление, чтобы страница оставалась отзывчивой при потенциально долгих загрузках.
      await new Promise<void>((r) => setTimeout(r, 0));
    }
    // Сбрасываем работу GPU, чтобы все промисы mapAsync staging-кольца разрешились до начала
    // цикла измерений (предотвращает голодание "slot pending" при обороте кольца).
    await device.queue.onSubmittedWorkDone();
  }

  // ---- Фаза измерений ------------------------------------------------------------------------------------------------------------
  const measureFrames = cfg.isStreaming ? STREAM_FRAMES : MEASURE_FRAMES;
  const frameTimes: number[] = [];
  let peakGPUMemory = streaming.budgetTracker?.totalUsed ?? 0;

  // ---- Настройка сходимости + проявления ------------------------------------------------------------------------------------------
  // Для A/B: сходимость уже произошла в конце фазы загрузки.
  // Для C/D/E: отслеживаем покадрово.
  let convergenceFrame: number | null = null;
  let convergenceTime:  number | null = null;
  let popInFrames = 0;
  const visibleTextureCount = objects.length; // плоская сетка - все тайлы видимы

  if (!cfg.isStreaming && loadStopReason === "complete") {
    // A/B: сходимость = завершение загрузки.
    convergenceFrame = loadFrameCount;
    convergenceTime  = loadTimeMs;
    // Проявление: в каждом кадре загрузки была хотя бы одна загрузка мипа.
    popInFrames = loadFrameCount;
  }

  // Для C/D/E: делаем снимок резидентных мипов перед началом цикла измерений.
  const prevMips = new Map<string, number>();
  if (cfg.isStreaming) {
    for (const [id, entry] of streaming.entries) {
      prevMips.set(id, entry.residentMip);
    }
  }

  const measureT0 = performance.now();

  for (let f = 0; f < measureFrames; f++) {
    const ft0 = performance.now();
    (ctx as Record<string, unknown>)["encoder"]    = device.createCommandEncoder();
    (ctx as Record<string, unknown>)["frameIndex"] = frameIndex++;
    streaming.prepareFrame(ctx);
    device.queue.submit([(ctx.encoder as GPUCommandEncoder).finish()]);
    await device.queue.onSubmittedWorkDone();
    const ft1 = performance.now();
    frameTimes.push(ft1 - ft0);

    const bt = streaming.budgetTracker;
    if (bt && bt.totalUsed > peakGPUMemory) peakGPUMemory = bt.totalUsed;

    // ---- Отслеживание сходимости и проявления (только для конфигов потоковой передачи) ----------------------
    if (cfg.isStreaming) {
      let anyChanged  = false;
      let allConverged = true;

      for (const [id, entry] of streaming.entries) {
        const prev = prevMips.get(id) ?? entry.residentMip;
        if (entry.residentMip !== prev) {
          anyChanged = true;
          prevMips.set(id, entry.residentMip);
        }

        // Критерий сходимости: residentMip ≤ desiredMip + 1
        const dist    = sphereDistance(entry.boundingSphere, ctx.camera.position);
        const desired = cfg.forceFullQuality
          ? 0
          : desiredMipLocal(dist, entry.parsed.pixelWidth, ctx.camera.fovY, ctx.camera.viewportHeight);
        if (entry.residentMip > desired + 1) {
          allConverged = false;
        }
      }

      if (anyChanged) popInFrames++;

      if (allConverged && convergenceFrame === null) {
        convergenceFrame = f;
        convergenceTime  = performance.now() - measureT0;
      }
    }
  }

  // Для конфигов потоковой передачи записываем суммарное прошедшее время как loadTimeMs.
  if (cfg.isStreaming) {
    loadTimeMs     = performance.now() - t0;
    loadStopReason = "streaming";
  }

  // Итоговая проверка полноты загрузки.
  let fullyCount = 0;
  for (const e of streaming.entries.values()) {
    if (e.residentMip === 0) fullyCount++;
  }

  // ---- Захват качества (до уничтожения) ----------------------------------------------------------------------------
  let qualityMeasurements: QualityMeasurement[] = [];
  if (qcap) {
    qualityMeasurements = await captureQuality(
      device, registry, qcap, objects.length, refFrames, outRefFrames,
      streaming, ctx, captureW, captureH,
    );
    qcap.destroy();
  }

  streaming.destroy();

  // Конфиг B: проецируем измеренную память RGBA8 вниз в 4× для оценки BC7.
  const isConfigB      = cfg.id === "B";
  const compressionFmt = isConfigB ? (hasBC ? "bc7" : "bc7-projected") : "rgba8";
  const compressRatio  = isConfigB ? 4 : 1;
  const rawMB          = peakGPUMemory / (1024 * 1024);
  const reportedMB     = rawMB / compressRatio;
  const memNote        = isConfigB
    ? `projected BC7 (actual RGBA8 upload = ${rawMB.toFixed(1)} MB, ÷4)`
    : "measured RGBA8";

  const sorted = [...frameTimes].sort((a, b) => a - b);

  return {
    config:               cfg.id,
    description:          cfg.label,
    peakGPUMemoryMB:      reportedMB,
    peakGPUMemoryNote:    memNote,
    loadTimeMs,
    loadCompleted:        loadStopReason === "complete",
    loadStopReason,
    frameMedianMs:        median(sorted),
    frameP95Ms:           percentile(sorted, 0.95),
    frameP99Ms:           percentile(sorted, 0.99),
    fullyLoadedPct:       (fullyCount / objects.length) * 100,
    textureCount:         objects.length,
    totalTextureBytesBudget: PRESET.expectedTotalBytes,
    compressionFormat:    compressionFmt,
    measurementFrames:    measureFrames,
    qualityMeasurements,
    convergenceMs:        convergenceTime,
    convergenceFrames:    convergenceFrame,
    popInDurationFrames:  popInFrames,
    visibleTextureCount,
  };
}

// ---- Привязка DOM --------------------------------------------------------------------------------------------------------------------------------

const runBtn    = document.getElementById("run-btn")     as HTMLButtonElement;
const exportBtn = document.getElementById("export-btn")  as HTMLButtonElement;
const progressEl= document.getElementById("progress")   as HTMLSpanElement;
const genProgEl = document.getElementById("gen-progress") as HTMLSpanElement;
const tbody     = document.getElementById("results-body")!;
const rawJsonEl = document.getElementById("raw-json")!;

const canvasEl = document.getElementById("bench-canvas") as HTMLElement & {
  addResourceManager: (m: unknown) => Promise<void>;
  registry: ResourceRegistry;
  canvasElement: HTMLCanvasElement;
};

let allResults: BenchmarkResult[] = [];
let sceneObjects: StressObject[] | null = null;

function psnrStr(psnr: number): string {
  return psnr >= 100 ? "∞ (identical)" : `${psnr.toFixed(1)} dB`;
}

function renderRow(r: BenchmarkResult): void {
  const cls       = `cfg-${r.config.toLowerCase()}`;
  const loadStr   = r.loadTimeMs < 0 ? "-" : `${r.loadTimeMs.toFixed(0)} ms`;
  const statusStr =
      r.loadStopReason === "complete"  ? "✓ done"
    : r.loadStopReason === "oom"       ? "⚠ OOM"
    : r.loadStopReason === "timeout"   ? "⏱ timeout"
    : r.loadStopReason === "streaming" ? "≈ streaming"
    : "-";

  // Качество: показываем PSNR для каждого ракурса как "overview / closeup / midrange".
  const poseQ    = (name: string) => r.qualityMeasurements.find(q => q.pose === name);
  const overQ    = poseQ("overview");
  const closeQ   = poseQ("closeup");
  const midQ     = poseQ("midrange");
  const psnrCell = overQ
    ? `${psnrStr(overQ.psnrDb)} / ${psnrStr(closeQ?.psnrDb ?? 0)} / ${psnrStr(midQ?.psnrDb ?? 0)}`
    : "-";
  const ssimCell = overQ
    ? `${overQ.ssim.toFixed(3)} / ${(closeQ?.ssim ?? 0).toFixed(3)} / ${(midQ?.ssim ?? 0).toFixed(3)}`
    : "-";

  const convStr  = r.convergenceMs === null
    ? "-"
    : `${r.convergenceMs.toFixed(0)} ms (f${r.convergenceFrames ?? "?"})`;

  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td class="${cls}"><strong>${r.config}</strong></td>
    <td style="max-width:220px;color:#999;font-size:11px">${r.description}</td>
    <td>${r.peakGPUMemoryMB.toFixed(1)}</td>
    <td title="${r.peakGPUMemoryNote}">${r.compressionFormat}</td>
    <td>${loadStr}</td>
    <td>${statusStr}</td>
    <td>${r.frameMedianMs.toFixed(2)}</td>
    <td>${r.frameP95Ms.toFixed(2)}</td>
    <td>${r.frameP99Ms.toFixed(2)}</td>
    <td>${r.fullyLoadedPct.toFixed(0)}%</td>
    <td title="overview / closeup / midrange">${psnrCell}</td>
    <td title="overview / closeup / midrange">${ssimCell}</td>
    <td>${convStr}</td>
    <td>${r.popInDurationFrames} / ${r.measurementFrames + (r.convergenceFrames ?? 0)}</td>
  `;
  tbody.appendChild(tr);
}

canvasEl.addEventListener("webgpu-ready", async (e: Event) => {
  const { device } = (e as CustomEvent<WebGPUReadyDetail>).detail;
  const registry   = (canvasEl as unknown as { registry: ResourceRegistry }).registry;
  const hasBC      = device.features.has("texture-compression-bc");
  const hasASTC    = device.features.has("texture-compression-astc");

  progressEl.textContent = `Device ready (BC: ${hasBC}, ASTC: ${hasASTC}) - generating scene…`;

  // Предварительно генерируем сцену один раз, чтобы все конфиги использовали идентичные текстуры.
  sceneObjects = await generateStressScene(PRESET, (done, total, label) => {
    genProgEl.textContent = `Generating: ${label} (${Math.round(done/total*100)}%)`;
  });
  genProgEl.textContent =
    `Scene ready: ${PRESET.objectCount} objects, ` +
    `${(PRESET.expectedTotalBytes / 1024 / 1024).toFixed(1)} MB expected`;

  progressEl.textContent = "Ready - click Run to start";
  runBtn.disabled = false;

  runBtn.addEventListener("click", async () => {
    if (!sceneObjects) return;
    runBtn.disabled = true;
    tbody.innerHTML = "";
    allResults      = [];

    // Эталонные кадры, захваченные из конфига A, используются для вычисления PSNR/SSIM
    // для конфигов B–E на тех же трёх ракурсах измерений.
    const refFrames    = new Map<string, Uint8Array>();
    let   refPopulated = false;

    for (const cfg of CONFIGS) {
      progressEl.textContent =
        `Running Config ${cfg.id}… ` +
        `(${cfg.isStreaming ? STREAM_FRAMES : "load + " + MEASURE_FRAMES} frames` +
        ` + quality capture)`;

      const outRefFrames = cfg.id === "A" ? refFrames : null;
      const inRefFrames  = refPopulated ? refFrames : null;

      const result = await runConfig(
        cfg, device, registry, sceneObjects, hasBC, inRefFrames, outRefFrames,
        canvasEl.canvasElement.width,
        canvasEl.canvasElement.height,
      );

      if (cfg.id === "A") refPopulated = true;

      allResults.push(result);
      renderRow(result);
      // Уступаем управление между конфигами, чтобы браузер мог "отдышаться".
      await new Promise<void>((r) => setTimeout(r, 100));
    }

    progressEl.textContent =
      `Done - ${CONFIGS.length} configs. ` +
      `Expected scene: ${(PRESET.expectedTotalBytes / 1024 / 1024).toFixed(1)} MB RGBA8.`;
    exportBtn.disabled = false;
    runBtn.disabled    = false;

    rawJsonEl.textContent  = JSON.stringify(allResults, null, 2);
    rawJsonEl.style.display = "block";
  });

  exportBtn.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(allResults, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `benchmark-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
});

canvasEl.addEventListener("webgpu-error", (e) => {
  progressEl.textContent = `WebGPU error: ${(e as CustomEvent<string>).detail}`;
});
