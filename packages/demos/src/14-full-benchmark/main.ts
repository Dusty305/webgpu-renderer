/**
 * Демо 14 - Полный многократный бенчмарк (Фаза 5.1)
 *
 * Запускает все 5 конфигураций по 30 раз каждую с паузой 2 секунды между запусками.
 * Первые WARMUP_RUNS (3) запуска на конфигурацию отбрасываются.
 * Производит два JSON-экспорта:
 *   - Raw:     одна запись на запуск (150 итого) с полными массивами времён кадров
 *   - Summary: одна запись на конфигурацию со среднеарифметическим ± std за 27 эффективных запусков
 *
 * Порядок внешнего цикла: все 30 запусков конфигурации A, затем B, C, D, E.
 * Конфигурация A захватывает эталонные кадры PSNR в первый не-warmup запуск;
 * эти кадры повторно используются при сравнении всех последующих конфигураций.
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import {
  TextureStreamingManager,
  MATERIAL_BIND_GROUP_KEY,
} from "@webgpu-streaming/texture-streaming";
import type { ResourceRegistry, FrameContext } from "@webgpu-streaming/gpu-types";
import { MATERIAL_ENTRY_WGSL } from "@webgpu-streaming/gpu-types";
import {
  generateStressScene,
  SCENE_PRESETS,
  computeMaxLayersPerTier,
} from "../shared/StressSceneGenerator.js";
import type { StressObject, ScenePreset, ScenePresetName } from "../shared/StressSceneGenerator.js";
import {
  captureFrame,
  computePSNR,
  computeSSIM,
  MEASUREMENT_POSES,
} from "../shared/QualityMetrics.js";
import {
  runPairwiseTests,
  holmBonferroniCorrection,
} from "../shared/WilcoxonTest.js";
import type { WilcoxonResult, HolmResult } from "../shared/WilcoxonTest.js";

// ---- Константы --------------------------------------------------------------------------------------------------------------------------------

const TOTAL_RUNS        = 30;   // всего запусков на конфигурацию (включая прогрев)
const WARMUP_RUNS       = 3;    // первые N запусков на конфигурацию отбрасываются
const EFFECTIVE_RUNS    = TOTAL_RUNS - WARMUP_RUNS; // 27 учитываемых запусков
const COOLDOWN_MS       = 2_000;
const LOAD_TIMEOUT_MS   = 60_000;
const MEASURE_FRAMES    = 120;  // кадры устойчивого состояния для A/B
const STREAM_FRAMES     = 300;  // кадры стриминга для C/D/E
const POSE_WARMUP_FRAMES = 20;  // кадры прогрева на позу захвата качества
const ORBIT_FRAMES      = 300;  // кадры динамической орбиты камеры

// ---- WGSL-шейдеры --------------------------------------------------------------------------------------------------------------------------

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

// ---- Типы --------------------------------------------------------------------------------------------------------------------------------------------

/** Статистика по кадрам, собранная во время фазы динамической орбиты камеры. */
export interface DynamicCameraStats {
  orbitFrames:       number;
  totalUploads:      number;
  totalEvictions:    number;
  peakMemoryMB:      number;
  frameTimeMedianMs: number;
  frameTimeP99Ms:    number;
  finalPsnrDb:       number;
  finalSsim:         number;
  /** Количество вытеснений на кадр (длина = orbitFrames). */
  perFrameEvictions: number[];
}

/** Необработанные данные одного запуска бенчмарка. */
export interface RunData {
  runIndex:        number;   // индекс с 0 среди эффективных (не-warmup) запусков
  config:          string;
  peakGPUMemoryMB: number;
  /** Все измеренные времена кадров в мс (MEASURE_FRAMES или STREAM_FRAMES записей). */
  frameTimesMs:    number[];
  convergenceMs:   number | null;
  /** Доля [0,1] текстур с residentMip ≤ desiredMip+1 в конце фазы измерений. */
  fullyLoadedPct:  number;
  qualityPsnrDb:   { overview: number; closeup: number; midrange: number };
  qualitySsim:     { overview: number; closeup: number; midrange: number };
  uploadCount:     number;   // всего загрузок mip по всем кадрам этого запуска
  evictionCount:   number;   // всего LRU-вытеснений по всем кадрам этого запуска
  dynamicCamera:   DynamicCameraStats;
  timestamp:       string;   // ISO 8601 - момент завершения этого запуска
}

/** Среднее ± std для скалярной метрики. */
export interface StatSummary { mean: number; std: number }

/** Агрегированная статистика по EFFECTIVE_RUNS запускам для одной конфигурации. */
export interface AggregateStats {
  config:          string;
  runs:            number;  // должно равняться EFFECTIVE_RUNS
  /** Присутствует и равно true, когда данные о памяти - прогноз, а не измеренные значения. */
  projected?:      boolean;
  frameTimeMedian: StatSummary;
  frameTimeP95:    StatSummary;
  frameTimeP99:    StatSummary;
  peakMemoryMB:    StatSummary;
  /** null, если все запуски вернули null для convergence (ограничение бюджетом). */
  convergenceMs:   StatSummary | null;
  psnrDb: {
    overview: StatSummary;
    closeup:  StatSummary;
    midrange: StatSummary;
  };
  ssim: {
    overview: StatSummary;
    closeup:  StatSummary;
    midrange: StatSummary;
  };
  uploadCount:    StatSummary;
  evictionCount:  StatSummary;
  fullyLoadedPct: StatSummary;
  dynamicCamera: {
    totalEvictions:    StatSummary;
    totalUploads:      StatSummary;
    peakMemoryMB:      StatSummary;
    frameTimeMedianMs: StatSummary;
    frameTimeP99Ms:    StatSummary;
    finalPsnrDb:       StatSummary;
    finalSsim:         StatSummary;
  };
}

/** GPU и браузерное окружение, захваченное в начале бенчмарка. */
export interface BenchmarkEnvironment {
  userAgent:    string;
  gpu: {
    vendor:       string;
    architecture: string;
    device:       string;
    description:  string;
    features:     string[];
    limits:       Record<string, number>;
  };
  screen:  { width: number; height: number; devicePixelRatio: number };
  canvas:  { width: number; height: number };
  captureResolution: string;
  timestamp:    string;
  scenePreset:  string;
  runsPerConfig: number;
  effectiveRunsPerConfig: number;
}

export interface RawExport     { environment: BenchmarkEnvironment; runs:    RunData[];       }
export interface SummaryExport {
  environment:           BenchmarkEnvironment;
  configs:               AggregateStats[];
  pairwiseTests:         WilcoxonResult[];
  holmBonferroniResults: HolmResult[];
}

// ---- Таблица конфигураций ------------------------------------------------------------------------------------------------------------

interface ConfigDef {
  id:                string;
  label:             string;
  budgetBytes:       number;
  frameUploadBudget: number;
  forceFullQuality:  boolean;
  isStreaming:       boolean;
  /** True, когда данные о памяти - прогноз (напр. оценка экономии BC7), а не измеренные. */
  projected?:        boolean;
}

/**
 * Строит таблицу из 5 конфигураций с бюджетами, откалиброванными относительно размера сцены.
 *
 * Потоковые конфигурации C/D/E нуждаются в бюджетах НИЖЕ естественного
 * множества сходимости стриминга, чтобы бюджетное давление и LRU-вытеснение действительно срабатывали.
 * Прежние фиксированные бюджеты (4/12/8 МБ) превышали множество сходимости для всех
 * размеров сцены - вытеснение никогда не запускалось.
 *
 * Формула: clamp(totalBytes × pct, minBytes, maxBytes)
 *   C: 0.1% от total, min 256 КБ, max 1.5 МБ  → ~0.5 МБ (thesis), ~256 КБ (small)
 *   D: 0.25% от total, min 512 КБ, max 2.0 МБ → ~1.25 МБ (thesis), ~512 КБ (small)
 *   E: 0.4% от total, min 1 МБ, max 2.5 МБ    → ~2.0 МБ (thesis), ~1.0 МБ (small)
 *
 * Наблюдаемые множества сходимости: 0.83 МБ (small), 2.15 МБ (medium), 2.79 МБ (thesis).
 * Все три потоковые конфигурации попадают ниже этих значений, обеспечивая C < D < E по пиковой памяти.
 */
function makeConfigs(totalSceneBytes: number): ConfigDef[] {
  function clampBudget(pct: number, minB: number, maxB: number): number {
    return Math.max(minB, Math.min(maxB, Math.round(totalSceneBytes * pct)));
  }

  const unlimited = 512 * 1024 * 1024 * 1024;
  const budgetC = clampBudget(0.001,  256 * 1024,       1.5 * 1024 * 1024);
  const budgetD = clampBudget(0.0025, 512 * 1024,       2.0 * 1024 * 1024);
  const budgetE = clampBudget(0.004,  1024 * 1024,      2.5 * 1024 * 1024);
  const mb = (b: number) => (b / (1024 * 1024)).toFixed(2);

  return [
    {
      id: "A", label: "Наивный - безлимитный бюджет, полное качество, без стриминга",
      budgetBytes: unlimited, frameUploadBudget: unlimited,
      forceFullQuality: true,  isStreaming: false,
    },
    {
      id: "B", label: "Прогноз BC7 (оценочный, не измеренный) - ÷4 памяти, тот же путь загрузки, что у A",
      budgetBytes: unlimited, frameUploadBudget: unlimited,
      forceFullQuality: true,  isStreaming: false, projected: true,
    },
    {
      id: "C", label: `Mip-стриминг - бюджет ${mb(budgetC)} МБ, без ограничения на кадр (активное LRU)`,
      budgetBytes: budgetC, frameUploadBudget: unlimited,
      forceFullQuality: false, isStreaming: true,
    },
    {
      id: "D", label: `Array manager - бюджет ${mb(budgetD)} МБ, без ограничения на кадр (умеренное LRU)`,
      budgetBytes: budgetD, frameUploadBudget: unlimited,
      forceFullQuality: false, isStreaming: true,
    },
    {
      id: "E", label: `Полная система - бюджет ${mb(budgetE)} МБ + ограничение 2 МБ/кадр (медленный стриминг + LRU)`,
      budgetBytes: budgetE, frameUploadBudget: 2 * 1024 * 1024,
      forceFullQuality: false, isStreaming: true,
    },
  ];
}

// ---- Вспомогательные функции ------------------------------------------------------------------------------------------------------

function percentile(sorted: number[], p: number): number {
  const idx = Math.min(Math.ceil(sorted.length * p) - 1, sorted.length - 1);
  return sorted[idx] ?? 0;
}
function median(arr: number[]): number { return percentile(arr, 0.5); }

function computeStats(values: number[]): StatSummary {
  if (values.length === 0) return { mean: 0, std: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / (values.length - 1);
  return { mean, std: Math.sqrt(variance) };
}

function lerpCamera(
  start: readonly [number, number, number],
  end:   readonly [number, number, number],
  t: number,
): [number, number, number] {
  return [
    start[0] + (end[0] - start[0]) * t,
    start[1] + (end[1] - start[1]) * t,
    start[2] + (end[2] - start[2]) * t,
  ];
}

function sphereDistance(sphere: Float32Array, cam: Float32Array): number {
  const dx = (sphere[0]??0)-(cam[0]??0), dy = (sphere[1]??0)-(cam[1]??0), dz = (sphere[2]??0)-(cam[2]??0);
  return Math.max(0, Math.sqrt(dx*dx+dy*dy+dz*dz) - (sphere[3]??0));
}
function desiredMipLocal(dist: number, texWidth: number, fovY: number, screenHeight: number): number {
  if (dist <= 0 || texWidth <= 0 || screenHeight <= 0) return 0;
  const projPx = screenHeight / (2 * dist * Math.tan(fovY / 2));
  return projPx <= 0 ? 0 : Math.max(0, Math.floor(Math.log2(texWidth / projPx)));
}

function formatDuration(ms: number): string {
  if (ms < 0 || !isFinite(ms)) return "-";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

function statStr(s: StatSummary): string {
  return `${s.mean.toFixed(2)} ±${s.std.toFixed(2)}`;
}

function psnrStr(psnr: number): string {
  return psnr >= 100 ? "∞" : psnr.toFixed(1);
}

async function cooldown(): Promise<void> {
  await new Promise<void>(r => setTimeout(r, COOLDOWN_MS));
  // Подсказываем GC, если доступен Chrome-специфичный API.
  if ("measureUserAgentSpecificMemory" in performance) {
    try { await (performance as Record<string, unknown>)["measureUserAgentSpecificMemory"]; } catch {}
  }
}

function makeFakeCtx(device: GPUDevice, viewportWidth: number, viewportHeight: number): FrameContext {
  return {
    device,
    encoder:     device.createCommandEncoder(),
    camera: {
      viewMatrix:           new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      projectionMatrix:     new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      viewProjectionMatrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
      position:  new Float32Array([0, 0, 5]),
      fovY:      Math.PI / 4,
      near:      0.1,
      far:       1000,
      viewportWidth,
      viewportHeight,
    },
    scene:       { nodes: [] },
    frameIndex:  0,
    deltaTime:   1 / 60,
    colorAttachment: null as unknown as GPUTextureView,
    depthAttachment: null as unknown as GPUTextureView,
  };
}

// ---- Пайплайн захвата качества --------------------------------------------------------------------------------------------------

interface QualCapPipeline {
  pipeline:   GPURenderPipeline;
  gridUniBuf: GPUBuffer;
  gridBg:     GPUBindGroup;
  destroy:    () => void;
}

function createQualCapPipeline(
  device:       GPUDevice,
  streamingBgl: GPUBindGroupLayout,
  gridCols:     number,
  gridRows:     number,
): QualCapPipeline {
  const gridBgl = device.createBindGroupLayout({
    label: "qcap-grid-bgl",
    entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
  });
  const layout = device.createPipelineLayout({
    label: "qcap-layout",
    bindGroupLayouts: [streamingBgl, gridBgl],
  });
  const pipeline = device.createRenderPipeline({
    label: "qcap-pipeline", layout,
    vertex:   { module: device.createShaderModule({ code: QCAP_VERT_WGSL }), entryPoint: "main" },
    fragment: { module: device.createShaderModule({ code: makeQCapFragWgsl() }), entryPoint: "main",
                targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const gridUniBuf = device.createBuffer({
    label: "qcap-grid-uni", size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(gridUniBuf, 0, new Uint32Array([gridCols, gridRows]));
  const gridBg = device.createBindGroup({
    label: "qcap-grid-bg", layout: gridBgl,
    entries: [{ binding: 0, resource: { buffer: gridUniBuf } }],
  });
  return { pipeline, gridUniBuf, gridBg, destroy() { gridUniBuf.destroy(); } };
}

// ---- Захват качества ----------------------------------------------------------------------------------------------------------------------

interface PoseQuality {
  overview: number;
  closeup:  number;
  midrange: number;
}

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
): Promise<{ psnrDb: PoseQuality; ssim: PoseQuality }> {
  const psnrDb: PoseQuality = { overview: 0, closeup: 0, midrange: 0 };
  const ssim:   PoseQuality = { overview: 0, closeup: 0, midrange: 0 };

  for (const pose of MEASUREMENT_POSES) {
    ctx.camera.position.set(pose.position);
    for (let i = 0; i < POSE_WARMUP_FRAMES; i++) {
      (ctx as Record<string, unknown>)["encoder"]    = device.createCommandEncoder();
      (ctx as Record<string, unknown>)["frameIndex"] = (ctx.frameIndex as number) + 1;
      streaming.prepareFrame(ctx);
      device.queue.submit([(ctx.encoder as GPUCommandEncoder).finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    const matBg = registry.request<GPUBindGroup>(MATERIAL_BIND_GROUP_KEY);
    if (!matBg) continue;

    const pixels = await captureFrame(device, captureW, captureH, (encoder, colorView) => {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: colorView, clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 },
          loadOp: "clear", storeOp: "store",
        }],
      });
      pass.setPipeline(qcap.pipeline);
      pass.setBindGroup(0, matBg);
      pass.setBindGroup(1, qcap.gridBg);
      pass.draw(6, objectCount);
      pass.end();
    });

    outRefFrames?.set(pose.name, pixels);

    const ref = refFrames?.get(pose.name);
    const key = pose.name as keyof PoseQuality;
    if (!ref) {
      psnrDb[key] = 100; ssim[key] = 1.0;
    } else {
      const raw = computePSNR(ref, pixels);
      psnrDb[key] = isFinite(raw) ? raw : 100;
      ssim[key]   = computeSSIM(ref, pixels, captureW, captureH);
    }
  }

  return { psnrDb, ssim };
}

// ---- Одиночный запуск бенчмарка ------------------------------------------------------------------------------------------------

/** Выполняет один полный проход бенчмарка для одной конфигурации. Возвращает raw-данные запуска. */
async function runSingle(
  cfg:          ConfigDef,
  device:       GPUDevice,
  registry:     ResourceRegistry,
  objects:      StressObject[],
  preset:       ScenePreset,
  refFrames:    ReadonlyMap<string, Uint8Array> | null,
  outRefFrames: Map<string, Uint8Array> | null,
  captureW:     number,
  captureH:     number,
): Promise<Omit<RunData, "runIndex" | "config" | "timestamp">> {
  // hasBC напрямую не используется - конфигурация B применяет проекцию ÷4 к отчётным МБ.

  const gridCols  = preset.gridCols;
  const gridRows  = Math.ceil(preset.objectCount / gridCols);
  const maxLayers = computeMaxLayersPerTier(preset);

  const streaming = new TextureStreamingManager({
    budgetBytes:       cfg.budgetBytes,
    frameUploadBudget: cfg.frameUploadBudget,
    maxLayersPerTier:  maxLayers,
    forceFullQuality:  cfg.forceFullQuality,
  });
  await streaming.initialize({ device, registry });

  const bgl  = streaming.bindGroupLayout;
  const qcap = bgl ? createQualCapPipeline(device, bgl, gridCols, gridRows) : null;

  for (const obj of objects) {
    streaming.registerTexture(obj.id, obj.parsed, obj.ktx2Bytes, obj.materialId, obj.boundingSphere);
  }

  const ctx = makeFakeCtx(device, captureW, captureH);
  let frameIndex = 0;
  let totalUploads   = 0;
  let totalEvictions = 0;

  // ---- Фаза загрузки (только A/B) --------------------------------------------------------------------------------------
  let loadFrameCount = 0;
  let convergenceFrame: number | null = null;
  let convergenceTime:  number | null = null;
  let popInFrames = 0;

  if (!cfg.isStreaming) {
    let stuckFrames = 0, lastUsed = -1;
    const t0 = performance.now();
    while (true) {
      (ctx as Record<string, unknown>)["encoder"]    = device.createCommandEncoder();
      (ctx as Record<string, unknown>)["frameIndex"] = frameIndex++;
      streaming.prepareFrame(ctx);
      device.queue.submit([(ctx.encoder as GPUCommandEncoder).finish()]);
      loadFrameCount++;
      totalUploads   += streaming.uploadsLastFrame;
      totalEvictions += streaming.evictionsLastFrame;

      if ([...streaming.entries.values()].every(e => e.residentMip === 0)) {
        convergenceFrame = loadFrameCount;
        convergenceTime  = performance.now() - t0;
        popInFrames      = loadFrameCount; // every loading frame had mip changes
        break;
      }
      if (performance.now() - t0 >= LOAD_TIMEOUT_MS) break;

      const used = streaming.budgetTracker?.totalUsed ?? 0;
      if (streaming.uploadsLastFrame === 0 && used === lastUsed) {
        if (++stuckFrames >= 5) break;
      } else { stuckFrames = 0; }
      lastUsed = used;
      await new Promise<void>(r => setTimeout(r, 0));
    }
    await device.queue.onSubmittedWorkDone();
  }

  // ---- Фаза измерений ----------------------------------------------------------------------------------------------------------------
  const measureFrames = cfg.isStreaming ? STREAM_FRAMES : MEASURE_FRAMES;
  const frameTimes: number[] = [];
  let peakGPUMemory = streaming.budgetTracker?.totalUsed ?? 0;

  const prevMips = new Map<string, number>();
  if (cfg.isStreaming) {
    for (const [id, e] of streaming.entries) prevMips.set(id, e.residentMip);
  }
  const measureT0 = performance.now();

  for (let f = 0; f < measureFrames; f++) {
    const ft0 = performance.now();
    (ctx as Record<string, unknown>)["encoder"]    = device.createCommandEncoder();
    (ctx as Record<string, unknown>)["frameIndex"] = frameIndex++;
    streaming.prepareFrame(ctx);
    device.queue.submit([(ctx.encoder as GPUCommandEncoder).finish()]);
    await device.queue.onSubmittedWorkDone();
    frameTimes.push(performance.now() - ft0);

    const bt = streaming.budgetTracker;
    if (bt && bt.totalUsed > peakGPUMemory) peakGPUMemory = bt.totalUsed;
    totalUploads   += streaming.uploadsLastFrame;
    totalEvictions += streaming.evictionsLastFrame;

    if (cfg.isStreaming) {
      let anyChanged = false, allConverged = true;
      for (const [id, e] of streaming.entries) {
        const prev = prevMips.get(id) ?? e.residentMip;
        if (e.residentMip !== prev) { anyChanged = true; prevMips.set(id, e.residentMip); }
        const dist    = sphereDistance(e.boundingSphere, ctx.camera.position);
        const desired = desiredMipLocal(dist, e.parsed.pixelWidth, ctx.camera.fovY, ctx.camera.viewportHeight);
        if (e.residentMip > desired + 1) allConverged = false;
      }
      if (anyChanged) popInFrames++;
      if (allConverged && convergenceFrame === null) {
        convergenceFrame = f;
        convergenceTime  = performance.now() - measureT0;
      }
    }
  }

  // Конфигурация B: делим отчётную память на 4 для прогноза BC7.
  const isConfigB    = cfg.id === "B";
  const compressRatio = isConfigB ? 4 : 1;
  const reportedMB   = (peakGPUMemory / (1024 * 1024)) / compressRatio;

  // ---- Доля полностью загруженных текстур ------------------------------------------------------------------------
  let fullyLoaded = 0;
  for (const [, e] of streaming.entries) {
    const dist    = sphereDistance(e.boundingSphere, ctx.camera.position);
    const desired = desiredMipLocal(dist, e.parsed.pixelWidth, ctx.camera.fovY, ctx.camera.viewportHeight);
    if (e.residentMip <= desired + 1) fullyLoaded++;
  }
  const fullyLoadedPct = streaming.entries.size > 0
    ? fullyLoaded / streaming.entries.size
    : 0;

  // ---- Захват качества --------------------------------------------------------------------------------------------------------------
  let qualityPsnrDb: PoseQuality = { overview: 0, closeup: 0, midrange: 0 };
  let qualitySsim:   PoseQuality = { overview: 0, closeup: 0, midrange: 0 };
  if (qcap) {
    ({ psnrDb: qualityPsnrDb, ssim: qualitySsim } = await captureQuality(
      device, registry, qcap, objects.length, refFrames, outRefFrames, streaming, ctx,
      captureW, captureH,
    ));
  }

  // ---- Динамическая орбита камеры ----------------------------------------------------------------------------------------
  // Камера перемещается из одного угла сетки сцены в противоположный,
  // принудительно перерасставляя приоритеты и вызывая LRU-вытеснение при жёстких бюджетах.
  const halfX = (preset.gridCols - 1) / 2 * preset.spacing;
  const halfY = (gridRows        - 1) / 2 * preset.spacing;
  const orbitStart: readonly [number, number, number] = [-halfX, -halfY, 8];
  const orbitEnd:   readonly [number, number, number] = [+halfX, +halfY, 2];

  let orbitPeak = streaming.budgetTracker?.totalUsed ?? 0;
  let orbitUploads = 0, orbitEvictions = 0;
  const orbitFrameTimes: number[] = [];
  const perFrameEvictions: number[] = [];

  for (let f = 0; f < ORBIT_FRAMES; f++) {
    const t = ORBIT_FRAMES > 1 ? f / (ORBIT_FRAMES - 1) : 0;
    const pos = lerpCamera(orbitStart, orbitEnd, t);
    ctx.camera.position[0] = pos[0];
    ctx.camera.position[1] = pos[1];
    ctx.camera.position[2] = pos[2];

    const ft0 = performance.now();
    (ctx as Record<string, unknown>)["encoder"]    = device.createCommandEncoder();
    (ctx as Record<string, unknown>)["frameIndex"] = frameIndex++;
    streaming.prepareFrame(ctx);
    device.queue.submit([(ctx.encoder as GPUCommandEncoder).finish()]);
    await device.queue.onSubmittedWorkDone();
    orbitFrameTimes.push(performance.now() - ft0);

    const bt = streaming.budgetTracker;
    if (bt && bt.totalUsed > orbitPeak) orbitPeak = bt.totalUsed;
    orbitUploads   += streaming.uploadsLastFrame;
    orbitEvictions += streaming.evictionsLastFrame;
    perFrameEvictions.push(streaming.evictionsLastFrame);
  }

  // Захватываем один обзорный образец качества в конце орбиты.
  let orbitPsnr = 0, orbitSsim = 0;
  if (qcap) {
    const firstPose = MEASUREMENT_POSES[0];
    if (firstPose) {
      ctx.camera.position.set(firstPose.position);
      // Один кадр прогрева в конечной позиции.
      (ctx as Record<string, unknown>)["encoder"]    = device.createCommandEncoder();
      (ctx as Record<string, unknown>)["frameIndex"] = frameIndex++;
      streaming.prepareFrame(ctx);
      device.queue.submit([(ctx.encoder as GPUCommandEncoder).finish()]);
      await device.queue.onSubmittedWorkDone();

      const matBg = registry.request<GPUBindGroup>(MATERIAL_BIND_GROUP_KEY);
      if (matBg) {
        const pixels = await captureFrame(device, captureW, captureH, (encoder, colorView) => {
          const pass = encoder.beginRenderPass({
            colorAttachments: [{
              view: colorView, clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1 },
              loadOp: "clear", storeOp: "store",
            }],
          });
          pass.setPipeline(qcap.pipeline);
          pass.setBindGroup(0, matBg);
          pass.setBindGroup(1, qcap.gridBg);
          pass.draw(6, objects.length);
          pass.end();
        });
        const ref = refFrames?.get(firstPose.name);
        if (!ref) {
          orbitPsnr = 100; orbitSsim = 1.0;
        } else {
          const raw = computePSNR(ref, pixels);
          orbitPsnr = isFinite(raw) ? raw : 100;
          orbitSsim = computeSSIM(ref, pixels, captureW, captureH);
        }
      }
    }
    qcap.destroy();
  }

  const orbitSorted = [...orbitFrameTimes].sort((a, b) => a - b);
  const dynamicCamera: DynamicCameraStats = {
    orbitFrames:       ORBIT_FRAMES,
    totalUploads:      orbitUploads,
    totalEvictions:    orbitEvictions,
    peakMemoryMB:      (isConfigB ? orbitPeak / 4 : orbitPeak) / (1024 * 1024),
    frameTimeMedianMs: median(orbitSorted),
    frameTimeP99Ms:    percentile(orbitSorted, 0.99),
    finalPsnrDb:       orbitPsnr,
    finalSsim:         orbitSsim,
    perFrameEvictions,
  };

  streaming.destroy();

  return {
    peakGPUMemoryMB: reportedMB,
    frameTimesMs:    frameTimes,
    convergenceMs:   convergenceTime,
    fullyLoadedPct,
    qualityPsnrDb,
    qualitySsim,
    uploadCount:     totalUploads,
    evictionCount:   totalEvictions,
    dynamicCamera,
  };
}

// ---- Агрегированная статистика --------------------------------------------------------------------------------------------------

function computeAggregate(configId: string, runs: RunData[], projected?: boolean): AggregateStats {
  const sorted = (r: RunData) => [...r.frameTimesMs].sort((a, b) => a - b);

  const medians = runs.map(r => median(sorted(r)));
  const p95s    = runs.map(r => percentile(sorted(r), 0.95));
  const p99s    = runs.map(r => percentile(sorted(r), 0.99));
  const mems    = runs.map(r => r.peakGPUMemoryMB);
  const convVals = runs.map(r => r.convergenceMs).filter((v): v is number => v !== null);

  return {
    config:          configId,
    runs:            runs.length,
    ...(projected ? { projected: true } : {}),
    frameTimeMedian: computeStats(medians),
    frameTimeP95:    computeStats(p95s),
    frameTimeP99:    computeStats(p99s),
    peakMemoryMB:    computeStats(mems),
    convergenceMs:   convVals.length > 0 ? computeStats(convVals) : null,
    psnrDb: {
      overview: computeStats(runs.map(r => r.qualityPsnrDb.overview)),
      closeup:  computeStats(runs.map(r => r.qualityPsnrDb.closeup)),
      midrange: computeStats(runs.map(r => r.qualityPsnrDb.midrange)),
    },
    ssim: {
      overview: computeStats(runs.map(r => r.qualitySsim.overview)),
      closeup:  computeStats(runs.map(r => r.qualitySsim.closeup)),
      midrange: computeStats(runs.map(r => r.qualitySsim.midrange)),
    },
    uploadCount:    computeStats(runs.map(r => r.uploadCount)),
    evictionCount:  computeStats(runs.map(r => r.evictionCount)),
    fullyLoadedPct: computeStats(runs.map(r => r.fullyLoadedPct)),
    dynamicCamera: {
      totalEvictions:    computeStats(runs.map(r => r.dynamicCamera.totalEvictions)),
      totalUploads:      computeStats(runs.map(r => r.dynamicCamera.totalUploads)),
      peakMemoryMB:      computeStats(runs.map(r => r.dynamicCamera.peakMemoryMB)),
      frameTimeMedianMs: computeStats(runs.map(r => r.dynamicCamera.frameTimeMedianMs)),
      frameTimeP99Ms:    computeStats(runs.map(r => r.dynamicCamera.frameTimeP99Ms)),
      finalPsnrDb:       computeStats(runs.map(r => r.dynamicCamera.finalPsnrDb)),
      finalSsim:         computeStats(runs.map(r => r.dynamicCamera.finalSsim)),
    },
  };
}

// ---- Метаданные окружения ------------------------------------------------------------------------------------------------------------

/**
 * Собирает все числовые ключи из GPUSupportedLimits.
 * Перечисляем известные имена ограничений вместо итерации по непрозрачному объекту,
 * поскольку GPUSupportedLimits не имеет публичного итератора.
 */
const ALL_LIMIT_KEYS: ReadonlyArray<keyof GPUSupportedLimits> = [
  "maxTextureDimension1D",
  "maxTextureDimension2D",
  "maxTextureDimension3D",
  "maxTextureArrayLayers",
  "maxBindGroups",
  "maxBindGroupsPlusVertexBuffers",
  "maxBindingsPerBindGroup",
  "maxDynamicUniformBuffersPerPipelineLayout",
  "maxDynamicStorageBuffersPerPipelineLayout",
  "maxSampledTexturesPerShaderStage",
  "maxSamplersPerShaderStage",
  "maxStorageBuffersPerShaderStage",
  "maxStorageTexturesPerShaderStage",
  "maxUniformBuffersPerShaderStage",
  "maxUniformBufferBindingSize",
  "maxStorageBufferBindingSize",
  "minUniformBufferOffsetAlignment",
  "minStorageBufferOffsetAlignment",
  "maxVertexBuffers",
  "maxBufferSize",
  "maxVertexAttributes",
  "maxVertexBufferArrayStride",
  "maxInterStageShaderVariables",
  "maxColorAttachments",
  "maxColorAttachmentBytesPerSample",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupSizeY",
  "maxComputeWorkgroupSizeZ",
  "maxComputeWorkgroupsPerDimension",
];

function collectEnvironment(
  device:      GPUDevice,
  adapterInfo: GPUAdapterInfo | null,
  canvasEl:    HTMLElement,
  presetName:  string,
): BenchmarkEnvironment {
  const info = adapterInfo ?? {} as GPUAdapterInfo;

  // Собираем все доступные числовые ограничения.
  const allLimits: Record<string, number> = {};
  for (const key of ALL_LIMIT_KEYS) {
    const v = device.limits[key];
    if (typeof v === "number") allLimits[key] = v;
  }

  return {
    userAgent: navigator.userAgent,
    gpu: {
      vendor:       info.vendor       ?? "unknown",
      architecture: info.architecture ?? "unknown",
      device:       info.device       ?? "unknown",
      description:  info.description  ?? "unknown",
      features:     [...device.features].sort(),
      limits:       allLimits,
    },
    screen: {
      width: screen.width, height: screen.height,
      devicePixelRatio: devicePixelRatio,
    },
    canvas: {
      width:  canvasEl.canvasElement.width,
      height: canvasEl.canvasElement.height,
    },
    captureResolution: `${canvasEl.canvasElement.width}×${canvasEl.canvasElement.height}`,
    timestamp:              new Date().toISOString(),
    scenePreset:            presetName,
    runsPerConfig:          TOTAL_RUNS,
    effectiveRunsPerConfig: EFFECTIVE_RUNS,
  };
}

// ---- Отображение результатов ------------------------------------------------------------------------------------------------------

const tbody = document.getElementById("results-body")!;

function renderAggregateRow(agg: AggregateStats): void {
  const cls   = `cfg-${agg.config.toLowerCase()}`;
  const conv  = agg.convergenceMs
    ? `${statStr(agg.convergenceMs)}`
    : "-";
  const psnrO = psnrStr(agg.psnrDb.overview.mean);
  const psnrC = psnrStr(agg.psnrDb.closeup.mean);
  const psnrM = psnrStr(agg.psnrDb.midrange.mean);
  const ssimO = agg.ssim.overview.mean.toFixed(3);
  const ssimC = agg.ssim.closeup.mean.toFixed(3);
  const ssimM = agg.ssim.midrange.mean.toFixed(3);

  // Удаляем строку-заглушку при первой вставке.
  const placeholder = tbody.querySelector("tr td[colspan]");
  if (placeholder) placeholder.parentElement!.remove();

  const flPct = (agg.fullyLoadedPct.mean * 100).toFixed(1) + "%";
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td class="${cls}"><strong>${agg.config}</strong></td>
    <td class="stat-cell">${statStr(agg.peakMemoryMB)}</td>
    <td class="stat-cell">${statStr(agg.frameTimeMedian)}</td>
    <td class="stat-cell">${statStr(agg.frameTimeP95)}</td>
    <td class="stat-cell">${statStr(agg.frameTimeP99)}</td>
    <td class="stat-cell">${conv}</td>
    <td title="overview / closeup / midrange">${psnrO} / ${psnrC} / ${psnrM}</td>
    <td title="overview / closeup / midrange">${ssimO} / ${ssimC} / ${ssimM}</td>
    <td class="stat-cell">${statStr(agg.uploadCount)}</td>
    <td class="stat-cell">${statStr(agg.evictionCount)}</td>
    <td class="stat-cell">${flPct}</td>
  `;
  tbody.appendChild(tr);
}

// ---- Рендер таблицы Вилкоксона --------------------------------------------------------------------------------------------------

const statsTbody = document.getElementById("stats-body") as HTMLTableSectionElement | null;

function renderWilcoxonTable(tests: WilcoxonResult[], holm?: HolmResult[]): void {
  if (!statsTbody) return;
  statsTbody.innerHTML = "";

  for (let i = 0; i < tests.length; i++) {
    const t = tests[i]!;
    const holmEntry = holm?.[i];
    const rawSig  = t.significant;
    const holmSig = holmEntry?.significant ?? rawSig;
    const pStr = t.pValue < 0.001 ? "<0.001" : t.pValue.toFixed(3);
    const zStr = t.z.toFixed(2);
    const rStr = t.effectSize.toFixed(3);
    const alphaStr = holmEntry ? holmEntry.holmAdjustedAlpha.toFixed(4) : "0.0500";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.metric}</td>
      <td>${t.configA} vs ${t.configB}</td>
      <td>${t.n}</td>
      <td>${t.W.toFixed(0)}</td>
      <td>${zStr}</td>
      <td class="${rawSig ? "sig-yes" : "sig-no"}">${pStr}</td>
      <td>${alphaStr}</td>
      <td class="${holmSig ? "sig-yes" : "sig-no"}">${holmSig ? "✓" : "✗"}</td>
      <td>${t.direction}</td>
      <td>${rStr}</td>
    `;
    statsTbody.appendChild(tr);
  }

  const footnote = document.getElementById("wilcoxon-footnote");
  if (footnote) footnote.style.display = "block";

  const section = document.getElementById("stats-section");
  if (section) (section as HTMLElement).style.display = "block";
}

// ---- DOM-элементы --------------------------------------------------------------------------------------------------------------------------

const runBtn       = document.getElementById("run-btn")       as HTMLButtonElement;
const statusLabel  = document.getElementById("status-label")  as HTMLSpanElement;
const progressLabel= document.getElementById("progress-label")as HTMLDivElement;
const progressBar  = document.getElementById("progress-bar")  as HTMLDivElement;
const etaEl        = document.getElementById("eta")           as HTMLDivElement;
const envBox       = document.getElementById("env-box")       as HTMLDivElement;
const presetSel    = document.getElementById("preset-sel")    as HTMLSelectElement;
const downloadsEl  = document.getElementById("downloads")     as HTMLDivElement;
const dlRawBtn     = document.getElementById("dl-raw-btn")    as HTMLButtonElement;
const dlSumBtn     = document.getElementById("dl-sum-btn")    as HTMLButtonElement;
const rawJsonEl    = document.getElementById("raw-json")      as HTMLPreElement;
const runCountNote = document.getElementById("run-count-note")as HTMLParagraphElement;

const canvasEl = document.getElementById("bench-canvas") as HTMLElement & { canvasElement: HTMLCanvasElement };

function setProgress(label: string, fraction: number, eta?: string): void {
  progressLabel.textContent  = label;
  progressBar.style.width    = `${Math.round(fraction * 100)}%`;
  etaEl.textContent          = eta ?? "";
}

function downloadJson(obj: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  URL.revokeObjectURL(url);
}

// ---- Точка входа ------------------------------------------------------------------------------------------------------------------------------

canvasEl.addEventListener("webgpu-ready", async (e: Event) => {
  const { device, adapterInfo } = (e as CustomEvent<WebGPUReadyDetail>).detail;
  const registry = (canvasEl as unknown as { registry: ResourceRegistry }).registry;
  const hasBC   = device.features.has("texture-compression-bc");
  const hasASTC = device.features.has("texture-compression-astc");

  envBox.textContent =
    `GPU: ${adapterInfo?.description ?? "unknown"} | ` +
    `BC7: ${hasBC} | ASTC: ${hasASTC} | ` +
    `Browser: ${navigator.userAgent.split(" ").pop() ?? "unknown"}`;

  statusLabel.textContent = "Готово - выберите пресет и нажмите «Запустить»";
  runBtn.disabled = false;

  runBtn.addEventListener("click", async () => {
    runBtn.disabled    = true;
    presetSel.disabled = true;
    tbody.innerHTML    = "";
    downloadsEl.style.display = "none";
    rawJsonEl.style.display   = "none";
    runCountNote.textContent  = "";

    const presetName = presetSel.value as ScenePresetName;
    const preset     = SCENE_PRESETS[presetName];

    // Строим таблицу конфигураций относительно сцены, когда пресет известен.
    const CONFIGS = makeConfigs(preset.expectedTotalBytes);

    // Выводим окружение в консоль согласно спецификации.
    const env = collectEnvironment(device, adapterInfo, canvasEl, presetName);
    const estMinutes = Math.round(CONFIGS.length * TOTAL_RUNS * 15 / 60);
    console.info(
      `=== ОКРУЖЕНИЕ БЕНЧМАРКА ===\n` +
      `GPU:              ${env.gpu.description}\n` +
      `Вендор:           ${env.gpu.vendor}  Архитектура: ${env.gpu.architecture}\n` +
      `BC7:              ${hasBC}  ASTC: ${hasASTC}\n` +
      `Браузер:          ${env.userAgent}\n` +
      `Canvas:           ${env.canvas.width}×${env.canvas.height}\n` +
      `Сцена:            пресет "${presetName}" (${preset.objectCount} объектов)\n` +
      `Конфигурации:     A, B, C, D, E\n` +
      `Запусков/конф.:   ${TOTAL_RUNS} (${WARMUP_RUNS} прогрев + ${EFFECTIVE_RUNS} эффективных)\n` +
      `Ожид. время:      ~${estMinutes} мин.\n` +
      `Метка времени:    ${env.timestamp}\n` +
      `===========================`,
    );

    setProgress("Generating scene…", 0);
    statusLabel.textContent = "Generating scene…";

    const objects = await generateStressScene(preset, (done, total) => {
      setProgress(`Generating scene… ${Math.round(done/total*100)}%`, 0);
    });

    if (objects.length !== preset.objectCount) {
      const msg = `Scene mismatch: expected ${preset.objectCount} objects for "${presetName}", got ${objects.length}`;
      console.error(msg);
      statusLabel.textContent = msg;
      runBtn.disabled = false;
      presetSel.disabled = false;
      return;
    }

    const actualBytes = objects.reduce((sum, o) => {
      let bytes = 0;
      for (let s = o.texSize; s >= 1; s >>= 1) bytes += s * s * 4;
      return sum + bytes;
    }, 0);
    const expectedMB = (preset.expectedTotalBytes / (1024 * 1024)).toFixed(1);
    const actualMB   = (actualBytes / (1024 * 1024)).toFixed(1);
    console.info(`[Benchmark] preset="${presetName}" objects=${objects.length} expectedMB=${expectedMB} actualMB=${actualMB}`);

    // All raw per-run data.
    const allRuns:    RunData[] = [];
    // Aggregates computed after each config's 30 runs complete.
    const aggregates: AggregateStats[] = [];

    // Reference frames from Config A (first non-warmup run).
    let refFrames:    Map<string, Uint8Array> | null = null;
    let refPopulated = false;

    const totalSteps = CONFIGS.length * TOTAL_RUNS;
    let   stepsDone  = 0;
    const benchStart = performance.now();

    for (const cfg of CONFIGS) {
      const configRuns: RunData[] = [];

      for (let run = 0; run < TOTAL_RUNS; run++) {
        stepsDone++;
        const isWarmup = run < WARMUP_RUNS;

        const elapsed  = performance.now() - benchStart;
        const avgPerStep = stepsDone > 1 ? elapsed / (stepsDone - 1) : elapsed;
        const remaining  = (totalSteps - stepsDone) * avgPerStep;

        setProgress(
          `Config ${cfg.id} - run ${run + 1}/${TOTAL_RUNS}` +
          (isWarmup ? " (warmup, discarded)" : ` (effective ${run - WARMUP_RUNS + 1}/${EFFECTIVE_RUNS})`),
          stepsDone / totalSteps,
          `ETA ${formatDuration(remaining)}`,
        );
        statusLabel.textContent = `Config ${cfg.id}, run ${run + 1}/${TOTAL_RUNS}`;

        // Cooldown between runs.
        if (run > 0) await cooldown();

        // For Config A's first non-warmup run: capture PSNR reference frames.
        const outRef: Map<string, Uint8Array> | null =
          (cfg.id === "A" && !isWarmup && !refPopulated) ? new Map() : null;

        const raw = await runSingle(cfg, device, registry, objects, preset, refFrames, outRef,
          canvasEl.canvasElement.width,
          canvasEl.canvasElement.height,
        );

        if (outRef && outRef.size > 0) {
          refFrames    = outRef;
          refPopulated = true;
        }

        if (!isWarmup) {
          configRuns.push({
            ...raw,
            runIndex:  run - WARMUP_RUNS,
            config:    cfg.id,
            timestamp: new Date().toISOString(),
          });
        }

        // Yield to browser.
        await new Promise<void>(r => setTimeout(r, 0));
      }

      allRuns.push(...configRuns);
      const agg = computeAggregate(cfg.id, configRuns, cfg.projected);
      aggregates.push(agg);
      renderAggregateRow(agg);
    }

    // ---- Done --------------------------------------------------------------------------------------------------------------------------------
    const totalElapsed = performance.now() - benchStart;
    setProgress("Complete", 1, `Total: ${formatDuration(totalElapsed)}`);
    statusLabel.textContent = `Done - ${CONFIGS.length} configs × ${EFFECTIVE_RUNS} effective runs`;

    // Group runs by config for Wilcoxon tests.
    const runsByConfig = new Map<string, RunData[]>();
    for (const r of allRuns) {
      let arr = runsByConfig.get(r.config);
      if (!arr) { arr = []; runsByConfig.set(r.config, arr); }
      arr.push(r);
    }
    const pairwiseTests = runPairwiseTests(runsByConfig);
    const holmBonferroniResults = holmBonferroniCorrection(
      pairwiseTests.map((t, i) => ({
        testId: `${i}_${t.metric}_${t.configA}_vs_${t.configB}`,
        pValue: t.pValue,
      })),
    );
    renderWilcoxonTable(pairwiseTests, holmBonferroniResults);

    const ts = new Date().toISOString().replace(/[:.]/g, "-");
    const rawExport:     RawExport     = { environment: env, runs:    allRuns    };
    const summaryExport: SummaryExport = { environment: env, configs: aggregates, pairwiseTests, holmBonferroniResults };

    downloadsEl.style.display = "flex";
    dlRawBtn.disabled  = false;
    dlSumBtn.disabled  = false;
    dlRawBtn.addEventListener("click", () =>
      downloadJson(rawExport,     `benchmark-raw-${ts}.json`));
    dlSumBtn.addEventListener("click", () =>
      downloadJson(summaryExport, `benchmark-summary-${ts}.json`));

    runCountNote.textContent =
      `${CONFIGS.length} configs × ${EFFECTIVE_RUNS} effective runs ` +
      `(${WARMUP_RUNS} warmup discarded per config). ` +
      `Total time: ${formatDuration(totalElapsed)}.`;

    rawJsonEl.textContent  = JSON.stringify(summaryExport, null, 2);
    rawJsonEl.style.display = "block";

    runBtn.disabled    = false;
    presetSel.disabled = false;
  });
});

canvasEl.addEventListener("webgpu-error", (e) => {
  statusLabel.textContent = `WebGPU error: ${(e as CustomEvent<string>).detail}`;
});
