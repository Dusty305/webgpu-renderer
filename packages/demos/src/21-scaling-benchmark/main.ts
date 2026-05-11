/**
 * Демо 21 — Бенчмарк масштабирования draw calls
 *
 * Каждый объект рисуется отдельным drawIndexed(36, 1, 0, 0, objectIndex),
 * что позволяет изолировать CPU-overhead от записи команд рисования.
 *
 * Объектов: 100 / 500 / 1000 / 5000.
 * На каждый вариант: 3 прогрева + 5 прогонов × 120 кадров.
 * Замер включает: запись команд CPU + ожидание завершения GPU.
 */

// ---- DOM-ссылки ----------------------------------------------------------------

const canvasEl      = document.getElementById("canvas")         as HTMLCanvasElement;
const canvasOverlay = document.getElementById("canvas-overlay") as HTMLDivElement;
const runBtn        = document.getElementById("run-btn")         as HTMLButtonElement;
const statusLabel   = document.getElementById("status-label")   as HTMLSpanElement;
const progressLabel = document.getElementById("progress-label") as HTMLDivElement;
const progressBar   = document.getElementById("progress-bar")   as HTMLDivElement;
const etaEl         = document.getElementById("eta")            as HTMLDivElement;
const envBox        = document.getElementById("env-box")        as HTMLDivElement;
const resultsSection = document.getElementById("results-section") as HTMLDivElement;
const resultsBody   = document.getElementById("results-body")   as HTMLTableSectionElement;
const downloads     = document.getElementById("downloads")      as HTMLDivElement;
const dlBtn         = document.getElementById("dl-btn")         as HTMLButtonElement;
const errorBox      = document.getElementById("error-box")      as HTMLDivElement;

// ---- Константы ----------------------------------------------------------------

const CANVAS_W      = 800;
const CANVAS_H      = 450;
const OBJECT_COUNTS = [100, 500, 1000, 5000] as const;
const WARMUP_RUNS   = 3;
const MEASURE_RUNS  = 5;
const FRAMES_PER_RUN = 120;
const COOLDOWN_MS   = 400;
const GRID_SPACING  = 1.5;
const FOV_Y         = Math.PI / 4;

// ---- WGSL-шейдер (единый модуль с двумя точками входа) -----------------------

const WGSL_SOURCE = /* wgsl */`
struct Camera {
  viewProj: mat4x4<f32>,
}
struct Object {
  model: mat4x4<f32>,
  color: vec4<f32>,
}

@group(0) @binding(0) var<uniform>       cam:  Camera;
@group(1) @binding(0) var<storage, read> objs: array<Object>;

struct VSOut {
  @builtin(position)              pos:    vec4<f32>,
  @location(0) @interpolate(flat) objIdx: u32,
  @location(1)                    norm:   vec3<f32>,
}

@vertex fn vs(
  @builtin(instance_index) objIdx: u32,
  @location(0)             pos:    vec3<f32>,
  @location(1)             norm:   vec3<f32>,
) -> VSOut {
  let worldPos = objs[objIdx].model * vec4<f32>(pos, 1.0);
  var o: VSOut;
  o.pos    = cam.viewProj * worldPos;
  o.objIdx = objIdx;
  o.norm   = norm;
  return o;
}

@fragment fn fs(
  @location(0) @interpolate(flat) objIdx: u32,
  @location(1)                    norm:   vec3<f32>,
) -> @location(0) vec4<f32> {
  let L    = normalize(vec3<f32>(0.6, 1.0, 0.4));
  let diff = max(dot(normalize(norm), L), 0.0) * 0.75 + 0.25;
  return vec4<f32>(objs[objIdx].color.rgb * diff, 1.0);
}
`;

// ---- Меш куба (24 вершины, 36 индексов) --------------------------------------
// Каждая вершина: позиция (3 float) + нормаль (3 float), шаг 24 байта.

const CUBE_VERTS = new Float32Array([
  // +Z грань, нормаль (0,0,1)
  -0.5,-0.5, 0.5,  0, 0, 1,   0.5,-0.5, 0.5,  0, 0, 1,
   0.5, 0.5, 0.5,  0, 0, 1,  -0.5, 0.5, 0.5,  0, 0, 1,
  // -Z грань, нормаль (0,0,-1)
   0.5,-0.5,-0.5,  0, 0,-1,  -0.5,-0.5,-0.5,  0, 0,-1,
  -0.5, 0.5,-0.5,  0, 0,-1,   0.5, 0.5,-0.5,  0, 0,-1,
  // +Y грань, нормаль (0,1,0)
  -0.5, 0.5, 0.5,  0, 1, 0,   0.5, 0.5, 0.5,  0, 1, 0,
   0.5, 0.5,-0.5,  0, 1, 0,  -0.5, 0.5,-0.5,  0, 1, 0,
  // -Y грань, нормаль (0,-1,0)
  -0.5,-0.5,-0.5,  0,-1, 0,   0.5,-0.5,-0.5,  0,-1, 0,
   0.5,-0.5, 0.5,  0,-1, 0,  -0.5,-0.5, 0.5,  0,-1, 0,
  // +X грань, нормаль (1,0,0)
   0.5,-0.5, 0.5,  1, 0, 0,   0.5,-0.5,-0.5,  1, 0, 0,
   0.5, 0.5,-0.5,  1, 0, 0,   0.5, 0.5, 0.5,  1, 0, 0,
  // -X грань, нормаль (-1,0,0)
  -0.5,-0.5,-0.5, -1, 0, 0,  -0.5,-0.5, 0.5, -1, 0, 0,
  -0.5, 0.5, 0.5, -1, 0, 0,  -0.5, 0.5,-0.5, -1, 0, 0,
]);

// 2 треугольника на грань, CCW-обход при взгляде снаружи
const CUBE_IDXS = new Uint16Array([
   0, 1, 2,   0, 2, 3,   // +Z
   4, 5, 6,   4, 6, 7,   // -Z
   8, 9,10,   8,10,11,   // +Y
  12,13,14,  12,14,15,   // -Y
  16,17,18,  16,18,19,   // +X
  20,21,22,  20,22,23,   // -X
]);
const CUBE_INDEX_COUNT = CUBE_IDXS.length; // 36

// ---- Матричная математика (column-major, как в CameraController) --------------

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += a[k * 4 + j]! * b[i * 4 + k]!;
      out[i * 4 + j] = sum;
    }
  return out;
}

function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovY / 2);
  const ri = 1 / (near - far);
  // prettier-ignore
  return new Float32Array([
    f / aspect, 0, 0,                        0,
    0,          f, 0,                        0,
    0,          0, (far + near) * ri,        -1,
    0,          0, 2 * far * near * ri,       0,
  ]);
}

function mat4LookAt(
  ex: number, ey: number, ez: number,
  cx: number, cy: number, cz: number,
): Float32Array {
  let fx = cx - ex, fy = cy - ey, fz = cz - ez;
  const fl = Math.sqrt(fx * fx + fy * fy + fz * fz);
  fx /= fl; fy /= fl; fz /= fl;

  // cross(f, up=[0,1,0]) = (-fz, 0, fx)
  let rx = -fz, ry = 0, rz = fx;
  const rl = Math.sqrt(rx * rx + ry * ry + rz * rz);
  rx /= rl; ry /= rl; rz /= rl;

  const ux = ry * fz - rz * fy;
  const uy = rz * fx - rx * fz;
  const uz = rx * fy - ry * fx;

  // prettier-ignore
  return new Float32Array([
    rx,  ux, -fx, 0,
    ry,  uy, -fy, 0,
    rz,  uz, -fz, 0,
    -(rx*ex + ry*ey + rz*ez), -(ux*ex + uy*ey + uz*ez), (fx*ex + fy*ey + fz*ez), 1,
  ]);
}

// ---- Вспомогательные функции --------------------------------------------------

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    default: return [v, p, q];
  }
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.min(Math.ceil(sorted.length * p) - 1, sorted.length - 1);
  return sorted[idx] ?? 0;
}
function arrMean(arr: number[]): number {
  return arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
}
function arrStd(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = arrMean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}

// ---- Генерация сцены ----------------------------------------------------------

function computeGrid(count: number): { cols: number; rows: number } {
  const cols = Math.ceil(Math.sqrt(count));
  return { cols, rows: Math.ceil(count / cols) };
}

/** Генерирует буфер per-object данных: mat4 (64 байта) + vec4 color (16 байт) = 80 байт/объект. */
function makeObjectData(count: number): Float32Array {
  const { cols, rows } = computeGrid(count);
  const data = new Float32Array(count * 20); // 20 float = 80 байт

  for (let i = 0; i < count; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const tx  = (col - (cols - 1) / 2) * GRID_SPACING;
    const ty  = (row - (rows - 1) / 2) * GRID_SPACING;

    const base = i * 20;
    // Матрица модели: трансляция (column-major)
    data[base +  0] = 1; data[base +  1] = 0; data[base +  2] = 0; data[base +  3] = 0;
    data[base +  4] = 0; data[base +  5] = 1; data[base +  6] = 0; data[base +  7] = 0;
    data[base +  8] = 0; data[base +  9] = 0; data[base + 10] = 1; data[base + 11] = 0;
    data[base + 12] = tx; data[base + 13] = ty; data[base + 14] = 0; data[base + 15] = 1;

    // Цвет объекта: HSV → RGB
    const [r, g, b] = hsvToRgb(i / count, 0.7, 0.9);
    data[base + 16] = r; data[base + 17] = g; data[base + 18] = b; data[base + 19] = 1;
  }
  return data;
}

/** Вычисляет матрицу view-projection для сцены с count объектами. */
function makeCameraMatrix(count: number): Float32Array {
  const { cols, rows } = computeGrid(count);
  const sceneHalfX = ((cols - 1) * GRID_SPACING) / 2;
  const sceneHalfY = ((rows - 1) * GRID_SPACING) / 2;
  const dist = Math.max(sceneHalfX, sceneHalfY) / Math.tan(FOV_Y / 2) + 4;

  const view = mat4LookAt(0, dist * 0.25, dist, 0, 0, 0);
  const proj = mat4Perspective(FOV_Y, CANVAS_W / CANVAS_H, 0.1, dist * 5);
  return mat4Multiply(proj, view);
}

// ---- Типы данных результатов --------------------------------------------------

export interface ScaleResult {
  objCount:    number;
  drawCalls:   number;
  medianMs:    number;
  medianStd:   number;
  p99Ms:       number;
  p99Std:      number;
  gridCols:    number;
  gridRows:    number;
}

interface ExportData {
  timestamp:    string;
  gpu:          string;
  canvas:       string;
  warmupRuns:   number;
  measureRuns:  number;
  framesPerRun: number;
  results:      ScaleResult[];
}

// ---- GPU ресурсы --------------------------------------------------------------

interface GpuState {
  device:      GPUDevice;
  format:      GPUTextureFormat;
  gpuCtx:      GPUCanvasContext;
  pipeline:    GPURenderPipeline;
  vertBuf:     GPUBuffer;
  idxBuf:      GPUBuffer;
  camBuf:      GPUBuffer;
  camBg:       GPUBindGroup;
  objBgl:      GPUBindGroupLayout;
  depthTex:    GPUTexture;
  depthView:   GPUTextureView;
}

async function initGpu(): Promise<{ state: GpuState; gpuLabel: string }> {
  if (!navigator.gpu) throw new Error("WebGPU не поддерживается в этом браузере");

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Не удалось получить GPU-адаптер. Проверьте, включён ли WebGPU.");

  const device = await adapter.requestDevice({ label: "bench-device" });
  device.lost.then(info => showError(`Потеряно GPU-устройство: ${info.message}`));

  const info = await adapter.requestAdapterInfo();
  const gpuLabel = [info.vendor, info.architecture, info.description]
    .filter(Boolean).join(" / ") || "GPU";

  const format = navigator.gpu.getPreferredCanvasFormat();
  const gpuCtx = canvasEl.getContext("webgpu") as GPUCanvasContext;
  gpuCtx.configure({ device, format, alphaMode: "opaque" });

  // Depth texture
  device.pushErrorScope("out-of-memory");
  const depthTex = device.createTexture({
    label: "bench-depth",
    size: [CANVAS_W, CANVAS_H],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  const depthErr = await device.popErrorScope();
  if (depthErr) throw new Error(`OOM: ${depthErr.message}`);
  const depthView = depthTex.createView();

  // Вершинный буфер куба
  device.pushErrorScope("out-of-memory");
  const vertBuf = device.createBuffer({
    label: "cube-verts",
    size: CUBE_VERTS.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });
  const vertErr = await device.popErrorScope();
  if (vertErr) throw new Error(`OOM: ${vertErr.message}`);
  device.queue.writeBuffer(vertBuf, 0, CUBE_VERTS);

  // Индексный буфер куба
  device.pushErrorScope("out-of-memory");
  const idxBuf = device.createBuffer({
    label: "cube-idxs",
    size: Math.ceil(CUBE_IDXS.byteLength / 4) * 4, // выравнивание до 4 байт
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  });
  const idxErr = await device.popErrorScope();
  if (idxErr) throw new Error(`OOM: ${idxErr.message}`);
  device.queue.writeBuffer(idxBuf, 0, CUBE_IDXS);

  // Камера: uniform буфер + bind group layout
  device.pushErrorScope("out-of-memory");
  const camBuf = device.createBuffer({
    label: "camera-uniform",
    size: 64, // mat4x4<f32> = 64 байта
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const camBufErr = await device.popErrorScope();
  if (camBufErr) throw new Error(`OOM: ${camBufErr.message}`);

  const camBgl = device.createBindGroupLayout({
    label: "cam-bgl",
    entries: [{
      binding: 0, visibility: GPUShaderStage.VERTEX,
      buffer: { type: "uniform" },
    }],
  });
  const camBg = device.createBindGroup({
    label: "cam-bg", layout: camBgl,
    entries: [{ binding: 0, resource: { buffer: camBuf } }],
  });

  // Objects: storage bind group layout (без самого буфера — создаётся для каждого теста)
  const objBgl = device.createBindGroupLayout({
    label: "objs-bgl",
    entries: [{
      binding: 0,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: { type: "read-only-storage" },
    }],
  });

  // Рендер-пайплайн
  const shaderMod = device.createShaderModule({ label: "bench-shader", code: WGSL_SOURCE });
  const pipeline = await device.createRenderPipelineAsync({
    label: "bench-pipeline",
    layout: device.createPipelineLayout({
      label: "bench-layout",
      bindGroupLayouts: [camBgl, objBgl],
    }),
    vertex: {
      module: shaderMod,
      entryPoint: "vs",
      buffers: [{
        arrayStride: 24, // 6 × float32
        attributes: [
          { shaderLocation: 0, offset:  0, format: "float32x3" }, // позиция
          { shaderLocation: 1, offset: 12, format: "float32x3" }, // нормаль
        ],
      }],
    },
    fragment: {
      module: shaderMod,
      entryPoint: "fs",
      targets: [{ format }],
    },
    primitive:    { topology: "triangle-list", cullMode: "back" },
    depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
  });

  return {
    state: { device, format, gpuCtx, pipeline, vertBuf, idxBuf, camBuf, camBg, objBgl, depthTex, depthView },
    gpuLabel,
  };
}

// ---- Один прогон бенчмарка для одного числа объектов -------------------------

async function runOneCount(
  gpu:      GpuState,
  count:    number,
  onFrame:  (run: number, frame: number) => void,
): Promise<ScaleResult> {
  const { device, gpuCtx, pipeline, vertBuf, idxBuf, camBuf, camBg, objBgl, depthView } = gpu;
  const { cols, rows } = computeGrid(count);

  // Загрузить матрицу камеры для этой сцены
  device.queue.writeBuffer(camBuf, 0, makeCameraMatrix(count));

  // Создать буфер объектов
  const objData = makeObjectData(count);
  device.pushErrorScope("out-of-memory");
  const objBuf = device.createBuffer({
    label: `objs-${count}`,
    size:  objData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const objBufErr = await device.popErrorScope();
  if (objBufErr) throw new Error(`OOM при создании буфера ${count} объектов: ${objBufErr.message}`);
  device.queue.writeBuffer(objBuf, 0, objData);

  const objBg = device.createBindGroup({
    label: `objs-bg-${count}`, layout: objBgl,
    entries: [{ binding: 0, resource: { buffer: objBuf } }],
  });

  const runMedians: number[] = [];
  const runP99s:    number[] = [];
  const totalRuns = WARMUP_RUNS + MEASURE_RUNS;

  for (let run = 0; run < totalRuns; run++) {
    const frameTimes: number[] = [];

    for (let f = 0; f < FRAMES_PER_RUN; f++) {
      onFrame(run, f);

      const colorView = gpuCtx.getCurrentTexture().createView();

      const t0 = performance.now();

      const encoder = device.createCommandEncoder({ label: `frame-${run}-${f}` });
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: colorView,
          clearValue: { r: 0.05, g: 0.05, b: 0.07, a: 1 },
          loadOp: "clear", storeOp: "store",
        }],
        depthStencilAttachment: {
          view: depthView,
          depthClearValue: 1.0,
          depthLoadOp: "clear", depthStoreOp: "discard",
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, camBg);
      pass.setBindGroup(1, objBg);
      pass.setVertexBuffer(0, vertBuf);
      pass.setIndexBuffer(idxBuf, "uint16");

      for (let i = 0; i < count; i++) {
        pass.drawIndexed(CUBE_INDEX_COUNT, 1, 0, 0, i);
      }

      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();

      frameTimes.push(performance.now() - t0);
    }

    if (run >= WARMUP_RUNS) {
      const sorted = [...frameTimes].sort((a, b) => a - b);
      runMedians.push(percentile(sorted, 0.5));
      runP99s.push(percentile(sorted, 0.99));
    }

    // Короткая пауза между прогонами
    await new Promise<void>(r => setTimeout(r, COOLDOWN_MS));
  }

  objBuf.destroy();

  return {
    objCount:  count,
    drawCalls: count,
    medianMs:  arrMean(runMedians),
    medianStd: arrStd(runMedians),
    p99Ms:     arrMean(runP99s),
    p99Std:    arrStd(runP99s),
    gridCols:  cols,
    gridRows:  rows,
  };
}

// ---- Оркестратор бенчмарка ---------------------------------------------------

async function runBenchmark(gpu: GpuState): Promise<ScaleResult[]> {
  const totalSteps = OBJECT_COUNTS.length * (WARMUP_RUNS + MEASURE_RUNS) * FRAMES_PER_RUN;
  let doneSteps = 0;
  const t0 = performance.now();

  const results: ScaleResult[] = [];

  for (let ci = 0; ci < OBJECT_COUNTS.length; ci++) {
    const count = OBJECT_COUNTS[ci];

    const result = await runOneCount(gpu, count, (run, _frame) => {
      doneSteps++;
      const pct = doneSteps / totalSteps;
      const elapsed = performance.now() - t0;
      const eta = pct > 0.01 ? (elapsed / pct - elapsed) : 0;
      const isWarmup = run < WARMUP_RUNS;
      const runLabel = isWarmup
        ? `прогрев ${run + 1}/${WARMUP_RUNS}`
        : `прогон ${run - WARMUP_RUNS + 1}/${MEASURE_RUNS}`;
      setProgress(`${count} объектов — ${runLabel}`, pct, eta);
    });

    results.push(result);
    updateTable(results);

    // Пауза перед следующим вариантом
    if (ci < OBJECT_COUNTS.length - 1) {
      await new Promise<void>(r => setTimeout(r, COOLDOWN_MS * 3));
    }
  }

  return results;
}

// ---- UI ----------------------------------------------------------------------

function showError(msg: string): void {
  errorBox.textContent = `Ошибка: ${msg}`;
  errorBox.style.display = "block";
  statusLabel.textContent = "Ошибка";
}

function setProgress(label: string, pct: number, etaMs: number): void {
  progressLabel.textContent = label;
  progressBar.style.width   = `${(pct * 100).toFixed(1)}%`;
  if (etaMs > 1000) {
    const s = Math.round(etaMs / 1000);
    etaEl.textContent = `Осталось ~${s < 60 ? s + "с" : Math.floor(s / 60) + "м " + (s % 60) + "с"}`;
  } else {
    etaEl.textContent = "";
  }
}

function fmt(v: number, std: number): string {
  return `<span class="val">${v.toFixed(2)}</span> <span class="std">±${std.toFixed(2)}</span>`;
}

function updateTable(results: ScaleResult[]): void {
  resultsSection.style.display = "block";
  resultsBody.innerHTML = results.map(r => `
    <tr>
      <td>${r.objCount.toLocaleString("ru")}</td>
      <td>${r.drawCalls.toLocaleString("ru")}</td>
      <td>${fmt(r.medianMs, r.medianStd)}</td>
      <td>${fmt(r.p99Ms, r.p99Std)}</td>
      <td style="color:#666;font-size:11px">${r.gridCols}×${r.gridRows}</td>
    </tr>
  `).join("");
}

function setupDownload(results: ScaleResult[], gpuLabel: string): void {
  downloads.style.display = "flex";
  dlBtn.disabled = false;

  const data: ExportData = {
    timestamp:    new Date().toISOString(),
    gpu:          gpuLabel,
    canvas:       `${CANVAS_W}×${CANVAS_H}`,
    warmupRuns:   WARMUP_RUNS,
    measureRuns:  MEASURE_RUNS,
    framesPerRun: FRAMES_PER_RUN,
    results,
  };

  dlBtn.addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = `bench-scaling-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

// ---- Инициализация -----------------------------------------------------------

async function main(): Promise<void> {
  let gpu: GpuState;
  let gpuLabel: string;

  try {
    ({ state: gpu, gpuLabel } = await initGpu());
  } catch (e) {
    showError(String(e));
    return;
  }

  envBox.textContent = `GPU: ${gpuLabel} | Холст: ${CANVAS_W}×${CANVAS_H} | `
    + `Прогонов: ${MEASURE_RUNS} × ${FRAMES_PER_RUN} кадров (+ ${WARMUP_RUNS} прогрев)`;

  statusLabel.textContent = "Готово к запуску";
  runBtn.disabled = false;

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    canvasOverlay.textContent = "Бенчмарк выполняется…";
    statusLabel.textContent = "Выполняется…";
    progressBar.style.width = "0%";

    let results: ScaleResult[];
    try {
      results = await runBenchmark(gpu);
    } catch (e) {
      showError(String(e));
      runBtn.disabled = false;
      return;
    }

    canvasOverlay.textContent = `Готово — ${results.length} вариантов`;
    statusLabel.textContent   = "Готово";
    setProgress("Завершено", 1, 0);
    setupDownload(results, gpuLabel);
  });
}

main().catch(e => showError(String(e)));
