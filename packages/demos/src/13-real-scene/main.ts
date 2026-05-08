/**
 * Демо 13 - Реальная сцена: glTF + потоковая передача текстур
 *
 * Загружает DamagedHelmet или Sponza glTF, регистрирует все альбедо-текстуры в
 * TextureStreamingManager и рендерит сцену с орбитальной камерой.
 *
 * Схема bind group для рендера:
 *   Группа 0 - streaming bind group (tier0, tier1, tier2, samp0–2, materialBuffer)
 *   Группа 1 - camera uniform (viewProj mat4 + position vec4 = 80 байт)
 *   Группа 2 - per-object storage buffer (model mat4 + materialId u32 на объект)
 *
 * Каждый примитив рисуется через drawIndexed(count, 1, 0, 0, objectIndex),
 * так что @builtin(instance_index) в вершинном шейдере выбирает данные объекта.
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import { CameraController } from "@webgpu-streaming/core";
import {
  TextureStreamingManager,
  MATERIAL_BIND_GROUP_KEY,
} from "@webgpu-streaming/texture-streaming";
import type {
  IRenderPass,
  RenderPassInitContext,
  FrameContext,
  ResourceRegistry,
} from "@webgpu-streaming/gpu-types";
import { MATERIAL_ENTRY_WGSL } from "@webgpu-streaming/gpu-types";
import { loadGLTF } from "../shared/GLTFLoader.js";
import type { LoadedScene } from "../shared/GLTFLoader.js";
import { FpsTracker } from "../shared/overlay.js";

// ---- DOM-ссылки --------------------------------------------------------------------------------------------------------------------------------

const el         = document.getElementById("canvas")        as HTMLElement;
const loadBtn    = document.getElementById("load-btn")       as HTMLButtonElement;
const sceneSel   = document.getElementById("scene-select")   as HTMLSelectElement;
const budgetSel  = document.getElementById("budget-select")  as HTMLSelectElement;
const warnEl     = document.getElementById("warn")           as HTMLSpanElement;
const ovStatus   = document.getElementById("ov-status")      as HTMLSpanElement;
const ovScene    = document.getElementById("ov-scene")       as HTMLSpanElement;
const ovPrims    = document.getElementById("ov-prims")       as HTMLSpanElement;
const ovTextures = document.getElementById("ov-textures")    as HTMLSpanElement;
const ovGpuMem   = document.getElementById("ov-gpu-mem")     as HTMLSpanElement;
const ovBudget   = document.getElementById("ov-budget")      as HTMLSpanElement;
const ovUploads  = document.getElementById("ov-uploads")     as HTMLSpanElement;
const ovEvict    = document.getElementById("ov-evictions")   as HTMLSpanElement;
const ovDraws    = document.getElementById("ov-draws")       as HTMLSpanElement;
const ovFps      = document.getElementById("ov-fps")         as HTMLSpanElement;
const progLabel  = document.getElementById("progress-label") as HTMLSpanElement;
const progBar    = document.getElementById("progress-bar")   as HTMLDivElement;

type WebGPUEl = HTMLElement & {
  setCameraController(c: unknown): void;
  addResourceManager(m: unknown): Promise<void>;
  addRenderPass(p: unknown): Promise<void>;
  canvasElement: HTMLCanvasElement;
  device: GPUDevice | null;
};

const webgpuEl = el as WebGPUEl;

function mb(bytes: number): string { return (bytes / 1048576).toFixed(1) + " MB"; }
function prog(msg: string, pct?: number): void {
  progLabel.textContent = msg;
  if (pct !== undefined) progBar.style.width = `${pct * 100}%`;
}

// ---- Константы структуры объекта ----------------------------------------------------------------------------------------------

/** mat4(64 байта) + materialId(u32, 4 байта) + 3×pad(u32) = 80 байт */
const OBJECT_STRIDE = 80;

// ---- WGSL-шейдеры --------------------------------------------------------------------------------------------------------------------------

const VERT_WGSL = /* wgsl */`
struct CameraUni {
  viewProj: mat4x4<f32>,
  position: vec4<f32>,
}

struct ObjectData {
  model:      mat4x4<f32>,
  materialId: u32,
  _p0: u32, _p1: u32, _p2: u32,
}

@group(1) @binding(0) var<uniform>       camera:  CameraUni;
@group(2) @binding(0) var<storage, read> objects: array<ObjectData>;

struct VSOut {
  @builtin(position)              clipPos:    vec4<f32>,
  @location(0)                    worldPos:   vec3<f32>,
  @location(1)                    worldNorm:  vec3<f32>,
  @location(2)                    uv:         vec2<f32>,
  @location(3) @interpolate(flat) materialId: u32,
}

@vertex
fn main(
  @builtin(instance_index) objIdx: u32,
  @location(0) pos:  vec3<f32>,
  @location(1) norm: vec3<f32>,
  @location(2) uv:   vec2<f32>,
) -> VSOut {
  let obj    = objects[objIdx];
  let worldP = obj.model * vec4<f32>(pos, 1.0);
  let worldN = normalize((obj.model * vec4<f32>(norm, 0.0)).xyz);
  var o: VSOut;
  o.clipPos    = camera.viewProj * worldP;
  o.worldPos   = worldP.xyz;
  o.worldNorm  = worldN;
  o.uv         = uv;
  o.materialId = obj.materialId;
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

fn sampleAlbedo(matId: u32, uv: vec2<f32>) -> vec4<f32> {
  let m   = materials[matId];
  let lod = f32(m.residentMip);
  if m.tierIndex == 0u {
    return textureSampleLevel(tier0, samp0, uv, m.layerIndex, lod);
  } else if m.tierIndex == 1u {
    return textureSampleLevel(tier1, samp1, uv, m.layerIndex, lod);
  }
  return textureSampleLevel(tier2, samp2, uv, m.layerIndex, lod);
}

@fragment
fn main(
  @location(0)                    worldPos:   vec3<f32>,
  @location(1)                    worldNorm:  vec3<f32>,
  @location(2)                    uv:         vec2<f32>,
  @location(3) @interpolate(flat) materialId: u32,
) -> @location(0) vec4<f32> {
  let albedo = sampleAlbedo(materialId, uv);
  let L      = normalize(vec3<f32>(0.6, 1.0, 0.4));
  let N      = normalize(worldNorm);
  let diff   = max(dot(N, L), 0.0) * 0.85 + 0.15; // диффузное + рассеянное освещение
  return vec4<f32>(albedo.rgb * diff, 1.0);
}
`;

// ---- Структура GPU-меша ----------------------------------------------------------------------------------------------------------------

interface MeshGPU {
  vertexBuf:   GPUBuffer;
  indexBuf:    GPUBuffer;
  indexCount:  number;
  indexFormat: GPUIndexFormat;
  objectIndex: number;
}

// ---- Render pass ------------------------------------------------------------------------------------------------------------------------------

class RealSceneRenderPass implements IRenderPass {
  readonly name = "real-scene-pass";

  private _device:       GPUDevice | null = null;
  private _registry:     ResourceRegistry | null = null;
  private _pipeline:     GPURenderPipeline | null = null;
  private _camBuf:       GPUBuffer | null = null;
  private _camBg:        GPUBindGroup | null = null;
  private _objBuf:       GPUBuffer | null = null;
  private _objBg:        GPUBindGroup | null = null;

  meshes: MeshGPU[] = [];
  drawCount = 0;

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, registry, presentationFormat } = ctx;
    this._device   = device;
    this._registry = registry;

    // Camera uniform: viewProj(mat4=64) + position(vec4=16) = 80 байт
    this._camBuf = device.createBuffer({
      label: "rs-camera-uni",
      size:  80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Схема bind group материалов - точно соответствует BindGroupManager (группа 0)
    const matBgl = device.createBindGroupLayout({
      label: "rs-mat-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture:  { viewDimension: "2d-array" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture:  { viewDimension: "2d-array" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture:  { viewDimension: "2d-array" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler:  {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler:  {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler:  {} },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer:   { type: "read-only-storage" } },
      ],
    });

    // Схема bind group камеры (группа 1)
    const camBgl = device.createBindGroupLayout({
      label: "rs-cam-bgl",
      entries: [
        { binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" } },
      ],
    });

    this._camBg = device.createBindGroup({
      layout: camBgl,
      entries: [{ binding: 0, resource: { buffer: this._camBuf } }],
    });

    // Схема bind group хранилища объектов (группа 2) - создаётся позже, когда известен objBuf
    const objBgl = device.createBindGroupLayout({
      label: "rs-obj-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX,
          buffer: { type: "read-only-storage" } },
      ],
    });

    // Пайплайн
    device.pushErrorScope("validation");
    this._pipeline = device.createRenderPipeline({
      label: "rs-pipeline",
      layout: device.createPipelineLayout({
        label: "rs-pipeline-layout",
        bindGroupLayouts: [matBgl, camBgl, objBgl],
      }),
      vertex: {
        module: device.createShaderModule({ code: VERT_WGSL }),
        entryPoint: "main",
        buffers: [{
          arrayStride: 32,
          attributes: [
            { shaderLocation: 0, offset: 0,  format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x2" },
          ],
        }],
      },
      fragment: {
        module: device.createShaderModule({ code: FRAG_WGSL }),
        entryPoint: "main",
        targets: [{ format: presentationFormat }],
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
    });
    void device.popErrorScope().then((err) => {
      if (err) console.error("[RealSceneRenderPass] Ошибка валидации пайплайна:", err);
    });

    // Сохраняем BGL объектов для использования в uploadObjects()
    this._objBglRef = objBgl;
  }

  // Сохраняем BGL объектов для построения bind group в uploadObjects()
  private _objBglRef: GPUBindGroupLayout | null = null;

  /** Загружает все per-object данные и создаёт bind group хранилища объектов. */
  uploadObjects(
    device: GPUDevice,
    objects: { model: Float32Array; materialId: number }[]
  ): void {
    if (!this._objBglRef) return;
    this._objBuf?.destroy();

    const count = objects.length;
    const cpuData = new ArrayBuffer(Math.max(OBJECT_STRIDE, count * OBJECT_STRIDE));
    const f32 = new Float32Array(cpuData);
    const u32 = new Uint32Array(cpuData);

    for (let i = 0; i < count; i++) {
      const base = i * (OBJECT_STRIDE / 4);
      for (let j = 0; j < 16; j++) f32[base + j] = objects[i]!.model[j]!;
      u32[base + 16] = objects[i]!.materialId;
    }

    this._objBuf = device.createBuffer({
      label: "rs-objects",
      size:  cpuData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this._objBuf, 0, cpuData);

    this._objBg = device.createBindGroup({
      layout: this._objBglRef,
      entries: [{ binding: 0, resource: { buffer: this._objBuf } }],
    });
  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._camBuf || !this._camBg ||
        !this._objBg || !this._registry || this.meshes.length === 0) return;

    const matBg = this._registry.request<GPUBindGroup>(MATERIAL_BIND_GROUP_KEY);
    if (!matBg) return;

    // Обновляем camera uniform из состояния камеры FrameContext
    const camData = new Float32Array(20);
    const vp = ctx.camera.viewProjectionMatrix;
    const cp = ctx.camera.position;
    for (let i = 0; i < 16; i++) camData[i] = vp[i]!;
    camData[16] = cp[0]!; camData[17] = cp[1]!; camData[18] = cp[2]!; camData[19] = 0;
    ctx.device.queue.writeBuffer(this._camBuf, 0, camData);

    const pass = ctx.encoder.beginRenderPass({
      colorAttachments: [{
        view:     ctx.colorAttachment,
        loadOp:   "load",
        storeOp:  "store",
      }],
      depthStencilAttachment: {
        view:              ctx.depthAttachment,
        depthLoadOp:       "load",
        depthStoreOp:      "store",
      },
    });

    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, matBg);
    pass.setBindGroup(1, this._camBg);
    pass.setBindGroup(2, this._objBg);

    this.drawCount = 0;
    for (const mesh of this.meshes) {
      pass.setVertexBuffer(0, mesh.vertexBuf);
      pass.setIndexBuffer(mesh.indexBuf, mesh.indexFormat);
      // first_instance = objectIndex → @builtin(instance_index) = objectIndex
      pass.drawIndexed(mesh.indexCount, 1, 0, 0, mesh.objectIndex);
      this.drawCount++;
    }

    pass.end();
  }

  onResize(_w: number, _h: number): void {}

  destroy(): void {
    for (const m of this.meshes) {
      m.vertexBuf.destroy();
      m.indexBuf.destroy();
    }
    this.meshes = [];
    this._camBuf?.destroy();
    this._objBuf?.destroy();
    this._pipeline    = null;
    this._camBg       = null;
    this._objBg       = null;
    this._objBglRef   = null;
    this._device      = null;
    this._registry    = null;
  }
}

// ---- Состояние --------------------------------------------------------------------------------------------------------------------------------

let streaming:  TextureStreamingManager | null = null;
let renderPass: RealSceneRenderPass | null = null;
let cameraCtrl: CameraController | null = null;
const fps = new FpsTracker();
let isInitialized = false;

// Доступно через консоль для отладки: streaming.frameUploadBudget = 50000 замедлит стриминг
(window as Record<string, unknown>)["__demo13"] = { get streaming() { return streaming; } };

// ---- WebGPU готов --------------------------------------------------------------------------------------------------------------------------

el.addEventListener("webgpu-ready", (e: Event) => {
  const { device } = (e as CustomEvent<WebGPUReadyDetail>).detail;
  if (!device) return;

  cameraCtrl = new CameraController(webgpuEl.canvasElement);
  cameraCtrl.setRadius(3);
  webgpuEl.setCameraController(cameraCtrl);

  isInitialized = true;
  ovStatus.textContent = "Готово";
  loadBtn.disabled = false;
  prog("Готово - выберите сцену и нажмите «Загрузить»");
});

el.addEventListener("webgpu-error", (e: Event) => {
  ovStatus.textContent = `Ошибка: ${(e as CustomEvent<string>).detail}`;
});

// ---- Кнопка загрузки ----------------------------------------------------------------------------------------------------------------------

loadBtn.addEventListener("click", async () => {
  if (!isInitialized || !webgpuEl.device) return;
  loadBtn.disabled  = true;
  sceneSel.disabled = true;
  warnEl.textContent = "";

  // Уничтожаем предыдущую сцену при повторной загрузке
  if (streaming) {
    streaming.destroy();
    streaming = null;
  }
  // Примечание: удалить render pass из plugin host после первой загрузки нельзя -
  // для повторной загрузки требуется перезагрузка страницы. Блокируем элементы управления.

  const sceneName  = sceneSel.value;
  const budgetBytes = parseInt(budgetSel.value, 10);
  ovScene.textContent  = sceneName;
  ovBudget.textContent = mb(budgetBytes);
  ovStatus.textContent = "Загрузка…";

  try {
    await loadScene(sceneName, budgetBytes);
    ovStatus.textContent = "Стриминг";
  } catch (err) {
    console.error("[Demo13]", err);
    const msg = err instanceof Error ? err.message : String(err);
    warnEl.textContent   = "Ошибка: " + msg;
    ovStatus.textContent = "Ошибка";
    loadBtn.disabled     = false;
    sceneSel.disabled    = false;
  }
  // Элементы управления остаются заблокированными после загрузки (повторное добавление
  // менеджеров в plugin host во время выполнения не поддерживается - для смены сцены перезагрузите страницу).
  progLabel.textContent = "Загружено - перезагрузите страницу для смены сцены";
});

// ---- Загрузка сцены ------------------------------------------------------------------------------------------------------------------------

async function loadScene(sceneName: string, budgetBytes: number): Promise<void> {
  const device = webgpuEl.device!;

  // ---- 1. Загружаем glTF --------------------------------------------------------------------------------------------------------
  const gltfUrl = `../../assets/${sceneName}/${sceneName}.gltf`;
  prog("Загрузка glTF…", 0);

  const scene: LoadedScene = await loadGLTF(gltfUrl, (msg) => prog(msg));

  if (scene.textures.length === 0) {
    warnEl.textContent = "Предупреждение: альбедо-текстуры не найдены.";
  }

  ovPrims.textContent    = String(scene.meshes.length);
  ovTextures.textContent = String(scene.textures.length);
  prog("Инициализация GPU-ресурсов…", 0.6);

  // ---- 2. Вычисляем количество слоёв тиров из реальных размеров изображений
  let t0 = 0, t1 = 0, t2 = 0;
  for (const tex of scene.textures) {
    const s = Math.max(tex.width, tex.height);
    if      (s <= 512)  t0++;
    else if (s <= 1024) t1++;
    else                t2++;
  }
  const maxLayersPerTier: [number, number, number] = [
    Math.max(2, t0 + 2),
    Math.max(2, t1 + 2),
    Math.max(2, t2 + 2),
  ];
  console.info(`[Demo13] Распределение по тирам: tier0=${t0}, tier1=${t1}, tier2=${t2}`);

  // ---- 3. Создаём менеджер потоковой передачи и добавляем как resource manager
  streaming = new TextureStreamingManager({
    budgetBytes,
    frameUploadBudget: 8 * 1024 * 1024,
    maxLayersPerTier,
  });

  await webgpuEl.addResourceManager(streaming);

  // ---- 4. Регистрируем текстуры ------------------------------------------------------------------------------------
  for (let i = 0; i < scene.textures.length; i++) {
    const tex = scene.textures[i]!;
    prog(`Регистрация текстуры ${i + 1}/${scene.textures.length}…`,
      0.6 + (i / scene.textures.length) * 0.2);

    // Используем первый меш, ссылающийся на эту текстуру, для bounding sphere
    const refMesh = scene.meshes.find((m) => m.textureIndex === i);
    const bs = refMesh?.boundingSphere ?? new Float32Array([0, 0, 0, 1]);

    streaming.registerTexture(tex.id, tex.parsed, tex.ktx2Bytes, i, bs);
  }

  // ---- 5. Создаём и добавляем render pass --------------------------------------------------------------
  if (!renderPass) {
    renderPass = new RealSceneRenderPass();
    await webgpuEl.addRenderPass(renderPass);
  }

  // ---- 6. Создаём GPU буферы вершин/индексов + per-object данные ----------------
  prog("Загрузка данных меша…", 0.85);

  const objectDatas: { model: Float32Array; materialId: number }[] = [];

  for (let i = 0; i < scene.meshes.length; i++) {
    const mesh = scene.meshes[i]!;

    const vb = device.createBuffer({
      label: `rs-vb-${i}`,
      size:  mesh.vertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vb, 0, mesh.vertexData);

    // WebGPU требует, чтобы данные writeBuffer были кратны 4 байтам.
    // Массивы индексов Uint16 с нечётным количеством индексов (например, 15 индексов = 30 байт)
    // нарушают это требование, поэтому дополняем до ближайшей границы 4 байт.
    const rawIbBytes = mesh.indexData.byteLength;
    const alignedIbBytes = (rawIbBytes + 3) & ~3;
    let ibData: ArrayBufferView = mesh.indexData;
    if (rawIbBytes !== alignedIbBytes) {
      const padded = new Uint8Array(alignedIbBytes);
      padded.set(new Uint8Array(mesh.indexData.buffer, mesh.indexData.byteOffset, rawIbBytes));
      ibData = padded;
    }
    const ib = device.createBuffer({
      label: `rs-ib-${i}`,
      size:  alignedIbBytes,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(ib, 0, ibData);

    renderPass.meshes.push({
      vertexBuf:   vb,
      indexBuf:    ib,
      indexCount:  mesh.indexCount,
      indexFormat: mesh.indexData instanceof Uint32Array ? "uint32" : "uint16",
      objectIndex: i,
    });

    // materialId = textureIndex (отсчёт с 0 по потоковым текстурам)
    // Если текстуры нет - используем 0 (первую зарегистрированную текстуру) как запасной вариант
    const materialId = mesh.textureIndex >= 0 ? mesh.textureIndex : 0;
    objectDatas.push({ model: mesh.worldTransform, materialId });
  }

  renderPass.uploadObjects(device, objectDatas);

  // ---- 7. Устанавливаем радиус камеры в зависимости от сцены ------------------------
  if (cameraCtrl) {
    cameraCtrl.setRadius(sceneName === "DamagedHelmet" ? 2.5 : 8);
  }

  // ---- 8. Запускаем таймер статистики ----------------------------------------------------------------------
  function tick(): void {
    fps.tick();
    if (streaming) {
      const bt = streaming.budgetTracker;
      ovGpuMem.textContent  = bt ? mb(bt.totalUsed) : "-";
      ovUploads.textContent = String(streaming.uploadsLastFrame);
      ovEvict.textContent   = String(streaming.evictionsLastFrame);
    }
    ovDraws.textContent = renderPass ? String(renderPass.drawCount) : "-";
    ovFps.textContent   = String(fps.fps);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  prog("Готово", 1.0);
}
