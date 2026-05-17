import type {
  CreateRendererOptions,
  Renderer,
  TextureHandle,
  TextureOptions,
  GeometryDescriptor,
  MaterialDescriptor,
  MeshHandle,
  CameraOptions,
  StreamingStats,
  CameraState,
  ResourceRegistry,
} from "@webgpu-streaming/gpu-types";
import {
  TextureStreamingManager,
  MATERIAL_BIND_GROUP_KEY,
  buildMipPyramid,
  packKtx2,
  decodeImage,
  parseKTX2,
} from "@webgpu-streaming/texture-streaming/internal";
import { DeviceManager } from "./DeviceManager.js";
import { CameraController } from "./CameraController.js";
import { RenderLoop } from "./RenderLoop.js";
import { ResourceRegistryImpl } from "./ResourceRegistryImpl.js";
import { PluginHost } from "./PluginHost.js";
import { SceneGraph } from "./SceneGraph.js";
import {
  mat4Identity,
  makeCubeGeometry,
  buildInterleavedVertices,
  toUint16Indices,
  computeBoundingSphere,
  MeshHandleImpl,
  FrameStatsTracker,
} from "./_scene-utils.js";
import { BUILTIN_WGSL, NO_TEXTURE_MATERIAL_ID } from "./_builtin-wgsl.js";
import { detectFormat } from "./loaders/FormatDetector.js";
import { parseOBJBuffer } from "./loaders/OBJLoader.js";
import { parseSTL } from "./loaders/STLLoader.js";

// ---- Шаг / лимиты ----------------------------------------------------------------------------------------------------------------------------

const SCENE_UBO_SIZE = 128; // viewProj(64) + cameraPos(16) + lightDir(16) + lightColor(16) = 128
const OBJ_STRIDE     = 256; // >= minUniformBufferOffsetAlignment
const MAX_OBJECTS    = 256;

// ---- ManualCameraController --------------------------------------------------------------------------------------------------------

/** Камера, состояние которой задаётся императивно (без ввода мышью). */
export class ManualCameraController {
  private _pos    = new Float32Array([0, 1, 4]);
  private _target = new Float32Array([0, 0, 0]);
  private _fovY   = Math.PI / 4;
  private _near   = 0.1;
  private _far    = 1000;
  private _aspect = 1;

  constructor(aspect = 1) {
    this._aspect = aspect;
  }

  get currentAspect(): number { return this._aspect; }

  setOptions(opt: CameraOptions, aspect: number): void {
    this._aspect = aspect;
    if (opt.position) this._pos    = new Float32Array(opt.position);
    if (opt.target)   this._target = new Float32Array(opt.target);
    if (opt.fov  != null) this._fovY = opt.fov;
    if (opt.near != null) this._near = opt.near;
    if (opt.far  != null) this._far  = opt.far;
  }

  setAspect(a: number): void { this._aspect = a; }

  getCameraState(): CameraState {
    const v = _lookAt(this._pos, this._target);
    const p = _perspective(this._fovY, this._aspect, this._near, this._far);
    return {
      viewMatrix: v,
      projectionMatrix: p,
      viewProjectionMatrix: _mul4(p, v),
      position: new Float32Array(this._pos),
      fovY: this._fovY,
      near: this._near,
      far: this._far,
      viewportWidth: 0,   // переопределяется в WebGPUElement реальным размером холста
      viewportHeight: 0,
    };
  }
}

// ---- Встроенный проход рендеринга --------------------------------------------------------------------------------------------

interface MeshGPU {
  vbo: GPUBuffer;
  ibo: GPUBuffer;
  indexCount: number;
}

interface ObjectEntry {
  meshId: string;
  color: Float32Array;
  materialId: number;
}

class BuiltinRenderPass {
  private _device: GPUDevice;
  private _pipeline: GPURenderPipeline | null = null;
  private _sceneUbo: GPUBuffer | null = null;
  private _sceneBg: GPUBindGroup | null = null;
  private _objUbo: GPUBuffer | null = null;
  private _objBg: GPUBindGroup | null = null;
  private _bgl2: GPUBindGroupLayout | null = null;
  private _registry: ResourceRegistry | null = null;
  private _aspect = 1;
  private _fmt: GPUTextureFormat;

  private readonly _meshes  = new Map<string, MeshGPU>();
  private readonly _objects = new Map<string, ObjectEntry>();

  private readonly _lightDir   = new Float32Array([0.5773, -0.5773, -0.5773]);
  private readonly _lightColor = new Float32Array([1.2, 1.1, 1.0]);

  constructor(device: GPUDevice, fmt: GPUTextureFormat) {
    this._device = device;
    this._fmt    = fmt;
    this._init();
  }

  private _init(): void {
    const d = this._device;

    // ---- Буферы ------------------------------------------------------------------------------------------------------------------------------
    d.pushErrorScope("out-of-memory");
    this._sceneUbo = d.createBuffer({
      label: "builtin-scene-ubo",
      size: SCENE_UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this._objUbo = d.createBuffer({
      label: "builtin-obj-ubo",
      size: OBJ_STRIDE * MAX_OBJECTS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    void d.popErrorScope().then((err) => { if (err) console.error("[api] Переполнение памяти GPU при выделении UBO:", err.message); });

    // ---- Макеты групп привязок --------------------------------------------------------------------------------------------------
    const bgl0 = d.createBindGroupLayout({
      label: "builtin-bgl0",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
    });
    const bgl1 = d.createBindGroupLayout({
      label: "builtin-bgl1",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform", hasDynamicOffset: true } }],
    });
    // Group 2: streaming bind group — совпадает с BindGroupManager.createLayout()
    const bgl2 = d.createBindGroupLayout({
      label: "builtin-bgl2-streaming",
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
    this._bgl2 = bgl2;

    this._sceneBg = d.createBindGroup({
      label: "builtin-scene-bg",
      layout: bgl0,
      entries: [{ binding: 0, resource: { buffer: this._sceneUbo! } }],
    });

    this._objBg = d.createBindGroup({
      label: "builtin-obj-bg",
      layout: bgl1,
      entries: [{ binding: 0, resource: { buffer: this._objUbo!, size: OBJ_STRIDE } }],
    });

    const module = d.createShaderModule({ label: "builtin-shader", code: BUILTIN_WGSL });

    this._pipeline = d.createRenderPipeline({
      label: "builtin-pipeline",
      layout: d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1, bgl2] }),
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 32,
          attributes: [
            { shaderLocation: 0, offset:  0, format: "float32x3" },
            { shaderLocation: 1, offset: 12, format: "float32x3" },
            { shaderLocation: 2, offset: 24, format: "float32x2" },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{ format: this._fmt }],
      },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      primitive: { topology: "triangle-list", cullMode: "back" },
    });
  }

  setRegistry(registry: ResourceRegistry): void {
    this._registry = registry;
  }

  registerMesh(id: string, verts: Float32Array<ArrayBuffer>, idxs: Uint16Array<ArrayBuffer>): void {
    const d = this._device;
    const old = this._meshes.get(id);
    if (old) { old.vbo.destroy(); old.ibo.destroy(); }

    d.pushErrorScope("out-of-memory");
    const vbo = d.createBuffer({ label: `vbo-${id}`, size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    void d.popErrorScope().then((err) => { if (err) console.error(`[api] Переполнение памяти GPU при выделении VBO для ${id}:`, err.message); });

    d.pushErrorScope("out-of-memory");
    const ibo = d.createBuffer({ label: `ibo-${id}`, size: Math.ceil(idxs.byteLength / 4) * 4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    void d.popErrorScope().then((err) => { if (err) console.error(`[api] Переполнение памяти GPU при выделении IBO для ${id}:`, err.message); });

    d.queue.writeBuffer(vbo, 0, verts);
    if (idxs.byteLength % 4 === 0) {
      d.queue.writeBuffer(ibo, 0, idxs);
    } else {
      const padded = new Uint16Array(Math.ceil(idxs.length / 2) * 2);
      padded.set(idxs);
      d.queue.writeBuffer(ibo, 0, padded);
    }
    this._meshes.set(id, { vbo, ibo, indexCount: idxs.length });
  }

  addObject(
    nodeId: string,
    meshId: string,
    color: Float32Array<ArrayBuffer> = new Float32Array([0.7, 0.5, 0.3, 1]) as Float32Array<ArrayBuffer>,
    materialId: number = NO_TEXTURE_MATERIAL_ID,
  ): void {
    this._objects.set(nodeId, { meshId, color, materialId });
  }

  removeObject(nodeId: string): void {
    this._objects.delete(nodeId);
  }

  onResize(w: number, h: number): void {
    this._aspect = h > 0 ? w / h : 1;
  }

  execute(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    camera: CameraState,
    nodes: import("@webgpu-streaming/gpu-types").SceneGraphReadView,
  ): void {
    if (!this._pipeline || !this._sceneUbo || !this._objUbo || !this._sceneBg || !this._objBg) return;

    const streamingBg = this._registry?.request<GPUBindGroup>(MATERIAL_BIND_GROUP_KEY) ?? null;
    if (!streamingBg) return; // не готов до инициализации streaming manager

    const d = this._device;

    // Обновить UBO сцены
    const sceneData = new Float32Array(32);
    sceneData.set(camera.viewProjectionMatrix, 0);
    sceneData.set(camera.position, 16);
    sceneData.set(this._lightDir, 20);
    sceneData.set(this._lightColor, 24);
    d.queue.writeBuffer(this._sceneUbo, 0, sceneData);

    // Обновить UBO для каждого объекта
    const objBuf = new Float32Array(MAX_OBJECTS * OBJ_STRIDE / 4);
    const u32view = new Uint32Array(objBuf.buffer);
    let objCount = 0;
    for (const node of nodes.nodes) {
      if (!node.visible || objCount >= MAX_OBJECTS) continue;
      const off = (objCount * OBJ_STRIDE) / 4;
      objBuf.set(node.worldTransform, off);           // model:     float[off..off+15]
      const entry = this._objects.get(node.id);
      const color = entry?.color ?? new Float32Array([0.7, 0.5, 0.3, 1]);
      objBuf.set(color, off + 16);                    // baseColor: float[off+16..off+19]
      // materialId находится на байтовом смещении 80 = float-смещение 20 от начала объекта
      u32view[off + 20] = entry?.materialId ?? NO_TEXTURE_MATERIAL_ID;
      objCount++;
    }
    d.queue.writeBuffer(this._objUbo, 0, objBuf, 0, objCount * OBJ_STRIDE / 4);

    const objBg = this._objBg;

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: { view: depthView, depthLoadOp: "load", depthStoreOp: "store" },
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._sceneBg);
    pass.setBindGroup(2, streamingBg);

    let drawn = 0;
    for (const node of nodes.nodes) {
      if (!node.visible || drawn >= MAX_OBJECTS) continue;
      const entry = this._objects.get(node.id);
      if (!entry) { drawn++; continue; }
      const mesh = this._meshes.get(entry.meshId);
      if (!mesh)  { drawn++; continue; }

      pass.setBindGroup(1, objBg, [drawn * OBJ_STRIDE]);
      pass.setVertexBuffer(0, mesh.vbo);
      pass.setIndexBuffer(mesh.ibo, "uint16");
      pass.drawIndexed(mesh.indexCount);
      drawn++;
    }
    pass.end();
  }

  destroy(): void {
    this._sceneUbo?.destroy();
    this._objUbo?.destroy();
    for (const m of this._meshes.values()) { m.vbo.destroy(); m.ibo.destroy(); }
    this._meshes.clear();
  }
}

// ---- TextureHandleImpl ------------------------------------------------------------------------------------------------------------------

class TextureHandleImpl implements TextureHandle {
  private _onDestroy: (() => void) | null;
  constructor(
    readonly id: string,
    readonly width: number,
    readonly height: number,
    readonly mipLevels: number,
    readonly residentMip: number,
    onDestroy?: () => void,
  ) {
    this._onDestroy = onDestroy ?? null;
  }
  destroy(): void {
    this._onDestroy?.();
    this._onDestroy = null;
  }
}

// ---- RendererImpl ----------------------------------------------------------------------------------------------------------------------------

class RendererImpl implements Renderer {
  private _meshCounter = 0;
  private _texCounter  = 0;
  private _texLoadedCount = 0;
  private readonly _tracker = new FrameStatsTracker();
  private _disposed = false;

  // materialId → streaming manager ID; id → materialId
  private readonly _matIdMap = new Map<string, number>();
  private _nextMatId = 0;

  constructor(
    private readonly _device: GPUDevice,
    private readonly _dm: DeviceManager,
    private readonly _loop: RenderLoop,
    private readonly _sceneGraph: SceneGraph,
    private readonly _pass: BuiltinRenderPass,
    private readonly _camera: ManualCameraController,
    private readonly _orbit: CameraController,
    private readonly _budgetMB: number,
    private readonly _streaming: TextureStreamingManager,
  ) {}

  get device(): GPUDevice { return this._device; }
  get ready(): boolean    { return !this._disposed; }

  async loadScene(source?: string | ArrayBuffer): Promise<void> {
    if (source == null) {
      const { vertices, indices } = makeCubeGeometry();
      const id = `scene-cube-${Date.now()}`;
      this._pass.registerMesh(id, vertices as Float32Array<ArrayBuffer>, indices as Uint16Array<ArrayBuffer>);
      this._sceneGraph.addNode(id, 0, mat4Identity(), new Float32Array([0, 0, 0, 1.8]));
      this._pass.addObject(id, id);
      return;
    }

    let format = detectFormat(source);
    let data: ArrayBuffer;
    let mtlText: string | undefined;

    if (typeof source === "string") {
      const resp = await fetch(source);
      if (!resp.ok) throw new Error(`[loadScene] Ошибка загрузки: ${source} (${resp.status})`);
      data = await resp.arrayBuffer();
      const dataFormat = detectFormat(data);
      if (dataFormat !== "unknown") format = dataFormat;

      if (format === "obj") {
        const mtlMatch = new TextDecoder().decode(data).match(/^mtllib\s+(.+)$/m);
        if (mtlMatch?.[1]) {
          const mtlUrl = new URL(mtlMatch[1].trim(), source).href;
          try {
            const mtlResp = await fetch(mtlUrl);
            if (mtlResp.ok) mtlText = await mtlResp.text();
          } catch {
            console.warn(`[loadScene] Не удалось загрузить MTL: ${mtlUrl}`);
          }
        }
      }
    } else {
      data = source;
    }

    let meshes: import("./loaders/OBJLoader.js").ParsedMesh[];

    switch (format) {
      case "obj":
        meshes = parseOBJBuffer(data, mtlText).meshes;
        break;
      case "stl-binary":
      case "stl-ascii":
        meshes = parseSTL(data).meshes;
        break;
      default:
        throw new Error(`[loadScene] Неподдерживаемый формат: ${format}. Поддерживаются: obj, stl, glb, gltf.`);
    }

    for (const mesh of meshes) {
      this.addMesh(mesh.geometry, mesh.material);
    }
  }

  async loadTexture(src: string | ArrayBuffer, _opt?: TextureOptions): Promise<TextureHandle> {
    let data: ArrayBuffer;
    if (typeof src === "string") {
      const resp = await fetch(src);
      if (!resp.ok) throw new Error(`[loadTexture] Не удалось загрузить: ${src} (${resp.status})`);
      data = await resp.arrayBuffer();
    } else {
      data = src;
    }

    const { data: rgba, width, height } = await decodeImage(new Blob([data]));
    const { mips } = buildMipPyramid(rgba, width, height);
    const ktx2Bytes = packKtx2(mips, width, height);
    const parsed = parseKTX2(ktx2Bytes);

    const id = `tex-${++this._texCounter}`;
    const matId = this._nextMatId++;

    this._streaming.registerTexture(id, parsed, ktx2Bytes, matId, new Float32Array([0, 0, 0, 1]));
    this._matIdMap.set(id, matId);
    this._texLoadedCount++;

    return new TextureHandleImpl(id, width, height, mips.length, parsed.levelCount - 1, () => {
      this._streaming.unregisterTexture(id);
      this._matIdMap.delete(id);
      this._texLoadedCount = Math.max(0, this._texLoadedCount - 1);
    });
  }

  addMesh(geometry: GeometryDescriptor, material: MaterialDescriptor): MeshHandle {
    const nodeId  = `node-${++this._meshCounter}`;
    const verts   = buildInterleavedVertices(geometry);
    const vertexCount = geometry.positions.length / 3;
    const idxs    = toUint16Indices(geometry.indices, vertexCount);
    const sphere  = computeBoundingSphere(geometry.positions);

    let color: Float32Array<ArrayBuffer>;
    let materialId = NO_TEXTURE_MATERIAL_ID;

    if (material.baseColor && typeof material.baseColor !== "string") {
      color = new Float32Array([1, 1, 1, 1]) as Float32Array<ArrayBuffer>;
      const texId = material.baseColor.id;
      materialId = this._matIdMap.get(texId) ?? NO_TEXTURE_MATERIAL_ID;
    } else {
      color = _parseColor(material.baseColor as string | undefined);
    }

    this._pass.registerMesh(nodeId, verts as Float32Array<ArrayBuffer>, idxs as Uint16Array<ArrayBuffer>);
    this._sceneGraph.addNode(nodeId, 0, mat4Identity(), sphere);
    this._pass.addObject(nodeId, nodeId, color, materialId);

    return new MeshHandleImpl(nodeId, this._sceneGraph, this._pass);
  }

  removeMesh(handle: MeshHandle): void { handle.destroy(); }

  setCamera(options: CameraOptions): void {
    if (options.position && options.target) {
      const [px, py, pz] = options.position;
      const [tx, ty, tz] = options.target;
      const dx = px! - tx!, dy = py! - ty!, dz = pz! - tz!;
      this._orbit.setRadius(Math.hypot(dx, dy, dz));
      this._orbit.setTarget(tx!, ty!, tz!);
    }
    if (options.fov)  this._orbit.setFov(options.fov * Math.PI / 180);
    if (options.near != null && options.far != null) this._orbit.setClip(options.near, options.far);
    this._camera.setOptions(options, this._camera.currentAspect);
  }

  getStats(): StreamingStats {
    const bt = this._streaming.budgetTracker;
    return {
      memoryUsedMB:   bt ? bt.totalUsed / (1024 * 1024) : 0,
      memoryBudgetMB: this._budgetMB,
      texturesLoaded: this._texLoadedCount,
      texturesTotal:  this._texLoadedCount,
      residentMipDistribution: {},
      fps: this._tracker.fps(),
      frameTimeP99Ms: this._tracker.p99Ms(),
    };
  }

  dispose(): void {
    if (this._disposed) return;
    this._disposed = true;
    this._loop.destroy();
    this._pass.destroy();
    this._orbit.destroy();
    this._dm.destroy();
  }

  _recordFrame(now: number): void {
    this._tracker.record(now);
  }
}

// ---- createRenderer ------------------------------------------------------------------------------------------------------------------------

/**
 * Создать полностью настроенный WebGPU-рендерер, выполняющий рендеринг в существующий элемент canvas.
 * Инициализирует GPU-устройство, встроенный Phong-рендерер и бюджет потоковой передачи текстур.
 *
 * @example
 * const renderer = await createRenderer({ canvas, memoryBudget: 256 });
 * renderer.setCamera({ position: [0, 2, 5], target: [0, 0, 0] });
 * await renderer.loadScene();
 *
 * // Или добавить геометрию напрямую:
 * const mesh = renderer.addMesh(geometry, { baseColor: '#e88033' });
 * mesh.setPosition(1, 0, 0);
 */
export async function createRenderer(options: CreateRendererOptions): Promise<Renderer> {
  const {
    canvas,
    memoryBudget = 256,
    frameUploadCap = 8,
    powerPreference: _pref = "high-performance",
  } = options;

  if (!navigator.gpu) throw new Error("WebGPU не поддерживается в этом браузере.");

  const dm = new DeviceManager();
  await dm.initialize();
  const device = dm.device!;

  const gpuCtx = canvas.getContext("webgpu") as GPUCanvasContext | null;
  if (!gpuCtx) throw new Error("Не удалось получить GPUCanvasContext.");
  const fmt = navigator.gpu.getPreferredCanvasFormat();
  gpuCtx.configure({ device, format: fmt, alphaMode: "opaque" });

  let w = canvas.width  || canvas.clientWidth  || 300;
  let h = canvas.height || canvas.clientHeight || 150;
  canvas.width = w; canvas.height = h;

  let depthTex  = _makeDepth(device, w, h);
  let depthView = depthTex.createView();

  const sceneGraph = new SceneGraph();
  const camera     = new ManualCameraController(w / Math.max(h, 1));
  const orbit      = new CameraController(canvas);
  const pass       = new BuiltinRenderPass(device, fmt);
  pass.onResize(w, h);

  const registry   = new ResourceRegistryImpl();
  const pluginHost = new PluginHost();
  await pluginHost.initialize(device, registry, fmt);

  // Инициализировать TextureStreamingManager — он регистрирует bind group в registry
  const streaming = new TextureStreamingManager({
    budgetBytes: memoryBudget * 1024 * 1024,
    frameUploadBudget: frameUploadCap * 1024 * 1024,
    maxLayersPerTier: [32, 16, 4],
  });
  await streaming.initialize({ device, registry });

  pass.setRegistry(registry);

  const loop = new RenderLoop();

  const impl = new RendererImpl(device, dm, loop, sceneGraph, pass, camera, orbit, memoryBudget, streaming);

  const ro = new ResizeObserver((entries) => {
    for (const e of entries) {
      const nw = Math.round(e.contentRect.width);
      const nh = Math.round(e.contentRect.height);
      if (nw === w && nh === h) continue;
      w = nw; h = nh;
      canvas.width = w; canvas.height = h;
      depthTex.destroy();
      depthTex  = _makeDepth(device, w, h);
      depthView = depthTex.createView();
      pass.onResize(w, h);
      camera.setAspect(w / Math.max(h, 1));
    }
  });
  ro.observe(canvas);

  loop.addCallback((deltaTime, frameIndex) => {
    if (w <= 0 || h <= 0) return;
    impl._recordFrame(performance.now());

    const colorView = gpuCtx.getCurrentTexture().createView();
    const encoder   = device.createCommandEncoder({ label: `frame-${frameIndex}` });

    // Очистка фона
    const clearPass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, loadOp: "clear", storeOp: "store", clearValue: { r: 0.05, g: 0.05, b: 0.08, a: 1 } }],
      depthStencilAttachment: { view: depthView, depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 1 },
    });
    clearPass.end();

    const camState = orbit.getCameraState();

    // Обновить streaming manager (загружает мипы, вытесняет, перестраивает bind group)
    streaming.prepareFrame({
      device,
      encoder,
      camera: { ...camState, viewportWidth: w, viewportHeight: h },
      scene: sceneGraph.getReadView(),
      frameIndex,
      deltaTime,
      colorAttachment: colorView,
      depthAttachment: depthView,
    });

    pass.execute(encoder, colorView, depthView, camState, sceneGraph.getReadView());

    device.pushErrorScope("validation");
    const cmd = encoder.finish();
    void device.popErrorScope().then((e) => {
      if (e) console.error(`[createRenderer] кадр ${frameIndex}:`, e);
    });
    device.queue.submit([cmd]);
    void deltaTime;
  });

  loop.start();

  void device.lost.then(() => { ro.disconnect(); });

  return impl;
}

// ---- Вспомогательные функции ------------------------------------------------------------------------------------------------------

function _makeDepth(device: GPUDevice, w: number, h: number): GPUTexture {
  device.pushErrorScope("out-of-memory");
  const t = device.createTexture({
    label: "depth", size: [Math.max(w, 1), Math.max(h, 1)],
    format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
  void device.popErrorScope().then((err) => { if (err) console.error("[api] Переполнение памяти GPU при создании текстуры глубины:", err.message); });
  return t;
}

function _parseColor(c?: string | TextureHandle): Float32Array<ArrayBuffer> {
  if (!c || typeof c !== "string") return new Float32Array([0.7, 0.5, 0.3, 1]) as Float32Array<ArrayBuffer>;
  const hex = c.replace("#", "");
  if (hex.length === 6) {
    const r = parseInt(hex.slice(0, 2), 16) / 255;
    const g = parseInt(hex.slice(2, 4), 16) / 255;
    const b = parseInt(hex.slice(4, 6), 16) / 255;
    return new Float32Array([r, g, b, 1]) as Float32Array<ArrayBuffer>;
  }
  return new Float32Array([0.7, 0.5, 0.3, 1]) as Float32Array<ArrayBuffer>;
}

// ---- Вспомогательные функции mat4 (локальные, чтобы избежать циклических импортов) ------

function _lookAt(eye: Float32Array, center: Float32Array): Float32Array {
  const up = new Float32Array([0, 1, 0]);
  const fx = center[0]!-eye[0]!, fy = center[1]!-eye[1]!, fz = center[2]!-eye[2]!;
  const fl = Math.sqrt(fx*fx+fy*fy+fz*fz) || 1;
  const f = new Float32Array([fx/fl, fy/fl, fz/fl]);
  const r = _norm(_cross(f, up));
  const u = _cross(r, f);
  const m = mat4Identity();
  m[0]=r[0]!; m[4]=r[1]!; m[8]=r[2]!;
  m[1]=u[0]!; m[5]=u[1]!; m[9]=u[2]!;
  m[2]=-f[0]!; m[6]=-f[1]!; m[10]=-f[2]!;
  m[12]=-(r[0]!*eye[0]!+r[1]!*eye[1]!+r[2]!*eye[2]!);
  m[13]=-(u[0]!*eye[0]!+u[1]!*eye[1]!+u[2]!*eye[2]!);
  m[14]= (f[0]!*eye[0]!+f[1]!*eye[1]!+f[2]!*eye[2]!);
  return m;
}

function _perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1/Math.tan(fovY/2);
  const ri = 1/(near-far);
  // prettier-ignore
  return new Float32Array([
    f/aspect, 0, 0,                 0,
    0,        f, 0,                 0,
    0,        0, (far+near)*ri,    -1,
    0,        0, 2*far*near*ri,     0,
  ]);
}

function _mul4(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k*4+j]! * b[i*4+k]!;
    o[i*4+j] = s;
  }
  return o;
}

function _norm(v: Float32Array): Float32Array {
  const l = Math.sqrt(v[0]!*v[0]!+v[1]!*v[1]!+v[2]!*v[2]!) || 1;
  return new Float32Array([v[0]!/l, v[1]!/l, v[2]!/l]);
}

function _cross(a: Float32Array, b: Float32Array): Float32Array {
  return new Float32Array([a[1]!*b[2]!-a[2]!*b[1]!, a[2]!*b[0]!-a[0]!*b[2]!, a[0]!*b[1]!-a[1]!*b[0]!]);
}
