import type {
  TextureHandle,
  TextureOptions,
  GeometryDescriptor,
  MaterialDescriptor,
  MeshHandle,
  CameraOptions,
  StreamingStats,
} from "@webgpu-streaming/gpu-types";
import { WebGPUElement } from "./WebGPUElement.js";
import { CameraController } from "./CameraController.js";
import {
  mat4Identity,
  makeCubeGeometry,
  buildInterleavedVertices,
  toUint16Indices,
  computeBoundingSphere,
  MeshHandleImpl,
  FrameStatsTracker,
} from "./_scene-utils.js";
import { ManualCameraController } from "./api.js";
import { BUILTIN_WGSL } from "./_builtin-wgsl.js";
import { detectFormat } from "./loaders/FormatDetector.js";
import { parseOBJBuffer } from "./loaders/OBJLoader.js";
import { parseSTL } from "./loaders/STLLoader.js";

const SCENE_UBO_SIZE = 128;
const OBJ_STRIDE     = 256;
const MAX_OBJECTS    = 256;

// ---- ElementRenderPass ------------------------------------------------------------------------------------------------------------------

interface MeshGPU { vbo: GPUBuffer; ibo: GPUBuffer; indexCount: number }

class ElementRenderPass {
  private _pipeline: GPURenderPipeline | null = null;
  private _sceneUbo: GPUBuffer | null = null;
  private _sceneBg: GPUBindGroup | null = null;
  private _objUbo: GPUBuffer | null = null;
  private _objBg: GPUBindGroup | null = null;

  private readonly _meshes  = new Map<string, MeshGPU>();
  private readonly _objects = new Map<string, { meshId: string; color: Float32Array }>();

  private readonly _light = new Float32Array([0.5773, -0.5773, -0.5773]);

  constructor(private readonly _dev: GPUDevice, private readonly _fmt: GPUTextureFormat) {
    this._setup();
  }

  private _setup(): void {
    const d = this._dev;

    d.pushErrorScope("out-of-memory");
    this._sceneUbo = d.createBuffer({ label: "el-scene-ubo", size: SCENE_UBO_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._objUbo   = d.createBuffer({ label: "el-obj-ubo", size: OBJ_STRIDE * MAX_OBJECTS, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    void d.popErrorScope().then((err) => { if (err) console.error("[WebGPUCanvasPublic] Переполнение памяти GPU:", err.message); });

    const bgl0 = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: {} }] });
    const bgl1 = d.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { hasDynamicOffset: true } }] });

    this._sceneBg = d.createBindGroup({ layout: bgl0, entries: [{ binding: 0, resource: { buffer: this._sceneUbo! } }] });
    this._objBg   = d.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: this._objUbo!,  size: OBJ_STRIDE } }] });

    const mod = d.createShaderModule({ label: "el-shader", code: BUILTIN_WGSL });
    this._pipeline = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] }),
      vertex: { module: mod, entryPoint: "vs_main", buffers: [{ arrayStride: 32, attributes: [
        { shaderLocation: 0, offset:  0, format: "float32x3" },
        { shaderLocation: 1, offset: 12, format: "float32x3" },
        { shaderLocation: 2, offset: 24, format: "float32x2" },
      ] }] },
      fragment: { module: mod, entryPoint: "fs_main", targets: [{ format: this._fmt }] },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      primitive: { topology: "triangle-list", cullMode: "back" },
    });
  }

  registerMesh(id: string, verts: Float32Array<ArrayBuffer>, idxs: Uint16Array<ArrayBuffer>): void {
    const d = this._dev;
    const old = this._meshes.get(id);
    if (old) { old.vbo.destroy(); old.ibo.destroy(); }
    d.pushErrorScope("out-of-memory");
    const vbo = d.createBuffer({ size: verts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    const ibo = d.createBuffer({ size: Math.ceil(idxs.byteLength/4)*4, usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
    void d.popErrorScope().then((err) => { if (err) console.error("[WebGPUCanvasPublic] Переполнение памяти GPU при регистрации меша:", err.message); });
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

  addObject(nodeId: string, meshId: string, color: Float32Array<ArrayBuffer> = new Float32Array([0.7, 0.5, 0.3, 1]) as Float32Array<ArrayBuffer>): void {
    this._objects.set(nodeId, { meshId, color });
  }

  removeObject(nodeId: string): void { this._objects.delete(nodeId); }

  execute(
    encoder: GPUCommandEncoder,
    colorView: GPUTextureView,
    depthView: GPUTextureView,
    camera: import("@webgpu-streaming/gpu-types").CameraState,
    nodes: import("@webgpu-streaming/gpu-types").SceneGraphReadView,
  ): void {
    if (!this._pipeline || !this._sceneUbo || !this._objUbo || !this._sceneBg || !this._objBg) return;
    const d = this._dev;

    const sd = new Float32Array(32);
    sd.set(camera.viewProjectionMatrix, 0);
    sd.set(camera.position, 16);
    sd.set(this._light, 20);
    sd.set(new Float32Array([1.2, 1.1, 1.0, 1]), 24);
    d.queue.writeBuffer(this._sceneUbo, 0, sd);

    const od = new Float32Array(MAX_OBJECTS * OBJ_STRIDE / 4);
    let cnt = 0;
    for (const n of nodes.nodes) {
      if (!n.visible || cnt >= MAX_OBJECTS) continue;
      const off = cnt * OBJ_STRIDE / 4;
      od.set(n.worldTransform, off);
      const c = this._objects.get(n.id)?.color ?? new Float32Array([0.7, 0.5, 0.3, 1]);
      od.set(c, off + 16);
      cnt++;
    }
    d.queue.writeBuffer(this._objUbo, 0, od, 0, cnt * OBJ_STRIDE / 4);

    const objBg = this._objBg;

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: colorView, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: { view: depthView, depthLoadOp: "load", depthStoreOp: "store" },
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._sceneBg);

    let drawn = 0;
    for (const n of nodes.nodes) {
      if (!n.visible || drawn >= MAX_OBJECTS) continue;
      const entry = this._objects.get(n.id);
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

// ---- Вспомогательная функция разбора цвета --------------------------------------------------------------------------

function parseColor(c?: string | TextureHandle): Float32Array<ArrayBuffer> {
  const mk = (r: number, g: number, b: number) =>
    new Float32Array([r, g, b, 1]) as Float32Array<ArrayBuffer>;
  if (!c || typeof c !== "string") return mk(0.7, 0.5, 0.3);
  const hex = c.replace("#", "");
  if (hex.length === 6) {
    return mk(
      parseInt(hex.slice(0, 2), 16) / 255,
      parseInt(hex.slice(2, 4), 16) / 255,
      parseInt(hex.slice(4, 6), 16) / 255,
    );
  }
  return mk(0.7, 0.5, 0.3);
}

// ---- WebGPUCanvasElement --------------------------------------------------------------------------------------------------------------

/**
 * Публичный Custom Element `<webgpu-canvas-streaming>`.
 *
 * Расширяет WebGPUElement:
 *  - Отражаемые HTML-атрибуты для настройки рендеринга
 *  - DOM-события для потребителей: gpu-ready, gpu-lost, streaming-stats
 *  - Высокоуровневый императивный API: loadScene, loadTexture, addMesh, setCamera, getStats, reset
 *
 * @example
 * ```html
 * <webgpu-canvas-streaming memory-budget="256" show-stats></webgpu-canvas-streaming>
 * <script type="module">
 *   import "@webgpu-streaming/core";
 *   const el = document.querySelector("webgpu-canvas-streaming");
 *   el.addEventListener("gpu-ready", async () => {
 *     await el.loadScene();
 *   });
 * </script>
 * ```
 */
export class WebGPUCanvasElement extends WebGPUElement {
  static get observedAttributes(): string[] {
    return ["memory-budget", "frame-upload-cap", "texture-tiers", "max-layers-per-tier", "show-stats", "camera-mode", "power-preference"];
  }

  // ---- Отражаемые атрибуты ------------------------------------------------------------------------------------------------------

  /** Бюджет памяти GPU для текстур в МБ. По умолчанию: 256 */
  get memoryBudget(): number  { return Number(this.getAttribute("memory-budget")  ?? 256); }
  set memoryBudget(v: number) { this.setAttribute("memory-budget", String(v)); }

  /** Максимальный объём данных текстур, загружаемых за кадр, в МБ. По умолчанию: 8 */
  get frameUploadCap(): number  { return Number(this.getAttribute("frame-upload-cap") ?? 8); }
  set frameUploadCap(v: number) { this.setAttribute("frame-upload-cap", String(v)); }

  /** Размеры уровней разрешения через запятую. По умолчанию: "512,1024,2048" */
  get textureTiers(): string  { return this.getAttribute("texture-tiers") ?? "512,1024,2048"; }
  set textureTiers(v: string) { this.setAttribute("texture-tiers", v); }

  /** Максимальное число слоёв в одном уровне текстурного массива. По умолчанию: 64 */
  get maxLayersPerTier(): number  { return Number(this.getAttribute("max-layers-per-tier") ?? 64); }
  set maxLayersPerTier(v: number) { this.setAttribute("max-layers-per-tier", String(v)); }

  /** Показывать наложение статистики потоковой передачи поверх холста. По умолчанию: false */
  get showStats(): boolean  { return this.hasAttribute("show-stats"); }
  set showStats(v: boolean) { v ? this.setAttribute("show-stats", "") : this.removeAttribute("show-stats"); }

  /** Режим управления камерой. По умолчанию: "orbit" */
  get cameraMode(): string  { return this.getAttribute("camera-mode") ?? "orbit"; }
  set cameraMode(v: string) { this.setAttribute("camera-mode", v); }

  /** Предпочтение мощности GPU. По умолчанию: "high-performance" */
  get powerPreference(): string  { return this.getAttribute("power-preference") ?? "high-performance"; }
  set powerPreference(v: string) { this.setAttribute("power-preference", v); }

  /** true после того, как сработал gpu-ready и элемент выполняет рендеринг. */
  get publicReady(): boolean { return this._publicReady; }

  // ---- Приватное состояние ------------------------------------------------------------------------------------------------------

  private _elPass: ElementRenderPass | null = null;
  private _manualCamera: ManualCameraController | null = null;
  private _orbitCamera: CameraController | null = null;
  private _meshCounter = 0;
  private _texCounter  = 0;
  private _publicReady = false;
  private readonly _tracker = new FrameStatsTracker();
  private _statsRafId: number | null = null;

  // ---- Жизненный цикл --------------------------------------------------------------------------------------------------------------

  override async connectedCallback(): Promise<void> {
    await super.connectedCallback();
    if (!this.device) return; // устройство недоступно

    const device = this.device;
    const fmt    = navigator.gpu.getPreferredCanvasFormat();

    this._elPass = new ElementRenderPass(device, fmt);

    // Подключить колбэк рендеринга поверх цикла базового элемента.
    // Мы подключаемся к детали "webgpu-ready", чтобы получать colorAttachment каждый кадр.
    // Вместо этого переопределяем через setCameraController + используем перенаправление addRenderPass.
    //
    // Простейший подход: создать обёртку, совместимую с IRenderPass, и зарегистрировать её.
    const self = this;
    await this.addRenderPass({
      name: "element-builtin-pass",
      async initialize() {},
      execute(ctx) {
        if (!self._elPass) return;
        self._elPass.execute(
          ctx.encoder,
          ctx.colorAttachment,
          ctx.depthAttachment,
          ctx.camera,
          ctx.scene,
        );
      },
      onResize() {},
      destroy() { self._elPass?.destroy(); self._elPass = null; },
    });

    // Настройка камеры
    if (this.cameraMode === "orbit") {
      const cam = new CameraController(this.canvasElement);
      this._orbitCamera = cam;
      this.setCameraController(cam);
    } else {
      const cam = new ManualCameraController(this.canvasWidth / Math.max(this.canvasHeight, 1));
      this._manualCamera = cam;
      this.setCameraController(cam);
    }

    this._publicReady = true;

    // Вызвать gpu-ready с публичными данными
    this.dispatchEvent(new CustomEvent("gpu-ready", {
      detail: {
        adapter: this.adapterInfo,
        features: Array.from(device.features),
        limits: {},
      },
      bubbles: true,
      composed: true,
    }));

    // Запустить события статистики каждый кадр
    this._startStatsLoop();
  }

  override disconnectedCallback(): void {
    this._publicReady = false;
    if (this._statsRafId !== null) {
      cancelAnimationFrame(this._statsRafId);
      this._statsRafId = null;
    }
    this._orbitCamera = null;
    this._manualCamera = null;
    this._meshCounter = 0;
    super.disconnectedCallback();
    this._elPass = null;
  }

  // ---- Высокоуровневый API ------------------------------------------------------------------------------------------------------

  /**
   * Загрузить сцену. На фазе 6.1 создаётся простой процедурный куб.
   * Полная поддержка glTF запланирована на фазу 6.4.
   */
  async loadScene(source?: string | ArrayBuffer): Promise<void> {
    const pass = this._elPass;
    const sg   = this.sceneGraph;
    if (!pass || !sg) {
      throw new Error("[WebGPUCanvasElement] loadScene вызван до gpu-ready");
    }

    if (source == null) {
      const { vertices, indices } = makeCubeGeometry();
      const id = `scene-cube-${Date.now()}`;
      pass.registerMesh(id, vertices as Float32Array<ArrayBuffer>, indices as Uint16Array<ArrayBuffer>);
      sg.addNode(id, 0, mat4Identity(), new Float32Array([0, 0, 0, 1.8]));
      pass.addObject(id, id);
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

  /** Загрузить текстуру для использования в материалах. Возвращает TextureHandle. */
  async loadTexture(_src: string | ArrayBuffer, _opt?: TextureOptions): Promise<TextureHandle> {
    return new TextureHandleImpl(`tex-${++this._texCounter}`, 1, 1, 1, 0);
  }

  /** Добавить геометрию в сцену. Возвращает дескриптор для трансформаций / удаления. */
  addMesh(geometry: GeometryDescriptor, material: MaterialDescriptor): MeshHandle {
    const pass = this._elPass;
    const sg   = this.sceneGraph;
    if (!pass || !sg) {
      throw new Error("[WebGPUCanvasElement] addMesh вызван до gpu-ready");
    }
    const nodeId = `node-${++this._meshCounter}`;
    const verts  = buildInterleavedVertices(geometry);
    const idxs   = toUint16Indices(geometry.indices, geometry.positions.length / 3);
    const sphere = computeBoundingSphere(geometry.positions);
    const color  = parseColor(material.baseColor);

    pass.registerMesh(nodeId, verts as Float32Array<ArrayBuffer>, idxs as Uint16Array<ArrayBuffer>);
    sg.addNode(nodeId, 0, mat4Identity(), sphere);
    pass.addObject(nodeId, nodeId, color);

    return new MeshHandleImpl(nodeId, sg, pass);
  }

  /** Удалить меш. */
  removeMesh(handle: MeshHandle): void { handle.destroy(); }

  /** Обновить параметры камеры. */
  setCamera(options: CameraOptions): void {
    const orbit  = this._orbitCamera;
    const manual = this._manualCamera;
    if (orbit && options.position && options.target) {
      const [px,py,pz] = options.position;
      const [tx,ty,tz] = options.target;
      const dist = Math.sqrt((px!-tx!)**2 + (py!-ty!)**2 + (pz!-tz!)**2);
      orbit.setRadius(dist);
    }
    if (manual) {
      manual.setOptions(options, this.canvasWidth / Math.max(this.canvasHeight, 1));
    }
  }

  /** Вернуть текущую статистику рендеринга. */
  getStats(): StreamingStats {
    return {
      memoryUsedMB: 0,
      memoryBudgetMB: this.memoryBudget,
      texturesLoaded: 0,
      texturesTotal: 0,
      residentMipDistribution: {},
      fps: this._tracker.fps(),
      frameTimeP99Ms: this._tracker.p99Ms(),
    };
  }

  /** Сбросить состояние сцены. */
  reset(): void {
    this._meshCounter = 0;
  }

  // ---- Приватные методы ----------------------------------------------------------------------------------------------------------

  private _startStatsLoop(): void {
    const tick = (now: number) => {
      if (!this._publicReady) return;
      this._tracker.record(now);
      this.dispatchEvent(new CustomEvent("streaming-stats", {
        detail: this.getStats(),
        bubbles: true,
        composed: true,
      }));
      this._statsRafId = requestAnimationFrame(tick);
    };
    this._statsRafId = requestAnimationFrame(tick);
  }
}

// Зарегистрировать публичный Custom Element под именем, отличным от внутреннего <webgpu-canvas>.
if (!customElements.get("webgpu-canvas-streaming")) {
  customElements.define("webgpu-canvas-streaming", WebGPUCanvasElement);
}
