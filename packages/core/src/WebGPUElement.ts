import { DeviceManager } from "./DeviceManager.js";
import { RenderLoop } from "./RenderLoop.js";
import { ResourceRegistryImpl } from "./ResourceRegistryImpl.js";
import { PluginHost } from "./PluginHost.js";
import { SceneGraph } from "./SceneGraph.js";
import type { CameraState } from "@webgpu-streaming/gpu-types";

/** Данные события "webgpu-ready". */
export interface WebGPUReadyDetail {
  device: GPUDevice;
  adapterInfo: GPUAdapterInfo | null;
  presentationFormat: GPUTextureFormat;
}

/**
 * Custom Element <webgpu-canvas>.
 *
 * Жизненный цикл:
 *   connectedCallback → DeviceManager.initialize → PluginHost.initialize → RenderLoop.start
 *   disconnectedCallback → RenderLoop.stop → PluginHost.destroy → DeviceManager.destroy
 *
 * События:
 *   "webgpu-ready" - вызывается после настройки устройства и контекста холста.
 *   "webgpu-error" - вызывается при ошибке инициализации.
 */
export class WebGPUElement extends HTMLElement {
  private _shadow: ShadowRoot;
  private _canvas: HTMLCanvasElement;
  private _context: GPUCanvasContext | null = null;
  private _deviceManager: DeviceManager | null = null;
  private _renderLoop: RenderLoop | null = null;
  private _registry: ResourceRegistryImpl | null = null;
  private _pluginHost: PluginHost | null = null;
  private _sceneGraph: SceneGraph | null = null;
  private _depthTexture: GPUTexture | null = null;
  private _depthView: GPUTextureView | null = null;
  private _resizeObserver: ResizeObserver;
  private _width = 0;
  private _height = 0;
  private _cameraController: { getCameraState(): CameraState } | null = null;
  private _deviceLost = false;

  /** Цвет очистки фона, применяемый каждый кадр перед запуском проходов плагинов. */
  clearColor: GPUColorDict = { r: 0.05, g: 0.05, b: 0.05, a: 1.0 };

  constructor() {
    super();
    this._shadow = this.attachShadow({ mode: "open" });

    const style = document.createElement("style");
    style.textContent = ":host { display: block; } canvas { display: block; width: 100%; height: 100%; }";
    this._shadow.appendChild(style);

    this._canvas = document.createElement("canvas");
    this._shadow.appendChild(this._canvas);

    this._resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        this._onResize(Math.round(width), Math.round(height));
      }
    });
  }

  get device(): GPUDevice | null {
    return this._deviceManager?.device ?? null;
  }

  get gpuContext(): GPUCanvasContext | null {
    return this._context;
  }

  get adapterInfo(): GPUAdapterInfo | null {
    return this._deviceManager?.adapterInfo ?? null;
  }

  get canvasWidth(): number {
    return this._width;
  }

  get canvasHeight(): number {
    return this._height;
  }

  get registry(): ResourceRegistryImpl | null {
    return this._registry;
  }

  get pluginHost(): PluginHost | null {
    return this._pluginHost;
  }

  get sceneGraph(): SceneGraph | null {
    return this._sceneGraph;
  }

  /** Внутренний элемент холста (полезен для подключения CameraController). */
  get canvasElement(): HTMLCanvasElement {
    return this._canvas;
  }

  /**
   * Подключить контроллер орбитальной камеры, чей getCameraState() вызывается каждый кадр.
   * Передайте null, чтобы вернуться к камере по умолчанию (единичная матрица).
   */
  setCameraController(controller: { getCameraState(): CameraState } | null): void {
    this._cameraController = controller;
  }

  /**
   * Зарегистрировать и инициализировать IResourceManager после запуска элемента.
   * Безопасно вызывать из обработчика события "webgpu-ready".
   */
  async addResourceManager(manager: import("@webgpu-streaming/gpu-types").IResourceManager): Promise<void> {
    if (!this._pluginHost || !this._deviceManager?.device || !this._registry) {
      throw new Error("[WebGPUElement] addResourceManager вызван до готовности устройства.");
    }
    this._pluginHost.registerResourceManager(manager);
    await manager.initialize({ device: this._deviceManager.device, registry: this._registry });
  }

  /**
   * Зарегистрировать и инициализировать IRenderPass после запуска элемента.
   * Безопасно вызывать из обработчика события "webgpu-ready".
   */
  async addRenderPass(pass: import("@webgpu-streaming/gpu-types").IRenderPass): Promise<void> {
    if (!this._pluginHost || !this._deviceManager?.device || !this._registry) {
      throw new Error("[WebGPUElement] addRenderPass вызван до готовности устройства.");
    }
    this._pluginHost.registerRenderPass(pass);
    await pass.initialize({
      device: this._deviceManager.device,
      registry: this._registry,
      presentationFormat: navigator.gpu.getPreferredCanvasFormat(),
    });
    pass.onResize(this._width, this._height);
  }

  async connectedCallback(): Promise<void> {
    this._resizeObserver.observe(this._canvas);

    const deviceManager = new DeviceManager();
    this._deviceManager = deviceManager;

    try {
      await deviceManager.initialize();
    } catch (err) {
      this._showError(String(err));
      this.dispatchEvent(new CustomEvent("webgpu-error", { detail: String(err), bubbles: true }));
      return;
    }

    const device = deviceManager.device!;

    this._context = this._canvas.getContext("webgpu") as GPUCanvasContext | null;
    if (!this._context) {
      const msg = "Не удалось получить контекст WebGPU холста.";
      this._showError(msg);
      this.dispatchEvent(new CustomEvent("webgpu-error", { detail: msg, bubbles: true }));
      return;
    }

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this._context.configure({ device, format: presentationFormat, alphaMode: "opaque" });

    const rect = this._canvas.getBoundingClientRect();
    this._onResize(Math.round(rect.width) || 300, Math.round(rect.height) || 150);
    // Защита: ResizeObserver мог сработать до готовности устройства, установив
    // _width/_height, но пропустив _recreateDepthTexture. Если _depthView всё ещё
    // равен null после _onResize (который прерывается при неизменившихся размерах),
    // принудительно создать текстуру глубины, пока устройство доступно.
    if (!this._depthView) {
      this._recreateDepthTexture(device, this._width || 300, this._height || 150);
    }

    this._registry = new ResourceRegistryImpl();
    this._pluginHost = new PluginHost();
    this._sceneGraph = new SceneGraph();

    await this._pluginHost.initialize(device, this._registry, presentationFormat);

    // Отслеживать потерю устройства, чтобы корректно остановить цикл рендеринга.
    this._deviceLost = false;
    void device.lost.then((info) => {
      if (this._deviceManager === deviceManager) { // всё ещё активное устройство
        this._deviceLost = true;
        this._renderLoop?.destroy();
        this._renderLoop = null;
        this.dispatchEvent(new CustomEvent("webgpu-lost", {
          detail: `${info.reason}: ${info.message}`,
          bubbles: true,
        }));
      }
    });

    const detail: WebGPUReadyDetail = {
      device,
      adapterInfo: deviceManager.adapterInfo,
      presentationFormat,
    };
    this.dispatchEvent(new CustomEvent<WebGPUReadyDetail>("webgpu-ready", { detail, bubbles: true }));

    const renderLoop = new RenderLoop();
    this._renderLoop = renderLoop;

    renderLoop.addCallback((deltaTime, frameIndex) => {
      if (!this._context || !this._depthView || this._deviceLost) return;

      const colorAttachment = this._context.getCurrentTexture().createView();
      const encoder = device.createCommandEncoder({ label: `frame-${frameIndex}` });

      // Всегда очищать холст до clearColor перед любыми проходами плагинов.
      const clearPass = encoder.beginRenderPass({
        label: "background-clear",
        colorAttachments: [{
          view: colorAttachment,
          loadOp: "clear",
          storeOp: "store",
          clearValue: this.clearColor,
        }],
        depthStencilAttachment: {
          view: this._depthView,
          depthLoadOp: "clear",
          depthStoreOp: "store",
          depthClearValue: 1.0,
        },
      });
      clearPass.end();

      const camera: CameraState = {
        ...(this._cameraController?.getCameraState() ?? {
          viewMatrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
          projectionMatrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
          viewProjectionMatrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]),
          position: new Float32Array(3),
          fovY: Math.PI / 4,
          near: 0.1,
          far: 1000,
        }),
        viewportWidth: this._width,
        viewportHeight: this._height,
      };

      const frameCtx = {
        device,
        encoder,
        camera,
        scene: this._sceneGraph!.getReadView(),
        frameIndex,
        deltaTime,
        colorAttachment,
        depthAttachment: this._depthView,
      };

      // Проходы плагинов используют loadOp:"load" для рендеринга поверх очищенного фона.
      device.pushErrorScope("validation");
      this._pluginHost!.runFrame(frameCtx);
      const finished = encoder.finish();
      void device.popErrorScope().then((e) => {
        if (e) console.error(`[WebGPUElement] кадр-${frameIndex} ошибка валидации:`, e);
      });
      device.queue.submit([finished]);
    });

    renderLoop.start();
  }

  disconnectedCallback(): void {
    this._resizeObserver.disconnect();
    this._renderLoop?.destroy();
    this._renderLoop = null;
    this._pluginHost?.destroy();
    this._pluginHost = null;
    this._depthTexture?.destroy();
    this._depthTexture = null;
    this._depthView = null;
    this._deviceManager?.destroy();
    this._deviceManager = null;
    this._context = null;
    this._registry = null;
    this._sceneGraph = null;
    this._deviceLost = false;
  }

  /**
   * Восстановиться после потери устройства: сбросить GPU-состояние и переинициализироваться.
   * Вызывать после получения события "webgpu-lost".
   * Событие "webgpu-ready" сработает повторно по завершении восстановления.
   */
  async recover(): Promise<void> {
    this._renderLoop?.destroy();
    this._renderLoop = null;
    this._pluginHost?.destroy();
    this._pluginHost = null;
    this._depthTexture?.destroy();
    this._depthTexture = null;
    this._depthView = null;
    this._deviceManager?.destroy();
    this._deviceManager = null;
    this._context = null;
    this._registry = null;
    this._sceneGraph = null;
    this._deviceLost = false;
    this._cameraController = null;
    await this.connectedCallback();
  }

  private _onResize(width: number, height: number): void {
    if (width === this._width && height === this._height) return;
    this._width = width;
    this._height = height;
    this._canvas.width = width;
    this._canvas.height = height;

    if (this._deviceManager?.device) {
      this._recreateDepthTexture(this._deviceManager.device, width, height);
    }

    this._pluginHost?.onResize(width, height);
  }

  private _recreateDepthTexture(device: GPUDevice, width: number, height: number): void {
    this._depthTexture?.destroy();
    device.pushErrorScope("out-of-memory");
    this._depthTexture = device.createTexture({
      label: "depth-texture",
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    void device.popErrorScope().then((err) => {
      if (err) console.error("[WebGPUElement] Переполнение памяти при создании текстуры глубины:", err);
    });
    this._depthView = this._depthTexture.createView();
  }

  private _showError(message: string): void {
    const div = document.createElement("div");
    div.style.cssText = [
      "position:absolute", "inset:0", "display:flex", "align-items:center",
      "justify-content:center", "background:#1a0000", "color:#ff6b6b",
      "font:14px/1.5 monospace", "padding:2em", "text-align:center",
    ].join(";");
    div.textContent = `WebGPU недоступен: ${message}`;
    this._shadow.appendChild(div);
  }
}

customElements.define("webgpu-canvas", WebGPUElement);
