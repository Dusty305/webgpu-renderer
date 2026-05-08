import type {
  IResourceManager,
  IRenderPass,
  FrameContext,
  ResourceManagerInitContext,
  RenderPassInitContext,
} from "@webgpu-streaming/gpu-types";
import type { ResourceRegistryImpl } from "./ResourceRegistryImpl.js";

/**
 * Оркестрирует IResourceManager и IRenderPass с правильным порядком жизненного цикла.
 *
 * Порядок кадра:
 *   1. manager.prepareFrame() для всех менеджеров (в порядке регистрации)
 *   2. pass.execute() для всех проходов (в порядке регистрации)
 *
 * Порядок уничтожения - обратный порядку регистрации.
 */
export class PluginHost {
  private readonly _managers: IResourceManager[] = [];
  private readonly _passes: IRenderPass[] = [];

  /**
   * Зарегистрировать IResourceManager. Менеджеры инициализируются немедленно,
   * если хост уже инициализирован; иначе - при следующем вызове initialize().
   */
  registerResourceManager(manager: IResourceManager): void {
    this._managers.push(manager);
  }

  /**
   * Зарегистрировать IRenderPass.
   */
  registerRenderPass(pass: IRenderPass): void {
    this._passes.push(pass);
  }

  /**
   * Инициализировать все зарегистрированные менеджеры, затем все проходы, в порядке регистрации.
   */
  async initialize(
    device: GPUDevice,
    registry: ResourceRegistryImpl,
    presentationFormat: GPUTextureFormat
  ): Promise<void> {
    const managerCtx: ResourceManagerInitContext = { device, registry };
    for (const manager of this._managers) {
      await manager.initialize(managerCtx);
    }

    const passCtx: RenderPassInitContext = { device, registry, presentationFormat };
    for (const pass of this._passes) {
      await pass.initialize(passCtx);
    }
  }

  /**
   * Вызывается каждый кадр. Выполняет все prepareFrame, затем все execute.
   */
  runFrame(ctx: FrameContext): void {
    for (const manager of this._managers) {
      manager.prepareFrame(ctx);
    }
    for (const pass of this._passes) {
      pass.execute(ctx);
    }
  }

  /**
   * Уведомить проходы о событии изменения размера.
   */
  onResize(width: number, height: number): void {
    for (const pass of this._passes) {
      pass.onResize(width, height);
    }
  }

  /**
   * Уничтожить все плагины в обратном порядке регистрации.
   */
  destroy(): void {
    for (const pass of [...this._passes].reverse()) {
      pass.destroy();
    }
    for (const manager of [...this._managers].reverse()) {
      manager.destroy();
    }
    this._passes.length = 0;
    this._managers.length = 0;
  }
}
