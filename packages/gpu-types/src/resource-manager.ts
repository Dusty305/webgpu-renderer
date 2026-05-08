import type { FrameContext } from "./frame-context.js";
import type { ResourceRegistry } from "./resource-registry.js";

/**
 * Контекст, предоставляемый IResourceManager при инициализации.
 */
export interface ResourceManagerInitContext {
  readonly device: GPUDevice;
  readonly registry: ResourceRegistry;
}

/**
 * Плагин менеджера ресурсов. Менеджеры ресурсов владеют GPU-ресурсами (текстурами,
 * буферами, семплерами) и предоставляют их через ResourceRegistry для использования
 * рендер-проходами.
 */
export interface IResourceManager {
  /** Уникальное имя для отладки */
  readonly name: string;

  /**
   * Вызывается один раз после того, как GPU-устройство становится доступным.
   * Выделяет GPU-ресурсы и регистрирует их в реестре.
   */
  initialize(ctx: ResourceManagerInitContext): Promise<void>;

  /**
   * Вызывается каждый кадр до любого IRenderPass.execute.
   * Обновляет ресурсы, записывает команды загрузки в ctx.encoder.
   */
  prepareFrame(ctx: FrameContext): void;

  /**
   * Вызывается при изменении сцены (добавление/удаление объектов).
   * Реализации должны выделять или освобождать ресурсы по необходимости.
   */
  onSceneChange?(ctx: ResourceManagerInitContext): void;

  /**
   * Освобождает все GPU-ресурсы. Вызывается в disconnectedCallback.
   */
  destroy(): void;
}
