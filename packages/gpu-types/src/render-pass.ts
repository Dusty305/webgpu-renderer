import type { FrameContext } from "./frame-context.js";
import type { ResourceRegistry } from "./resource-registry.js";

/**
 * Контекст, предоставляемый IRenderPass при инициализации.
 */
export interface RenderPassInitContext {
  readonly device: GPUDevice;
  readonly registry: ResourceRegistry;
  /** Формат текстуры поверхности цепочки обмена */
  readonly presentationFormat: GPUTextureFormat;
}

/**
 * Плагин рендер-прохода. Рендер-проходы отправляют команды отрисовки, используя ресурсы,
 * подготовленные IResourceManager ранее в том же кадре.
 */
export interface IRenderPass {
  /** Уникальное имя для отладки и поиска в реестре */
  readonly name: string;

  /**
   * Вызывается один раз после того, как GPU-устройство становится доступным.
   * Здесь следует создавать конвейеры, макеты групп привязок и статические буферы.
   */
  initialize(ctx: RenderPassInitContext): Promise<void>;

  /**
   * Вызывается каждый кадр, после того как все IResourceManager выполнили prepareFrame.
   * Записывает команды рендер-прохода в ctx.encoder.
   */
  execute(ctx: FrameContext): void;

  /**
   * Вызывается при изменении размера холста.
   */
  onResize(width: number, height: number): void;

  /**
   * Освобождает все GPU-ресурсы. Вызывается в disconnectedCallback.
   */
  destroy(): void;
}
