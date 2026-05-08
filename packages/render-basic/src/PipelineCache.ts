/**
 * Простой кэш GPURenderPipeline с ключом в виде строкового дескриптора.
 * Предотвращает избыточное создание пайплайнов для идентичных конфигураций.
 */
export class PipelineCache {
  private readonly _cache = new Map<string, GPURenderPipeline>();

  /** Получить кэшированный пайплайн или создать и закэшировать новый. */
  getOrCreate(
    key: string,
    device: GPUDevice,
    descriptor: GPURenderPipelineDescriptor
  ): GPURenderPipeline {
    const cached = this._cache.get(key);
    if (cached) return cached;

    const pipeline = device.createRenderPipeline(descriptor);
    this._cache.set(key, pipeline);
    return pipeline;
  }

  /** Удалить все кэшированные пайплайны. */
  clear(): void {
    this._cache.clear();
  }

  destroy(): void {
    this.clear();
  }
}
