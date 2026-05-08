/** Размеры тиров разрешения в текселях (предполагается ширина == высота). */
export const TIER_SIZES = [512, 1024, 2048] as const;
export type TierIndex = 0 | 1 | 2;

/** Максимальное число слоёв на тир по умолчанию [тир0=512, тир1=1024, тир2=2048].
 *  Расход памяти при значениях по умолчанию: тир0 ≈ 44 МБ, тир1 ≈ 90 МБ, тир2 ≈ 90 МБ.
 */
const DEFAULT_MAX_LAYERS: [number, number, number] = [32, 16, 4];

/**
 * Управляет одним texture_2d_array на каждый тир разрешения.
 * Обеспечивает аллокацию слоёв внутри каждого тира через список свободных.
 */
export class TierAllocator {
  private readonly _textures: GPUTexture[] = [];
  private readonly _views: GPUTextureView[] = [];
  private readonly _freeLists: number[][] = [];
  private readonly _maxLayersPerTier: [number, number, number];

  constructor(
    private readonly _device: GPUDevice,
    private readonly _format: GPUTextureFormat,
    maxLayersPerTier: [number, number, number] = DEFAULT_MAX_LAYERS
  ) {
    this._maxLayersPerTier = maxLayersPerTier;

    for (const [tierIdx, size] of TIER_SIZES.entries()) {
      const layers = maxLayersPerTier[tierIdx as TierIndex];
      _device.pushErrorScope("out-of-memory");
      const texture = _device.createTexture({
        label: `tier-array-${size}`,
        size: [size, size, layers],
        mipLevelCount: Math.floor(Math.log2(size)) + 1,
        format: _format,
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST,
        dimension: "2d",
      });
      void _device.popErrorScope().then((err) => {
        if (err) console.error(`[TierAllocator] Нехватка памяти при создании тира ${tierIdx} (${size}px, ${layers} слоёв):`, err);
      });

      this._textures.push(texture);
      this._views.push(texture.createView({ dimension: "2d-array" }));

      // Заполняем список свободных всеми слоями.
      const freeList: number[] = [];
      for (let i = layers - 1; i >= 0; i--) {
        freeList.push(i);
      }
      this._freeLists.push(freeList);
    }
  }

  /**
   * Выделить слой в заданном тире.
   * @returns Индекс слоя или -1, если тир заполнен.
   */
  allocateLayer(tier: TierIndex): number {
    return this._freeLists[tier]!.pop() ?? -1;
  }

  /**
   * Вернуть слой в список свободных тира.
   */
  freeLayer(tier: TierIndex, layer: number): void {
    this._freeLists[tier]!.push(layer);
  }

  /** Количество слоёв, используемых в данный момент в тире. */
  usedLayers(tier: TierIndex): number {
    return this._maxLayersPerTier[tier] - this._freeLists[tier]!.length;
  }

  /** Получить GPUTextureView для тира (вид 2d-array). */
  getView(tier: TierIndex): GPUTextureView {
    return this._views[tier]!;
  }

  /** Получить сырой GPUTexture для тира. */
  getTexture(tier: TierIndex): GPUTexture {
    return this._textures[tier]!;
  }

  destroy(): void {
    for (const tex of this._textures) {
      tex.destroy();
    }
    this._textures.length = 0;
    this._views.length = 0;
    this._freeLists.length = 0;
  }
}
