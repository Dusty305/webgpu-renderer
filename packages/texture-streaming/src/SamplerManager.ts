import type { TierIndex } from "./TierAllocator.js";

/**
 * Управляет одним GPUSampler на каждый тир разрешения.
 *
 * Сэмплер всегда имеет lodMinClamp=0. Защита от обращения к нулинициализированной
 * памяти мипмапов полностью реализована в шейдере через
 * textureSampleLevel с явным LOD = max(approxLod, residentMip).
 * Установка lodMinClamp в сэмплере мешала бы textureSampleLevel,
 * поскольку ограничения GPUSampler применяются ко всем операциям семплирования, включая
 * явные LOD - это не позволило бы обращаться к детальным мипмапам, даже когда
 * они уже загружены.
 *
 * Отслеживание текстур по тирам сохраняется, чтобы getTierLodMinClamp() мог сообщать
 * концептуальный максимум для тира для буфера материалов и отладочных оверлеев.
 */
export class SamplerManager {
  private readonly _samplers: (GPUSampler | null)[] = [null, null, null];
  /** Отображение textureId → текущий residentMip по тирам (для getTierLodMinClamp) */
  private readonly _tierMips: [Map<string, number>, Map<string, number>, Map<string, number>] = [
    new Map(), new Map(), new Map(),
  ];
  private readonly _tierMaxMip: [number, number, number] = [0, 0, 0];

  constructor(private readonly _device: GPUDevice) {}

  /**
   * Зарегистрировать или обновить residentMip текстуры в её тире.
   * Обновляет концептуальный максимум для тира (используется getTierLodMinClamp).
   */
  setTextureMip(tier: TierIndex, textureId: string, residentMip: number): void {
    this._tierMips[tier].set(textureId, residentMip);
    this._recomputeMax(tier);
  }

  /** Удалить текстуру из отслеживания тира. */
  removeTexture(tier: TierIndex, textureId: string): void {
    this._tierMips[tier].delete(textureId);
    this._recomputeMax(tier);
  }

  /** Получить (или лениво создать) сэмплер для тира. lodMinClamp всегда равен 0. */
  getSampler(tier: TierIndex): GPUSampler {
    if (this._samplers[tier] === null) {
      this._samplers[tier] = this._device.createSampler({
        label: `tier-${tier}-sampler`,
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
        // Использовать режим повторения - в glTF по умолчанию GL_REPEAT (10497) и многие
        // сетки (например, DamagedHelmet) имеют UV-координаты вне диапазона [0,1].
        addressModeU: "repeat",
        addressModeV: "repeat",
        lodMinClamp: 0,   // должен быть 0 - явный LOD в шейдере выполняет ограничение
        lodMaxClamp: 1000,
      });
    }
    return this._samplers[tier]!;
  }

  /**
   * Концептуальный lodMinClamp для тира = max(residentMip) по всем текстурам
   * в этом тире. Хранится в MaterialEntry для отображения в оверлее и документации диссертации.
   * НЕ управляет реальным GPUSampler.
   */
  getTierLodMinClamp(tier: TierIndex): number {
    return this._tierMaxMip[tier];
  }

  private _recomputeMax(tier: TierIndex): void {
    let max = 0;
    for (const mip of this._tierMips[tier].values()) {
      if (mip > max) max = mip;
    }
    this._tierMaxMip[tier] = max;
  }

  destroy(): void {
    // У GPUSampler нет метода .destroy() - сборщик мусора обрабатывает их.
    this._samplers.fill(null);
    for (const m of this._tierMips) m.clear();
  }
}
