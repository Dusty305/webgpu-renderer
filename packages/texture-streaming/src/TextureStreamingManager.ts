import type {
  IResourceManager,
  ResourceManagerInitContext,
  FrameContext,
} from "@webgpu-streaming/gpu-types";
import { BudgetTracker } from "./BudgetTracker.js";
import { LRUEvictionPolicy } from "./LRUEvictionPolicy.js";
import { MipPriorityQueue } from "./MipPriorityQueue.js";
import { StagingRingBuffer } from "./StagingRingBuffer.js";
import { TierAllocator, TIER_SIZES } from "./TierAllocator.js";
import type { TierIndex } from "./TierAllocator.js";
import { SamplerManager } from "./SamplerManager.js";
import { BindGroupManager } from "./BindGroupManager.js";
import { FormatRouter } from "./FormatRouter.js";
import { MaterialBufferWriter } from "./MaterialBufferWriter.js";
import { computeDesiredMip, computeScreenCoverage } from "./mip-math.js";
import type { KTX2ParseResult } from "./KTX2Parser.js";
import { TranscodePipeline } from "./TranscodePipeline.js";

/** Запись зарегистрированной текстуры, отслеживаемой менеджером стриминга. */
export interface TextureEntry {
  id: string;
  parsed: KTX2ParseResult;
  ktx2Bytes: ArrayBuffer;
  tier: TierIndex;
  layer: number;
  materialId: number;
  /** Наиболее детальный уровень мипа, в данный момент резидентный на GPU. */
  residentMip: number;
  /** Ограничивающая сфера объекта для расчёта приоритета: [cx, cy, cz, radius]. */
  boundingSphere: Float32Array;
}

export interface StreamingManagerConfig {
  /** Общий бюджет памяти GPU для текстур в байтах. По умолчанию 256 МБ. */
  budgetBytes?: number;
  /** Максимум байт для загрузки за кадр. По умолчанию 8 МБ. */
  frameUploadBudget?: number;
  /**
   * Максимальное число слоёв на тир [тир0, тир1, тир2]. По умолчанию [32, 16, 4].
   * Настраивается под ожидаемое число текстур и доступный объём VRAM.
   * Расход памяти: тир0 ≈ 44 МБ, тир1 ≈ 90 МБ, тир2 ≈ 90 МБ при значениях по умолчанию.
   */
  maxLayersPerTier?: [number, number, number];
  /**
   * Если true, всегда целиться в мип 0 (полное разрешение) для каждой текстуры,
   * игнорируя расстояние до камеры. Используется для бенчмарков Config A (наивный / без стриминга),
   * цель которых - измерить пиковое потребление GPU-памяти при максимальном качестве.
   * По умолчанию false.
   */
  forceFullQuality?: boolean;
}

/**
 * Центральный IResourceManager, оркестрирующий все подсистемы стриминга текстур.
 *
 * Порядок инициализации:
 *   FormatRouter → TierAllocator → StagingRingBuffer → BudgetTracker →
 *   SamplerManager → BindGroupManager → TranscodePipeline → MaterialBufferWriter
 *
 * Порядок выполнения prepareFrame():
 *   1. Обновить временны́е метки LRU
 *   2. Вычислить приоритеты мипов по камере
 *   3. Отобрать загрузки в пределах бюджета кадра
 *   4. Вытеснить при превышении бюджета
 *   5. Загрузить через кольцевой буфер стейджинга
 *   6. Обновить сэмплеры и буфер материалов
 *   7. Перестроить группу привязок, если она помечена как устаревшая
 */
export class TextureStreamingManager implements IResourceManager {
  readonly name = "texture-streaming-manager";

  private _device: GPUDevice | null = null;
  private _formatRouter: FormatRouter | null = null;
  private _tierAllocator: TierAllocator | null = null;
  private _stagingRing: StagingRingBuffer | null = null;
  private _budgetTracker: BudgetTracker | null = null;
  private _samplerManager: SamplerManager | null = null;
  private _bindGroupManager: BindGroupManager | null = null;
  private _materialWriter: MaterialBufferWriter | null = null;
  private _lruPolicy: LRUEvictionPolicy | null = null;
  private _priorityQueue: MipPriorityQueue | null = null;
  private _transcode: TranscodePipeline | null = null;

  private readonly _entries = new Map<string, TextureEntry>();
  private readonly _budgetBytes: number;
  private _frameUploadBudget: number;
  private readonly _maxLayersPerTier: [number, number, number];
  private readonly _forceFullQuality: boolean;
  private _bindGroupDirty = false;

  // Наблюдаемая статистика.
  totalUploaded = 0;
  evictionsLastFrame = 0;
  uploadsLastFrame = 0;

  constructor(config: StreamingManagerConfig = {}) {
    this._budgetBytes       = config.budgetBytes       ?? 256 * 1024 * 1024;
    this._frameUploadBudget = config.frameUploadBudget ??   8 * 1024 * 1024;
    this._maxLayersPerTier  = config.maxLayersPerTier  ?? [32, 16, 4];
    this._forceFullQuality  = config.forceFullQuality  ?? false;
  }

  get budgetTracker(): BudgetTracker | null {
    return this._budgetTracker;
  }

  get entries(): ReadonlyMap<string, TextureEntry> {
    return this._entries;
  }

  get frameUploadBudget(): number {
    return this._frameUploadBudget;
  }

  set frameUploadBudget(value: number) {
    this._frameUploadBudget = value;
  }

  /** Общий макет группы привязок для группы 0, доступен после initialize(). */
  get bindGroupLayout(): GPUBindGroupLayout | null {
    return this._bindGroupManager?.layout ?? null;
  }

  async initialize(ctx: ResourceManagerInitContext): Promise<void> {
    this._device = ctx.device;

    this._formatRouter    = new FormatRouter(ctx.device.features);
    // Используем rgba8unorm для массивов тиров - _uploadMipSync всегда записывает сырые данные RGBA8
    // и не выполняет транскодирование. Сжатые форматы FormatRouter (BC7, ASTC)
    // зарезервированы для будущего использования при подключении TranscodePipeline.
    const colorFormat     = "rgba8unorm" as GPUTextureFormat;

    this._tierAllocator   = new TierAllocator(ctx.device, colorFormat, this._maxLayersPerTier);
    // Размер слота должен вмещать наибольшую одиночную загрузку мипа (2048² RGBA8 = 16 МБ).
    // frameUploadBudget - логический лимит на кадр, а не размер физического буфера;
    // ограничение 64 МБ предотвращает запрос многогигабайтных буферов при «неограниченном» бюджете.
    const largestMip0Bytes = Math.max(...TIER_SIZES.map(s => s * s * 4));
    const stagingSlotBytes = Math.max(largestMip0Bytes, Math.min(this._frameUploadBudget, 64 * 1024 * 1024));
    this._stagingRing     = new StagingRingBuffer(ctx.device, 4, stagingSlotBytes);
    this._budgetTracker   = new BudgetTracker(this._budgetBytes);
    this._samplerManager  = new SamplerManager(ctx.device);
    this._bindGroupManager = new BindGroupManager(ctx.device, ctx.registry);
    this._materialWriter  = new MaterialBufferWriter(ctx.device);
    this._lruPolicy       = new LRUEvictionPolicy();
    this._priorityQueue   = new MipPriorityQueue();
    this._transcode       = new TranscodePipeline(2);

    this._bindGroupManager.createLayout();

    // Создаём начальную (пустую) группу привязок, чтобы проходы рендеринга не получали null.
    this._rebuildBindGroup();

    console.info("[TextureStreamingManager] Инициализирован. Цветовой формат:", colorFormat);
  }

  /**
   * Зарегистрировать текстуру KTX2 для стриминга. Назначает тир и слой.
   * Должен вызываться после initialize().
   *
   * @param id - Уникальный идентификатор текстуры.
   * @param parsed - Разобранный заголовок KTX2.
   * @param ktx2Bytes - Сырые байты файла KTX2.
   * @param materialId - Индекс в буфере материалов.
   * @param boundingSphere - Ограничивающая сфера в мировом пространстве [cx, cy, cz, radius].
   */
  registerTexture(
    id: string,
    parsed: KTX2ParseResult,
    ktx2Bytes: ArrayBuffer,
    materialId: number,
    boundingSphere: Float32Array
  ): void {
    if (!this._tierAllocator || !this._lruPolicy || !this._materialWriter || !this._samplerManager) {
      throw new Error("[TextureStreamingManager] registerTexture вызван до initialize()");
    }

    // Выбрать тир по размеру текстуры.
    const size = Math.max(parsed.pixelWidth, parsed.pixelHeight);
    const tier: TierIndex = size <= 512 ? 0 : size <= 1024 ? 1 : 2;
    const layer = this._tierAllocator.allocateLayer(tier);
    if (layer === -1) {
      console.warn(`[TextureStreamingManager] Тир ${tier} заполнен, невозможно зарегистрировать текстуру "${id}"`);
      return;
    }

    const coarsestMip = parsed.levelCount - 1;
    const entry: TextureEntry = {
      id, parsed, ktx2Bytes, tier, layer, materialId,
      residentMip: coarsestMip,
      boundingSphere: new Float32Array(boundingSphere),
    };
    this._entries.set(id, entry);

    this._lruPolicy.registerTexture(id, parsed.levelCount);
    // Регистрируем в менеджере сэмплеров на самом грубом мипе (консервативный нижний предел).
    this._samplerManager.setTextureMip(tier, id, coarsestMip);

    const tierLodMinClamp = this._samplerManager.getTierLodMinClamp(tier);

    // Записываем начальную запись MaterialEntry (самый грубый мип).
    this._materialWriter.write(materialId, {
      tierIndex: tier, layerIndex: layer, residentMip: coarsestMip, tierLodMinClamp,
    });

    // Немедленно загружаем 2 самых грубых уровня мипа (гарантия постоянного резидентства).
    // Используем writeTexture напрямую, так как энкодер при регистрации недоступен.
    const uploadStart = Math.max(0, coarsestMip - 1);
    for (let level = uploadStart; level <= coarsestMip; level++) {
      this._uploadMipImmediate(entry, level);
    }

    this._bindGroupDirty = true;
  }

  /**
   * Отменить регистрацию текстуры и освободить её слой тира.
   * Все отслеживаемые данные мипов удаляются из бюджета и политики LRU.
   */
  unregisterTexture(textureId: string): void {
    const entry = this._entries.get(textureId);
    if (!entry) return;
    if (this._tierAllocator) {
      this._tierAllocator.freeLayer(entry.tier, entry.layer);
    }
    if (this._budgetTracker && this._lruPolicy) {
      for (let level = entry.residentMip; level < entry.parsed.levelCount; level++) {
        this._budgetTracker.recordEviction(entry.id, level);
      }
      this._lruPolicy.removeTexture(entry.id);
    }
    this._entries.delete(textureId);
    this._bindGroupDirty = true;
  }

  prepareFrame(ctx: FrameContext): void {
    if (!this._lruPolicy || !this._priorityQueue || !this._budgetTracker ||
        !this._materialWriter || !this._transcode || !this._samplerManager) return;

    const frameIndex = ctx.frameIndex;
    this.evictionsLastFrame = 0;
    this.uploadsLastFrame   = 0;
    this._priorityQueue.clear();

    // ---- 1. Вычислить приоритеты + избирательно обновить LRU ----------------------------------
    for (const entry of this._entries.values()) {
      const dist    = this._distanceToCamera(entry.boundingSphere, ctx.camera.position);
      const desired = this._forceFullQuality
        ? 0
        : computeDesiredMip(dist, entry.parsed.pixelWidth, ctx.camera.viewportHeight, ctx.camera.fovY);
      const gap     = entry.residentMip - desired; // положительное = нужен более детальный мип
      if (gap > 0) {
        // Обновляем только текстуры, которые камера видит в данный момент -
        // необновлённые текстуры становятся кандидатами LRU при смещении камеры.
        this._lruPolicy.touch(entry.id, entry.residentMip, frameIndex);
        const coverage = computeScreenCoverage(
          entry.boundingSphere,
          ctx.camera.viewProjectionMatrix,
          ctx.camera.viewportWidth,
          ctx.camera.viewportHeight,
        );
        this._priorityQueue.push({
          textureId: entry.id,
          mipLevel: entry.residentMip - 1,
          priority: gap * coverage,
        });
      }
    }

    // ---- 2. Загрузить мипы с наивысшим приоритетом, вытесняя для освобождения места ----
    this._stagingRing!.beginFrame();
    let frameBytes = 0;
    while (this._priorityQueue.size > 0) {
      const req = this._priorityQueue.pop();
      if (!req) break;

      const entry = this._entries.get(req.textureId);
      if (!entry || entry.residentMip <= req.mipLevel) continue;

      const mipW = Math.max(1, entry.parsed.pixelWidth  >> req.mipLevel);
      const mipH = Math.max(1, entry.parsed.pixelHeight >> req.mipLevel);
      const bytes = mipW * mipH * 4;
      if (frameBytes + bytes > this._frameUploadBudget) continue;

      // Вытеснять мипы с низким приоритетом для освобождения места при заполненном бюджете.
      if (!this._budgetTracker.canUpload(bytes)) {
        const toEvict = this._lruPolicy.selectEvictions(bytes);
        let freed = 0;
        for (const candidate of toEvict) {
          // Никогда не вытесняем мип, который собираемся загружать.
          if (candidate.textureId === req.textureId) continue;
          const victim = this._entries.get(candidate.textureId);
          if (!victim) continue;
          this._budgetTracker.recordEviction(candidate.textureId, candidate.mipLevel);
          this._lruPolicy.forget(candidate.textureId, candidate.mipLevel);
          freed += candidate.bytes;
          if (candidate.mipLevel <= victim.residentMip) {
            victim.residentMip = candidate.mipLevel + 1;
            this._samplerManager.setTextureMip(victim.tier, victim.id, victim.residentMip);
            this._bindGroupDirty = true;
          }
          this.evictionsLastFrame++;
        }
        if (freed < bytes || !this._budgetTracker.canUpload(bytes)) continue;
      }

      this._uploadMipStaged(ctx.encoder, entry, req.mipLevel);
      frameBytes += bytes;
      this.uploadsLastFrame++;
    }
    this._stagingRing!.endFrame();

    // ---- 4. Обновить буфер материалов с текущими residentMip + tierLodMinClamp --
    for (const entry of this._entries.values()) {
      const tierLodMinClamp = this._samplerManager.getTierLodMinClamp(entry.tier);
      this._materialWriter.write(entry.materialId, {
        tierIndex: entry.tier, layerIndex: entry.layer,
        residentMip: entry.residentMip, tierLodMinClamp,
      });
    }
    this._materialWriter.flush();

    // ---- 5. Перестроить группу привязок, если она помечена как устаревшая ----------------
    if (this._bindGroupDirty) {
      this._rebuildBindGroup();
      this._bindGroupDirty = false;
    }

    this.totalUploaded = this._budgetTracker.totalUsed;
  }

  /**
   * Загрузить уровень мипа через кольцевой буфер стейджинга.
   * Записывает команду copyBufferToTexture в предоставленный энкодер.
   * Вызывается из prepareFrame(), где доступен энкодер.
   */
  private _uploadMipStaged(encoder: GPUCommandEncoder, entry: TextureEntry, level: number): void {
    if (!this._device || !this._tierAllocator || !this._budgetTracker ||
        !this._lruPolicy || !this._samplerManager || !this._stagingRing) return;

    const mipW = Math.max(1, entry.parsed.pixelWidth  >> level);
    const mipH = Math.max(1, entry.parsed.pixelHeight >> level);
    const levelInfo = entry.parsed.levels[level];
    if (!levelInfo) return;

    const rawData = new Uint8Array(
      entry.ktx2Bytes, levelInfo.byteOffset, levelInfo.byteLength
    );
    const tex = this._tierAllocator.getTexture(entry.tier);

    const staged = this._stagingRing.stageUpload(
      encoder, rawData, tex, level, entry.layer, mipW, mipH
    );

    if (!staged) {
      // Слот стейджинга заполнен или недоступен - откат к writeTexture.
      const bytesPerRow = Math.max(256, Math.ceil(mipW * 4 / 256) * 256);
      const padded = new Uint8Array(bytesPerRow * mipH);
      for (let row = 0; row < mipH; row++) {
        padded.set(rawData.subarray(row * mipW * 4, (row + 1) * mipW * 4), row * bytesPerRow);
      }
      this._device.queue.writeTexture(
        { texture: tex, mipLevel: level, origin: { x: 0, y: 0, z: entry.layer } },
        padded, { bytesPerRow, rowsPerImage: mipH }, [mipW, mipH, 1]
      );
    }

    this._recordMipResident(entry, level);
  }

  /**
   * Загрузить уровень мипа через writeTexture напрямую.
   * Используется только во время registerTexture(), когда энкодер недоступен.
   */
  private _uploadMipImmediate(entry: TextureEntry, level: number): void {
    if (!this._device || !this._tierAllocator || !this._budgetTracker ||
        !this._lruPolicy || !this._samplerManager) return;

    const mipW = Math.max(1, entry.parsed.pixelWidth  >> level);
    const mipH = Math.max(1, entry.parsed.pixelHeight >> level);
    const levelInfo = entry.parsed.levels[level];
    if (!levelInfo) return;

    const rawData = new Uint8Array(entry.ktx2Bytes, levelInfo.byteOffset, levelInfo.byteLength);
    const bytesPerRow = Math.max(256, Math.ceil(mipW * 4 / 256) * 256);
    const padded = new Uint8Array(bytesPerRow * mipH);
    for (let row = 0; row < mipH; row++) {
      padded.set(rawData.subarray(row * mipW * 4, (row + 1) * mipW * 4), row * bytesPerRow);
    }

    const tex = this._tierAllocator.getTexture(entry.tier);
    this._device.queue.writeTexture(
      { texture: tex, mipLevel: level, origin: { x: 0, y: 0, z: entry.layer } },
      padded, { bytesPerRow, rowsPerImage: mipH }, [mipW, mipH, 1]
    );

    this._recordMipResident(entry, level);
  }

  /** Обновить состояние отслеживания после любой загрузки мипа. */
  private _recordMipResident(entry: TextureEntry, level: number): void {
    const bytes = Math.max(1, entry.parsed.pixelWidth >> level) *
                  Math.max(1, entry.parsed.pixelHeight >> level) * 4;
    this._budgetTracker!.recordUpload(entry.id, level, bytes);
    this._lruPolicy!.recordSize(entry.id, level, bytes);

    const prevMip = entry.residentMip;
    entry.residentMip = Math.min(entry.residentMip, level);
    if (entry.residentMip !== prevMip) {
      this._samplerManager!.setTextureMip(entry.tier, entry.id, entry.residentMip);
      this._bindGroupDirty = true;
    }
  }

  private _rebuildBindGroup(): void {
    if (!this._tierAllocator || !this._samplerManager || !this._materialWriter || !this._bindGroupManager) return;

    const samplers: [GPUSampler, GPUSampler, GPUSampler] = [
      this._samplerManager.getSampler(0),
      this._samplerManager.getSampler(1),
      this._samplerManager.getSampler(2),
    ];
    const views: [GPUTextureView, GPUTextureView, GPUTextureView] = [
      this._tierAllocator.getView(0),
      this._tierAllocator.getView(1),
      this._tierAllocator.getView(2),
    ];
    this._bindGroupManager.rebuild(views, samplers, this._materialWriter.buffer);
  }

  private _distanceToCamera(boundingSphere: Float32Array, cameraPos: Float32Array): number {
    const dx = (boundingSphere[0] ?? 0) - (cameraPos[0] ?? 0);
    const dy = (boundingSphere[1] ?? 0) - (cameraPos[1] ?? 0);
    const dz = (boundingSphere[2] ?? 0) - (cameraPos[2] ?? 0);
    return Math.max(0, Math.sqrt(dx*dx + dy*dy + dz*dz) - (boundingSphere[3] ?? 0));
  }

  destroy(): void {
    this._materialWriter?.destroy();
    this._bindGroupManager?.destroy();
    this._stagingRing?.destroy();
    this._tierAllocator?.destroy();
    this._lruPolicy?.destroy();
    this._budgetTracker?.destroy();
    this._transcode?.destroy();
    this._samplerManager?.destroy();
    this._priorityQueue?.clear();
    this._entries.clear();
    this._device = null;
  }
}
