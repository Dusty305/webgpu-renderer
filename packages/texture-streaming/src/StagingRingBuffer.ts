/** Слот, полученный из кольцевого буфера стейджинга для записи. */
export interface StagingSlot {
  /** Базовый GPUBuffer (MAP_WRITE | COPY_SRC). */
  buffer: GPUBuffer;
  /** Отображённый ArrayBuffer - записывайте данные текстуры сюда перед вызовом submit(). */
  arrayBuffer: ArrayBuffer;
  /** Всегда 0 при полном отображении буфера (смещение внутри отображённого диапазона). */
  offset: number;
}

type BufferState = "mapped" | "pending" | "in-flight";

/**
 * Кольцевой буфер из N отображённых GPUBuffer для стейджинга загрузки мипов текстур.
 *
 * ### Паттерн множественных загрузок за кадр (фаза 7.4+)
 *
 * Каждый кадр упаковывает все загрузки мипов в текущий слот через суб-аллокацию:
 *
 *   ring.beginFrame()
 *   for each mip:
 *     ring.stageUpload(encoder, data, texture, mipLevel, layerIndex, mipW, mipH)
 *   ring.endFrame()   // разотображение, продвижение кольца
 *   // затем: device.queue.submit([encoder.finish()])
 *
 * Размер слота должен быть >= суммарного объёма загрузок за кадр (по умолчанию 16 МБ
 * с запасом для бюджета 8 МБ/кадр).
 *
 * ### Устаревший паттерн одиночной загрузки (сохранён для демо)
 *
 *   slot = ring.acquire()
 *   запись данных в slot.arrayBuffer
 *   ring.recordCopy(encoder, texture, mipLevel, size, bytesPerRow, dataByteLength)
 *   ring.submit()
 */
export class StagingRingBuffer {
  private readonly _buffers: GPUBuffer[];
  private readonly _arrayBuffers: (ArrayBuffer | null)[];
  private readonly _states: BufferState[];
  private _current = 0;
  private readonly _count: number;
  readonly slotSizeBytes: number;

  /** Смещение записи внутри текущего слота для суб-аллокации. */
  private _writeOffset = 0;

  constructor(
    private readonly _device: GPUDevice,
    slotCount: number = 4,
    slotSizeBytes: number = 16 * 1024 * 1024
  ) {
    this._count = slotCount;
    this.slotSizeBytes = slotSizeBytes;
    this._buffers = [];
    this._arrayBuffers = [];
    this._states = [];

    for (let i = 0; i < slotCount; i++) {
      _device.pushErrorScope("out-of-memory");
      const buf = _device.createBuffer({
        label: `staging-ring-${i}`,
        size: slotSizeBytes,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      void _device.popErrorScope().then((err) => {
        if (err) console.error(`[StagingRingBuffer] Нехватка памяти для слота ${i}:`, err);
      });
      this._buffers.push(buf);
      this._arrayBuffers.push(buf.getMappedRange());
      this._states.push("mapped");
    }
  }

  get currentIndex(): number {
    return this._current;
  }

  /** Состояния слотов для отладочных оверлеев. */
  getStates(): readonly BufferState[] {
    return this._states;
  }

  // ---- API множественных загрузок за кадр ----------------------------------------------------------------------------

  /**
   * Начать пакет загрузок нового кадра. Гарантирует, что текущий слот отображён.
   * Сбрасывает смещение записи суб-аллокации.
   * Выбрасывает исключение, если текущий слот ещё выполняется на GPU.
   */
  beginFrame(): void {
    const state = this._states[this._current];
    if (state !== "mapped" && import.meta.env.DEV) {
      console.warn(`[StagingRingBuffer] Слот ${this._current} в состоянии ${state} - загрузки стейджинга в этом кадре пропущены.`);
    }
    this._writeOffset = 0;
  }

  /**
   * Поставить в стейджинг один уровень мипа в текущий слот через суб-аллокацию.
   * Записывает команду copyBufferToTexture в предоставленный энкодер.
   *
   * @param encoder    GPUCommandEncoder текущего кадра.
   * @param data       Сырые данные пикселей (должны поместиться в слот вместе с предыдущими загрузками).
   * @param texture    Целевой массив текстур.
   * @param mipLevel   Целевой уровень мипа.
   * @param layerIndex Индекс слоя массива (z-начало).
   * @param mipW       Ширина мипа в текселях.
   * @param mipH       Высота мипа в текселях.
   * @returns true при успешном стейджинге, false если слот недоступен или заполнен.
   */
  stageUpload(
    encoder: GPUCommandEncoder,
    data: Uint8Array,
    texture: GPUTexture,
    mipLevel: number,
    layerIndex: number,
    mipW: number,
    mipH: number
  ): boolean {
    if (this._states[this._current] !== "mapped") return false;
    const ab = this._arrayBuffers[this._current];
    if (!ab) return false;

    const bytesPerRow = Math.max(256, Math.ceil(mipW * 4 / 256) * 256);
    const totalBytes  = bytesPerRow * mipH;

    if (totalBytes > this.slotSizeBytes) {
      throw new Error(
        `[StagingRingBuffer] данные (${totalBytes} Б) превышают размер слота ` +
        `(${this.slotSizeBytes} Б). Увеличьте slotByteSize.`
      );
    }

    // Выравниваем смещение записи до 512 байт (смещение источника copyBufferToTexture
    // должно быть кратно размеру блока текселей × выравниванию bytesPerRow).
    const alignedOffset = Math.ceil(this._writeOffset / 512) * 512;
    if (alignedOffset + totalBytes > this.slotSizeBytes) {
      console.warn(`[StagingRingBuffer] Слот ${this._current} заполнен - для этого мипа используется writeTexture.`);
      return false;
    }

    // Копируем данные пикселей в отображённую область строка за строкой с учётом выравнивания.
    const dst = new Uint8Array(ab, alignedOffset, totalBytes);
    const srcRowBytes = mipW * 4;
    for (let row = 0; row < mipH; row++) {
      dst.set(data.subarray(row * srcRowBytes, (row + 1) * srcRowBytes), row * bytesPerRow);
    }

    encoder.copyBufferToTexture(
      { buffer: this._buffers[this._current]!, offset: alignedOffset, bytesPerRow, rowsPerImage: mipH },
      { texture, mipLevel, origin: { x: 0, y: 0, z: layerIndex } },
      [mipW, mipH, 1]
    );

    this._writeOffset = alignedOffset + totalBytes;
    return true;
  }

  /**
   * Завершить пакет загрузок кадра. Разотображает текущий слот, продвигает кольцо
   * и асинхронно запускает повторное отображение следующего слота.
   * Вызывать ДО device.queue.submit(), чтобы буфер был разотображён при выполнении команд.
   */
  endFrame(): void {
    if (this._writeOffset === 0) return; // в этом кадре ничего не поставлено в стейджинг
    if (this._states[this._current] !== "mapped") return;

    const current = this._current;
    this._buffers[current]!.unmap();
    this._arrayBuffers[current] = null;
    this._states[current] = "in-flight";

    this._current = (this._current + 1) % this._count;
    this._writeOffset = 0;

    // Повторно отображаем слот, на который только что перешли, если он вернулся из in-flight.
    if (this._states[this._current] === "in-flight") {
      this._states[this._current] = "pending";
      void this._buffers[this._current]!.mapAsync(GPUMapMode.WRITE)
        .then(() => {
          this._arrayBuffers[this._current] = this._buffers[this._current]!.getMappedRange();
          this._states[this._current] = "mapped";
        })
        .catch((err) => { console.error("[StagingRingBuffer] mapAsync завершился с ошибкой:", err); });
    }
  }

  // ---- Устаревший API одиночной загрузки (сохранён для демо) --------------------------------------

  /**
   * Получить текущий слот кольца. Возвращённый arrayBuffer доступен для записи
   * до вызова submit(). Выбрасывает исключение, если текущий слот ещё не отображён.
   */
  acquire(): StagingSlot {
    const state = this._states[this._current];
    if (state !== "mapped") {
      throw new Error(`[StagingRingBuffer] Слот ${this._current} в состоянии ${state}, не отображён. GPU отстаёт.`);
    }
    return {
      buffer: this._buffers[this._current]!,
      arrayBuffer: this._arrayBuffers[this._current]!,
      offset: 0,
    };
  }

  /**
   * Записать команду copyBufferToTexture в энкодер для текущего слота.
   * Должен вызываться после acquire() и перед submit().
   */
  recordCopy(
    encoder: GPUCommandEncoder,
    texture: GPUTexture,
    mipLevel: number,
    size: [number, number],
    bytesPerRow: number,
    dataByteLength: number
  ): void {
    const slot = this._current;
    const buf = this._buffers[slot]!;
    encoder.copyBufferToTexture(
      { buffer: buf, bytesPerRow, rowsPerImage: size[1] },
      { texture, mipLevel },
      [size[0], size[1], 1]
    );
    void dataByteLength;
  }

  /**
   * Разотобразить текущий слот, продвинуть кольцевой указатель и начать повторное
   * отображение следующего слота, чтобы он был готов к следующему обходу.
   */
  submit(): void {
    const current = this._current;
    const buf = this._buffers[current]!;
    buf.unmap();
    this._arrayBuffers[current] = null;
    this._states[current] = "in-flight";

    this._current = (this._current + 1) % this._count;

    const prevState = this._states[this._current];
    if (prevState === "in-flight") {
      this._states[this._current] = "pending";
      void this._buffers[this._current]!.mapAsync(GPUMapMode.WRITE)
        .then(() => {
          this._arrayBuffers[this._current] = this._buffers[this._current]!.getMappedRange();
          this._states[this._current] = "mapped";
        })
        .catch((err) => { console.error("[StagingRingBuffer] mapAsync завершился с ошибкой:", err); });
    }
  }

  destroy(): void {
    for (const buf of this._buffers) {
      buf.destroy();
    }
    this._buffers.length = 0;
    this._arrayBuffers.length = 0;
    this._states.length = 0;
  }
}
