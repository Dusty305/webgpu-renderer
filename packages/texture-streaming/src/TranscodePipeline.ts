import type { KTX2ParseResult } from "./KTX2Parser.js";

/**
 * Целевые форматы транскодирования, поддерживаемые Basis Universal.
 */
export type BasisTargetFormat =
  | "bc7-rgba-unorm"
  | "bc7-rgba-unorm-srgb"
  | "astc-4x4-unorm"
  | "astc-4x4-unorm-srgb"
  | "rgba8unorm"
  | "rgba8unorm-srgb";

export interface TranscodeRequest {
  /** Полные байты файла KTX2. */
  ktx2Bytes: ArrayBuffer;
  /** Разобранный заголовок KTX2 (чтобы не разбирать повторно в воркере). */
  parsed: KTX2ParseResult;
  /** Целевой формат GPU-текстуры. */
  targetFormat: BasisTargetFormat;
  /** Уровень мипа для транскодирования (0 = полное разрешение). */
  mipLevel: number;
  /** Необязательный сигнал отмены. */
  signal?: AbortSignal;
}

export interface TranscodeResult {
  mipLevel: number;
  /** Транскодированные блочные данные, готовые для copyBufferToTexture. */
  data: ArrayBuffer;
  width: number;
  height: number;
  /** Байт на строку (уже выровнено до 256 байт для COPY_DST). */
  bytesPerRow: number;
}

/**
 * Транскодирует данные мипов KTX2 с использованием пула Web Workers,
 * выполняющих транскодер Basis Universal WASM.
 *
 * Для несжатых KTX2 (без суперсжатия, vkFormat = RGBA8)
 * транскодирование является холостым копированием - сырые байты возвращаются напрямую.
 * Этого достаточно для валидации фазы 2.6.
 *
 * Полное транскодирование ETC1S/UASTC через basis_transcoder.wasm отложено
 * до фазы 3 (в экспериментах диссертации используются заранее конвертированные файлы KTX2).
 */
export class TranscodePipeline {
  private readonly _workerCount: number;

  constructor(workerCount: number = 2) {
    this._workerCount = workerCount;
    // Инициализация пула воркеров отложена до интеграции basis_transcoder.wasm
    // в фазе 3. Пока резервный путь обрабатывает несжатые KTX2.
    void this._workerCount;
  }

  /**
   * Транскодировать один уровень мипа.
   *
   * Если файл KTX2 не использует суперсжатие (сырой RGBA8), данные
   * возвращаются напрямую без вызова воркера.
   */
  async transcode(request: TranscodeRequest): Promise<TranscodeResult> {
    const { parsed, ktx2Bytes, mipLevel } = request;

    if (request.signal?.aborted) {
      throw new DOMException("Транскодирование отменено", "AbortError");
    }

    const levelInfo = parsed.levels[mipLevel];
    if (!levelInfo) {
      throw new RangeError(`[TranscodePipeline] mipLevel ${mipLevel} вне диапазона (${parsed.levelCount} уровней)`);
    }

    // Несжатый путь: supercompressionScheme === 0 (None).
    if (parsed.supercompressionScheme === 0) {
      const width  = Math.max(1, parsed.pixelWidth  >> mipLevel);
      const height = Math.max(1, parsed.pixelHeight >> mipLevel);
      const rawSlice = ktx2Bytes.slice(levelInfo.byteOffset, levelInfo.byteOffset + levelInfo.byteLength);
      const bytesPerRow = Math.ceil((width * 4) / 256) * 256;

      // Переупаковать строки до выравнивания в 256 байт при необходимости.
      if (width * 4 === bytesPerRow) {
        return { mipLevel, data: rawSlice, width, height, bytesPerRow };
      }

      const padded = new ArrayBuffer(bytesPerRow * height);
      const src = new Uint8Array(rawSlice);
      const dst = new Uint8Array(padded);
      for (let row = 0; row < height; row++) {
        dst.set(src.subarray(row * width * 4, (row + 1) * width * 4), row * bytesPerRow);
      }
      return { mipLevel, data: padded, width, height, bytesPerRow };
    }

    // Путь суперсжатия: требуются воркеры + basis_transcoder.wasm.
    throw new Error(
      `[TranscodePipeline] Схема суперсжатия ${parsed.supercompressionScheme} ` +
      `требует basis_transcoder.wasm (ещё не интегрирован). ` +
      `Используйте несжатый KTX2 для валидации фазы 2.6.`
    );
  }

  destroy(): void {
    // Завершение работы воркеров отложено до их создания в фазе 3.
  }
}
