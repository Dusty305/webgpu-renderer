/** Магические байты файла KTX2. */
const KTX2_MAGIC = new Uint8Array([0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a]);

/** Значения схем суперсжатия из спецификации KTX2. */
export const enum Supercompression {
  None = 0,
  BasisLZ = 1,
  Zstd = 2,
  UASTC = 3,
}

/** Данные одного уровня мипмапа, извлечённые из файла KTX2. */
export interface MipLevelData {
  mipLevel: number;
  byteOffset: number;
  byteLength: number;
  uncompressedByteLength: number;
}

/** Результат разбора заголовка файла KTX2. */
export interface KTX2ParseResult {
  vkFormat: number;
  pixelWidth: number;
  pixelHeight: number;
  layerCount: number;
  faceCount: number;
  levelCount: number;
  supercompressionScheme: Supercompression;
  isSrgb: boolean;
  levels: MipLevelData[];
}

/**
 * Разобрать бинарный буфер KTX2 и извлечь заголовок + смещения байт каждого мипмапа.
 * НЕ выполняет транскодирование - возвращает сырые (возможно суперсжатые) данные мипмапов.
 *
 * @throws если файл не является корректным KTX2.
 */
export function parseKTX2(buffer: ArrayBuffer): KTX2ParseResult {
  const bytes = new Uint8Array(buffer);

  // Проверить магические байты.
  for (let i = 0; i < KTX2_MAGIC.length; i++) {
    if (bytes[i] !== KTX2_MAGIC[i]) {
      throw new Error("[KTX2Parser] Неверные магические байты KTX2.");
    }
  }

  const view = new DataView(buffer);
  let offset = 12; // После магических байт

  const vkFormat               = view.getUint32(offset, true); offset += 4;
  const typeSize               = view.getUint32(offset, true); offset += 4; // не используется
  void typeSize;
  const pixelWidth             = view.getUint32(offset, true); offset += 4;
  const pixelHeight            = view.getUint32(offset, true); offset += 4;
  const pixelDepth             = view.getUint32(offset, true); offset += 4; // не используется
  void pixelDepth;
  const layerCount             = view.getUint32(offset, true); offset += 4;
  const faceCount              = view.getUint32(offset, true); offset += 4;
  const levelCount             = view.getUint32(offset, true); offset += 4;
  const supercompressionScheme = view.getUint32(offset, true); offset += 4;

  // Индекс (байтовые смещения до DFD, KVD, SGD)
  const dfdByteOffset          = view.getUint32(offset, true); offset += 4;
  const dfdByteLength          = view.getUint32(offset, true); offset += 4;
  void dfdByteOffset; void dfdByteLength;
  offset += 24; // kvdByteOffset(4), kvdByteLength(4), sgdByteOffset(8), sgdByteLength(8)

  // Индекс уровней (одна запись на уровень мипмапа).
  const levels: MipLevelData[] = [];
  for (let i = 0; i < levelCount; i++) {
    const byteOffset             = Number(view.getBigUint64(offset, true)); offset += 8;
    const byteLength             = Number(view.getBigUint64(offset, true)); offset += 8;
    const uncompressedByteLength = Number(view.getBigUint64(offset, true)); offset += 8;
    levels.push({ mipLevel: i, byteOffset, byteLength, uncompressedByteLength });
  }

  // Определить sRGB по vkFormat (упрощённо - охватывает распространённые случаи).
  const isSrgb = isSrgbFormat(vkFormat);

  return {
    vkFormat,
    pixelWidth,
    pixelHeight,
    layerCount: Math.max(1, layerCount),
    faceCount: Math.max(1, faceCount),
    levelCount,
    supercompressionScheme: supercompressionScheme as Supercompression,
    isSrgb,
    levels,
  };
}

/**
 * Возвращает true для значений перечисления Vulkan формата sRGB.
 * Справочник: https://registry.khronos.org/vulkan/specs/1.3/html/vkspec.html#VkFormat
 */
function isSrgbFormat(vkFormat: number): boolean {
  // Значения VK_FORMAT_*_SRGB: 43 (R8_SRGB) и многие блочные форматы.
  // На практике проверяем известные идентификаторы vkFormat для sRGB.
  const SRGB_FORMATS = new Set([
    43, 44, 50, 56, 57, 144, 145, 148, 149, 152, 153, 156, 157,
  ]);
  return SRGB_FORMATS.has(vkFormat);
}
