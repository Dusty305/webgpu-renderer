/**
 * Генерирует минимальный корректный KTX2-бинарник в памяти для тестирования.
 *
 * Спецификация: https://registry.khronos.org/KTX/specs/2.0/ktxspec.v2.html
 *
 * Формат: несжатый RGBA8 (vkFormat = VK_FORMAT_R8G8B8A8_SRGB = 43),
 * без суперсжатия, N мип-уровней с данными сплошного цвета.
 */

const KTX2_MAGIC = new Uint8Array([
  0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
]);

/** Генерирует мип-уровень с одним сплошным цветом. */
function solidMipData(width: number, height: number, r: number, g: number, b: number): Uint8Array {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4]     = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return data;
}

const MIP_COLORS: [number, number, number][] = [
  [255, 255, 255], // 0 - белый
  [153, 0,   204], // 1 - фиолетовый
  [26,  26,  255], // 2 - синий
  [0,   204, 204], // 3 - голубой
  [26,  204, 26],  // 4 - зелёный
  [255, 255, 0],   // 5 - жёлтый
  [255, 128, 0],   // 6 - оранжевый
  [255, 26,  26],  // 7 - красный
];

export function generateTestKtx2(baseWidth: number, baseHeight: number): ArrayBuffer {
  const mipCount = Math.floor(Math.log2(Math.min(baseWidth, baseHeight))) + 1;

  // Собираем пиксельные данные мипов.
  const mipDatas: Uint8Array[] = [];
  for (let level = 0; level < mipCount; level++) {
    const w = Math.max(1, baseWidth  >> level);
    const h = Math.max(1, baseHeight >> level);
    const [r, g, b] = MIP_COLORS[level] ?? [128, 128, 128];
    mipDatas.push(solidMipData(w, h, r!, g!, b!));
  }

  // Заголовок KTX2 занимает 80 байт.
  // Индекс уровней: mipCount × 24 байта (3 × u64: byteOffset, byteLength, uncompressedByteLength).
  // DFD: минимальный 28-байтный дескриптор формата данных.
  // Без KVD и SGD.
  // Далее следуют данные мипов.

  const HEADER_SIZE     = 80;
  const LEVEL_INDEX_SIZE = mipCount * 24;
  const DFD_SIZE        = 28;
  const DFD_OFFSET      = HEADER_SIZE + LEVEL_INDEX_SIZE;
  const MIPS_START      = DFD_OFFSET + DFD_SIZE;

  // Вычисляем смещения байт мипов (KTX2 хранит мипы от грубых к детальным в индексе уровней,
  // но физическое расположение может быть от детальных к грубым; для простоты используем второй вариант).
  const mipOffsets: number[] = [];
  let offset = MIPS_START;
  for (let level = 0; level < mipCount; level++) {
    mipOffsets.push(offset);
    offset += mipDatas[level]!.byteLength;
  }

  const totalSize = offset;
  const buf = new ArrayBuffer(totalSize);
  const bytes = new Uint8Array(buf);
  const view = new DataView(buf);

  // ---- Магическое число ------------------------------------------------------------------------------------------------------------
  bytes.set(KTX2_MAGIC, 0);

  // ---- Заголовок (смещения согласно таблице 2 спецификации) ------------------------------------
  let p = 12;
  const VK_FORMAT_R8G8B8A8_SRGB = 43;
  view.setUint32(p, VK_FORMAT_R8G8B8A8_SRGB, true); p += 4; // vkFormat
  view.setUint32(p, 1,           true); p += 4; // typeSize
  view.setUint32(p, baseWidth,   true); p += 4; // pixelWidth
  view.setUint32(p, baseHeight,  true); p += 4; // pixelHeight
  view.setUint32(p, 0,           true); p += 4; // pixelDepth (0 = 2D)
  view.setUint32(p, 0,           true); p += 4; // layerCount (0 = не массив)
  view.setUint32(p, 1,           true); p += 4; // faceCount
  view.setUint32(p, mipCount,    true); p += 4; // levelCount
  view.setUint32(p, 0,           true); p += 4; // supercompressionScheme (нет)

  // Индекс:
  view.setUint32(p, DFD_OFFSET,  true); p += 4; // dfdByteOffset
  view.setUint32(p, DFD_SIZE,    true); p += 4; // dfdByteLength
  view.setUint32(p, 0,           true); p += 4; // kvdByteOffset
  view.setUint32(p, 0,           true); p += 4; // kvdByteLength
  view.setBigUint64(p, 0n,       true); p += 8; // sgdByteOffset
  view.setBigUint64(p, 0n,       true); p += 8; // sgdByteLength
  // p теперь равен 80 - конец заголовка.

  // ---- Индекс уровней ----------------------------------------------------------------------------------------------------------------
  for (let level = 0; level < mipCount; level++) {
    const byteLen = mipDatas[level]!.byteLength;
    view.setBigUint64(p, BigInt(mipOffsets[level]!), true); p += 8; // byteOffset (смещение байт)
    view.setBigUint64(p, BigInt(byteLen),            true); p += 8; // byteLength (длина в байтах)
    view.setBigUint64(p, BigInt(byteLen),            true); p += 8; // uncompressedByteLength (несжатая длина)
  }

  // ---- Минимальный DFD (Data Format Descriptor) ------------------------------------------------------------
  // Записываем минимальный 28-байтный DFD. Читатели используют его для подтверждения RGBA8 sRGB.
  view.setUint32(p, DFD_SIZE,    true); p += 4; // totalSize (общий размер)
  view.setUint32(p, DFD_SIZE - 4, true); p += 4; // descriptor block size (размер блока дескриптора)
  view.setUint32(p, 0,           true); p += 4; // model=RGBSDA, colorPrimaries=BT709 и др.
  view.setUint32(p, 1,           true); p += 4; // texelBlockDimension0=1 (RGBA)
  view.setUint32(p, 0,           true); p += 4;
  view.setUint32(p, 0,           true); p += 4;
  view.setUint32(p, 0,           true); p += 4;

  // ---- Пиксельные данные мипов ----------------------------------------------------------------------------------------------
  for (let level = 0; level < mipCount; level++) {
    bytes.set(mipDatas[level]!, mipOffsets[level]!);
  }

  return buf;
}
