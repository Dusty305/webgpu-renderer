/**
 * Генерирует KTX2-бинарник для тестов потоковой загрузки с бюджетом.
 *
 * Мип 0 (самый детальный) = заданный базовый цвет.
 * Более грубые мипы линейно переходят к белому, чтобы незагруженные текстуры
 * отображались белыми, а затем загружались до своего уникального цвета.
 */

const KTX2_MAGIC = new Uint8Array([
  0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
]);

/** Преобразует HSV (все значения в [0,1]) в байты RGB [0,255]. */
export function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  let r: number, g: number, b: number;
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    default: r = v; g = p; b = q; break;
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function solidMip(width: number, height: number, r: number, g: number, b: number): Uint8Array {
  const data = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4]     = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return data;
}

/**
 * Генерирует квадратный KTX2-файл, в котором:
 *   уровень 0 = базовый цвет (самый детальный мип)
 *   уровень n = линейная интерполяция к белому (грубее = белее)
 */
export function generateColorKtx2(size: number, baseR: number, baseG: number, baseB: number): ArrayBuffer {
  const mipCount = Math.floor(Math.log2(size)) + 1;

  const mipDatas: Uint8Array[] = [];
  for (let level = 0; level < mipCount; level++) {
    const w = Math.max(1, size >> level);
    const t = level / Math.max(1, mipCount - 1);
    const r = Math.round(baseR + (255 - baseR) * t);
    const g = Math.round(baseG + (255 - baseG) * t);
    const b = Math.round(baseB + (255 - baseB) * t);
    mipDatas.push(solidMip(w, w, r, g, b));
  }

  const HEADER_SIZE      = 80;
  const LEVEL_INDEX_SIZE = mipCount * 24;
  const DFD_SIZE         = 28;
  const DFD_OFFSET       = HEADER_SIZE + LEVEL_INDEX_SIZE;
  const MIPS_START       = DFD_OFFSET + DFD_SIZE;

  const mipOffsets: number[] = [];
  let offset = MIPS_START;
  for (let level = 0; level < mipCount; level++) {
    mipOffsets.push(offset);
    offset += mipDatas[level]!.byteLength;
  }

  const buf  = new ArrayBuffer(offset);
  const bytes = new Uint8Array(buf);
  const view  = new DataView(buf);

  bytes.set(KTX2_MAGIC, 0);

  let p = 12;
  const VK_FORMAT_R8G8B8A8_SRGB = 43;
  view.setUint32(p, VK_FORMAT_R8G8B8A8_SRGB, true); p += 4;
  view.setUint32(p, 1,         true); p += 4;
  view.setUint32(p, size,      true); p += 4;
  view.setUint32(p, size,      true); p += 4;
  view.setUint32(p, 0,         true); p += 4;
  view.setUint32(p, 0,         true); p += 4;
  view.setUint32(p, 1,         true); p += 4;
  view.setUint32(p, mipCount,  true); p += 4;
  view.setUint32(p, 0,         true); p += 4;

  view.setUint32(p, DFD_OFFSET, true); p += 4;
  view.setUint32(p, DFD_SIZE,   true); p += 4;
  view.setUint32(p, 0,          true); p += 4;
  view.setUint32(p, 0,          true); p += 4;
  view.setBigUint64(p, 0n,      true); p += 8;
  view.setBigUint64(p, 0n,      true); p += 8;

  for (let level = 0; level < mipCount; level++) {
    const byteLen = mipDatas[level]!.byteLength;
    view.setBigUint64(p, BigInt(mipOffsets[level]!), true); p += 8;
    view.setBigUint64(p, BigInt(byteLen),            true); p += 8;
    view.setBigUint64(p, BigInt(byteLen),            true); p += 8;
  }

  view.setUint32(p, DFD_SIZE,       true); p += 4;
  view.setUint32(p, DFD_SIZE - 4,   true); p += 4;
  view.setUint32(p, 0,              true); p += 4;
  view.setUint32(p, 1,              true); p += 4;
  view.setUint32(p, 0,              true); p += 4;
  view.setUint32(p, 0,              true); p += 4;
  view.setUint32(p, 0,              true); p += 4;

  for (let level = 0; level < mipCount; level++) {
    bytes.set(mipDatas[level]!, mipOffsets[level]!);
  }

  return buf;
}
