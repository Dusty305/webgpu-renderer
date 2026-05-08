/**
 * Процедурный генератор сцены для стресс-тестирования.
 *
 * Генерирует визуально различимые текстуры RGBA8 с использованием FBM-шума на основе хэшей.
 * Вся генерация выполняется на CPU - загрузка из интернета не требуется.
 * Текстуры упаковываются в тот же бинарный формат KTX2, что используется в остальном
 * конвейере стриминга, поэтому они проходят через TextureStreamingManager без изменений.
 */
import { parseKTX2 } from "@webgpu-streaming/texture-streaming";
import type { KTX2ParseResult } from "@webgpu-streaming/texture-streaming";

// ---- Типы ------------------------------------------------------------------------------------------------------------------------------------------

export type ScenePresetName = "small" | "medium" | "large" | "thesis";

export interface ScenePreset {
  readonly name: ScenePresetName;
  readonly objectCount: number;
  /** Одна запись на объект - length === objectCount. */
  readonly textureSizes: readonly number[];
  /** Байты полной цепочки мипов для всех текстур в RGBA8. */
  readonly expectedTotalBytes: number;
  readonly gridCols: number;
  readonly spacing: number;
  /** Рекомендуемое начальное расстояние камеры от начала координат. */
  readonly cameraDistance: number;
}

export interface StressObject {
  readonly id: string;
  readonly materialId: number;
  readonly ktx2Bytes: ArrayBuffer;
  readonly parsed: KTX2ParseResult;
  /** [cx, cy, cz, radius] ограничивающая сфера в мировом пространстве. */
  readonly boundingSphere: Float32Array;
  readonly texSize: number;
}

/** Вызывается после генерации каждой текстуры: (done, total, label). */
export type ProgressCallback = (done: number, total: number, label: string) => void;

// ---- Пресеты --------------------------------------------------------------------------------------------------------------------------------------

/** Байты полной цепочки мипов RGBA8 для квадратной текстуры. */
function mipChainBytes(size: number): number {
  let total = 0;
  for (let s = size; s >= 1; s >>= 1) total += s * s * 4;
  return total;
}

export const SCENE_PRESETS: Record<ScenePresetName, ScenePreset> = {
  small: {
    name:              "small",
    objectCount:        25,
    textureSizes:       Array<number>(25).fill(512),
    expectedTotalBytes: 25 * mipChainBytes(512),        // ≈ 33 МБ
    gridCols:           5,
    spacing:            1.5,
    cameraDistance:     6,
  },
  medium: {
    name:              "medium",
    objectCount:        100,
    textureSizes:       Array<number>(100).fill(1024),
    expectedTotalBytes: 100 * mipChainBytes(1024),      // ≈ 533 МБ
    gridCols:           10,
    spacing:            1.5,
    cameraDistance:     14,
  },
  large: {
    name:              "large",
    objectCount:        100,
    textureSizes:       Array<number>(100).fill(2048),
    expectedTotalBytes: 100 * mipChainBytes(2048),      // ≈ 2.1 ГБ
    gridCols:           10,
    spacing:            1.5,
    cameraDistance:     14,
  },
  thesis: {
    name:              "thesis",
    objectCount:        200,
    textureSizes: [
      ...Array<number>(50).fill(512),
      ...Array<number>(100).fill(1024),
      ...Array<number>(50).fill(2048),
    ],
    expectedTotalBytes:
      50  * mipChainBytes(512) +
      100 * mipChainBytes(1024) +
      50  * mipChainBytes(2048),                        // ≈ 1.7 ГБ
    gridCols:           15,
    spacing:            1.5,
    cameraDistance:     20,
  },
};

/**
 * Вычисляет количество слоёв на уровень, необходимых для пресета.
 * Возвращает [tier0Count, tier1Count, tier2Count], каждое значение ≥ 1
 * (TierAllocator требует не менее 1 слоя на уровень).
 */
export function computeMaxLayersPerTier(preset: ScenePreset): [number, number, number] {
  let t0 = 0, t1 = 0, t2 = 0;
  for (const s of preset.textureSizes) {
    if      (s <= 512)  t0++;
    else if (s <= 1024) t1++;
    else                t2++;
  }
  return [Math.max(1, t0), Math.max(1, t1), Math.max(1, t2)];
}

// ---- Шум ------------------------------------------------------------------------------------------------------------------------------------------

/** Целочисленный хэш с хорошими лавинными свойствами. */
function ihash(x: number, y: number, seed: number): number {
  let v = (x * 1664525 + y * 22695477 + seed * 1013904223) | 0;
  v = Math.imul(v ^ (v >>> 16), 0x45d9f3b);
  v = Math.imul(v ^ (v >>> 16), 0x45d9f3b);
  return (v ^ (v >>> 16)) >>> 0;
}

/** Число с плавающей точкой в [0, 1] по координатам решётки. */
function hashF(x: number, y: number, seed: number): number {
  return ihash(x, y, seed) / 4294967296;
}

/** Smoothstep для интерполяции шума. */
function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Билинейный value-шум (на основе хэшей, без предвычисленной решётки). */
function valueNoise(wx: number, wy: number, seed: number): number {
  const ix = Math.floor(wx), iy = Math.floor(wy);
  const fx = smooth(wx - ix), fy = smooth(wy - iy);
  const v00 = hashF(ix,     iy,     seed);
  const v10 = hashF(ix + 1, iy,     seed);
  const v01 = hashF(ix,     iy + 1, seed);
  const v11 = hashF(ix + 1, iy + 1, seed);
  return (v00 * (1 - fx) + v10 * fx) * (1 - fy)
       + (v01 * (1 - fx) + v11 * fx) * fy;
}

/** Дробное броуновское движение на 3 октавы, результат ограничен до [0, 1]. */
function fbm(wx: number, wy: number, seed: number): number {
  const n = valueNoise(wx,         wy,         seed)       * 0.50
          + valueNoise(wx * 2.1,   wy * 2.1,   seed + 100) * 0.30
          + valueNoise(wx * 4.37,  wy * 4.37,  seed + 200) * 0.20;
  return Math.min(1, Math.max(0, n));
}

// ---- Цвет ----------------------------------------------------------------------------------------------------------------------------------------

/** HSV (все [0, 1]) → байты RGB [0, 255]. */
function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
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

// ---- Генерация текстур ----------------------------------------------------------------------------------------------------------------

/**
 * Генерирует изображение RGBA8 уровня мипа 0 размером `size × size`.
 * Узор: FBM-шум, окрашенный уникальным цветом объекта.
 * Яркость в диапазоне [0.35, 1.0], чтобы оттенок всегда был различим.
 */
function generateMip0(
  size: number, seed: number,
  tr: number, tg: number, tb: number
): Uint8Array {
  const data = new Uint8Array(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      // Масштабируем UV для ~4 периодов шума по текстуре
      const n = fbm((x / size) * 4, (y / size) * 4, seed);
      const b = 0.35 + n * 0.65;
      const i = (y * size + x) * 4;
      data[i]     = Math.min(255, tr * b) | 0;
      data[i + 1] = Math.min(255, tg * b) | 0;
      data[i + 2] = Math.min(255, tb * b) | 0;
      data[i + 3] = 255;
    }
  }
  return data;
}

/** Уменьшение масштаба фильтром 2×2 box для генерации мипов. */
function downsampleMip(src: Uint8Array, srcSize: number): Uint8Array {
  const d = srcSize >> 1;
  const dst = new Uint8Array(d * d * 4);
  for (let y = 0; y < d; y++) {
    for (let x = 0; x < d; x++) {
      const o = (y * d + x) * 4;
      const s = (y * 2 * srcSize + x * 2) * 4;
      for (let c = 0; c < 4; c++) {
        dst[o + c] = (
          src[s + c]! + src[s + 4 + c]! +
          src[s + srcSize * 4 + c]! + src[s + srcSize * 4 + 4 + c]!
        ) >> 2;
      }
    }
  }
  return dst;
}

/** Строит полную пирамиду мипов из уровня 0 последовательным уменьшением 2×2. */
function buildMipPyramid(mip0: Uint8Array, size: number): Uint8Array[] {
  const mips: Uint8Array[] = [mip0];
  let cur = mip0, s = size;
  while (s > 1) { cur = downsampleMip(cur, s); mips.push(cur); s >>= 1; }
  return mips;
}

// ---- Упаковка KTX2 --------------------------------------------------------------------------------------------------------------------------

const KTX2_MAGIC = new Uint8Array([
  0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a,
]);

/** Упаковывает пирамиду мипов в корректный бинарный файл KTX2 (несжатый RGBA8). */
function packKtx2(mips: Uint8Array[], size: number): ArrayBuffer {
  const mc = mips.length;
  const DFD_OFFSET = 80 + mc * 24;
  const DFD_SIZE   = 28;
  const MIPS_START = DFD_OFFSET + DFD_SIZE;

  const offsets: number[] = [];
  let off = MIPS_START;
  for (let l = 0; l < mc; l++) { offsets.push(off); off += mips[l]!.byteLength; }

  const buf = new ArrayBuffer(off);
  const u8  = new Uint8Array(buf);
  const dv  = new DataView(buf);

  u8.set(KTX2_MAGIC, 0);
  let p = 12;
  dv.setUint32(p, 43,       true); p += 4; // VK_FORMAT_R8G8B8A8_SRGB
  dv.setUint32(p, 1,        true); p += 4; // размер типа
  dv.setUint32(p, size,     true); p += 4; // ширина в пикселях
  dv.setUint32(p, size,     true); p += 4; // высота в пикселях
  dv.setUint32(p, 0,        true); p += 4; // глубина в пикселях
  dv.setUint32(p, 0,        true); p += 4; // количество слоёв
  dv.setUint32(p, 1,        true); p += 4; // количество граней
  dv.setUint32(p, mc,       true); p += 4; // количество уровней
  dv.setUint32(p, 0,        true); p += 4; // схема суперсжатия
  dv.setUint32(p, DFD_OFFSET, true); p += 4;
  dv.setUint32(p, DFD_SIZE,   true); p += 4;
  dv.setUint32(p, 0,          true); p += 4; // kvd offset
  dv.setUint32(p, 0,          true); p += 4; // kvd length
  dv.setBigUint64(p, 0n,      true); p += 8; // sgd offset
  dv.setBigUint64(p, 0n,      true); p += 8; // sgd length

  for (let l = 0; l < mc; l++) {
    const bl = BigInt(mips[l]!.byteLength);
    dv.setBigUint64(p, BigInt(offsets[l]!), true); p += 8;
    dv.setBigUint64(p, bl,                  true); p += 8;
    dv.setBigUint64(p, bl,                  true); p += 8;
  }

  // Minimal DFD block
  dv.setUint32(p, DFD_SIZE,     true); p += 4;
  dv.setUint32(p, DFD_SIZE - 4, true); p += 4;
  for (let i = 0; i < 5; i++) { dv.setUint32(p, 0, true); p += 4; }

  for (let l = 0; l < mc; l++) u8.set(mips[l]!, offsets[l]!);
  return buf;
}

// ---- Public API --------------------------------------------------------------------------------------------------------------------------------

/**
 * Generate all objects for a stress-test scene.
 * Yields to the browser every 4 textures so progress callbacks can update the UI.
 *
 * @param preset - Scene preset (use SCENE_PRESETS[name])
 * @param onProgress - Optional progress callback
 */
export async function generateStressScene(
  preset: ScenePreset,
  onProgress?: ProgressCallback,
): Promise<StressObject[]> {
  const objects: StressObject[] = [];
  const { objectCount, textureSizes, gridCols, spacing } = preset;
  const gridRows = Math.ceil(objectCount / gridCols);

  for (let i = 0; i < objectCount; i++) {
    const size   = textureSizes[i] ?? 512;
    const seed   = i * 7919 + 12345;              // deterministic per object
    const [tr, tg, tb] = hsvToRgb(i / objectCount, 0.8, 0.9);

    const mip0      = generateMip0(size, seed, tr, tg, tb);
    const mips      = buildMipPyramid(mip0, size);
    const ktx2Bytes = packKtx2(mips, size);
    const parsed    = parseKTX2(ktx2Bytes);

    const col = i % gridCols;
    const row = Math.floor(i / gridCols);
    const cx  = (col - (gridCols - 1) / 2) * spacing;
    const cy  = (row - (gridRows - 1) / 2) * spacing;

    objects.push({
      id:             `obj-${i}`,
      materialId:     i,
      ktx2Bytes,
      parsed,
      boundingSphere: new Float32Array([cx, cy, 0, spacing * 0.48]),
      texSize:        size,
    });

    onProgress?.(i + 1, objectCount, `${i + 1}/${objectCount} (${size}×${size})`);

    // Yield every 4 textures so the browser can repaint the progress display.
    if ((i + 1) % 4 === 0) await new Promise<void>((r) => setTimeout(r, 0));
  }

  return objects;
}
