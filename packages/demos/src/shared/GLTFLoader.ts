/**
 * Минимальный загрузчик glTF 2.0 для Demo 13 (потоковая передача реальной сцены).
 *
 * Поддерживается:
 *   - Примитивы меша с атрибутами POSITION, NORMAL, TEXCOORD_0
 *   - Буферы индексов (u16 / u32)
 *   - Материалы: pbrMetallicRoughness.baseColorTexture (только albedo)
 *   - Внешние изображения (PNG / JPEG), декодируемые в RGBA8 через OffscreenCanvas
 *   - Иерархия узлов с преобразованиями TRS или матрицей
 *   - Внешние бинарные буферы (.bin)
 *
 * НЕ поддерживается: анимации, скины, морф-таргеты, расширения glTF.
 *
 * Выходные текстуры упакованы как KTX2-совместимые буферы RGBA8, пригодные
 * для прямой передачи в TextureStreamingManager.registerTexture().
 */

import { parseKTX2, buildMipPyramid, packKtx2, decodeImage } from "@webgpu-streaming/texture-streaming";
import type { KTX2ParseResult } from "@webgpu-streaming/texture-streaming";

// ---- Публичные типы ----------------------------------------------------------------------------------------------------------------------------

/** Один GPU-готовый примитив меша. */
export interface LoadedMesh {
  /** Человекочитаемая метка для отладки. */
  name: string;
  /**
   * Чередующиеся данные вершин: [px, py, pz, nx, ny, nz, u, v] × vertexCount.
   * Шаг = 32 байта (8 × float32).
   */
  vertexData: Float32Array;
  /** Индексы списка треугольников (u16 или u32 в зависимости от числа вершин). */
  indexData: Uint16Array | Uint32Array;
  indexCount: number;
  /** Индекс (с нуля) в LoadedScene.textures. -1, если нет текстуры albedo. */
  textureIndex: number;
  /** Ограничивающая сфера в мировом пространстве [cx, cy, cz, radius]. */
  boundingSphere: Float32Array;
  /** Мировое преобразование 4×4 в столбцовом порядке (Float32Array из 16 элементов). */
  worldTransform: Float32Array;
}

/** Одна текстура albedo, упакованная для TextureStreamingManager. */
export interface LoadedTexture {
  /** Уникальный идентификатор (например, "tex-0"). */
  id: string;
  /** KTX2-бинарник с пирамидой мипов RGBA8 SRGB. */
  ktx2Bytes: ArrayBuffer;
  parsed: KTX2ParseResult;
  /** Ширина изображения mip-0. */
  width: number;
  /** Высота изображения mip-0. */
  height: number;
}

export interface LoadedScene {
  meshes: LoadedMesh[];
  /** По одному элементу на уникальную текстуру albedo, индексируется через LoadedMesh.textureIndex. */
  textures: LoadedTexture[];
}

/** Колбэк прогресса: (message) => void. */
export type LoadProgress = (msg: string) => void;

// ---- Типы JSON glTF (минимальные) --------------------------------------------------------------------------------------------------

interface GltfAccessor {
  bufferView?: number;
  byteOffset?: number;
  componentType: number; // 5120..5126
  count: number;
  type: string; // SCALAR / VEC2 / VEC3 / VEC4 / MAT4
  min?: number[];
  max?: number[];
}

interface GltfBufferView {
  buffer: number;
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
}

interface GltfBuffer { byteLength: number; uri?: string; }

interface GltfImage { uri?: string; bufferView?: number; mimeType?: string; }

interface GltfTexture { source?: number; sampler?: number; }

interface GltfMaterial {
  name?: string;
  pbrMetallicRoughness?: {
    baseColorTexture?: { index: number };
  };
}

interface GltfPrimitive {
  attributes: Record<string, number>;
  indices?: number;
  material?: number;
}

interface GltfMesh { name?: string; primitives: GltfPrimitive[]; }

interface GltfNode {
  mesh?: number;
  children?: number[];
  translation?: [number, number, number];
  rotation?: [number, number, number, number];
  scale?: [number, number, number];
  matrix?: number[];
  name?: string;
}

interface GltfScene { nodes?: number[]; }

interface GltfRoot {
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: GltfBuffer[];
  images?: GltfImage[];
  textures?: GltfTexture[];
  materials?: GltfMaterial[];
  meshes?: GltfMesh[];
  nodes?: GltfNode[];
  scene?: number;
  scenes?: GltfScene[];
}

// ---- Размеры типов компонентов ------------------------------------------------------------------------------------------------------------

const COMPONENT_SIZES: Record<number, number> = {
  5120: 1, // BYTE
  5121: 1, // UNSIGNED_BYTE
  5122: 2, // SHORT
  5123: 2, // UNSIGNED_SHORT
  5124: 4, // INT
  5125: 4, // UNSIGNED_INT
  5126: 4, // FLOAT
};

const ACCESSOR_TYPE_COUNTS: Record<string, number> = {
  SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4,
  MAT2: 4, MAT3: 9, MAT4: 16,
};

// ---- Чтение аксессора ----------------------------------------------------------------------------------------------------------------------

/**
 * Читает аксессор glTF в типизированный массив.
 * Обрабатывает ненулевой byteOffset, byteStride и все типы компонентов.
 */
function readAccessor(
  gltf: GltfRoot,
  accIdx: number,
  buffers: ArrayBuffer[]
): Float32Array | Uint16Array | Uint32Array {
  const acc = gltf.accessors![accIdx]!;
  const compSize = COMPONENT_SIZES[acc.componentType]!;
  const numComps = ACCESSOR_TYPE_COUNTS[acc.type]!;
  const count = acc.count;

  if (acc.bufferView === undefined) {
    // Аксессор, заполненный нулями (редко, но допустимо по спецификации)
    return new Float32Array(count * numComps);
  }

  const bv = gltf.bufferViews![acc.bufferView]!;
  const buf = buffers[bv.buffer]!;
  const bvOffset = bv.byteOffset ?? 0;
  const accOffset = acc.byteOffset ?? 0;
  const stride = bv.byteStride ?? (compSize * numComps);

  // Если данные плотно упакованы - вернуть прямое представление
  const isTight = stride === compSize * numComps;
  if (isTight) {
    const byteOffset = bvOffset + accOffset;
    const byteLen = count * numComps * compSize;
    if (acc.componentType === 5126) return new Float32Array(buf, byteOffset, count * numComps);
    if (acc.componentType === 5123) return new Uint16Array(buf, byteOffset, count * numComps);
    if (acc.componentType === 5125) return new Uint32Array(buf, byteOffset, count * numComps);
  }

  // Чтение с шагом - копируем в новый массив
  if (acc.componentType === 5126) {
    const out = new Float32Array(count * numComps);
    const view = new DataView(buf);
    let base = bvOffset + accOffset;
    for (let i = 0; i < count; i++) {
      for (let c = 0; c < numComps; c++) {
        out[i * numComps + c] = view.getFloat32(base + c * 4, true);
      }
      base += stride;
    }
    return out;
  }
  if (acc.componentType === 5123) {
    const out = new Uint16Array(count * numComps);
    const view = new DataView(buf);
    let base = bvOffset + accOffset;
    for (let i = 0; i < count; i++) {
      for (let c = 0; c < numComps; c++) {
        out[i * numComps + c] = view.getUint16(base + c * 2, true);
      }
      base += stride;
    }
    return out;
  }
  if (acc.componentType === 5125) {
    const out = new Uint32Array(count * numComps);
    const view = new DataView(buf);
    let base = bvOffset + accOffset;
    for (let i = 0; i < count; i++) {
      for (let c = 0; c < numComps; c++) {
        out[i * numComps + c] = view.getUint32(base + c * 4, true);
      }
      base += stride;
    }
    return out;
  }

  throw new Error(`[GLTFLoader] Unsupported componentType ${acc.componentType}`);
}

// ---- Матричные вычисления ------------------------------------------------------------------------------------------------------------------------------

function mat4Identity(): Float32Array {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[row + k * 4]! * b[k + col * 4]!;
      }
      out[row + col * 4] = sum;
    }
  }
  return out;
}

/**
 * Строит матрицу 4×4 в столбцовом порядке из компонентов TRS.
 * Вращение задаётся кватернионом [x, y, z, w].
 */
function mat4FromTRS(
  t?: [number, number, number],
  r?: [number, number, number, number],
  s?: [number, number, number]
): Float32Array {
  const m = mat4Identity();

  // Вращение
  if (r) {
    const [x, y, z, w] = r;
    m[0]  = 1 - 2 * (y * y + z * z);
    m[1]  = 2 * (x * y + z * w);
    m[2]  = 2 * (x * z - y * w);
    m[4]  = 2 * (x * y - z * w);
    m[5]  = 1 - 2 * (x * x + z * z);
    m[6]  = 2 * (y * z + x * w);
    m[8]  = 2 * (x * z + y * w);
    m[9]  = 2 * (y * z - x * w);
    m[10] = 1 - 2 * (x * x + y * y);
    m[15] = 1;
  }

  // Масштаб (запекается в столбцы вращения)
  if (s) {
    m[0] *= s[0]; m[1] *= s[0]; m[2] *= s[0];
    m[4] *= s[1]; m[5] *= s[1]; m[6] *= s[1];
    m[8] *= s[2]; m[9] *= s[2]; m[10] *= s[2];
  }

  // Перемещение
  if (t) {
    m[12] = t[0]; m[13] = t[1]; m[14] = t[2];
  }

  return m;
}

function nodeLocalTransform(node: GltfNode): Float32Array {
  if (node.matrix) {
    const m = new Float32Array(16);
    for (let i = 0; i < 16; i++) m[i] = node.matrix[i]!;
    return m;
  }
  return mat4FromTRS(node.translation, node.rotation, node.scale);
}

// ---- Ограничивающая сфера ----------------------------------------------------------------------------------------------------------------------

function computeBoundingSphere(
  positions: Float32Array,
  worldTransform: Float32Array
): Float32Array {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  const count = positions.length / 3;
  for (let i = 0; i < count; i++) {
    const lx = positions[i * 3]!;
    const ly = positions[i * 3 + 1]!;
    const lz = positions[i * 3 + 2]!;

    // Преобразуем в мировое пространство
    const wx = worldTransform[0]! * lx + worldTransform[4]! * ly + worldTransform[8]!  * lz + worldTransform[12]!;
    const wy = worldTransform[1]! * lx + worldTransform[5]! * ly + worldTransform[9]!  * lz + worldTransform[13]!;
    const wz = worldTransform[2]! * lx + worldTransform[6]! * ly + worldTransform[10]! * lz + worldTransform[14]!;

    if (wx < minX) minX = wx; if (wx > maxX) maxX = wx;
    if (wy < minY) minY = wy; if (wy > maxY) maxY = wy;
    if (wz < minZ) minZ = wz; if (wz > maxZ) maxZ = wz;
  }

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const r = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.5;

  return new Float32Array([cx, cy, cz, Math.max(r, 0.001)]);
}

// ---- Обход узлов ------------------------------------------------------------------------------------------------------------------------

interface PrimitiveResult {
  meshIdx: number;
  primIdx: number;
  worldTransform: Float32Array;
}

function collectPrimitives(
  gltf: GltfRoot,
  nodeIdx: number,
  parentTransform: Float32Array,
  results: PrimitiveResult[]
): void {
  const node = gltf.nodes![nodeIdx]!;
  const local = nodeLocalTransform(node);
  const world = mat4Multiply(parentTransform, local);

  if (node.mesh !== undefined) {
    const mesh = gltf.meshes![node.mesh]!;
    for (let p = 0; p < mesh.primitives.length; p++) {
      results.push({ meshIdx: node.mesh, primIdx: p, worldTransform: world });
    }
  }

  for (const childIdx of node.children ?? []) {
    collectPrimitives(gltf, childIdx, world, results);
  }
}

// ---- Основной загрузчик ------------------------------------------------------------------------------------------------------------------------------

/**
 * Загружает сцену glTF по URL.
 *
 * Возвращает все примитивы меша в виде чередующихся данных вершин (позиция+нормаль+uv),
 * а также текстуры albedo, упакованные как KTX2-буферы, готовые для TextureStreamingManager.
 *
 * Для каждого материала загружается только pbrMetallicRoughness baseColorTexture.
 * Остальные PBR-карты (normal, ORM, emissive) игнорируются.
 *
 * @param url - URL .gltf-файла.
 * @param onProgress - Необязательный колбэк прогресса.
 */
export async function loadGLTF(
  url: string,
  onProgress?: LoadProgress
): Promise<LoadedScene> {
  const base = url.substring(0, url.lastIndexOf("/") + 1);

  // ---- 1. Загружаем JSON glTF ----------------------------------------------------------------------------------------------------
  onProgress?.("Fetching glTF JSON…");
  const gltfResp = await fetch(url);
  if (!gltfResp.ok) throw new Error(`[GLTFLoader] Failed to fetch ${url}: ${gltfResp.statusText}`);
  const gltf = (await gltfResp.json()) as GltfRoot;

  // ---- 2. Загружаем бинарные буферы --------------------------------------------------------------------------------------------
  const rawBuffers: ArrayBuffer[] = [];
  for (const buf of gltf.buffers ?? []) {
    if (!buf.uri) {
      rawBuffers.push(new ArrayBuffer(0));
      continue;
    }
    onProgress?.(`Loading buffer: ${buf.uri}`);
    const resp = await fetch(base + buf.uri);
    if (!resp.ok) throw new Error(`[GLTFLoader] Failed to fetch buffer ${buf.uri}`);
    rawBuffers.push(await resp.arrayBuffer());
  }

  // ---- 3. Определяем, какие изображения нужны (только albedo) ----------------------------------
  const materials = gltf.materials ?? [];
  const gltfTextures = gltf.textures ?? [];
  const gltfImages = gltf.images ?? [];

  // Отображение из индекса текстуры glTF → индекс загруженной текстуры (без дублей)
  const gltfTexIdxToLoadedIdx = new Map<number, number>();
  const imagesToLoad: number[] = []; // индексы источников изображений glTF в порядке загрузки

  for (const mat of materials) {
    const albedo = mat.pbrMetallicRoughness?.baseColorTexture;
    if (albedo === undefined) continue;

    const gltfTexIdx = albedo.index;
    if (gltfTexIdxToLoadedIdx.has(gltfTexIdx)) continue;

    const srcIdx = gltfTextures[gltfTexIdx]?.source ?? -1;
    if (srcIdx === -1) continue;

    gltfTexIdxToLoadedIdx.set(gltfTexIdx, imagesToLoad.length);
    imagesToLoad.push(srcIdx);
  }

  // ---- 4. Загружаем и обрабатываем изображения ------------------------------------------------------------------------------------
  const loadedTextures: LoadedTexture[] = [];

  for (let i = 0; i < imagesToLoad.length; i++) {
    const srcIdx = imagesToLoad[i]!;
    const gltfImg = gltfImages[srcIdx]!;
    onProgress?.(`Loading image ${i + 1}/${imagesToLoad.length}: ${gltfImg.uri ?? "(embedded)"}`);

    let blob: Blob;
    if (gltfImg.uri) {
      const resp = await fetch(base + gltfImg.uri);
      if (!resp.ok) throw new Error(`[GLTFLoader] Failed to fetch image ${gltfImg.uri}`);
      blob = await resp.blob();
    } else if (gltfImg.bufferView !== undefined) {
      const bv = gltf.bufferViews![gltfImg.bufferView]!;
      const buf = rawBuffers[bv.buffer]!;
      const slice = buf.slice(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength);
      blob = new Blob([slice], { type: gltfImg.mimeType ?? "image/jpeg" });
    } else {
      console.warn("[GLTFLoader] Image has no uri or bufferView, skipping.");
      continue;
    }

    const { data: rgba, width, height } = await decodeImage(blob);

    // Строим пирамиду мипов
    const { mips } = buildMipPyramid(rgba, width, height);

    // Упаковываем в KTX2
    const ktx2Bytes = packKtx2(mips, width, height);
    const parsed = parseKTX2(ktx2Bytes);

    loadedTextures.push({
      id: `tex-${i}`,
      ktx2Bytes,
      parsed,
      width,
      height,
    });
  }

  // ---- 5. Строим отображение материал → индекс загруженной текстуры ------------------------------------------------
  // Отображает индекс материала glTF → индекс загруженной текстуры (или -1, если нет albedo)
  const matToTexIdx: number[] = materials.map((mat) => {
    const albedo = mat.pbrMetallicRoughness?.baseColorTexture;
    if (!albedo) return -1;
    return gltfTexIdxToLoadedIdx.get(albedo.index) ?? -1;
  });

  // ---- 6. Собираем все примитивы с мировыми преобразованиями ------------------------------------------
  onProgress?.("Parsing mesh primitives…");
  const scene = gltf.scenes?.[gltf.scene ?? 0];
  const primitives: PrimitiveResult[] = [];
  const identity = mat4Identity();
  for (const rootIdx of scene?.nodes ?? []) {
    collectPrimitives(gltf, rootIdx, identity, primitives);
  }

  // ---- 7. Строим чередующиеся буферы вершин и индексов --------------------------------------------------
  const loadedMeshes: LoadedMesh[] = [];

  for (const { meshIdx, primIdx, worldTransform } of primitives) {
    const mesh = gltf.meshes![meshIdx]!;
    const prim = mesh.primitives[primIdx]!;

    const posAccIdx  = prim.attributes["POSITION"];
    const normAccIdx = prim.attributes["NORMAL"];
    const uvAccIdx   = prim.attributes["TEXCOORD_0"];

    if (posAccIdx === undefined) continue; // пропускаем примитивы без позиций

    const positions = readAccessor(gltf, posAccIdx, rawBuffers) as Float32Array;
    const normals   = normAccIdx !== undefined
      ? readAccessor(gltf, normAccIdx, rawBuffers) as Float32Array
      : new Float32Array(positions.length).fill(0);
    const uvs       = uvAccIdx !== undefined
      ? readAccessor(gltf, uvAccIdx, rawBuffers) as Float32Array
      : new Float32Array((positions.length / 3) * 2).fill(0);

    const vertexCount = positions.length / 3;
    const vertexData = new Float32Array(vertexCount * 8);
    for (let v = 0; v < vertexCount; v++) {
      const vo = v * 8;
      vertexData[vo]     = positions[v * 3]!;
      vertexData[vo + 1] = positions[v * 3 + 1]!;
      vertexData[vo + 2] = positions[v * 3 + 2]!;
      vertexData[vo + 3] = normals[v * 3]!;
      vertexData[vo + 4] = normals[v * 3 + 1]!;
      vertexData[vo + 5] = normals[v * 3 + 2]!;
      vertexData[vo + 6] = uvs[v * 2]!;
      vertexData[vo + 7] = uvs[v * 2 + 1]!;
    }

    // Буфер индексов
    let indexData: Uint16Array | Uint32Array;
    let indexCount: number;
    if (prim.indices !== undefined) {
      const raw = readAccessor(gltf, prim.indices, rawBuffers);
      if (raw instanceof Uint32Array) {
        indexData = raw;
      } else {
        // Конвертируем Uint16 - проверяем, не превышают ли значения 65535
        indexData = new Uint16Array(raw);
      }
      indexCount = indexData.length;
    } else {
      // Без индексов: синтезируем последовательные индексы
      indexCount = vertexCount;
      if (vertexCount > 65535) {
        indexData = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) indexData[i] = i;
      } else {
        indexData = new Uint16Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) indexData[i] = i;
      }
    }

    const textureIndex = prim.material !== undefined ? (matToTexIdx[prim.material] ?? -1) : -1;
    const name = `${mesh.name ?? `mesh-${meshIdx}`}-${primIdx}`;

    loadedMeshes.push({
      name,
      vertexData,
      indexData,
      indexCount,
      textureIndex,
      boundingSphere: computeBoundingSphere(positions, worldTransform),
      worldTransform,
    });
  }

  onProgress?.(`Loaded ${loadedMeshes.length} primitives, ${loadedTextures.length} textures.`);
  return { meshes: loadedMeshes, textures: loadedTextures };
}

// ---- GLB binary loader -----------------------------------------------------------------------------------------------------------------------

/**
 * Загружает сцену из GLB-буфера (drag & drop / FileReader).
 * Конвертирует встроенные JPEG/PNG в KTX2 с мипирамидой, пригодной для
 * TextureStreamingManager.registerTexture().
 */
export async function loadGLTFFromBuffer(
  buffer: ArrayBuffer,
  onProgress?: LoadProgress,
): Promise<LoadedScene> {
  // ---- 1. Парсим GLB-контейнер -------------------------------------------------------------------------------------------
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error("[GLTFLoader] Not a valid GLB file");

  const jsonLen  = dv.getUint32(12, true);
  const jsonType = dv.getUint32(16, true);
  if (jsonType !== 0x4E4F534A) throw new Error("[GLTFLoader] Expected JSON chunk first in GLB");

  const gltf = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLen))
  ) as GltfRoot;

  let binBuf: ArrayBuffer = new ArrayBuffer(0);
  const binOff = 20 + jsonLen;
  if (binOff + 8 <= buffer.byteLength) {
    const bLen  = dv.getUint32(binOff,     true);
    const bType = dv.getUint32(binOff + 4, true);
    if (bType === 0x004E4942) {
      binBuf = buffer.slice(binOff + 8, binOff + 8 + bLen);
    }
  }

  // В GLB все bufferView ссылаются на buffer 0 — передаём его как единственный элемент
  const rawBuffers: ArrayBuffer[] = [binBuf];

  const materials    = gltf.materials   ?? [];
  const gltfTextures = gltf.textures    ?? [];
  const gltfImages   = gltf.images      ?? [];

  // ---- 2. Определяем нужные изображения (только albedo) -------------------------------------------------------------------
  const gltfTexIdxToLoadedIdx = new Map<number, number>();
  const imagesToLoad: number[] = [];

  for (const mat of materials) {
    const albedo = mat.pbrMetallicRoughness?.baseColorTexture;
    if (albedo === undefined) continue;
    const gltfTexIdx = albedo.index;
    if (gltfTexIdxToLoadedIdx.has(gltfTexIdx)) continue;
    const srcIdx = gltfTextures[gltfTexIdx]?.source ?? -1;
    if (srcIdx === -1) continue;
    gltfTexIdxToLoadedIdx.set(gltfTexIdx, imagesToLoad.length);
    imagesToLoad.push(srcIdx);
  }

  // ---- 3. Декодируем изображения → KTX2 -----------------------------------------------------------------------------------
  const loadedTextures: LoadedTexture[] = [];

  for (let i = 0; i < imagesToLoad.length; i++) {
    const srcIdx  = imagesToLoad[i]!;
    const gltfImg = gltfImages[srcIdx]!;
    onProgress?.(`Обработка текстуры ${i + 1}/${imagesToLoad.length}…`);

    let blob: Blob | undefined;
    if (gltfImg.bufferView !== undefined) {
      const bv    = gltf.bufferViews![gltfImg.bufferView]!;
      const slice = binBuf.slice(bv.byteOffset ?? 0, (bv.byteOffset ?? 0) + bv.byteLength);
      blob = new Blob([slice], { type: gltfImg.mimeType ?? "image/jpeg" });
    } else if (gltfImg.uri?.startsWith("data:")) {
      const comma = gltfImg.uri.indexOf(",");
      if (comma !== -1) {
        const mime = gltfImg.uri.slice(5, gltfImg.uri.indexOf(";")) || "image/jpeg";
        const bin  = atob(gltfImg.uri.slice(comma + 1));
        const bytes = new Uint8Array(bin.length);
        for (let j = 0; j < bin.length; j++) bytes[j] = bin.charCodeAt(j);
        blob = new Blob([bytes], { type: mime });
      }
    }

    if (!blob) continue;

    const { data: rgba, width, height } = await decodeImage(blob);
    const { mips } = buildMipPyramid(rgba, width, height);
    const ktx2Bytes = packKtx2(mips, width, height);
    const parsed    = parseKTX2(ktx2Bytes);

    loadedTextures.push({ id: `tex-${i}`, ktx2Bytes, parsed, width, height });
  }

  // ---- 4. Маппинг материал → текстура -------------------------------------------------------------------------------------
  const matToTexIdx: number[] = materials.map((mat) => {
    const albedo = mat.pbrMetallicRoughness?.baseColorTexture;
    if (!albedo) return -1;
    return gltfTexIdxToLoadedIdx.get(albedo.index) ?? -1;
  });

  // ---- 5. Обход иерархии узлов → примитивы -------------------------------------------------------------------------------
  onProgress?.("Парсинг мешей…");
  const scene = gltf.scenes?.[gltf.scene ?? 0];
  const primitives: PrimitiveResult[] = [];
  const identity = mat4Identity();
  for (const rootIdx of scene?.nodes ?? []) {
    collectPrimitives(gltf, rootIdx, identity, primitives);
  }

  // ---- 6. Формируем LoadedMesh[] ------------------------------------------------------------------------------------------
  const loadedMeshes: LoadedMesh[] = [];

  for (const { meshIdx, primIdx, worldTransform } of primitives) {
    const mesh = gltf.meshes![meshIdx]!;
    const prim = mesh.primitives[primIdx]!;

    const posAccIdx  = prim.attributes["POSITION"];
    if (posAccIdx === undefined) continue;

    const normAccIdx = prim.attributes["NORMAL"];
    const uvAccIdx   = prim.attributes["TEXCOORD_0"];

    const positions = readAccessor(gltf, posAccIdx, rawBuffers) as Float32Array;
    const normals   = normAccIdx !== undefined
      ? readAccessor(gltf, normAccIdx, rawBuffers) as Float32Array
      : new Float32Array(positions.length).fill(0);
    const uvs       = uvAccIdx !== undefined
      ? readAccessor(gltf, uvAccIdx, rawBuffers) as Float32Array
      : new Float32Array((positions.length / 3) * 2).fill(0);

    const vertexCount = positions.length / 3;
    const vertexData  = new Float32Array(vertexCount * 8);
    for (let v = 0; v < vertexCount; v++) {
      const vo = v * 8;
      vertexData[vo]     = positions[v * 3]!;
      vertexData[vo + 1] = positions[v * 3 + 1]!;
      vertexData[vo + 2] = positions[v * 3 + 2]!;
      vertexData[vo + 3] = normals[v * 3]!;
      vertexData[vo + 4] = normals[v * 3 + 1]!;
      vertexData[vo + 5] = normals[v * 3 + 2]!;
      vertexData[vo + 6] = uvs[v * 2]!;
      vertexData[vo + 7] = uvs[v * 2 + 1]!;
    }

    let indexData: Uint16Array | Uint32Array;
    let indexCount: number;
    if (prim.indices !== undefined) {
      const raw = readAccessor(gltf, prim.indices, rawBuffers);
      indexData  = raw instanceof Uint32Array ? raw : new Uint16Array(raw);
      indexCount = indexData.length;
    } else {
      indexCount = vertexCount;
      indexData  = vertexCount > 65535
        ? new Uint32Array(vertexCount).map((_, i) => i)
        : new Uint16Array(vertexCount).map((_, i) => i);
    }

    const textureIndex = prim.material !== undefined ? (matToTexIdx[prim.material] ?? -1) : -1;
    const name = `${mesh.name ?? `mesh-${meshIdx}`}-${primIdx}`;

    loadedMeshes.push({
      name,
      vertexData,
      indexData,
      indexCount,
      textureIndex,
      boundingSphere: computeBoundingSphere(positions, worldTransform),
      worldTransform,
    });
  }

  onProgress?.(`Загружено ${loadedMeshes.length} примитивов, ${loadedTextures.length} текстур.`);
  return { meshes: loadedMeshes, textures: loadedTextures };
}
