/**
 * Внутренние утилиты геометрии и математики, разделяемые между api.ts и WebGPUCanvasPublic.ts.
 * Не являются частью публичного API.
 */

import type { GeometryDescriptor, MeshHandle, StreamingStats } from "@webgpu-streaming/gpu-types";
import type { SceneGraph } from "./SceneGraph.js";

/** Минимальный интерфейс рендер-прохода, отслеживающего сопоставление меш-узел. */
interface RenderPassWithMesh {
  removeObject(nodeId: string): void;
}

// ---- вспомогательные функции mat4 ------------------------------------------------------------------------------------------

export function mat4Identity(): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]);
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 4; j++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + j]! * b[i * 4 + k]!;
      out[i * 4 + j] = s;
    }
  }
  return out;
}

function mat4Translation(x: number, y: number, z: number): Float32Array {
  const m = mat4Identity();
  m[12] = x; m[13] = y; m[14] = z;
  return m;
}

function mat4Scale(x: number, y: number, z: number): Float32Array {
  const m = mat4Identity();
  m[0] = x; m[5] = y; m[10] = z;
  return m;
}

function mat4RotX(rad: number): Float32Array {
  const c = Math.cos(rad), s = Math.sin(rad);
  const m = mat4Identity();
  m[5] = c; m[6] = s; m[9] = -s; m[10] = c;
  return m;
}

function mat4RotY(rad: number): Float32Array {
  const c = Math.cos(rad), s = Math.sin(rad);
  const m = mat4Identity();
  m[0] = c; m[2] = -s; m[8] = s; m[10] = c;
  return m;
}

function mat4RotZ(rad: number): Float32Array {
  const c = Math.cos(rad), s = Math.sin(rad);
  const m = mat4Identity();
  m[0] = c; m[1] = s; m[4] = -s; m[5] = c;
  return m;
}

/** Построить матрицу TRS из позиции/поворота (углы Эйлера)/масштаба. */
export function buildTRS(
  pos: Float32Array, rot: Float32Array, scale: Float32Array
): Float32Array {
  const t = mat4Translation(pos[0]!, pos[1]!, pos[2]!);
  const rx = mat4RotX(rot[0]!);
  const ry = mat4RotY(rot[1]!);
  const rz = mat4RotZ(rot[2]!);
  const s = mat4Scale(scale[0]!, scale[1]!, scale[2]!);
  const r = mat4Multiply(mat4Multiply(rz, ry), rx);
  return mat4Multiply(t, mat4Multiply(r, s));
}

// ---- вспомогательные функции геометрии --------------------------------------------------------------------------------

/**
 * Построить единичный куб с чередующимися данными position(vec3)/normal(vec3)/uv(vec2).
 * 24 вершины (4 на грань × 6 граней), 36 индексов.
 */
export function makeCubeGeometry(): { vertices: Float32Array; indices: Uint16Array } {
  // Каждая строка: px py pz  nx ny nz  u v
  // prettier-ignore
  const faceData: Array<{ n: [number, number, number]; pts: Array<[number, number, number]> }> = [
    { n: [ 0,  0,  1], pts: [[-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]] }, // +Z
    { n: [ 0,  0, -1], pts: [[ 1,-1,-1],[-1,-1,-1],[-1, 1,-1],[ 1, 1,-1]] }, // -Z
    { n: [-1,  0,  0], pts: [[-1,-1,-1],[-1,-1, 1],[-1, 1, 1],[-1, 1,-1]] }, // -X
    { n: [ 1,  0,  0], pts: [[ 1,-1, 1],[ 1,-1,-1],[ 1, 1,-1],[ 1, 1, 1]] }, // +X
    { n: [ 0,  1,  0], pts: [[-1, 1, 1],[ 1, 1, 1],[ 1, 1,-1],[-1, 1,-1]] }, // +Y
    { n: [ 0, -1,  0], pts: [[-1,-1,-1],[ 1,-1,-1],[ 1,-1, 1],[-1,-1, 1]] }, // -Y
  ];
  const uvs: Array<[number, number]> = [[0,1],[1,1],[1,0],[0,0]];

  const verts: number[] = [];
  const idxs: number[] = [];
  let base = 0;

  for (const face of faceData) {
    for (let i = 0; i < 4; i++) {
      const p = face.pts[i]!;
      const uv = uvs[i]!;
      verts.push(p[0], p[1], p[2], face.n[0], face.n[1], face.n[2], uv[0], uv[1]);
    }
    idxs.push(base, base+1, base+2, base, base+2, base+3);
    base += 4;
  }

  return { vertices: new Float32Array(verts), indices: new Uint16Array(idxs) };
}

/**
 * Преобразовать GeometryDescriptor в чередующийся формат вершин, ожидаемый BasicRenderPass:
 * position(vec3f), normal(vec3f), uv(vec2f) = 8 вещественных чисел на вершину.
 */
export function buildInterleavedVertices(geo: GeometryDescriptor): Float32Array {
  const n = geo.positions.length / 3;
  const out = new Float32Array(n * 8);
  for (let i = 0; i < n; i++) {
    out[i*8+0] = geo.positions[i*3+0] ?? 0;
    out[i*8+1] = geo.positions[i*3+1] ?? 0;
    out[i*8+2] = geo.positions[i*3+2] ?? 0;
    out[i*8+3] = geo.normals?.[i*3+0] ?? 0;
    out[i*8+4] = geo.normals?.[i*3+1] ?? 1;
    out[i*8+5] = geo.normals?.[i*3+2] ?? 0;
    out[i*8+6] = geo.uvs?.[i*2+0] ?? 0;
    out[i*8+7] = geo.uvs?.[i*2+1] ?? 0;
  }
  return out;
}

/** Преобразовать индексы в Uint16Array (предполагается < 65536 вершин).
 *  Если индексы не предоставлены, используется число вершин для генерации последовательного списка.
 *  Результат всегда дополнен до кратного 4 байтам для GPUQueue.writeBuffer. */
export function toUint16Indices(indices?: Uint16Array | Uint32Array, vertexCount?: number): Uint16Array {
  let arr: Uint16Array;
  if (!indices) {
    const count = vertexCount ?? 3;
    arr = new Uint16Array(count);
    for (let i = 0; i < count; i++) arr[i] = i;
  } else if (indices instanceof Uint16Array) {
    arr = indices;
  } else {
    arr = new Uint16Array(indices);
  }
  // Дополнить до кратного 4 байтам (2 элемента = 4 байта)
  if (arr.length % 2 !== 0) {
    const padded = new Uint16Array(arr.length + 1);
    padded.set(arr);
    return padded;
  }
  return arr;
}

/** Вычислить ограничивающую сферу [cx, cy, cz, radius] для массива позиций. */
export function computeBoundingSphere(positions: Float32Array): Float32Array {
  const n = positions.length / 3;
  if (n === 0) return new Float32Array([0, 0, 0, 1]);

  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) {
    cx += positions[i*3+0]!;
    cy += positions[i*3+1]!;
    cz += positions[i*3+2]!;
  }
  cx /= n; cy /= n; cz /= n;

  let r = 0;
  for (let i = 0; i < n; i++) {
    const dx = positions[i*3+0]! - cx;
    const dy = positions[i*3+1]! - cy;
    const dz = positions[i*3+2]! - cz;
    r = Math.max(r, Math.sqrt(dx*dx + dy*dy + dz*dz));
  }

  return new Float32Array([cx, cy, cz, Math.max(r, 0.01)]);
}

// ---- MeshHandleImpl ------------------------------------------------------------------------------------------------------------------------

/** Внутренняя реализация MeshHandle. */
export class MeshHandleImpl implements MeshHandle {
  private readonly _pos   = new Float32Array(3);
  private readonly _rot   = new Float32Array(3);
  private readonly _scale = new Float32Array([1, 1, 1]);
  private _visible = true;
  private _destroyed = false;

  constructor(
    readonly id: string,
    private readonly _scene: SceneGraph,
    private readonly _pass: RenderPassWithMesh,
  ) {}

  get visible(): boolean { return this._visible; }

  setTransform(matrix: Float32Array): void {
    this._scene.updateTransform(this.id, matrix);
  }

  setPosition(x: number, y: number, z: number): void {
    this._pos[0] = x; this._pos[1] = y; this._pos[2] = z;
    this._scene.updateTransform(this.id, buildTRS(this._pos, this._rot, this._scale));
  }

  setRotation(x: number, y: number, z: number): void {
    this._rot[0] = x; this._rot[1] = y; this._rot[2] = z;
    this._scene.updateTransform(this.id, buildTRS(this._pos, this._rot, this._scale));
  }

  setScale(x: number, y: number, z: number): void {
    this._scale[0] = x; this._scale[1] = y; this._scale[2] = z;
    this._scene.updateTransform(this.id, buildTRS(this._pos, this._rot, this._scale));
  }

  setVisible(v: boolean): void {
    this._visible = v;
    this._scene.setVisible(this.id, v);
  }

  destroy(): void {
    if (this._destroyed) return;
    this._destroyed = true;
    this._pass.removeObject(this.id);
    this._scene.removeNode(this.id);
  }
}

// ---- Трекер статистики кадров ----------------------------------------------------------------------------------------------------

const STATS_WINDOW = 120; // хранить последние N значений времени кадра

/** Трекер для вычисления fps / p99 времени кадра. */
export class FrameStatsTracker {
  private readonly _times: number[] = [];
  private _lastTime = 0;

  record(now: number): void {
    if (this._lastTime > 0) {
      this._times.push(now - this._lastTime);
      if (this._times.length > STATS_WINDOW) this._times.shift();
    }
    this._lastTime = now;
  }

  fps(): number {
    if (this._times.length < 2) return 0;
    const total = this._times.reduce((a, b) => a + b, 0);
    return total > 0 ? (this._times.length * 1000) / total : 0;
  }

  p99Ms(): number {
    if (this._times.length === 0) return 0;
    const sorted = [...this._times].sort((a, b) => a - b);
    const idx = Math.floor(sorted.length * 0.99);
    return sorted[Math.min(idx, sorted.length - 1)] ?? 0;
  }
}

/** Построить снимок StreamingStats из TextureStreamingManager и трекера кадров. */
export function buildStats(
  sm: { budgetTracker: { totalUsed: number; budget: number } | null; entries: ReadonlyMap<string, { residentMip: number }> } | null,
  budgetMB: number,
  tracker: FrameStatsTracker,
): StreamingStats {
  const used = sm?.budgetTracker?.totalUsed ?? 0;
  const total = sm?.budgetTracker?.budget ?? (budgetMB * 1024 * 1024);
  const entries = sm ? Array.from(sm.entries.values()) : [];

  const dist: Record<number, number> = {};
  for (const e of entries) {
    dist[e.residentMip] = (dist[e.residentMip] ?? 0) + 1;
  }

  return {
    memoryUsedMB: used / (1024 * 1024),
    memoryBudgetMB: total / (1024 * 1024),
    texturesLoaded: entries.filter(e => e.residentMip < 999).length,
    texturesTotal: entries.length,
    residentMipDistribution: dist,
    fps: tracker.fps(),
    frameTimeP99Ms: tracker.p99Ms(),
  };
}
