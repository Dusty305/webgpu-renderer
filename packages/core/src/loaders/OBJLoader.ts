import type { GeometryDescriptor, MaterialDescriptor } from "@webgpu-streaming/gpu-types";

export interface ParsedMesh {
  geometry: GeometryDescriptor;
  material: MaterialDescriptor;
  name?: string;
}

export interface ParsedModel {
  meshes: ParsedMesh[];
}

// -- Разбор MTL --

interface MtlMaterial {
  name: string;
  baseColor: string;
  diffuseMap?: string;
  roughness?: number;
  opacity: number;
}

function parseMTL(text: string): Map<string, MtlMaterial> {
  const materials = new Map<string, MtlMaterial>();
  let current: MtlMaterial | null = null;

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const parts = line.split(/\s+/);
    const cmd = parts[0]!;

    if (cmd === "newmtl") {
      current = { name: parts[1] ?? "default", baseColor: "#808080", opacity: 1 };
      materials.set(current.name, current);
    } else if (current) {
      if (cmd === "Kd" && parts.length >= 4) {
        const r = Math.round(parseFloat(parts[1]!) * 255);
        const g = Math.round(parseFloat(parts[2]!) * 255);
        const b = Math.round(parseFloat(parts[3]!) * 255);
        current.baseColor = `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
      } else if (cmd === "map_Kd" && parts[1]) {
        current.diffuseMap = parts[1];
      } else if (cmd === "Ns" && parts[1]) {
        const ns = parseFloat(parts[1]);
        current.roughness = Math.max(0, Math.min(1, 1 - ns / 1000));
      } else if (cmd === "d" && parts[1]) {
        current.opacity = parseFloat(parts[1]);
      } else if (cmd === "Tr" && parts[1]) {
        current.opacity = 1 - parseFloat(parts[1]);
      }
    }
  }

  return materials;
}

function mtlToMaterialDescriptor(mtl: MtlMaterial | undefined): MaterialDescriptor {
  if (!mtl) return { baseColor: "#808080" };
  return { baseColor: mtl.baseColor };
}

// -- Вычисление плоских нормалей --

function computeFlatNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const normals = new Float32Array(positions.length);

  for (let i = 0; i < indices.length; i += 3) {
    const i0 = indices[i]! * 3;
    const i1 = indices[i + 1]! * 3;
    const i2 = indices[i + 2]! * 3;

    const ax = positions[i1]! - positions[i0]!;
    const ay = positions[i1 + 1]! - positions[i0 + 1]!;
    const az = positions[i1 + 2]! - positions[i0 + 2]!;
    const bx = positions[i2]! - positions[i0]!;
    const by = positions[i2 + 1]! - positions[i0 + 1]!;
    const bz = positions[i2 + 2]! - positions[i0 + 2]!;

    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;

    for (let v = 0; v < 3; v++) {
      const vi = indices[i + v]! * 3;
      normals[vi]     = nx / len;
      normals[vi + 1] = ny / len;
      normals[vi + 2] = nz / len;
    }
  }

  return normals;
}

// -- Состояние группы --

interface FaceVertex { p: number; t: number; n: number; }

interface GroupBuffer {
  name: string;
  materialName: string;
  faces: FaceVertex[][];
}

// -- Разбор OBJ --

/**
 * Разобрать текст OBJ. Возвращает один ParsedMesh на каждый объект/группу.
 * @param objText  - полное содержимое OBJ-файла
 * @param mtlText  - необязательное содержимое MTL-файла
 */
export function parseOBJ(objText: string, mtlText?: string): ParsedModel {
  const rawPositions: number[] = [];
  const rawUVs: number[]      = [];
  const rawNormals: number[]  = [];

  const materials = mtlText ? parseMTL(mtlText) : new Map<string, MtlMaterial>();

  const groups: GroupBuffer[] = [];
  let currentGroup: GroupBuffer = { name: "default", materialName: "", faces: [] };
  groups.push(currentGroup);

  function resolveIndex(raw: number, count: number): number {
    return raw < 0 ? count + raw : raw - 1; // OBJ использует индексацию с 1
  }

  for (const rawLine of objText.split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;

    const parts = line.split(/\s+/);
    const cmd = parts[0]!;

    if (cmd === "v") {
      rawPositions.push(parseFloat(parts[1]!), parseFloat(parts[2]!), parseFloat(parts[3]!));
    } else if (cmd === "vt") {
      rawUVs.push(parseFloat(parts[1]!), parseFloat(parts[2]!));
    } else if (cmd === "vn") {
      rawNormals.push(parseFloat(parts[1]!), parseFloat(parts[2]!), parseFloat(parts[3]!));
    } else if (cmd === "o" || cmd === "g") {
      const name = parts[1] ?? "default";
      // Начинаем новую группу только если в текущей уже есть грани
      if (currentGroup.faces.length > 0) {
        currentGroup = { name, materialName: currentGroup.materialName, faces: [] };
        groups.push(currentGroup);
      } else {
        currentGroup.name = name;
      }
    } else if (cmd === "usemtl") {
      const matName = parts[1] ?? "";
      if (currentGroup.faces.length > 0) {
        currentGroup = { name: currentGroup.name, materialName: matName, faces: [] };
        groups.push(currentGroup);
      } else {
        currentGroup.materialName = matName;
      }
    } else if (cmd === "f") {
      // Собираем все ссылки на вершины в этой грани
      const faceVerts: FaceVertex[] = [];
      for (let i = 1; i < parts.length; i++) {
        const token = parts[i]!;
        const indices = token.split("/");
        const p = resolveIndex(parseInt(indices[0]!, 10), rawPositions.length / 3);
        const t = indices[1] && indices[1] !== "" ? resolveIndex(parseInt(indices[1], 10), rawUVs.length / 2) : -1;
        const n = indices[2] && indices[2] !== "" ? resolveIndex(parseInt(indices[2], 10), rawNormals.length / 3) : -1;
        faceVerts.push({ p, t, n });
      }
      // Веерная триангуляция: (0,1,2), (0,2,3), (0,3,4), ...
      for (let i = 1; i < faceVerts.length - 1; i++) {
        currentGroup.faces.push([faceVerts[0]!, faceVerts[i]!, faceVerts[i + 1]!]);
      }
    }
    // mtllib: загрузка и передача mtlText - ответственность вызывающего кода
    // s: группы сглаживания - игнорируются
  }

  const posCount = rawPositions.length / 3;

  // Создаём один ParsedMesh для каждой группы, в которой есть грани
  const meshes: ParsedMesh[] = [];

  for (const group of groups) {
    if (group.faces.length === 0) continue;

    const vertexMap = new Map<string, number>();
    const outPositions: number[] = [];
    const outNormals: number[]   = [];
    const outUVs: number[]       = [];
    const outIndices: number[]   = [];
    const hasUVs     = group.faces.some(f => f.some(v => v.t >= 0));
    const hasNormals = group.faces.some(f => f.some(v => v.n >= 0));

    for (const tri of group.faces) {
      for (const fv of tri) {
        const key = `${fv.p}/${fv.t}/${fv.n}`;
        let idx = vertexMap.get(key);
        if (idx === undefined) {
          idx = outPositions.length / 3;
          vertexMap.set(key, idx);

          const pi = fv.p * 3;
          outPositions.push(
            rawPositions[pi]     ?? 0,
            rawPositions[pi + 1] ?? 0,
            rawPositions[pi + 2] ?? 0,
          );

          if (hasNormals) {
            if (fv.n >= 0) {
              const ni = fv.n * 3;
              outNormals.push(rawNormals[ni] ?? 0, rawNormals[ni + 1] ?? 0, rawNormals[ni + 2] ?? 0);
            } else {
              outNormals.push(0, 1, 0);
            }
          }

          if (hasUVs) {
            if (fv.t >= 0) {
              const ti = fv.t * 2;
              outUVs.push(rawUVs[ti] ?? 0, rawUVs[ti + 1] ?? 0);
            } else {
              outUVs.push(0, 0);
            }
          }
        }
        outIndices.push(idx);
      }
    }

    const positions = new Float32Array(outPositions);
    const indices   = outPositions.length / 3 > 65535
      ? new Uint32Array(outIndices)
      : new Uint16Array(outIndices);

    let normals: Float32Array | undefined;
    if (hasNormals) {
      normals = new Float32Array(outNormals);
    } else {
      // Вычисляем плоские нормали с помощью Uint32Array для арифметики
      normals = computeFlatNormals(positions, new Uint32Array(outIndices));
    }

    const geometry: GeometryDescriptor = {
      positions,
      normals,
      indices,
    };
    if (hasUVs) {
      geometry.uvs = new Float32Array(outUVs);
    }

    meshes.push({
      geometry,
      material: mtlToMaterialDescriptor(materials.get(group.materialName)),
      name: group.name,
    });
  }

  // Если ничего не распознано, возвращаем пустую модель вместо аварийного завершения
  if (meshes.length === 0) {
    return { meshes: [] };
  }

  return { meshes };
}

/**
 * Разобрать OBJ из ArrayBuffer (предполагается кодировка UTF-8).
 */
export function parseOBJBuffer(buffer: ArrayBuffer, mtlText?: string): ParsedModel {
  const text = new TextDecoder().decode(buffer);
  return parseOBJ(text, mtlText);
}
