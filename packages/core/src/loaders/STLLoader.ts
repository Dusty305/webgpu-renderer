import type { GeometryDescriptor, MaterialDescriptor } from "@webgpu-streaming/gpu-types";
import type { ParsedModel } from "./OBJLoader.js";

const DEFAULT_STL_MATERIAL: MaterialDescriptor = { baseColor: "#808080" };

// -- Бинарный STL --

function parseSTLBinary(buffer: ArrayBuffer): ParsedModel {
  const view = new DataView(buffer);
  const triangleCount = view.getUint32(80, true);

  const positions = new Float32Array(triangleCount * 9);
  const normals   = new Float32Array(triangleCount * 9);

  for (let i = 0; i < triangleCount; i++) {
    const base = 84 + i * 50;

    const nx = view.getFloat32(base,      true);
    const ny = view.getFloat32(base + 4,  true);
    const nz = view.getFloat32(base + 8,  true);

    for (let v = 0; v < 3; v++) {
      const vo = base + 12 + v * 12;
      const pi = i * 9 + v * 3;
      positions[pi]     = view.getFloat32(vo,      true);
      positions[pi + 1] = view.getFloat32(vo + 4,  true);
      positions[pi + 2] = view.getFloat32(vo + 8,  true);
      normals[pi]       = nx;
      normals[pi + 1]   = ny;
      normals[pi + 2]   = nz;
    }
  }

  const vertexCount = triangleCount * 3;
  const indices = vertexCount > 65535
    ? new Uint32Array(vertexCount).map((_, i) => i)
    : new Uint16Array(vertexCount).map((_, i) => i);

  const geometry: GeometryDescriptor = { positions, normals, indices };
  return { meshes: [{ geometry, material: DEFAULT_STL_MATERIAL, name: "stl_mesh" }] };
}

// ---- ASCII STL ----------------------------------------------------------------------------------------------------------------------------------

function parseSTLASCII(buffer: ArrayBuffer): ParsedModel {
  const text  = new TextDecoder().decode(buffer);
  const lines = text.split("\n");

  const positions: number[] = [];
  const normals:   number[] = [];
  let currentNormal: [number, number, number] = [0, 0, 0];
  let verticesInFacet = 0;

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (line.startsWith("facet normal")) {
      const parts = line.split(/\s+/);
      currentNormal = [
        parseFloat(parts[2]!),
        parseFloat(parts[3]!),
        parseFloat(parts[4]!),
      ];
      verticesInFacet = 0;
    } else if (line.startsWith("vertex")) {
      const parts = line.split(/\s+/);
      positions.push(parseFloat(parts[1]!), parseFloat(parts[2]!), parseFloat(parts[3]!));
      normals.push(currentNormal[0], currentNormal[1], currentNormal[2]);
      verticesInFacet++;
    }
  }

  const vertexCount = positions.length / 3;
  const indices = vertexCount > 65535
    ? new Uint32Array(vertexCount).map((_, i) => i)
    : new Uint16Array(vertexCount).map((_, i) => i);

  const geometry: GeometryDescriptor = {
    positions: new Float32Array(positions),
    normals:   new Float32Array(normals),
    indices,
  };
  return { meshes: [{ geometry, material: DEFAULT_STL_MATERIAL, name: "stl_mesh" }] };
}

// ---- Автоопределение и разбор ----------------------------------------------------------------------------------------------------

/**
 * Разобрать STL-файл (бинарный или ASCII) из ArrayBuffer.
 * Формат определяется автоматически по заголовку и ожидаемому бинарному размеру.
 */
export function parseSTL(buffer: ArrayBuffer): ParsedModel {
  if (buffer.byteLength === 0) {
    return { meshes: [{ geometry: { positions: new Float32Array(0) }, material: DEFAULT_STL_MATERIAL, name: "stl_mesh" }] };
  }

  if (buffer.byteLength >= 84) {
    // Проверяем ASCII-заголовок "solid"
    const headerBytes = new Uint8Array(buffer, 0, Math.min(80, buffer.byteLength));
    const headerStr = String.fromCharCode(...headerBytes.slice(0, 6));

    if (headerStr.startsWith("solid ") || headerStr === "solid\n" || headerStr.startsWith("solid\r")) {
      // Проверяем соответствие бинарному размеру, чтобы избежать ложного срабатывания
      const view = new DataView(buffer);
      const triangleCount = view.getUint32(80, true);
      const expectedBinarySize = 84 + triangleCount * 50;
      if (buffer.byteLength === expectedBinarySize && triangleCount > 0) {
        return parseSTLBinary(buffer);
      }
      return parseSTLASCII(buffer);
    }
  }

  return parseSTLBinary(buffer);
}
