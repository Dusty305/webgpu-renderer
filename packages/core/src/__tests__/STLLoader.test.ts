import { describe, it, expect } from "vitest";
import { parseSTL } from "../loaders/STLLoader.js";

/** Построить минимальный корректный бинарный STL: 80-байтный заголовок + счётчик треугольников + N треугольников. */
function makeBinarySTL(triangles: Array<{ normal: [number,number,number]; verts: [[number,number,number],[number,number,number],[number,number,number]] }>): ArrayBuffer {
  const buf = new ArrayBuffer(84 + triangles.length * 50);
  const view = new DataView(buf);
  view.setUint32(80, triangles.length, true);
  for (let i = 0; i < triangles.length; i++) {
    const base = 84 + i * 50;
    const { normal, verts } = triangles[i]!;
    view.setFloat32(base,      normal[0], true);
    view.setFloat32(base + 4,  normal[1], true);
    view.setFloat32(base + 8,  normal[2], true);
    for (let v = 0; v < 3; v++) {
      const vo = base + 12 + v * 12;
      view.setFloat32(vo,      verts[v]![0], true);
      view.setFloat32(vo + 4,  verts[v]![1], true);
      view.setFloat32(vo + 8,  verts[v]![2], true);
    }
    view.setUint16(base + 48, 0, true);
  }
  return buf;
}

const ASCII_STL = `solid tetra
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid tetra
`;

describe("parseSTL", () => {
  it("разбирает бинарный STL с 1 треугольником", () => {
    const buf = makeBinarySTL([{
      normal: [0, 0, 1],
      verts: [[0,0,0],[1,0,0],[0,1,0]],
    }]);

    const model = parseSTL(buf);
    expect(model.meshes).toHaveLength(1);
    const mesh = model.meshes[0]!;
    expect(mesh.geometry.positions).toHaveLength(9);
    expect(mesh.geometry.normals).toHaveLength(9);
    expect(mesh.geometry.indices!.length).toBe(3);
  });

  it("корректно копирует нормали в бинарном STL", () => {
    const buf = makeBinarySTL([{
      normal: [0, 0, 1],
      verts: [[0,0,0],[1,0,0],[0,1,0]],
    }]);
    const mesh = parseSTL(buf).meshes[0]!;
    // Все 3 вершины должны иметь нормаль грани
    for (let v = 0; v < 3; v++) {
      expect(mesh.geometry.normals![v * 3]!).toBeCloseTo(0);
      expect(mesh.geometry.normals![v * 3 + 1]!).toBeCloseTo(0);
      expect(mesh.geometry.normals![v * 3 + 2]!).toBeCloseTo(1);
    }
  });

  it("разбирает бинарный STL с несколькими треугольниками", () => {
    const buf = makeBinarySTL([
      { normal: [0,0,1], verts: [[0,0,0],[1,0,0],[0,1,0]] },
      { normal: [1,0,0], verts: [[0,0,0],[0,1,0],[0,0,1]] },
    ]);
    const mesh = parseSTL(buf).meshes[0]!;
    expect(mesh.geometry.positions.length).toBe(18); // 2 треугольника × 3 вершины × 3
    expect(mesh.geometry.indices!.length).toBe(6);
  });

  it("разбирает ASCII STL", () => {
    const buf = new TextEncoder().encode(ASCII_STL).buffer as ArrayBuffer;
    const model = parseSTL(buf);
    expect(model.meshes).toHaveLength(1);
    const mesh = model.meshes[0]!;
    expect(mesh.geometry.positions.length).toBe(9);
    expect(mesh.geometry.normals!.length).toBe(9);
  });

  it("выбирает ASCII-парсер когда заголовок 'solid ', но размер не совпадает с бинарным", () => {
    // ASCII STL не должен иметь ровно 84 + N*50 байт
    const buf = new TextEncoder().encode(ASCII_STL).buffer as ArrayBuffer;
    const model = parseSTL(buf);
    // Должно успешно разобраться как ASCII
    expect(model.meshes[0]!.geometry.positions.length).toBeGreaterThan(0);
  });

  it("обрабатывает пустой бинарный STL (0 треугольников)", () => {
    const buf = new ArrayBuffer(84);
    new DataView(buf).setUint32(80, 0, true);
    const model = parseSTL(buf);
    expect(model.meshes).toHaveLength(1);
    expect(model.meshes[0]!.geometry.positions.length).toBe(0);
  });

  it("использует Uint16Array для индексов когда число вершин вмещается в 16 бит", () => {
    const buf = makeBinarySTL([{
      normal: [0,0,1], verts: [[0,0,0],[1,0,0],[0,1,0]],
    }]);
    const mesh = parseSTL(buf).meshes[0]!;
    expect(mesh.geometry.indices).toBeInstanceOf(Uint16Array);
  });

  it("назначает серый материал по умолчанию", () => {
    const buf = makeBinarySTL([{
      normal: [0,0,1], verts: [[0,0,0],[1,0,0],[0,1,0]],
    }]);
    const mesh = parseSTL(buf).meshes[0]!;
    expect(mesh.material.baseColor).toBe("#808080");
  });
});
