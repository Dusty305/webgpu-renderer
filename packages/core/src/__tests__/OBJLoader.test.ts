import { describe, it, expect } from "vitest";
import { parseOBJ } from "../loaders/OBJLoader.js";

const TRIANGLE_OBJ = `
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
`;

const FULL_OBJ = `
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
vn 0 0 1
vn 0 0 1
vt 0 0
vt 1 0
vt 0 1
f 1/1/1 2/2/2 3/3/3
`;

describe("parseOBJ", () => {
  it("разбирает минимальный OBJ только с позициями", () => {
    const model = parseOBJ(TRIANGLE_OBJ);
    expect(model.meshes).toHaveLength(1);
    const mesh = model.meshes[0]!;
    expect(mesh.geometry.positions).toHaveLength(9); // 3 вершины x 3 компонента
    expect(mesh.geometry.indices).toBeDefined();
    expect(mesh.geometry.indices!.length).toBe(3);
  });

  it("преобразует 1-базированные OBJ-индексы в 0-базированные", () => {
    const model = parseOBJ(TRIANGLE_OBJ);
    const indices = Array.from(model.meshes[0]!.geometry.indices!);
    expect(indices).toEqual([0, 1, 2]);
  });

  it("разбирает позиции, нормали и UV из полного OBJ", () => {
    const model = parseOBJ(FULL_OBJ);
    const mesh = model.meshes[0]!;
    expect(mesh.geometry.normals).toBeDefined();
    expect(mesh.geometry.uvs).toBeDefined();
    expect(mesh.geometry.normals!.length).toBe(9);
    expect(mesh.geometry.uvs!.length).toBe(6);
  });

  it("триангулирует четырёхугольные грани в два треугольника", () => {
    const obj = `
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
`;
    const model = parseOBJ(obj);
    const mesh = model.meshes[0]!;
    expect(mesh.geometry.indices!.length).toBe(6); // 2 треугольника = 6 индексов
  });

  it("обрабатывает отрицательные индексы", () => {
    // -1 означает последнюю вершину, -2 - предпоследнюю
    const obj = `
v 0 0 0
v 1 0 0
v 0 1 0
f -3 -2 -1
`;
    const model = parseOBJ(obj);
    const mesh = model.meshes[0]!;
    // Должно разобраться без ошибок и дать 3 индекса
    expect(mesh.geometry.indices!.length).toBe(3);
    // Позиции должны ссылаться на 3 вершины
    expect(mesh.geometry.positions).toHaveLength(9);
  });

  it("вычисляет плоские нормали когда они не заданы", () => {
    const model = parseOBJ(TRIANGLE_OBJ);
    const normals = model.meshes[0]!.geometry.normals!;
    expect(normals).toBeDefined();
    // Треугольник в плоскости XY должен иметь нормаль, направленную по +Z
    // нормали всех 3 вершин должны быть [0, 0, 1]
    for (let i = 0; i < 3; i++) {
      expect(normals[i * 3]!).toBeCloseTo(0, 5);
      expect(normals[i * 3 + 1]!).toBeCloseTo(0, 5);
      expect(normals[i * 3 + 2]!).toBeCloseTo(1, 5);
    }
  });

  it("разделяет меши по ключевым словам объект/группа", () => {
    const obj = `
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
v 1 0 1
v 0 1 1
o ObjectA
f 1 2 3
o ObjectB
f 4 5 6
`;
    const model = parseOBJ(obj);
    expect(model.meshes.length).toBeGreaterThanOrEqual(2);
    const names = model.meshes.map(m => m.name);
    expect(names).toContain("ObjectA");
    expect(names).toContain("ObjectB");
  });

  it("назначает материал из usemtl", () => {
    const obj = `
v 0 0 0
v 1 0 0
v 0 1 0
usemtl MyMat
f 1 2 3
`;
    const mtl = `
newmtl MyMat
Kd 1.0 0.0 0.0
`;
    const model = parseOBJ(obj, mtl);
    expect(model.meshes[0]!.material.baseColor).toBe("#ff0000");
  });

  it("обрабатывает грани без UV (синтаксис с двойным слэшем)", () => {
    const obj = `
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
vn 0 0 1
vn 0 0 1
f 1//1 2//2 3//3
`;
    const model = parseOBJ(obj);
    const mesh = model.meshes[0]!;
    expect(mesh.geometry.normals).toBeDefined();
    expect(mesh.geometry.uvs).toBeUndefined();
  });

  it("возвращает пустой массив мешей для пустого OBJ", () => {
    const model = parseOBJ("# пустой файл\n");
    expect(model.meshes).toHaveLength(0);
  });
});
