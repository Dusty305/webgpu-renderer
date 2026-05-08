import { describe, it, expect } from "vitest";
import { detectFormat } from "../loaders/FormatDetector.js";

describe("detectFormat", () => {
  describe("определение по расширению", () => {
    it("определяет .obj", () => {
      expect(detectFormat("model.obj")).toBe("obj");
    });

    it("определяет .stl как stl-binary (перепроверяется после загрузки)", () => {
      expect(detectFormat("model.stl")).toBe("stl-binary");
    });

    it("определяет .glb", () => {
      expect(detectFormat("model.glb")).toBe("glb");
    });

    it("определяет .gltf", () => {
      expect(detectFormat("scene.gltf")).toBe("gltf");
    });

    it("возвращает unknown для нераспознанного расширения", () => {
      expect(detectFormat("model.fbx")).toBe("unknown");
    });

    it("регистронезависимо определяет расширение", () => {
      expect(detectFormat("MODEL.OBJ")).toBe("obj");
      expect(detectFormat("model.GLB")).toBe("glb");
    });
  });

  describe("определение по магическим байтам из ArrayBuffer", () => {
    it("определяет бинарный glTF по магическим байтам", () => {
      const buf = new ArrayBuffer(12);
      const view = new DataView(buf);
      view.setUint32(0, 0x46546c67, true); // "glTF" LE
      expect(detectFormat(buf)).toBe("glb");
    });

    it("определяет ASCII STL по заголовку 'solid '", () => {
      const text = "solid mypart\nfacet normal 0 0 1\n  outer loop\n    vertex 0 0 0\n    vertex 1 0 0\n    vertex 0 1 0\n  endloop\nendfacet\nendsolid mypart\n";
      const buf = new TextEncoder().encode(text).buffer;
      expect(detectFormat(buf)).toBe("stl-ascii");
    });

    it("определяет бинарный STL когда заголовок соответствует ожидаемому бинарному размеру", () => {
      // 1 треугольник: 80-байтный заголовок + 4-байтный счётчик + 50-байтный треугольник = 134 байта
      const buf = new ArrayBuffer(134);
      const view = new DataView(buf);
      // Записываем "solid \0\0..." в заголовок (6 байт), затем счётчик треугольников = 1
      const header = new Uint8Array(buf);
      header[0] = 115; header[1] = 111; header[2] = 108; header[3] = 105; header[4] = 100; header[5] = 32; // "solid "
      view.setUint32(80, 1, true); // 1 треугольник
      expect(detectFormat(buf)).toBe("stl-binary");
    });

    it("возвращает unknown для пустого буфера", () => {
      expect(detectFormat(new ArrayBuffer(0))).toBe("unknown");
    });
  });
});
