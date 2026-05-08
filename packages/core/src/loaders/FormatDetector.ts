export type ModelFormat = "gltf" | "glb" | "obj" | "stl-ascii" | "stl-binary" | "unknown";

/** Определить формат 3D-модели по URL-строке или ArrayBuffer. */
export function detectFormat(source: string | ArrayBuffer): ModelFormat {
  if (typeof source === "string") {
    const ext = source.split(".").pop()?.toLowerCase();
    if (ext === "glb") return "glb";
    if (ext === "gltf") return "gltf";
    if (ext === "obj") return "obj";
    if (ext === "stl") return "stl-binary"; // перепроверяется после загрузки
  }
  if (source instanceof ArrayBuffer) {
    if (source.byteLength < 4) return "unknown";

    // Магические байты бинарного glTF: "glTF" = 0x46546C67 LE
    const view = new DataView(source);
    if (view.getUint32(0, true) === 0x46546C67) return "glb";

    const headBytes = new Uint8Array(source, 0, Math.min(512, source.byteLength));
    const headerStr = String.fromCharCode(...headBytes.slice(0, 6));

    // ASCII STL начинается с "solid "
    if (headerStr.startsWith("solid ") || headerStr.startsWith("solid\n")) {
      // Проверяем соответствие ожидаемому бинарному размеру, чтобы не спутать с бинарным STL, у которого "solid" в заголовке
      if (source.byteLength >= 84) {
        const triangleCount = view.getUint32(80, true);
        const expectedBinarySize = 84 + triangleCount * 50;
        if (source.byteLength === expectedBinarySize && triangleCount > 0) {
          return "stl-binary";
        }
      }
      return "stl-ascii";
    }

    // OBJ: текстовый файл со строками "v ", "vn ", "vt ", "f " или "#"
    // Проверяем ДО запасного варианта с бинарным STL, чтобы не спутать OBJ с STL.
    const headText = String.fromCharCode(...headBytes);
    if (/^(#|v |vn |vt |f )/m.test(headText)) return "obj";

    // Бинарный STL: 80-байтный заголовок + 4-байтный счётчик + N*50 треугольников.
    // Считаем бинарным STL только если буфер содержит непечатаемые байты в начале
    // (признак настоящего бинарного файла), чтобы не спутать с текстовыми файлами.
    if (source.byteLength > 84) {
      const hasBinaryContent = headBytes.slice(0, 80).some(b => b < 9 || (b > 13 && b < 32));
      if (hasBinaryContent) return "stl-binary";
    }
  }
  return "unknown";
}
