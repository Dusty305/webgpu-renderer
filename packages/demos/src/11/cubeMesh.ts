/**
 * Генерирует меш единичного куба с нормалями и UV-координатами на грань.
 * Разметка вершины: position(vec3f) normal(vec3f) uv(vec2f) - 32 байта/вершина.
 * Возвращает 24 вершины (4 на грань) и 36 индексов (6 на грань).
 */
export function generateCubeMesh(): {
  vertices: Float32Array<ArrayBuffer>;
  indices: Uint16Array<ArrayBuffer>;
} {
  // prettier-ignore
  const FACES: Array<{ normal: [number,number,number]; verts: number[][] }> = [
    // +X (ПЧЧ при взгляде из +x: right = (0,0,-1), v0=сзади-снизу, v1=сзади-сверху, v2=спереди-сверху, v3=спереди-снизу)
    { normal: [1,0,0], verts: [[0.5,-0.5,-0.5],[0.5, 0.5,-0.5],[0.5, 0.5, 0.5],[0.5,-0.5, 0.5]] },
    // -X (ПЧЧ при взгляде из -x: right = (0,0,+1))
    { normal: [-1,0,0], verts: [[-0.5,-0.5, 0.5],[-0.5, 0.5, 0.5],[-0.5, 0.5,-0.5],[-0.5,-0.5,-0.5]] },
    // +Y (ПЧЧ при взгляде из +y: right = (1,0,0))
    { normal: [0,1,0], verts: [[-0.5, 0.5,-0.5],[-0.5, 0.5, 0.5],[0.5, 0.5, 0.5],[0.5, 0.5,-0.5]] },
    // -Y (ПЧЧ при взгляде из -y: right = (1,0,0))
    { normal: [0,-1,0], verts: [[-0.5,-0.5, 0.5],[-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,-0.5, 0.5]] },
    // +Z (без изменений - уже корректно)
    { normal: [0,0,1], verts: [[-0.5,-0.5, 0.5],[0.5,-0.5, 0.5],[0.5, 0.5, 0.5],[-0.5, 0.5, 0.5]] },
    // -Z (без изменений - уже корректно)
    { normal: [0,0,-1], verts: [[0.5,-0.5,-0.5],[-0.5,-0.5,-0.5],[-0.5, 0.5,-0.5],[0.5, 0.5,-0.5]] },
  ];

  const UVS: [number, number][] = [[0,1],[1,1],[1,0],[0,0]];

  const vertices = new Float32Array(new ArrayBuffer(24 * 8 * 4)); // 24 вершины × 8 float
  const indices  = new Uint16Array(new ArrayBuffer(36 * 2));      // 36 индексов

  let vi = 0;
  let ii = 0;
  let baseVert = 0;

  for (const face of FACES) {
    for (let v = 0; v < 4; v++) {
      const [px, py, pz] = face.verts[v]!;
      const [nx, ny, nz] = face.normal;
      const [u, uv_v]    = UVS[v]!;
      vertices[vi++] = px!; vertices[vi++] = py!; vertices[vi++] = pz!;
      vertices[vi++] = nx;  vertices[vi++] = ny;  vertices[vi++] = nz;
      vertices[vi++] = u;   vertices[vi++] = uv_v;
    }
    // Два треугольника на грань: 0,1,2  0,2,3
    indices[ii++] = baseVert;
    indices[ii++] = baseVert + 1;
    indices[ii++] = baseVert + 2;
    indices[ii++] = baseVert;
    indices[ii++] = baseVert + 2;
    indices[ii++] = baseVert + 3;
    baseVert += 4;
  }

  return { vertices, indices };
}

/** Строит матрицу трансляции 4×4 в column-major порядке. */
export function mat4Translation(x: number, y: number, z: number): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ]);
}
