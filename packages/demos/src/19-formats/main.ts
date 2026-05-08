/**
 * Демо 19 - Многоформатный загрузчик: OBJ / STL
 *
 * Три панели:
 *  1. OBJ  - загружает встроенную строку OBJ куба, принимает файловый ввод .obj
 *  2. STL  - загружает встроенный бинарный тетраэдр, принимает файловый ввод .stl
 *  3. Drop - принимает любой поддерживаемый формат через перетаскивание или файловый ввод
 *
 * После разбора вычисляем ограничивающую сферу и подгоняем камеру под неё,
 * чтобы модели с любой системой координат и масштабом всегда были видны.
 */

import { createRenderer, detectFormat, parseOBJBuffer, parseSTL } from "@webgpu-streaming/core";
import type { Renderer, GeometryDescriptor } from "@webgpu-streaming/core";

// ---- Встроенный OBJ: единичный куб ------------------------------------------------------------------------------------------------------

// Явные нормали на грань, обход CCW, внешние нормали верифицированы.
const CUBE_OBJ = `# Unit cube
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1
vn  0  0 -1
vn  0  0  1
vn  1  0  0
vn -1  0  0
vn  0  1  0
vn  0 -1  0
# Front (z=-1)
f 1//1 4//1 3//1
f 1//1 3//1 2//1
# Back (z=+1)
f 5//2 6//2 7//2
f 5//2 7//2 8//2
# Right (x=+1)
f 2//3 3//3 7//3
f 2//3 7//3 6//3
# Left (x=-1)
f 1//4 5//4 8//4
f 1//4 8//4 4//4
# Top (y=+1)
f 4//5 8//5 7//5
f 4//5 7//5 3//5
# Bottom (y=-1)
f 1//6 2//6 6//6
f 1//6 6//6 5//6
`;

// ---- Встроенный STL: бинарный тетраэдр ----------------------------------------------------------------------------------

function makeTetrahedronSTL(): ArrayBuffer {
  const H = 1.6329931618554521;
  const verts: Array<[number,number,number]> = [
    [ 0,  H,  0        ],
    [-1, -H/3, 1.1547  ],
    [ 1, -H/3, 1.1547  ],
    [ 0, -H/3, -2*1.1547/2 ],
  ];
  const tris: Array<[number,number,number]> = [
    [0,1,2], [0,2,3], [0,3,1], [1,3,2],
  ];

  const buf  = new ArrayBuffer(84 + tris.length * 50);
  const view = new DataView(buf);
  view.setUint32(80, tris.length, true);

  tris.forEach(([ai, bi, ci], i) => {
    const base = 84 + i * 50;
    const a = verts[ai]!, b = verts[bi]!, c = verts[ci]!;
    const ex = b[0]-a[0], ey = b[1]-a[1], ez = b[2]-a[2];
    const fx = c[0]-a[0], fy = c[1]-a[1], fz = c[2]-a[2];
    const nx = ey*fz - ez*fy, ny = ez*fx - ex*fz, nz = ex*fy - ey*fx;
    const len = Math.sqrt(nx*nx + ny*ny + nz*nz) || 1;
    view.setFloat32(base,      nx/len, true);
    view.setFloat32(base+4,    ny/len, true);
    view.setFloat32(base+8,    nz/len, true);
    for (const [vi, v] of [[0,a],[1,b],[2,c]] as [number,[number,number,number]][]) {
      const vo = base + 12 + vi * 12;
      view.setFloat32(vo,    v[0], true);
      view.setFloat32(vo+4,  v[1], true);
      view.setFloat32(vo+8,  v[2], true);
    }
    view.setUint16(base + 48, 0, true);
  });

  return buf;
}

// ---- Ограничивающая сфера ----------------------------------------------------------------------------------------------------------------------

interface Sphere { cx: number; cy: number; cz: number; radius: number; }

function boundingSphere(geometries: GeometryDescriptor[]): Sphere {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const g of geometries) {
    const p = g.positions;
    for (let i = 0; i < p.length; i += 3) {
      minX = Math.min(minX, p[i]!);   maxX = Math.max(maxX, p[i]!);
      minY = Math.min(minY, p[i+1]!); maxY = Math.max(maxY, p[i+1]!);
      minZ = Math.min(minZ, p[i+2]!); maxZ = Math.max(maxZ, p[i+2]!);
    }
  }

  const cx = (minX+maxX)/2, cy = (minY+maxY)/2, cz = (minZ+maxZ)/2;
  let r = 0;
  for (const g of geometries) {
    const p = g.positions;
    for (let i = 0; i < p.length; i += 3) {
      const dx = p[i]!-cx, dy = p[i+1]!-cy, dz = p[i+2]!-cz;
      r = Math.max(r, Math.sqrt(dx*dx+dy*dy+dz*dz));
    }
  }

  return { cx, cy, cz, radius: Math.max(r, 0.001) };
}

// ---- Вспомогательные функции --------------------------------------------------------------------------------------------------------------------------------------

function setStatus(el: HTMLElement, msg: string, isError = false): void {
  el.textContent = msg;
  el.className = isError ? "status error" : "status";
}

async function initRenderer(canvas: HTMLCanvasElement): Promise<Renderer> {
  return createRenderer({ canvas, memoryBudget: 256 });
}

/** Разбирает ArrayBuffer, добавляет все меши, подгоняет камеру под ограничивающую сферу. */
async function loadIntoRenderer(renderer: Renderer, data: ArrayBuffer, statusEl: HTMLElement): Promise<void> {
  try {
    const format = detectFormat(data);
    let model;

    if (format === "obj") {
      model = parseOBJBuffer(data);
    } else if (format === "stl-binary" || format === "stl-ascii") {
      model = parseSTL(data);
    } else {
      throw new Error(`Unrecognised format - drop an .obj or .stl file`);
    }

    if (model.meshes.length === 0) {
      setStatus(statusEl, "File parsed but contained no geometry", true);
      return;
    }

    for (const mesh of model.meshes) {
      renderer.addMesh(mesh.geometry, mesh.material);
    }

    // Подгоняем камеру под ограничивающую сферу, чтобы модель любого размера была всегда видна.
    const { cx, cy, cz, radius } = boundingSphere(model.meshes.map(m => m.geometry));
    const fovY = Math.PI / 4; // 45°
    const dist = radius / Math.tan(fovY / 2) * 1.2; // отступ 20%
    renderer.setCamera({
      position: [cx, cy + radius * 0.3, cz + dist],
      target:   [cx, cy, cz],
      fov:      45,
      near:     dist * 0.01,
      far:      dist * 20,
    });

    const kb = (data.byteLength / 1024).toFixed(0);
    const tris = model.meshes.reduce((s, m) => s + (m.geometry.indices?.length ?? 0) / 3, 0);
    setStatus(statusEl, `Loaded ${kb} KB · ${tris.toFixed(0)} triangles · r=${radius.toFixed(1)}`);
  } catch (err) {
    setStatus(statusEl, `Error: ${err instanceof Error ? err.message : String(err)}`, true);
  }
}

// ---- Привязка файлов и перетаскивания ----------------------------------------------------------------------------------------------------------------

function wireFileInput(
  fileInput: HTMLInputElement,
  canvas: HTMLCanvasElement,
  getRenderer: () => Renderer | null,
  statusEl: HTMLElement,
): void {
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const buf = await file.arrayBuffer();
    const renderer = getRenderer();
    if (renderer) await loadIntoRenderer(renderer, buf, statusEl);
    fileInput.value = "";
  });

  canvas.addEventListener("dragover", (e) => {
    e.preventDefault();
    canvas.classList.add("drag-over");
  });
  canvas.addEventListener("dragleave", () => canvas.classList.remove("drag-over"));
  canvas.addEventListener("drop", async (e) => {
    e.preventDefault();
    canvas.classList.remove("drag-over");
    const file = e.dataTransfer?.files[0];
    if (!file) return;
    setStatus(statusEl, `Loading ${file.name}…`);
    const buf = await file.arrayBuffer();
    const renderer = getRenderer();
    if (renderer) await loadIntoRenderer(renderer, buf, statusEl);
  });
}

// ---- Главная функция --------------------------------------------------------------------------------------------------------------------------------------------

async function main(): Promise<void> {
  const objCanvas  = document.getElementById("c-obj")  as HTMLCanvasElement;
  const stlCanvas  = document.getElementById("c-stl")  as HTMLCanvasElement;
  const dropCanvas = document.getElementById("c-gltf") as HTMLCanvasElement;

  const objStatus  = document.getElementById("st-obj")!;
  const stlStatus  = document.getElementById("st-stl")!;
  const dropStatus = document.getElementById("st-gltf")!;

  const fileObj  = document.getElementById("file-obj")  as HTMLInputElement;
  const fileStl  = document.getElementById("file-stl")  as HTMLInputElement;
  const fileDrop = document.getElementById("file-gltf") as HTMLInputElement;

  let objRenderer:  Renderer | null = null;
  let stlRenderer:  Renderer | null = null;
  let dropRenderer: Renderer | null = null;

  // ---- Панель OBJ ----------------------------------------------------------------------------------------------------------------------------

  try {
    objRenderer = await initRenderer(objCanvas);
    const buf = new TextEncoder().encode(CUBE_OBJ).buffer as ArrayBuffer;
    await loadIntoRenderer(objRenderer, buf, objStatus);
  } catch (err) {
    setStatus(objStatus, `Init error: ${err instanceof Error ? err.message : String(err)}`, true);
  }

  wireFileInput(fileObj, objCanvas, () => objRenderer, objStatus);

  // ---- Панель STL ----------------------------------------------------------------------------------------------------------------------------

  try {
    stlRenderer = await initRenderer(stlCanvas);
    const buf = makeTetrahedronSTL();
    await loadIntoRenderer(stlRenderer, buf, stlStatus);
  } catch (err) {
    setStatus(stlStatus, `Init error: ${err instanceof Error ? err.message : String(err)}`, true);
  }

  wireFileInput(fileStl, stlCanvas, () => stlRenderer, stlStatus);

  // ---- Панель Drop --------------------------------------------------------------------------------------------------------------------------

  try {
    dropRenderer = await initRenderer(dropCanvas);
    setStatus(dropStatus, "Drop or pick any .obj or .stl file to load");
  } catch (err) {
    setStatus(dropStatus, `Init error: ${err instanceof Error ? err.message : String(err)}`, true);
  }

  wireFileInput(fileDrop, dropCanvas, () => dropRenderer, dropStatus);
}

main().catch(console.error);
