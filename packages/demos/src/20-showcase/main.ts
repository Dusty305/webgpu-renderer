import { createRenderer, detectFormat, parseOBJBuffer, parseSTL } from "@webgpu-streaming/core";
import type { ParsedMesh } from "@webgpu-streaming/core";
import type { Renderer, MeshHandle, GeometryDescriptor, TextureHandle } from "@webgpu-streaming/gpu-types";

// ---- DOM refs ------------------------------------------------------------------------------------------------------------------------------------

const canvasEl      = document.getElementById("main-canvas")   as HTMLCanvasElement;
const dropOverlay   = document.getElementById("drop-overlay")!;
const statsPanel    = document.getElementById("stats-panel")!;
const loadingOv     = document.getElementById("loading-overlay")!;
const loadingMsg    = document.getElementById("loading-msg")!;
const loadingBar    = document.getElementById("loading-bar")!;
const webgpuError   = document.getElementById("webgpu-error")!;
const sceneSelect   = document.getElementById("scene-select")   as HTMLSelectElement;
const loadSceneBtn  = document.getElementById("load-scene-btn") as HTMLButtonElement;
const fileInput     = document.getElementById("file-input")     as HTMLInputElement;
const statsCheck    = document.getElementById("stats-check")    as HTMLInputElement;
const svFps         = document.getElementById("sv-fps")!;
const svP99         = document.getElementById("sv-p99")!;
const svMem         = document.getElementById("sv-mem")!;
const svBudget      = document.getElementById("sv-budget")!;
const svTex         = document.getElementById("sv-tex")!;
const svStream      = document.getElementById("sv-stream")!;
const ibFormat      = document.getElementById("ib-format")!;
const ibTris        = document.getElementById("ib-tris")!;
const ibMeshes      = document.getElementById("ib-meshes")!;
const ibStatus      = document.getElementById("ib-status")!;

// ---- State ------------------------------------------------------------------------------------------------------------------------------------------

let renderer: Renderer | null = null;
let currentMeshes: MeshHandle[] = [];
let currentTextures: TextureHandle[] = [];
let statsTimer: ReturnType<typeof setInterval> | null = null;

// ---- UI helpers --------------------------------------------------------------------------------------------------------------------------------

function setStatus(msg: string, isError = false): void {
  ibStatus.textContent = msg;
  ibStatus.className = "ib-val" + (isError ? " err" : "");
}

function showLoading(msg: string, pct?: number): void {
  loadingOv.classList.add("active");
  loadingMsg.textContent = msg;
  if (pct !== undefined) loadingBar.style.width = `${pct * 100}%`;
}

function hideLoading(): void {
  loadingOv.classList.remove("active");
  loadingBar.style.width = "0%";
}

function updateInfoBar(format: string, tris: number, meshCount: number): void {
  ibFormat.textContent = format;
  ibTris.textContent   = tris.toLocaleString();
  ibMeshes.textContent = String(meshCount);
}

// ---- Scene management --------------------------------------------------------------------------------------------------------------------

function clearScene(): void {
  for (const h of currentMeshes) renderer?.removeMesh(h);
  for (const t of currentTextures) t.destroy();
  currentMeshes = [];
  currentTextures = [];
}

function fitCamera(positions: Float32Array[]): void {
  if (!renderer || positions.length === 0) return;

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (const pos of positions) {
    for (let i = 0; i < pos.length; i += 3) {
      if (pos[i]! < minX) minX = pos[i]!;   if (pos[i]! > maxX) maxX = pos[i]!;
      if (pos[i+1]! < minY) minY = pos[i+1]!; if (pos[i+1]! > maxY) maxY = pos[i+1]!;
      if (pos[i+2]! < minZ) minZ = pos[i+2]!; if (pos[i+2]! > maxZ) maxZ = pos[i+2]!;
    }
  }

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  let r = 0.01;
  for (const pos of positions) {
    for (let i = 0; i < pos.length; i += 3) {
      const dx = pos[i]! - cx, dy = pos[i+1]! - cy, dz = pos[i+2]! - cz;
      const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
      if (d > r) r = d;
    }
  }

  const fovY = Math.PI / 4;
  const dist = (r / Math.tan(fovY / 2)) * 1.3;
  renderer.setCamera({
    position: [cx, cy + r * 0.25, cz + dist],
    target:   [cx, cy, cz],
    fov:      45,
    near:     Math.max(0.005, dist * 0.005),
    far:      dist * 30,
  });
}

// ---- Color utility --------------------------------------------------------------------------------------------------------------------------

function hslToHex(h: number, s: number, l: number): string {
  const hh = ((h % 360) + 360) % 360;
  const sl = s / 100;
  const ll = l / 100;
  const k  = (n: number): number => (n + hh / 30) % 12;
  const a  = sl * Math.min(ll, 1 - ll);
  const f  = (n: number): number => ll - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const r  = Math.round(f(0) * 255);
  const g  = Math.round(f(8) * 255);
  const b  = Math.round(f(4) * 255);
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`;
}

// ---- Cube geometry --------------------------------------------------------------------------------------------------------------------------

function makeCubeGeo(cx: number, cy: number, cz: number, half: number): GeometryDescriptor {
  // prettier-ignore
  const faces: Array<{ n: [number,number,number]; pts: Array<[number,number,number]> }> = [
    { n:[ 0, 0, 1], pts:[[-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]] },
    { n:[ 0, 0,-1], pts:[[ 1,-1,-1],[-1,-1,-1],[-1, 1,-1],[ 1, 1,-1]] },
    { n:[-1, 0, 0], pts:[[-1,-1,-1],[-1,-1, 1],[-1, 1, 1],[-1, 1,-1]] },
    { n:[ 1, 0, 0], pts:[[ 1,-1, 1],[ 1,-1,-1],[ 1, 1,-1],[ 1, 1, 1]] },
    { n:[ 0, 1, 0], pts:[[-1, 1, 1],[ 1, 1, 1],[ 1, 1,-1],[-1, 1,-1]] },
    { n:[ 0,-1, 0], pts:[[-1,-1,-1],[ 1,-1,-1],[ 1,-1, 1],[-1,-1, 1]] },
  ];
  const fuvs: [number,number][] = [[0,1],[1,1],[1,0],[0,0]];

  const pos: number[] = [], nrm: number[] = [], uv: number[] = [], idx: number[] = [];
  let base = 0;
  for (const face of faces) {
    for (let i = 0; i < 4; i++) {
      const p = face.pts[i]!;
      pos.push(cx + p[0]*half, cy + p[1]*half, cz + p[2]*half);
      nrm.push(face.n[0], face.n[1], face.n[2]);
      uv.push(fuvs[i]![0], fuvs[i]![1]);
    }
    idx.push(base, base+1, base+2, base, base+2, base+3);
    base += 4;
  }
  return {
    positions: new Float32Array(pos),
    normals:   new Float32Array(nrm),
    uvs:       new Float32Array(uv),
    indices:   new Uint16Array(idx),
  };
}

// ---- Procedural grid scene ----------------------------------------------------------------------------------------------------------

function loadProceduralScene(count: number): void {
  if (!renderer) return;
  clearScene();

  const cols    = Math.ceil(Math.sqrt(count));
  const rows    = Math.ceil(count / cols);
  const spacing = 2.4;
  const half    = 0.85;
  const offX    = -(cols - 1) * spacing / 2;
  const offZ    = -(rows - 1) * spacing / 2;
  const allPos: Float32Array[] = [];
  let tris = 0;

  for (let i = 0; i < count; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = offX + col * spacing;
    const z = offZ + row * spacing;
    const geo = makeCubeGeo(x, 0, z, half);
    const color = hslToHex((i / count) * 360, 62, 52);
    currentMeshes.push(renderer.addMesh(geo, { baseColor: color }));
    allPos.push(geo.positions);
    tris += 12;
  }

  fitCamera(allPos);
  updateInfoBar("Procedural", tris, count);
  setStatus("Готово");
}

// ---- GLB парсер с поддержкой текстур --------------------------------------------------------------------------------------

interface GltfDoc {
  meshes?:      Array<{ primitives: Array<GltfPrimitive> }>;
  accessors?:   GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?:     Array<{ byteLength: number; uri?: string }>;
  materials?:   Array<{ pbrMetallicRoughness?: { baseColorTexture?: { index: number } } }>;
  textures?:    Array<{ source?: number }>;
  images?:      Array<{ bufferView?: number; mimeType?: string; uri?: string }>;
}
interface GltfPrimitive  { attributes: Record<string, number>; indices?: number; material?: number }
interface GltfAccessor   { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string }
interface GltfBufferView { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }

interface GLBPrimResult {
  positions: Float32Array;
  normals?:  Float32Array;
  uvs?:      Float32Array;
  indices?:  Uint16Array | Uint32Array;
  imgIdx:    number | null; // индекс в массиве images glTF (null = нет текстуры)
}

interface GLBSceneResult {
  primitives: GLBPrimResult[];
  // raw bytes каждого изображения (null = не удалось извлечь)
  images: Array<{ data: ArrayBuffer; mimeType: string } | null>;
}

function glbCompBytes(ct: number): number {
  if (ct === 5120 || ct === 5121) return 1;
  if (ct === 5122 || ct === 5123) return 2;
  return 4;
}

function extractGLBScene(buffer: ArrayBuffer): GLBSceneResult {
  const dv = new DataView(buffer);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error("Not a valid GLB file");

  const jsonLen  = dv.getUint32(12, true);
  const jsonType = dv.getUint32(16, true);
  if (jsonType !== 0x4E4F534A) throw new Error("Expected JSON chunk first");

  const gltf = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 20, jsonLen))
  ) as GltfDoc;

  let binBuf: ArrayBuffer | undefined;
  const binOff = 20 + jsonLen;
  if (binOff + 8 <= buffer.byteLength) {
    const bLen  = dv.getUint32(binOff,     true);
    const bType = dv.getUint32(binOff + 4, true);
    if (bType === 0x004E4942) {
      binBuf = buffer.slice(binOff + 8, binOff + 8 + bLen);
    }
  }

  const accessors   = gltf.accessors   ?? [];
  const bufferViews = gltf.bufferViews ?? [];
  const materials   = gltf.materials   ?? [];
  const gltfTextures = gltf.textures   ?? [];
  const gltfImages  = gltf.images      ?? [];

  // ---- Извлечение изображений из бинарного чанка -------------------------------------------------------------------------
  const images: GLBSceneResult["images"] = gltfImages.map(img => {
    if (img.bufferView != null) {
      const bv = bufferViews[img.bufferView];
      if (!bv || !binBuf) return null;
      const start = bv.byteOffset ?? 0;
      return {
        data: binBuf.slice(start, start + bv.byteLength),
        mimeType: img.mimeType ?? "image/jpeg",
      };
    }
    if (img.uri?.startsWith("data:")) {
      const comma = img.uri.indexOf(",");
      if (comma === -1) return null;
      const mime = img.uri.slice(5, img.uri.indexOf(";")) || "image/jpeg";
      const b64  = img.uri.slice(comma + 1);
      const bin  = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return { data: bytes.buffer as ArrayBuffer, mimeType: mime };
    }
    return null; // внешний URI не поддерживается при drag-and-drop
  });

  // ---- Вспомогательные функции чтения аксессоров -------------------------------------------------------------------------
  function readVec3(accIdx: number): Float32Array | undefined {
    const acc = accessors[accIdx];
    if (!acc || acc.bufferView == null || acc.type !== "VEC3" || acc.componentType !== 5126) return undefined;
    const bv     = bufferViews[acc.bufferView]!;
    if (!binBuf) return undefined;
    const start  = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const stride = bv.byteStride ?? 12;
    if (stride === 12) return new Float32Array(binBuf.slice(start, start + acc.count * 12));
    const out = new Float32Array(acc.count * 3);
    for (let i = 0; i < acc.count; i++) {
      const s = start + i * stride;
      const v = new Float32Array(binBuf, s, 3);
      out[i*3] = v[0]!; out[i*3+1] = v[1]!; out[i*3+2] = v[2]!;
    }
    return out;
  }

  function readVec2(accIdx: number): Float32Array | undefined {
    const acc = accessors[accIdx];
    if (!acc || acc.bufferView == null || acc.type !== "VEC2" || acc.componentType !== 5126) return undefined;
    const bv     = bufferViews[acc.bufferView]!;
    if (!binBuf) return undefined;
    const start  = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const stride = bv.byteStride ?? 8;
    if (stride === 8) return new Float32Array(binBuf.slice(start, start + acc.count * 8));
    const out = new Float32Array(acc.count * 2);
    for (let i = 0; i < acc.count; i++) {
      const s = start + i * stride;
      const v = new Float32Array(binBuf, s, 2);
      out[i*2] = v[0]!; out[i*2+1] = v[1]!;
    }
    return out;
  }

  function readIndices(accIdx: number): Uint16Array | Uint32Array | undefined {
    const acc = accessors[accIdx];
    if (!acc || acc.bufferView == null || acc.type !== "SCALAR") return undefined;
    const bv    = bufferViews[acc.bufferView]!;
    if (!binBuf) return undefined;
    const start = (bv.byteOffset ?? 0) + (acc.byteOffset ?? 0);
    const bytes = glbCompBytes(acc.componentType);
    const sl    = binBuf.slice(start, start + acc.count * bytes);
    if (acc.componentType === 5123) return new Uint16Array(sl);
    if (acc.componentType === 5125) return new Uint32Array(sl);
    return undefined;
  }

  // ---- Сборка примитивов -----------------------------------------------------------------------------------------
  const primitives: GLBPrimResult[] = [];
  for (const mesh of gltf.meshes ?? []) {
    for (const prim of mesh.primitives) {
      try {
        const posIdx = prim.attributes["POSITION"];
        if (posIdx == null) continue;
        const positions = readVec3(posIdx);
        if (!positions || positions.length === 0) continue;

        const normIdx = prim.attributes["NORMAL"];
        const uvIdx   = prim.attributes["TEXCOORD_0"];
        const normals = normIdx != null ? readVec3(normIdx) : undefined;
        const uvs     = uvIdx   != null ? readVec2(uvIdx)   : undefined;
        const indices = prim.indices != null ? readIndices(prim.indices) : undefined;

        // Определяем индекс изображения через материал → текстуру → источник
        let imgIdx: number | null = null;
        if (prim.material != null) {
          const mat = materials[prim.material];
          const texIdx = mat?.pbrMetallicRoughness?.baseColorTexture?.index;
          if (texIdx != null) {
            const srcIdx = gltfTextures[texIdx]?.source;
            if (srcIdx != null) imgIdx = srcIdx;
          }
        }

        primitives.push({ positions, normals, uvs, indices, imgIdx });
      } catch {
        // пропускаем повреждённый примитив
      }
    }
  }

  return { primitives, images };
}

// ---- Load arbitrary 3D file --------------------------------------------------------------------------------------------------------

const MESH_COLORS = ["#a0a8b8", "#8898b0", "#90a0b8", "#b0a8a0", "#98b0a0"];

async function loadFile(filename: string, buffer: ArrayBuffer): Promise<void> {
  if (!renderer) return;
  clearScene();
  showLoading(`Загрузка ${filename}…`);

  try {
    const ext    = filename.split(".").pop()?.toLowerCase() ?? "";
    const bufFmt = detectFormat(buffer);
    const format = bufFmt !== "unknown" ? bufFmt : detectFormat(filename);

    // ---- GLB: обрабатываем отдельно, чтобы загрузить текстуры -------------------------------------------------------
    if (format === "glb") {
      const { primitives, images } = extractGLBScene(buffer);
      if (primitives.length === 0) throw new Error("В GLB-файле не найдена геометрия");

      // Загружаем изображения как GPU-текстуры (дедупликация по индексу)
      const texByImgIdx = new Map<number, TextureHandle>();
      const validImgs   = images.filter(img => img !== null);
      if (validImgs.length > 0) showLoading(`Загрузка текстур… 0/${validImgs.length}`, 0);

      for (let i = 0; i < images.length; i++) {
        const img = images[i];
        if (!img) continue;
        try {
          const handle = await renderer.loadTexture(img.data);
          texByImgIdx.set(i, handle);
          currentTextures.push(handle);
        } catch (e) {
          console.warn(`[showcase] Текстура ${i} не загружена:`, e);
        }
        showLoading(
          `Загрузка текстур… ${texByImgIdx.size}/${validImgs.length}`,
          texByImgIdx.size / Math.max(validImgs.length, 1),
        );
      }

      const allPos: Float32Array[] = [];
      let tris = 0;
      let colorIdx = 0;
      for (const prim of primitives) {
        const texHandle = prim.imgIdx != null ? texByImgIdx.get(prim.imgIdx) : undefined;
        const geo: GeometryDescriptor = {
          positions: prim.positions,
          normals:   prim.normals,
          uvs:       prim.uvs,
          indices:   prim.indices,
        };
        const material = texHandle
          ? { baseColor: texHandle }
          : { baseColor: MESH_COLORS[colorIdx % MESH_COLORS.length]! };
        currentMeshes.push(renderer.addMesh(geo, material));
        allPos.push(prim.positions);
        tris += Math.floor((prim.indices?.length ?? prim.positions.length / 3) / 3);
        colorIdx++;
      }

      fitCamera(allPos);
      hideLoading();
      const texInfo = texByImgIdx.size > 0 ? ` · ${texByImgIdx.size} текстур` : "";
      updateInfoBar(`GLB${texInfo}`, tris, primitives.length);
      setStatus("Загружено");
      return;
    }

    // ---- OBJ / STL ------------------------------------------------------------------------------------------------
    let geos: GeometryDescriptor[] = [];
    let fmtName = "";

    if (format === "obj") {
      geos    = parseOBJBuffer(buffer).meshes.map((m: ParsedMesh) => m.geometry);
      fmtName = "OBJ";
    } else if (format === "stl-binary" || format === "stl-ascii") {
      geos    = parseSTL(buffer).meshes.map((m: ParsedMesh) => m.geometry);
      fmtName = "STL";
    } else if (format === "gltf" || ext === "gltf") {
      const text = new TextDecoder().decode(new Uint8Array(buffer));
      const json = JSON.parse(text) as GltfDoc;
      const hasExternal = (json.buffers ?? []).some(b => b.uri && !b.uri.startsWith("data:"));
      if (hasExternal) throw new Error("glTF с внешними зависимостями не поддерживается - используйте .glb");
      throw new Error("glTF без встроенной геометрии - используйте формат .glb");
    } else {
      throw new Error(`Неподдерживаемый формат: .${ext} - поддерживаются: .glb .obj .stl`);
    }

    if (geos.length === 0) throw new Error("В файле не найдена геометрия");

    const allPos: Float32Array[] = [];
    let tris = 0;

    for (let i = 0; i < geos.length; i++) {
      const geo   = geos[i]!;
      const color = MESH_COLORS[i % MESH_COLORS.length]!;
      currentMeshes.push(renderer.addMesh(geo, { baseColor: color }));
      allPos.push(geo.positions);
      const idxCount = geo.indices?.length ?? (geo.positions.length / 3);
      tris += Math.floor(idxCount / 3);
    }

    fitCamera(allPos);
    hideLoading();
    updateInfoBar(fmtName, tris, geos.length);
    setStatus("Загружено");
  } catch (err) {
    hideLoading();
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(msg, true);
  }
}

// ---- Stats update ----------------------------------------------------------------------------------------------------------------------------

function startStats(): void {
  if (statsTimer !== null) clearInterval(statsTimer);
  statsTimer = setInterval(() => {
    if (!renderer) return;
    const s = renderer.getStats();
    svFps.textContent    = s.fps.toFixed(0);
    svP99.textContent    = `${s.frameTimeP99Ms.toFixed(1)} ms`;
    svMem.textContent    = s.memoryUsedMB > 0   ? `${s.memoryUsedMB.toFixed(1)} MB`   : "-";
    svBudget.textContent = s.memoryBudgetMB > 0 ? `${s.memoryBudgetMB.toFixed(0)} MB` : "-";
    svTex.textContent    = s.texturesTotal > 0  ? String(s.texturesLoaded)             : "-";
    svStream.textContent = s.texturesTotal > 0 ? `${s.texturesLoaded}/${s.texturesTotal}` : "-";
  }, 500);
}

// ---- Init --------------------------------------------------------------------------------------------------------------------------------------------

async function main(): Promise<void> {
  if (!navigator.gpu) {
    webgpuError.classList.add("active");
    setStatus("WebGPU не поддерживается", true);
    return;
  }

  try {
    showLoading("Инициализация WebGPU…");
    renderer = await createRenderer({ canvas: canvasEl, memoryBudget: 256 });
    hideLoading();
    loadSceneBtn.disabled = false;
    setStatus("Готово");
    startStats();

    // Default scene
    loadProceduralScene(200);
  } catch (err) {
    hideLoading();
    const msg = err instanceof Error ? err.message : String(err);
    webgpuError.querySelector("h2")!.textContent = "Ошибка инициализации";
    webgpuError.querySelector("p")!.textContent  = msg;
    webgpuError.classList.add("active");
    setStatus(msg, true);
    return;
  }

  // Load scene button
  loadSceneBtn.addEventListener("click", () => {
    loadSceneBtn.disabled = true;
    loadProceduralScene(parseInt(sceneSelect.value, 10));
    loadSceneBtn.disabled = false;
  });

  // File picker
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    if (!file) return;
    const buf = await file.arrayBuffer();
    await loadFile(file.name, buf);
    fileInput.value = "";
  });

  // Stats overlay toggle
  statsCheck.addEventListener("change", () => {
    statsPanel.classList.toggle("visible", statsCheck.checked);
  });

  // Drag-and-drop - whole canvas area is the drop zone
  const canvasArea = document.getElementById("canvas-area")!;
  canvasArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropOverlay.classList.add("active");
  });
  canvasArea.addEventListener("dragleave", (e) => {
    if (!canvasArea.contains(e.relatedTarget as Node | null)) {
      dropOverlay.classList.remove("active");
    }
  });
  canvasArea.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropOverlay.classList.remove("active");
    const file = e.dataTransfer?.files[0];
    if (!file) return;
    const buf = await file.arrayBuffer();
    await loadFile(file.name, buf);
  });
}

main().catch((err) => {
  console.error("[showcase]", err);
  setStatus(err instanceof Error ? err.message : String(err), true);
});
