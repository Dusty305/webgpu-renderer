/**
 * Демо 08 - LRU-вытеснение
 *
 * 64 текстурированных квада в сетке 8×8, уходящей вдаль.
 * Жёсткий лимит памяти вынуждает вытеснять по LRU детальные мип-уровни.
 * Орбитальная камера: ближние квады чёткие, дальние - размытые.
 *
 * У каждого квада своя текстура 256×256 с уникальным сплошным цветом.
 * Резидентный мип-уровень отображается на тепловой карте внизу.
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import type { IRenderPass, RenderPassInitContext, FrameContext } from "@webgpu-streaming/gpu-types";
import { LRUEvictionPolicy, BudgetTracker, computeDesiredMip } from "@webgpu-streaming/texture-streaming";
import { createOverlay, FpsTracker } from "../shared/overlay.js";

const overlay = createOverlay("08 - LRU Eviction");
const fps = new FpsTracker();
overlay.set("Status", "Initializing…");

// ---- Конфигурация ----------------------------------------------------------------------------------------------------------------------------

const GRID   = 8;          // 8×8 = 64 квадов
const TEX    = 256;        // базовый размер текстуры
const MIPS   = 8;          // log2(256)+1
const BUDGET = 4 * 1024 * 1024; // 4 МБ - вынуждает вытеснение при 256px × 64 текстуры × 4 байта × ~несколько мипов
const FRAME_UPLOAD_BUDGET = 512 * 1024; // 512 КБ за кадр

// ---- WGSL ------------------------------------------------------------------------------------------------------------------------------------------

const VERT_WGSL = /* wgsl */ `
struct Camera {
  viewProj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> cam: Camera;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv:        vec2<f32>,
  @location(1) instId:    u32,
}

@vertex
fn main(
  @builtin(vertex_index)   vi:   u32,
  @builtin(instance_index) inst: u32,
) -> VSOut {
  let col = inst % 8u;
  let row = inst / 8u;

  // Размещаем квады в сетке на плоскости XZ, уходящей от камеры.
  let x = (f32(col) - 3.5) * 2.5;
  let z = -f32(row) * 2.5;

  var corners = array<vec3<f32>, 6>(
    vec3(x-1, 0, z-1), vec3(x+1, 0, z-1), vec3(x-1, 0, z+1),
    vec3(x-1, 0, z+1), vec3(x+1, 0, z-1), vec3(x+1, 0, z+1),
  );
  var uvs = array<vec2<f32>, 6>(
    vec2(0,0), vec2(1,0), vec2(0,1),
    vec2(0,1), vec2(1,0), vec2(1,1),
  );

  var o: VSOut;
  o.pos    = cam.viewProj * vec4<f32>(corners[vi], 1.0);
  o.uv     = uvs[vi];
  o.instId = inst;
  return o;
}
`;

const FRAG_WGSL = /* wgsl */ `
struct LodEntry { lodMin: f32, _pad0: f32, _pad1: f32, _pad2: f32 }
@group(0) @binding(1) var<storage, read> lodTable: array<LodEntry>;
@group(0) @binding(2) var textures: texture_2d_array<f32>;
@group(0) @binding(3) var samp:     sampler;

@fragment
fn main(@location(0) uv: vec2<f32>, @location(1) instId: u32) -> @location(0) vec4<f32> {
  let entry = lodTable[instId];
  let s = createSamplerWithLodMin(entry.lodMin);
  // Use textureSampleLevel at the clamped lod to simulate lodMinClamp.
  let lod = max(entry.lodMin, 0.0);
  return textureSampleLevel(textures, samp, uv, instId, lod);
}
`;

// Примечание: WGSL не допускает динамическое создание семплеров во фрагментном шейдере.
// Используем другой подход: кодируем lodMin в UV и применяем textureSampleBias, или
// задаём равномерный LOD на экземпляр через textureSampleLevel.
// Исправленный фрагментный шейдер - просто используем textureSampleLevel с резидентным мипом:
const FRAG_WGSL2 = /* wgsl */ `
struct LodEntry { lodMin: f32, _pad0: f32, _pad1: f32, _pad2: f32 }
@group(0) @binding(1) var<storage, read> lodTable: array<LodEntry>;
@group(0) @binding(2) var textures: texture_2d_array<f32>;
@group(0) @binding(3) var samp:     sampler;

@fragment
fn main(@location(0) uv: vec2<f32>, @location(1) instId: u32) -> @location(0) vec4<f32> {
  let lod = lodTable[instId].lodMin;
  return textureSampleLevel(textures, samp, uv, instId, lod);
}
`;

// ---- Вспомогательные функции ------------------------------------------------------------------------------------------------------

function hsl(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

function solidMip(matId: number, mipLevel: number): Uint8Array<ArrayBuffer> {
  const size = Math.max(1, TEX >> mipLevel);
  const hue = (matId / (GRID * GRID)) * 360;
  // Варьируем яркость по мип-уровню, чтобы на тепловой карте грубее = темнее.
  const lightness = 0.3 + (mipLevel / MIPS) * 0.4;
  const [r, g, b] = hsl(hue, 0.8, lightness);
  const data = new Uint8Array(new ArrayBuffer(size * size * 4));
  for (let i = 0; i < size * size; i++) {
    data[i * 4]     = Math.round(r * 255);
    data[i * 4 + 1] = Math.round(g * 255);
    data[i * 4 + 2] = Math.round(b * 255);
    data[i * 4 + 3] = 255;
  }
  return data;
}

function uploadMip(device: GPUDevice, texture: GPUTexture, layer: number, level: number, matId: number): void {
  const size = Math.max(1, TEX >> level);
  const data = solidMip(matId, level);
  const bytesPerRow = Math.max(256, Math.ceil((size * 4) / 256) * 256);
  const buf = size * 4 === bytesPerRow ? data : (() => {
    const p = new Uint8Array(new ArrayBuffer(bytesPerRow * size));
    for (let r = 0; r < size; r++) p.set(data.subarray(r * size * 4, (r + 1) * size * 4), r * bytesPerRow);
    return p;
  })();
  device.queue.writeTexture(
    { texture, mipLevel: level, origin: { x: 0, y: 0, z: layer } },
    buf, { bytesPerRow, rowsPerImage: size }, [size, size, 1]
  );
}

// ---- Камера (простая перспектива + взгляд сверху) ----------------------------------------------------------

function mat4Mul(a: Float32Array, b: Float32Array): Float32Array {
  const o = new Float32Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + j]! * b[i * 4 + k]!;
    o[i * 4 + j] = s;
  }
  return o;
}

function perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovY / 2);
  const ri = 1 / (near - far);
  return new Float32Array([
    f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*ri,-1, 0,0,2*far*near*ri,0,
  ]);
}

function lookAt(ex: number, ey: number, ez: number, cx: number, cy: number, cz: number): Float32Array {
  const fx=cx-ex, fy=cy-ey, fz=cz-ez;
  const fl = Math.sqrt(fx*fx+fy*fy+fz*fz);
  const fx_=fx/fl, fy_=fy/fl, fz_=fz/fl;
  const rx=fy_*0-fz_*1, ry=fz_*0-fx_*0, rz=fx_*1-fy_*0;
  const rl=Math.sqrt(rx*rx+ry*ry+rz*rz);
  const rx_=rx/rl, ry_=ry/rl, rz_=rz/rl;
  const ux=ry_*fz_-rz_*fy_, uy=rz_*fx_-rx_*fz_, uz=rx_*fy_-ry_*fx_;
  return new Float32Array([
    rx_,ux,-fx_,0, ry_,uy,-fy_,0, rz_,uz,-fz_,0,
    -(rx_*ex+ry_*ey+rz_*ez), -(ux*ex+uy*ey+uz*ez), fx_*ex+fy_*ey+fz_*ez, 1,
  ]);
}

// ---- Рендер-проход --------------------------------------------------------------------------------------------------------------------------

const N = GRID * GRID; // 64 материала

class LRUPass implements IRenderPass {
  readonly name = "lru-pass";

  private _device: GPUDevice | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _camBuf: GPUBuffer | null = null;
  private _lodBuf: GPUBuffer | null = null;
  private _arrayTex: GPUTexture | null = null;
  private _width = 1; private _height = 1;

  // Состояние стриминга
  readonly lru = new LRUEvictionPolicy();
  readonly budget = new BudgetTracker(BUDGET);
  /** residentMip[i] = самый детальный загруженный мип-уровень для материала i */
  readonly residentMip: number[] = new Array(N).fill(MIPS - 1); // начинаем с самого грубого
  evictedThisFrame = 0;

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, presentationFormat } = ctx;
    this._device = device;

    // Единый texture_2d_array: 64 слоя × 256×256 × 8 мип-уровней
    device.pushErrorScope("out-of-memory");
    this._arrayTex = device.createTexture({
      label: "lru-tex-array",
      size: [TEX, TEX, N],
      mipLevelCount: MIPS,
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      dimension: "2d",
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[LRUPass] OOM:", e); });

    // Загружаем 2 самых грубых мипа для каждого материала (гарантия постоянного присутствия).
    for (let mat = 0; mat < N; mat++) {
      for (let level = MIPS - 2; level < MIPS; level++) {
        uploadMip(device, this._arrayTex, mat, level, mat);
        const bytes = Math.max(1, TEX >> level) ** 2 * 4;
        this.budget.recordUpload(`m${mat}`, level, bytes);
        this.lru.registerTexture(`m${mat}`, MIPS);
        this.lru.touch(`m${mat}`, level, 0);
      }
      this.residentMip[mat] = MIPS - 2;
      this.lru.recordSize(`m${mat}`, MIPS - 2, (TEX >> (MIPS - 2)) ** 2 * 4);
    }

    this._camBuf = device.createBuffer({ label: "cam-buf", size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this._lodBuf = device.createBuffer({ label: "lod-buf", size: N * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    this._pipeline = device.createRenderPipeline({
      label: "lru-pipeline",
      layout: "auto",
      vertex:   { module: device.createShaderModule({ code: VERT_WGSL }), entryPoint: "main" },
      fragment: { module: device.createShaderModule({ code: FRAG_WGSL2 }), entryPoint: "main", targets: [{ format: presentationFormat }] },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });

    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear" });
    this._bindGroup = device.createBindGroup({
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._camBuf } },
        { binding: 1, resource: { buffer: this._lodBuf } },
        { binding: 2, resource: this._arrayTex.createView({ dimension: "2d-array" }) },
        { binding: 3, resource: sampler },
      ],
    });
  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._bindGroup || !this._camBuf || !this._lodBuf || !this._device || !this._arrayTex) return;

    const frameIndex = ctx.frameIndex;
    const device = this._device;

    // ---- Камера ----------------------------------------------------------------------------------------------------------------------------
    const t = frameIndex * 0.008;
    const radius = 10;
    const ex = Math.sin(t) * radius;
    const ez = 5 + Math.cos(t) * radius * 0.3;
    const aspect = this._width / this._height;
    const proj = perspective(Math.PI / 3, aspect, 0.1, 200);
    const view = lookAt(ex, 6, ez, 0, 0, -8);
    const vp = mat4Mul(proj, view);
    device.queue.writeBuffer(this._camBuf, 0, vp);

    // ---- Вычисляем желаемый мип для каждого материала --------------------------------------------------
    const desiredMip: number[] = [];
    for (let i = 0; i < N; i++) {
      const col = i % GRID;
      const row = Math.floor(i / GRID);
      const wx = (col - 3.5) * 2.5;
      const wz = -row * 2.5;
      const dist = Math.sqrt((ex - wx) ** 2 + (ez - wz) ** 2 + 36); // +36 = 6² (высота камеры)
      desiredMip.push(computeDesiredMip(dist, TEX, this._height, Math.PI / 3));
    }

    // ---- Обновляем LRU для видимых текстур ------------------------------------------------------------------------
    for (let i = 0; i < N; i++) {
      this.lru.touch(`m${i}`, this.residentMip[i]!, frameIndex);
    }

    // ---- Вытесняем при превышении бюджета --------------------------------------------------------------------------
    this.evictedThisFrame = 0;
    if (!this.budget.canUpload(0)) {
      const toEvict = this.lru.selectEvictions(FRAME_UPLOAD_BUDGET);
      for (const e of toEvict) {
        this.budget.recordEviction(e.textureId, e.mipLevel);
        this.lru.forget(e.textureId, e.mipLevel);
        const matId = parseInt(e.textureId.slice(1));
        if (e.mipLevel <= this.residentMip[matId]!) {
          this.residentMip[matId] = e.mipLevel + 1; // огрубляем до следующего доступного
        }
        this.evictedThisFrame++;
      }
    }

    // ---- Загружаем более детальные мипы для ближних материалов --------------------------------
    let frameBytes = 0;
    for (let i = 0; i < N; i++) {
      const target = Math.max(0, desiredMip[i]!);
      if (target < this.residentMip[i]!) {
        const nextLevel = this.residentMip[i]! - 1;
        const size = Math.max(1, TEX >> nextLevel);
        const bytes = size * size * 4;
        if (frameBytes + bytes > FRAME_UPLOAD_BUDGET) continue;
        if (!this.budget.canUpload(bytes)) continue;

        uploadMip(device, this._arrayTex, i, nextLevel, i);
        this.budget.recordUpload(`m${i}`, nextLevel, bytes);
        this.lru.recordSize(`m${i}`, nextLevel, bytes);
        this.lru.touch(`m${i}`, nextLevel, frameIndex);
        this.residentMip[i] = nextLevel;
        frameBytes += bytes;
      }
    }

    // ---- Загружаем таблицу LOD --------------------------------------------------------------------------------------------------
    const lodData = new Float32Array(new ArrayBuffer(N * 16));
    for (let i = 0; i < N; i++) {
      lodData[i * 4] = this.residentMip[i]!;
    }
    device.queue.writeBuffer(this._lodBuf, 0, lodData);

    // ---- Отрисовка --------------------------------------------------------------------------------------------------------------------------
    const pass = ctx.encoder.beginRenderPass({
      colorAttachments: [{ view: ctx.colorAttachment, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: { view: ctx.depthAttachment, depthLoadOp: "load", depthStoreOp: "store" },
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.draw(6, N);
    pass.end();
  }

  onResize(w: number, h: number): void { this._width = w; this._height = h; }

  destroy(): void {
    this._arrayTex?.destroy();
    this._camBuf?.destroy();
    this._lodBuf?.destroy();
    this.budget.destroy();
    this.lru.destroy();
    this._arrayTex = null; this._pipeline = null; this._bindGroup = null; this._device = null;
  }
}

// ---- Тепловая карта ------------------------------------------------------------------------------------------------------------------------

function setupHeatmap(): (mips: number[]) => void {
  const container = document.getElementById("heatmap")!;
  const cells: HTMLElement[] = [];
  for (let i = 0; i < N; i++) {
    const cell = document.createElement("div");
    cell.className = "cell";
    container.appendChild(cell);
    cells.push(cell);
  }
  const COLORS = ["#ffffff","#cc88ff","#4444ff","#00cccc","#22cc22","#ffff00","#ff8800","#ff2222"];
  return (mips: number[]) => {
    for (let i = 0; i < N; i++) {
      const m = mips[i]!;
      cells[i]!.style.background = COLORS[m] ?? "#888";
      cells[i]!.textContent = String(m);
    }
  };
}

// ---- Подключение ------------------------------------------------------------------------------------------------------------------------------

const el = document.getElementById("canvas") as HTMLElement & {
  clearColor: GPUColorDict;
  addRenderPass: (p: IRenderPass) => Promise<void>;
};

el.addEventListener("webgpu-error", (e) => overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`));

el.addEventListener("webgpu-ready", async (e) => {
  const { device } = (e as CustomEvent<WebGPUReadyDetail>).detail;
  void device;
  el.clearColor = { r: 0.04, g: 0.04, b: 0.06, a: 1.0 };

  const pass = new LRUPass();
  await el.addRenderPass(pass);

  const updateHeatmap = setupHeatmap();
  overlay.set("Status", "Running");
  overlay.set("Бюджет", `${(BUDGET / 1024).toFixed(0)} КБ`);

  function tick() {
    fps.tick();
    overlay.set("FPS", fps.fps);
    overlay.set("GPU занято", `${(pass.budget.totalUsed / 1024).toFixed(1)} КБ`);
    overlay.set("Утилизация", `${(pass.budget.utilization * 100).toFixed(1)}%`);
    overlay.set("Вытеснений/кадр", pass.evictedThisFrame);
    updateHeatmap(pass.residentMip);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
