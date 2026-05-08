/**
 * Демо 07 - Tier Allocator + Texture Arrays
 *
 * Отрисовывает сетку квадов 5×4. У каждого квада уникальный материал,
 * отображающийся на слой в texture_2d_array с разделением по разрешению.
 * Шейдер ищет текстуру через буфер хранилища материалов (tierIndex, layerIndex).
 *
 * Размеры уровней: 512, 1024, 2048. Текстуры назначаются по кругу.
 * Для каждой текстуры загружаются все мип-уровни.
 */
import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import type { IRenderPass, RenderPassInitContext, FrameContext } from "@webgpu-streaming/gpu-types";
import { MATERIAL_ENTRY_WGSL } from "@webgpu-streaming/gpu-types";
import { createOverlay, FpsTracker } from "../shared/overlay.js";

const overlay = createOverlay("07 - Tier Arrays");
const fps = new FpsTracker();
overlay.set("Status", "Initializing…");

// ---- Конфигурация ----------------------------------------------------------------------------------------------------------------------------

const GRID_COLS = 5;
const GRID_ROWS = 4;
const MATERIAL_COUNT = GRID_COLS * GRID_ROWS; // 20
const TIER_SIZES = [512, 1024, 2048] as const;
const MAX_LAYERS = 8; // на уровень, для снижения нагрузки на GPU в демо

// ---- Процедурные текстуры ------------------------------------------------------------------------------------------------------------

/** HSL в линейный RGB. */
function hsl(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

/** Генерирует текстуру сплошного оттенка с шахматной рамкой. */
function makeProceduralTexture(materialId: number, size: number): Uint8Array<ArrayBuffer> {
  const hue = (materialId / MATERIAL_COUNT) * 360;
  const [r, g, b] = hsl(hue, 0.7, 0.4);
  const data = new Uint8Array(new ArrayBuffer(size * size * 4));
  const border = Math.max(4, size >> 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const onBorder = x < border || x >= size - border || y < border || y >= size - border;
      const i = (y * size + x) * 4;
      if (onBorder) {
        data[i] = data[i + 1] = data[i + 2] = 200;
      } else {
        data[i]     = Math.round(r * 255);
        data[i + 1] = Math.round(g * 255);
        data[i + 2] = Math.round(b * 255);
      }
      data[i + 3] = 255;
    }
  }
  return data;
}

/** Загружает все мип-уровни текстуры через writeTexture. */
function uploadAllMips(device: GPUDevice, texture: GPUTexture, baseData: Uint8Array, baseSize: number): void {
  let size = baseSize;
  let src = baseData;
  for (let level = 0; size >= 1; level++, size >>= 1) {
    const bytesPerRow = Math.max(256, Math.ceil((size * 4) / 256) * 256);
    if (size * 4 === bytesPerRow) {
      device.queue.writeTexture({ texture, mipLevel: level }, src, { bytesPerRow, rowsPerImage: size }, [size, size]);
    } else {
      const padded = new Uint8Array(new ArrayBuffer(bytesPerRow * size));
      for (let row = 0; row < size; row++) {
        padded.set(src.subarray(row * size * 4, (row + 1) * size * 4), row * bytesPerRow);
      }
      device.queue.writeTexture({ texture, mipLevel: level }, padded, { bytesPerRow, rowsPerImage: size }, [size, size]);
    }
    if (size <= 1) break;
    // Понижение разрешения для следующего мипа.
    const half = size >> 1;
    const dst = new Uint8Array(new ArrayBuffer(half * half * 4));
    for (let y = 0; y < half; y++) {
      for (let x = 0; x < half; x++) {
        const o = (y * half + x) * 4;
        const s = (y * 2 * size + x * 2) * 4;
        for (let c = 0; c < 4; c++) {
          dst[o + c] = ((src[s + c]! + src[s + 4 + c]! + src[s + size * 4 + c]! + src[s + size * 4 + 4 + c]!) / 4) | 0;
        }
      }
    }
    src = dst;
    size = half;
  }
}

// ---- WGSL ------------------------------------------------------------------------------------------------------------------------------------------

const VERT_WGSL = /* wgsl */ `
struct Uniforms {
  gridCols: u32,
  gridRows: u32,
}
@group(0) @binding(0) var<uniform> uni: Uniforms;

struct VSOut {
  @builtin(position)              pos:        vec4<f32>,
  @location(0)                    uv:         vec2<f32>,
  @location(1) @interpolate(flat) materialId: u32,
}

@vertex
fn main(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) inst: u32,
) -> VSOut {
  let col = inst % uni.gridCols;
  let row = inst / uni.gridCols;

  let cellW = 2.0 / f32(uni.gridCols);
  let cellH = 2.0 / f32(uni.gridRows);
  let pad   = 0.02;

  let x0 = -1.0 + f32(col) * cellW + pad;
  let y0 = -1.0 + f32(row) * cellH + pad;
  let x1 = x0 + cellW - pad * 2.0;
  let y1 = y0 + cellH - pad * 2.0;

  var corners = array<vec2<f32>, 6>(
    vec2(x0,y0), vec2(x1,y0), vec2(x0,y1),
    vec2(x0,y1), vec2(x1,y0), vec2(x1,y1),
  );
  var uvs = array<vec2<f32>, 6>(
    vec2(0,0), vec2(1,0), vec2(0,1),
    vec2(0,1), vec2(1,0), vec2(1,1),
  );

  var o: VSOut;
  o.pos        = vec4<f32>(corners[vi], 0.0, 1.0);
  o.uv         = uvs[vi];
  o.materialId = inst;
  return o;
}
`;

const FRAG_WGSL = /* wgsl */ `
${MATERIAL_ENTRY_WGSL}

@group(0) @binding(1) var<storage, read> materials: array<MaterialEntry>;
@group(0) @binding(2) var tier0: texture_2d_array<f32>;
@group(0) @binding(3) var tier1: texture_2d_array<f32>;
@group(0) @binding(4) var tier2: texture_2d_array<f32>;
@group(0) @binding(5) var samp:  sampler;

@fragment
fn main(
  @location(0)                    uv:         vec2<f32>,
  @location(1) @interpolate(flat) materialId: u32,
) -> @location(0) vec4<f32> {
  let m = materials[materialId];
  // textureSampleLevel используется вместо textureSample, так как каждый экземпляр
  // может иметь разный tierIndex, что делает поток управления if/else неравномерным.
  // textureSample требует равномерного потока управления во фрагментном шейдере.
  var color: vec4<f32>;
  if m.tierIndex == 0u {
    color = textureSampleLevel(tier0, samp, uv, m.layerIndex, 0.0);
  } else if m.tierIndex == 1u {
    color = textureSampleLevel(tier1, samp, uv, m.layerIndex, 0.0);
  } else {
    color = textureSampleLevel(tier2, samp, uv, m.layerIndex, 0.0);
  }
  return color;
}
`;

// ---- Рендер-проход --------------------------------------------------------------------------------------------------------------------------

class TierArrayPass implements IRenderPass {
  readonly name = "tier-array-pass";

  private _device: GPUDevice | null = null;
  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _uniformBuffer: GPUBuffer | null = null;
  private _materialBuffer: GPUBuffer | null = null;
  private _tierTextures: GPUTexture[] = [];
  private _tierViews: GPUTextureView[] = [];
  private _tierUsed: number[] = [0, 0, 0];

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, presentationFormat } = ctx;
    this._device = device;

    // Создаём массивы текстур для уровней.
    for (let t = 0; t < 3; t++) {
      const size = TIER_SIZES[t]!;
      const mipCount = Math.floor(Math.log2(size)) + 1;
      device.pushErrorScope("out-of-memory");
      const tex = device.createTexture({
        label: `tier-${t}-array`,
        size: [size, size, MAX_LAYERS],
        mipLevelCount: mipCount,
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        dimension: "2d",
      });
      void device.popErrorScope().then((e) => { if (e) console.error(`[TierArrayPass] OOM tier ${t}:`, e); });
      this._tierTextures.push(tex);
      this._tierViews.push(tex.createView({ dimension: "2d-array" }));
    }

    // Назначаем материалы по уровням по кругу и загружаем текстуры.
    const matData = new Uint32Array(new ArrayBuffer(MATERIAL_COUNT * 16)); // 4×u32 на запись
    for (let id = 0; id < MATERIAL_COUNT; id++) {
      const tier = id % 3;
      const layer = this._tierUsed[tier]!;
      this._tierUsed[tier]!++;

      const size = TIER_SIZES[tier]!;
      const baseData = makeProceduralTexture(id, size);

      // Записываем в 2d-массив на нужный слой.
      // Загружаем мип за мипом, указывая слой массива.
      let mipSrc = baseData;
      let mipSize = size;
      for (let level = 0; mipSize >= 1; level++) {
        const bytesPerRow = Math.max(256, Math.ceil((mipSize * 4) / 256) * 256);
        const upload = mipSize * 4 === bytesPerRow ? mipSrc : (() => {
          const p = new Uint8Array(new ArrayBuffer(bytesPerRow * mipSize));
          for (let row = 0; row < mipSize; row++) p.set(mipSrc.subarray(row * mipSize * 4, (row + 1) * mipSize * 4), row * bytesPerRow);
          return p;
        })();
        device.queue.writeTexture(
          { texture: this._tierTextures[tier]!, mipLevel: level, origin: { x: 0, y: 0, z: layer } },
          upload,
          { bytesPerRow, rowsPerImage: mipSize },
          [mipSize, mipSize, 1]
        );
        if (mipSize <= 1) break;
        const half = mipSize >> 1;
        const dst = new Uint8Array(new ArrayBuffer(half * half * 4));
        for (let y = 0; y < half; y++) for (let x = 0; x < half; x++) {
          const o = (y * half + x) * 4, s = (y * 2 * mipSize + x * 2) * 4;
          for (let c = 0; c < 4; c++) dst[o+c] = ((mipSrc[s+c]!+mipSrc[s+4+c]!+mipSrc[s+mipSize*4+c]!+mipSrc[s+mipSize*4+4+c]!)/4)|0;
        }
        mipSrc = dst; mipSize = half;
      }

      // Записываем запись материала: [tierIndex, layerIndex, residentMip=0, pad=0]
      const base = id * 4;
      matData[base]     = tier;
      matData[base + 1] = layer;
      matData[base + 2] = 0;
      matData[base + 3] = 0;
    }

    // Буфер форм: gridCols, gridRows.
    this._uniformBuffer = device.createBuffer({
      label: "grid-uniforms",
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this._uniformBuffer, 0, new Uint32Array([GRID_COLS, GRID_ROWS]));

    // Буфер хранилища материалов.
    this._materialBuffer = device.createBuffer({
      label: "material-buffer",
      size: matData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this._materialBuffer, 0, matData);

    const sampler = device.createSampler({ magFilter: "linear", minFilter: "linear", mipmapFilter: "linear" });

    device.pushErrorScope("validation");
    this._pipeline = device.createRenderPipeline({
      label: "tier-array-pipeline",
      layout: "auto",
      vertex: { module: device.createShaderModule({ label: "ta-vert", code: VERT_WGSL }), entryPoint: "main" },
      fragment: { module: device.createShaderModule({ label: "ta-frag", code: FRAG_WGSL }), entryPoint: "main", targets: [{ format: presentationFormat }] },
      primitive: { topology: "triangle-list" },
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[TierArrayPass] Pipeline error:", e); });

    device.pushErrorScope("validation");
    this._bindGroup = device.createBindGroup({
      label: "tier-array-bg",
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this._uniformBuffer } },
        { binding: 1, resource: { buffer: this._materialBuffer } },
        { binding: 2, resource: this._tierViews[0]! },
        { binding: 3, resource: this._tierViews[1]! },
        { binding: 4, resource: this._tierViews[2]! },
        { binding: 5, resource: sampler },
      ],
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[TierArrayPass] BindGroup error:", e); });

    overlay.set("Materials", MATERIAL_COUNT);
    overlay.set("Tier 0 (512px)", `${this._tierUsed[0]} layers`);
    overlay.set("Tier 1 (1024px)", `${this._tierUsed[1]} layers`);
    overlay.set("Tier 2 (2048px)", `${this._tierUsed[2]} layers`);
  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._bindGroup) return;
    const pass = ctx.encoder.beginRenderPass({
      colorAttachments: [{ view: ctx.colorAttachment, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    // Один вызов отрисовки: 6 вершин × MATERIAL_COUNT экземпляров.
    pass.draw(6, MATERIAL_COUNT);
    pass.end();
  }

  onResize(_w: number, _h: number): void {}

  destroy(): void {
    for (const t of this._tierTextures) t.destroy();
    this._uniformBuffer?.destroy();
    this._materialBuffer?.destroy();
    this._tierTextures = [];
    this._tierViews = [];
    this._pipeline = null;
    this._bindGroup = null;
    this._device = null;
  }
}

// ---- Подключение ------------------------------------------------------------------------------------------------------------------------------

const el = document.getElementById("canvas") as HTMLElement & {
  clearColor: GPUColorDict;
  addRenderPass: (p: IRenderPass) => Promise<void>;
};

el.addEventListener("webgpu-error", (e) => overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`));

el.addEventListener("webgpu-ready", async () => {
  el.clearColor = { r: 0.06, g: 0.06, b: 0.06, a: 1.0 };
  await el.addRenderPass(new TierArrayPass());
  overlay.set("Status", "Запущено");
  overlay.set("Draw calls", "1 (инстансинг)");

  function tick() {
    fps.tick();
    overlay.set("FPS", fps.fps);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
