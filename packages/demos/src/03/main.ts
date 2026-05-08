import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import type { IRenderPass, RenderPassInitContext, FrameContext } from "@webgpu-streaming/gpu-types";
import { createOverlay, FpsTracker } from "../shared/overlay.js";

const overlay = createOverlay("03 - Textured Quad");
const fps = new FpsTracker();
overlay.set("Status", "Initializing…");

// ---- WGSL-шейдеры --------------------------------------------------------------------------------------------------------------------------

const VERT_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

// Полноэкранный квад по индексу вершины (VBO не нужен).
@vertex
fn main(@builtin(vertex_index) vi: u32) -> VSOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
  );
  var uvs = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
  );
  var out: VSOut;
  out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
  out.uv  = uvs[vi];
  return out;
}
`;

const FRAG_WGSL = /* wgsl */ `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
  return textureSample(tex, samp, uv);
}
`;

// ---- Процедурная текстура «шахматная доска» ------------------------------------------------------------------------

/** Генерирует шахматную текстуру 256×256 RGBA8 с клетками 16 пикселей. */
function makeCheckerboard(size: number, tileSize: number): Uint8Array<ArrayBuffer> {
  const data = new Uint8Array(new ArrayBuffer(size * size * 4));
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const even = (Math.floor(x / tileSize) + Math.floor(y / tileSize)) % 2 === 0;
      const i = (y * size + x) * 4;
      data[i]     = even ? 255 : 30;
      data[i + 1] = even ? 128 : 128;
      data[i + 2] = even ? 0   : 180;
      data[i + 3] = 255;
    }
  }
  return data;
}

/** Генерация мип-уровней на CPU с понижением разрешения в 2× (box filter). */
function generateMips(baseData: Uint8Array, baseSize: number): Uint8Array[] {
  const mips: Uint8Array[] = [baseData];
  let size = baseSize;
  let src = baseData;
  while (size > 1) {
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
    mips.push(dst);
    src = dst;
    size = half;
  }
  return mips;
}

// ---- Рендер-проход текстурированного квада --------------------------------------------------------------------------

class TexturedQuadPass implements IRenderPass {
  readonly name = "textured-quad-pass";
  private _pipeline: GPURenderPipeline | null = null;
  private _bindGroup: GPUBindGroup | null = null;
  private _texture: GPUTexture | null = null;

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, presentationFormat } = ctx;

    const BASE_SIZE = 256;
    const mipCount = Math.floor(Math.log2(BASE_SIZE)) + 1; // 9

    const mipData = generateMips(makeCheckerboard(BASE_SIZE, 16), BASE_SIZE);

    device.pushErrorScope("out-of-memory");
    this._texture = device.createTexture({
      label: "checkerboard",
      size: [BASE_SIZE, BASE_SIZE],
      mipLevelCount: mipCount,
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[QuadPass] Нехватка памяти (текстура):", e); });

    // Загружаем каждый мип-уровень.
    for (let level = 0; level < mipCount; level++) {
      const mipSize = BASE_SIZE >> level;
      const data = mipData[level]!;
      const bytesPerRow = Math.ceil((mipSize * 4) / 256) * 256;
      // Переупаковываем, если mipSize * 4 не кратно 256.
      if (mipSize * 4 === bytesPerRow) {
        device.queue.writeTexture(
          { texture: this._texture, mipLevel: level },
          data,
          { bytesPerRow, rowsPerImage: mipSize },
          [mipSize, mipSize]
        );
      } else {
        // Дополняем строки до выравнивания 256 байт.
        const padded = new Uint8Array(new ArrayBuffer(bytesPerRow * mipSize));
        for (let row = 0; row < mipSize; row++) {
          padded.set(data.subarray(row * mipSize * 4, (row + 1) * mipSize * 4), row * bytesPerRow);
        }
        device.queue.writeTexture(
          { texture: this._texture, mipLevel: level },
          padded,
          { bytesPerRow, rowsPerImage: mipSize },
          [mipSize, mipSize]
        );
      }
    }

    const sampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
    });

    this._pipeline = device.createRenderPipeline({
      label: "textured-quad-pipeline",
      layout: "auto",
      vertex: {
        module: device.createShaderModule({ label: "quad-vert", code: VERT_WGSL }),
        entryPoint: "main",
      },
      fragment: {
        module: device.createShaderModule({ label: "quad-frag", code: FRAG_WGSL }),
        entryPoint: "main",
        targets: [{ format: presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    });

    this._bindGroup = device.createBindGroup({
      label: "textured-quad-bg",
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this._texture.createView() },
        { binding: 1, resource: sampler },
      ],
    });

    overlay.set("Format", "rgba8unorm");
    overlay.set("Size", `${BASE_SIZE}×${BASE_SIZE}`);
    overlay.set("Mip levels", mipCount);
  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._bindGroup) return;
    const pass = ctx.encoder.beginRenderPass({
      label: "textured-quad-rp",
      colorAttachments: [{ view: ctx.colorAttachment, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, this._bindGroup);
    pass.draw(6); // 2 треугольника, без VBO
    pass.end();
  }

  onResize(_w: number, _h: number): void {}

  destroy(): void {
    this._texture?.destroy();
    this._texture = null;
    this._pipeline = null;
    this._bindGroup = null;
  }
}

// ---- Подключение ------------------------------------------------------------------------------------------------------------------------------

const el = document.getElementById("canvas") as HTMLElement & {
  clearColor: GPUColorDict;
  addRenderPass: (pass: IRenderPass) => Promise<void>;
};

el.addEventListener("webgpu-error", (e) => {
  overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`);
});

el.addEventListener("webgpu-ready", async () => {
  el.clearColor = { r: 0.1, g: 0.1, b: 0.1, a: 1.0 };
  await el.addRenderPass(new TexturedQuadPass());
  overlay.set("Status", "Running");

  function tick() {
    fps.tick();
    overlay.set("FPS", fps.fps);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
