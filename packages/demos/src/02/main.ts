import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import type { IRenderPass, RenderPassInitContext, FrameContext } from "@webgpu-streaming/gpu-types";
import { createOverlay, FpsTracker } from "../shared/overlay.js";

const overlay = createOverlay("02 - Triangle");
const fps = new FpsTracker();
overlay.set("Status", "Initializing…");

// ---- Встроенные WGSL-шейдеры --------------------------------------------------------------------------------------------------

const VERTEX_WGSL = /* wgsl */ `
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec3<f32>,
}

@vertex
fn main(@location(0) pos: vec2<f32>, @location(1) color: vec3<f32>) -> VertexOutput {
  var out: VertexOutput;
  out.position = vec4<f32>(pos, 0.0, 1.0);
  out.color    = color;
  return out;
}
`;

const FRAGMENT_WGSL = /* wgsl */ `
@fragment
fn main(@location(0) color: vec3<f32>) -> @location(0) vec4<f32> {
  return vec4<f32>(color, 1.0);
}
`;

// ---- Данные вершин: [x, y, r, g, b] на вершину --------------------------------------------------------------
// prettier-ignore
const VERTICES = new Float32Array([
  // позиция      цвет (линейный)
   0.0,  0.7,   1.0, 0.2, 0.2,   // вверху        - красный
  -0.6, -0.5,   0.2, 1.0, 0.2,   // внизу слева   - зелёный
   0.6, -0.5,   0.2, 0.2, 1.0,   // внизу справа  - синий
]);
const VERTEX_STRIDE = 5 * 4; // 5 вещественных × 4 байта

// ---- Рендер-проход треугольника ----------------------------------------------------------------------------------------------

class TrianglePass implements IRenderPass {
  readonly name = "triangle-pass";
  private _pipeline: GPURenderPipeline | null = null;
  private _vertexBuffer: GPUBuffer | null = null;

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, presentationFormat } = ctx;

    device.pushErrorScope("out-of-memory");
    this._vertexBuffer = device.createBuffer({
      label: "triangle-vbo",
      size: VERTICES.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[TrianglePass] Нехватка памяти:", e); });
    device.queue.writeBuffer(this._vertexBuffer, 0, VERTICES);

    this._pipeline = device.createRenderPipeline({
      label: "triangle-pipeline",
      layout: "auto",
      vertex: {
        module: device.createShaderModule({ label: "tri-vert", code: VERTEX_WGSL }),
        entryPoint: "main",
        buffers: [{
          arrayStride: VERTEX_STRIDE,
          attributes: [
            { shaderLocation: 0, offset: 0,     format: "float32x2" },
            { shaderLocation: 1, offset: 2 * 4, format: "float32x3" },
          ],
        }],
      },
      fragment: {
        module: device.createShaderModule({ label: "tri-frag", code: FRAGMENT_WGSL }),
        entryPoint: "main",
        targets: [{ format: presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._vertexBuffer) return;
    const pass = ctx.encoder.beginRenderPass({
      label: "triangle-render-pass",
      colorAttachments: [{ view: ctx.colorAttachment, loadOp: "load", storeOp: "store" }],
      depthStencilAttachment: {
        view: ctx.depthAttachment,
        depthLoadOp: "load",
        depthStoreOp: "store",
      },
    });
    pass.setPipeline(this._pipeline);
    pass.setVertexBuffer(0, this._vertexBuffer);
    pass.draw(3);
    pass.end();
  }

  onResize(_w: number, _h: number): void {}

  destroy(): void {
    this._vertexBuffer?.destroy();
    this._vertexBuffer = null;
    this._pipeline = null;
  }
}

// ---- Подключение ----------------------------------------------------------------------------------------------------------------------------

const el = document.getElementById("canvas") as HTMLElement & {
  clearColor: GPUColorDict;
  addRenderPass: (pass: IRenderPass) => Promise<void>;
};

el.addEventListener("webgpu-error", (e) => {
  overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`);
});

el.addEventListener("webgpu-ready", async (e) => {
  const { device } = (e as CustomEvent<WebGPUReadyDetail>).detail;

  el.clearColor = { r: 0.05, g: 0.05, b: 0.08, a: 1.0 };
  await el.addRenderPass(new TrianglePass());

  overlay.set("Status", "Running");
  overlay.set("Adapter", device.label || "gpu");

  let frameCount = 0;
  function tick() {
    fps.tick();
    overlay.set("FPS", fps.fps);
    overlay.set("Frames", ++frameCount);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
});
