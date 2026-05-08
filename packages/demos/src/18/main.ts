/**
 * Demo 18 - Sampler Coherence Verification (Phase 7.6)
 *
 * 8 objects in the same tier (Tier 0, 256×256), each with a unique
 * color-coded mip texture, streamed at staggered start times:
 *   Object 0: starts immediately,  reaches full res in ~2 s
 *   Object 1: starts after 1 s
 *   Object 2: starts after 2 s
 *   ...
 *   Object 7: starts after 7 s
 *
 * The camera is FIXED (no orbit).
 * An overlay per-object shows residentMip and sampling path.
 * A timeline at the bottom shows mip history over time.
 * A "Force Evict Object 4" button resets it to coarsest mip.
 *
 * What this proves (Phase 7.1 fix):
 *   - No corruption at any point (no black pixels)
 *   - Object 0 sharpens while Object 7 stays coarse
 *   - lodMinClamp on the sampler = 0 always; safety floor is shader-side
 *   - textureSampleLevel with lod=max(approxLod, residentMip) is correct
 */

async function main(): Promise<void> {

// ---- Constants ----------------------------------------------------------------------------------------------------------------------------------

const N_OBJECTS = 8;
const TEX_SIZE  = 256;
const MIP_COUNT = 9; // log2(256) + 1  (mip 0 = finest, mip 8 = coarsest)
const MIP_INTERVAL_MS = 250; // ms between mip uploads once streaming starts
const STREAM_START_DELAY_MS = Array.from({ length: N_OBJECTS }, (_, i) => i * 1000);

// Color per mip level: fine→coarse = red→white
const MIP_COLORS: [number, number, number][] = [
  [230,  60,  60], // mip 0
  [230, 140,  60], // mip 1
  [230, 220,  60], // mip 2
  [100, 220,  60], // mip 3
  [ 60, 200, 200], // mip 4
  [ 60, 100, 240], // mip 5
  [150,  60, 240], // mip 6
  [210,  60, 195], // mip 7
  [210, 210, 210], // mip 8 (coarsest)
];

// ---- WebGPU init ------------------------------------------------------------------------------------------------------------------------------

const canvas = document.getElementById("c") as HTMLCanvasElement;
const overlayCanvas = document.getElementById("overlay") as HTMLCanvasElement;

const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) {
  document.getElementById("status")!.textContent = "WebGPU not available";
  throw new Error("no adapter");
}
const device = await adapter.requestDevice();
document.getElementById("status")!.textContent = "GPU ready";

const gpuCtx = canvas.getContext("webgpu")!;
const format = navigator.gpu.getPreferredCanvasFormat();

function resizeCanvases(): void {
  const area = document.getElementById("canvas-area")!;
  const w = Math.floor(area.clientWidth  * devicePixelRatio);
  const h = Math.floor(area.clientHeight * devicePixelRatio);
  canvas.width = w; canvas.height = h;
  overlayCanvas.width  = area.clientWidth;
  overlayCanvas.height = area.clientHeight;
  gpuCtx.configure({ device, format, alphaMode: "opaque" });
}
resizeCanvases();
window.addEventListener("resize", resizeCanvases);

// ---- Texture array (8 layers) ----------------------------------------------------------------------------------------------------

device.pushErrorScope("out-of-memory");
const texArray = device.createTexture({
  label: "tier0-tex-array",
  size: [TEX_SIZE, TEX_SIZE, N_OBJECTS],
  mipLevelCount: MIP_COUNT,
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
void device.popErrorScope().then(e => { if (e) console.error("[demo18] OOM texArray:", e); });

function buildMipPixels(mipLevel: number, w: number, h: number): Uint8Array {
  const [r, g, b] = MIP_COLORS[mipLevel] ?? [128, 128, 128];
  const out = new Uint8Array(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    out[i * 4] = r; out[i * 4 + 1] = g; out[i * 4 + 2] = b; out[i * 4 + 3] = 255;
  }
  return out;
}

function uploadMipLevel(layer: number, mipLevel: number): void {
  const w = Math.max(1, TEX_SIZE >> mipLevel);
  const h = Math.max(1, TEX_SIZE >> mipLevel);
  const bpr = Math.max(256, Math.ceil(w * 4 / 256) * 256);
  const src = buildMipPixels(mipLevel, w, h);
  const data = new Uint8Array(bpr * h);
  for (let row = 0; row < h; row++) {
    data.set(src.subarray(row * w * 4, (row + 1) * w * 4), row * bpr);
  }
  device.queue.writeTexture(
    { texture: texArray, mipLevel, origin: { x: 0, y: 0, z: layer } },
    data, { bytesPerRow: bpr, rowsPerImage: h }, [w, h, 1]
  );
}

// Initialize all layers to coarsest mip only
for (let layer = 0; layer < N_OBJECTS; layer++) {
  uploadMipLevel(layer, MIP_COUNT - 1);
}

// ---- Per-object streaming state ------------------------------------------------------------------------------------------------

const residentMip = new Array<number>(N_OBJECTS).fill(MIP_COUNT - 1);
const lastMipUploadAt = new Array<number>(N_OBJECTS).fill(0);

// ---- Sampler (lodMinClamp = 0 always; shader handles safety floor) --------------------------

device.pushErrorScope("out-of-memory");
const sampler = device.createSampler({
  label: "tier0-samp",
  magFilter: "linear",
  minFilter: "linear",
  mipmapFilter: "linear",
  lodMinClamp: 0,
  lodMaxClamp: 1000,
});
void device.popErrorScope().then(e => { if (e) console.error("[demo18] OOM sampler:", e); });

// ---- Material buffer: N_OBJECTS × 4 u32 = [tierIndex, layerIndex, residentMip, tierLodMinClamp] --

device.pushErrorScope("out-of-memory");
const matBuf = device.createBuffer({
  label: "demo18-mat",
  size: N_OBJECTS * 4 * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
void device.popErrorScope().then(e => { if (e) console.error("[demo18] OOM matBuf:", e); });

function computeTierMax(): number {
  return Math.max(...residentMip);
}

function updateMaterialBuffer(): void {
  const tierMax = computeTierMax();
  const data = new Uint32Array(N_OBJECTS * 4);
  for (let i = 0; i < N_OBJECTS; i++) {
    data[i * 4    ] = 0;
    data[i * 4 + 1] = i;
    data[i * 4 + 2] = residentMip[i]!;
    data[i * 4 + 3] = tierMax;
  }
  device.queue.writeBuffer(matBuf, 0, data);
}
updateMaterialBuffer();

// ---- WGSL --------------------------------------------------------------------------------------------------------------------------------------------

const texView = texArray.createView({ dimension: "2d-array" });

const VERT_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos:    vec4<f32>,
  @location(0)       uv:     vec2<f32>,
  @location(1) @interpolate(flat) matId: u32,
}

@vertex
fn vs_main(
  @builtin(vertex_index)   vi:   u32,
  @builtin(instance_index) inst: u32,
) -> VSOut {
  let quadW = 2.0 / f32(${N_OBJECTS});
  let x0 = -1.0 + f32(inst) * quadW;
  let x1 = x0 + quadW;

  var positions = array<vec2<f32>, 6>(
    vec2(x0, -0.8), vec2(x1, -0.8), vec2(x0,  0.8),
    vec2(x0,  0.8), vec2(x1, -0.8), vec2(x1,  0.8),
  );
  var uvs = array<vec2<f32>, 6>(
    vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
    vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(1.0, 0.0),
  );

  var out: VSOut;
  out.pos   = vec4(positions[vi], 0.0, 1.0);
  out.uv    = uvs[vi];
  out.matId = inst;
  return out;
}
`;

const FRAG_WGSL = /* wgsl */ `
struct MaterialEntry {
  tierIndex:       u32,
  layerIndex:      u32,
  residentMip:     u32,
  tierLodMinClamp: u32,
}

@group(0) @binding(0) var texArr:  texture_2d_array<f32>;
@group(0) @binding(1) var samp:    sampler;
@group(0) @binding(2) var<storage, read> materials: array<MaterialEntry>;

fn computeApproxLod(uv: vec2<f32>, texSize: vec2<f32>) -> f32 {
  let ddx_v = dpdx(uv) * texSize;
  let ddy_v = dpdy(uv) * texSize;
  let d     = max(dot(ddx_v, ddx_v), dot(ddy_v, ddy_v));
  return 0.5 * log2(max(d, 0.0001));
}

@fragment
fn fs_main(
  @location(0)       uv:    vec2<f32>,
  @location(1) @interpolate(flat) matId: u32,
) -> @location(0) vec4<f32> {
  let mat      = materials[matId];
  let layer    = i32(mat.layerIndex);
  let resident = mat.residentMip;

  let texSize = vec2<f32>(textureDimensions(texArr, 0).xy);
  let approxLod = computeApproxLod(uv, texSize);
  // Safety floor: never sample a mip finer than what's resident
  let lod = max(approxLod, f32(resident));
  return textureSampleLevel(texArr, samp, uv, layer, lod);
}
`;

const shaderMod = device.createShaderModule({ code: VERT_WGSL + FRAG_WGSL });

const bgl = device.createBindGroupLayout({
  label: "demo18-bgl",
  entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
  ],
});

const pipeline = device.createRenderPipeline({
  label: "demo18-pipe",
  layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
  vertex:   { module: shaderMod, entryPoint: "vs_main" },
  fragment: { module: shaderMod, entryPoint: "fs_main", targets: [{ format }] },
  primitive: { topology: "triangle-list" },
});

const bindGroup = device.createBindGroup({
  label: "demo18-bg",
  layout: bgl,
  entries: [
    { binding: 0, resource: texView },
    { binding: 1, resource: sampler },
    { binding: 2, resource: { buffer: matBuf } },
  ],
});

// ---- Timeline state ------------------------------------------------------------------------------------------------------------------------

/** residentMip history per object, sampled once per second */
const timeline: number[][] = Array.from({ length: N_OBJECTS }, () => [MIP_COUNT - 1]);
let lastTimelineSample = 0;

// ---- Overlay drawing ----------------------------------------------------------------------------------------------------------------------

const ov2d = overlayCanvas.getContext("2d")!;

function drawOverlay(elapsedMs: number): void {
  ov2d.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  const W = overlayCanvas.width;
  const H = overlayCanvas.height;
  const colW = W / N_OBJECTS;
  const tierMax = computeTierMax();

  ov2d.font = "10px monospace";
  ov2d.textAlign = "center";

  for (let i = 0; i < N_OBJECTS; i++) {
    const cx = colW * (i + 0.5);
    const rm = residentMip[i]!;
    const started = elapsedMs >= STREAM_START_DELAY_MS[i]!;
    const done = rm === 0;

    // Column background highlight if streaming has started
    if (started && !done) {
      ov2d.fillStyle = "rgba(255,255,100,0.04)";
      ov2d.fillRect(colW * i, 0, colW, H * 0.9);
    }

    // Object label
    ov2d.fillStyle = "#aaa";
    ov2d.fillText(`Object ${i}`, cx, 18);

    // Start delay label
    const delayS = STREAM_START_DELAY_MS[i]! / 1000;
    ov2d.fillStyle = "#666";
    ov2d.fillText(delayS === 0 ? "starts immediately" : `starts at t=${delayS}s`, cx, 32);

    // residentMip bar
    const barH = 16;
    const barY = H * 0.12;
    const barX = cx - 30;
    const [cr, cg, cb] = MIP_COLORS[rm] ?? [128, 128, 128];
    ov2d.fillStyle = `rgb(${cr},${cg},${cb})`;
    ov2d.fillRect(barX, barY, 60, barH);
    ov2d.fillStyle = "#000";
    ov2d.fillText(`mip ${rm}`, cx, barY + barH - 3);

    // residentMip numeric
    ov2d.fillStyle = "#ccc";
    ov2d.fillText(`residentMip = ${rm}`, cx, barY + barH + 14);

    // Sampling path label
    // Since lodMinClamp=0, always textureSampleLevel; for overlay we label
    // "HW aniso" when rm == tierMax (safety floor = tier max, lod clamp at max)
    // and "Manual LOD" when rm < tierMax (this texture is finer than floor)
    const path = (rm <= tierMax && rm < MIP_COUNT - 1 && rm < tierMax)
      ? "Manual LOD"
      : "HW aniso path";
    ov2d.fillStyle = rm < tierMax ? "#6ef" : "#8f8";
    ov2d.fillText(path, cx, barY + barH + 28);
  }

  // Top right: tier-wide status
  ov2d.textAlign = "right";
  ov2d.fillStyle = "#888";
  ov2d.fillText(`sampler.lodMinClamp = 0 (always)`, W - 8, H * 0.9 - 14);
  ov2d.fillText(`shader safety floor = max(approxLod, residentMip)`, W - 8, H * 0.9);
}

// ---- Timeline chart ------------------------------------------------------------------------------------------------------------------------

const tlCanvas = document.getElementById("timeline-canvas") as HTMLCanvasElement;
const tl2d = tlCanvas.getContext("2d")!;

function drawTimeline(elapsedS: number): void {
  const W = tlCanvas.width;
  const H = tlCanvas.height;
  tl2d.clearRect(0, 0, W, H);

  const maxS = Math.max(10, elapsedS + 1);
  const rowH = Math.floor((H - 10) / N_OBJECTS);
  const barH = Math.max(4, rowH - 3);

  for (let i = 0; i < N_OBJECTS; i++) {
    const y = 5 + i * rowH;
    const hist = timeline[i]!;

    // Draw each second's resident mip as a colored segment
    for (let t = 0; t < hist.length; t++) {
      const mip = hist[t]!;
      const x0 = (t / maxS) * (W - 40) + 36;
      const x1 = ((t + 1) / maxS) * (W - 40) + 36;
      const [r, g, b] = MIP_COLORS[mip] ?? [128, 128, 128];
      tl2d.fillStyle = `rgb(${r},${g},${b})`;
      tl2d.fillRect(x0, y, Math.max(1, x1 - x0), barH);
    }

    // Label
    tl2d.fillStyle = "#888";
    tl2d.font = "9px monospace";
    tl2d.textAlign = "right";
    tl2d.fillText(`O${i}`, 32, y + barH - 1);
  }

  // Time axis
  tl2d.fillStyle = "#555";
  tl2d.textAlign = "center";
  tl2d.font = "9px monospace";
  for (let t = 0; t <= Math.ceil(maxS); t += 2) {
    const x = (t / maxS) * (W - 40) + 36;
    tl2d.fillText(`${t}s`, x, H - 1);
  }
}

// ---- Force eviction button ----------------------------------------------------------------------------------------------------------

document.getElementById("evict-btn")!.addEventListener("click", () => {
  // Reset object 4 to coarsest mip
  residentMip[4] = MIP_COUNT - 1;
  uploadMipLevel(4, MIP_COUNT - 1);
  // Reset its streaming clock so it will re-stream from coarsest
  lastMipUploadAt[4] = performance.now();
  updateMaterialBuffer();
});

// ---- Render loop ------------------------------------------------------------------------------------------------------------------------------

const startTime = performance.now();

function frame(): void {
  const now = performance.now();
  const elapsed = now - startTime;

  // Per-object streaming: upload next mip if interval elapsed and streaming started
  let matDirty = false;
  for (let i = 0; i < N_OBJECTS; i++) {
    if (elapsed < STREAM_START_DELAY_MS[i]!) continue;
    if (residentMip[i]! <= 0) continue;
    if (now - lastMipUploadAt[i]! >= MIP_INTERVAL_MS) {
      const nextMip = residentMip[i]! - 1;
      uploadMipLevel(i, nextMip);
      residentMip[i] = nextMip;
      lastMipUploadAt[i] = now;
      matDirty = true;
    }
  }
  if (matDirty) updateMaterialBuffer();

  // Timeline sampling: once per second
  if (now - lastTimelineSample >= 1000) {
    lastTimelineSample = now;
    for (let i = 0; i < N_OBJECTS; i++) {
      timeline[i]!.push(residentMip[i]!);
    }
  }

  // GPU render
  const colorTex = gpuCtx.getCurrentTexture();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view:       colorTex.createView(),
      loadOp:     "clear",
      clearValue: { r: 0.08, g: 0.08, b: 0.08, a: 1 },
      storeOp:    "store",
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(6, N_OBJECTS);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // Overlay
  drawOverlay(elapsed);

  // Timeline
  const elapsedS = elapsed / 1000;
  drawTimeline(elapsedS);

  // HUD
  const tierMax = computeTierMax();
  document.getElementById("tier-clamp-val")!.textContent = "0 (lodMinClamp always 0)";
  document.getElementById("tier-max-val")!.textContent   = String(tierMax);
  document.getElementById("elapsed-val")!.textContent    = elapsedS.toFixed(1);

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

} // main()

main().catch(console.error);
