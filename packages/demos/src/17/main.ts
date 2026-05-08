/**
 * Демо 17 - Исправление когерентности сэмплера lodMinClamp
 *
 * Доказывает корректную работу двухуровневого исправления зажима.
 * 4 текстуры в одном тире (Tier 0 / 256×256 для скорости), каждая с потоковой
 * передачей с разной скоростью. Сэмплер тира lodMinClamp = max(residentMip).
 * Текстуры с более детальными мипами, чем нижняя граница тира, используют textureSampleLevel;
 * остальные используют textureSample (аппаратное анизо).
 *
 * Демо рендерит 4 четырёхугольника рядом с наложениями для каждого.
 * - Tex 0: поток одного мипа каждые 200 мс (самый быстрый)
 * - Tex 1: поток одного мипа каждые 500 мс
 * - Tex 2: поток одного мипа каждые 1000 мс
 * - Tex 3: никогда не передаётся потоком (остаётся на самом грубом мипе)
 *
 * Корректное поведение:
 *   - Все 4 четырёхугольника всегда показывают допустимый цвет (без чёрного / искажений)
 *   - Tex 0 заметно резче, пока Tex 3 остаётся размытым
 *   - tierLodMinClamp остаётся на уровне самого грубого мипа Tex 3 на всём протяжении
 *   - Детальные мипы Tex 0 ВИДНЫ (шейдер использует обход textureSampleLevel)
 *   - Когда все текстуры сходятся, зажим тира сбрасывается и используется textureSample
 */

async function main(): Promise<void> {

// ---- Параметры текстур ----------------------------------------------------------------------------------------------------------------

const TEX_SIZE = 256;
const MIP_COUNT = 9; // log2(256) + 1 = 9 (0..8, coarsest=8)

/** мс между загрузками мипов для каждой текстуры (Infinity = никогда) */
const STREAM_INTERVALS = [200, 500, 1000, Infinity];

// ---- Цветовая палитра: один цвет на уровень мипа (как в демо 04) ----------------------------------
// мип 0 = красный (самый детальный), мип 8 = белый (самый грубый)
const MIP_COLORS: [number, number, number][] = [
  [255,  50,  50], // mip 0
  [255, 150,  50], // mip 1
  [255, 230,  50], // mip 2
  [100, 230,  50], // mip 3
  [50,  200, 200], // mip 4
  [50,  100, 255], // mip 5
  [150,  50, 255], // mip 6
  [220,  50, 200], // mip 7
  [230, 230, 230], // мип 8 (самый грубый)
];

// ---- Инициализация WebGPU ----------------------------------------------------------------------------------------------------------------------------

const canvas = document.getElementById("c") as HTMLCanvasElement;
const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) { document.body.textContent = "WebGPU not available"; throw new Error("no adapter"); }
const device = await adapter.requestDevice();

const ctx = canvas.getContext("webgpu")!;
const format = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format });

function resizeCanvas(): void {
  canvas.width  = Math.floor(canvas.clientWidth  * devicePixelRatio);
  canvas.height = Math.floor(canvas.clientHeight * devicePixelRatio);
}
resizeCanvas();
window.addEventListener("resize", resizeCanvas);

// ---- Массив текстур (эквивалент Tier 0) --------------------------------------------------------------------------------

device.pushErrorScope("out-of-memory");
const texArray = device.createTexture({
  label: "tier0-tex-array",
  size: [TEX_SIZE, TEX_SIZE, 4], // 4 layers
  mipLevelCount: MIP_COUNT,
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
void device.popErrorScope().then((e) => { if (e) console.error("[demo17] OOM:", e); });

/** Создаёт изображение уровня мипа RGBA8 однородного цвета. */
function buildMipPixels(mipLevel: number, mipWidth: number, mipHeight: number): Uint8Array {
  const [r, g, b] = MIP_COLORS[mipLevel] ?? [128, 128, 128];
  const pixels = new Uint8Array(mipWidth * mipHeight * 4);
  for (let i = 0; i < mipWidth * mipHeight; i++) {
    pixels[i * 4    ] = r;
    pixels[i * 4 + 1] = g;
    pixels[i * 4 + 2] = b;
    pixels[i * 4 + 3] = 255;
  }
  return pixels;
}

/** Загружает один уровень мипа для конкретного слоя массива. */
function uploadMip(layer: number, mipLevel: number): void {
  const mipW = Math.max(1, TEX_SIZE >> mipLevel);
  const mipH = Math.max(1, TEX_SIZE >> mipLevel);
  const bytesPerRow = Math.max(256, Math.ceil(mipW * 4 / 256) * 256);

  const srcData = buildMipPixels(mipLevel, mipW, mipH);
  // Дополняем до bytesPerRow при необходимости.
  let uploadData: Uint8Array;
  if (mipW * 4 === bytesPerRow) {
    uploadData = srcData;
  } else {
    uploadData = new Uint8Array(bytesPerRow * mipH);
    for (let row = 0; row < mipH; row++) {
      uploadData.set(srcData.subarray(row * mipW * 4, (row + 1) * mipW * 4), row * bytesPerRow);
    }
  }

  device.queue.writeTexture(
    { texture: texArray, mipLevel, origin: { x: 0, y: 0, z: layer } },
    uploadData, { bytesPerRow, rowsPerImage: mipH }, [mipW, mipH, 1]
  );
}

// Инициализируем все слои только самым грубым мипом (воспроизводит гарантию постоянного резидента).
for (let layer = 0; layer < 4; layer++) {
  uploadMip(layer, MIP_COUNT - 1); // самый грубый = мип 8
}

// ---- Состояние потоковой передачи для каждой текстуры ----------------------------------------------------------------------------------------------

const residentMip = [MIP_COUNT - 1, MIP_COUNT - 1, MIP_COUNT - 1, MIP_COUNT - 1];
const lastUploadTime = [0, 0, 0, 0];

// ---- Состояние материала / тира ----------------------------------------------------------------------------------------------------------

/**
 * Вычисляет общий для тира lodMinClamp = max(residentMip) по всем 4 текстурам.
 * Это значение запекается в GPUSampler.
 */
function computeTierLodMinClamp(): number {
  return Math.max(...residentMip);
}

// ---- Управление сэмплером ----------------------------------------------------------------------------------------------------------------

let currentSampler: GPUSampler | null = null;

function getSampler(): GPUSampler {
  if (currentSampler) return currentSampler;
  // lodMinClamp должен быть 0, чтобы textureSampleLevel мог обращаться к детальным мипам.
  // Явный LOD = max(approxLod, residentMip) в шейдере - нижний предел безопасности.
  currentSampler = device.createSampler({
    label: "demo17-samp",
    magFilter: "linear",
    minFilter: "linear",
    mipmapFilter: "linear",
    lodMinClamp: 0,
    lodMaxClamp: 1000,
  });
  return currentSampler;
}

// ---- Буфер материалов: [tierIndex, layerIndex, residentMip, tierLodMinClamp] × 4 --

device.pushErrorScope("out-of-memory");
const matBuf = device.createBuffer({
  label: "demo17-mat-buf",
  size: 4 * 4 * 4, // 4 materials × 4 u32 × 4 bytes
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
void device.popErrorScope().then((e) => { if (e) console.error("[demo17] OOM matBuf:", e); });

function updateMaterialBuffer(): void {
  const tierClamp = computeTierLodMinClamp();
  const data = new Uint32Array(4 * 4);
  for (let i = 0; i < 4; i++) {
    data[i * 4    ] = 0;             // tierIndex (все в tier 0)
    data[i * 4 + 1] = i;             // layerIndex
    data[i * 4 + 2] = residentMip[i]!;
    data[i * 4 + 3] = tierClamp;     // tierLodMinClamp
  }
  device.queue.writeBuffer(matBuf, 0, data);
}
updateMaterialBuffer();

// ---- WGSL-шейдеры ----------------------------------------------------------------------------------------------------------------------------

const texView = texArray.createView({ dimension: "2d-array" });

const VERT_WGSL = /* wgsl */ `
struct VSOut {
  @builtin(position) pos:    vec4<f32>,
  @location(0)       uv:     vec2<f32>,
  @location(1) @interpolate(flat) matId: u32,
}

@vertex
fn main(
  @builtin(vertex_index)   vi:   u32,
  @builtin(instance_index) inst: u32,
) -> VSOut {
  // 4 четырёхугольника расположены горизонтально: x в [-1, 1]
  let quadW: f32 = 2.0 / 4.0;
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
  out.pos   = vec4<f32>(positions[vi], 0.0, 1.0);
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

@group(0) @binding(0) var tex:  texture_2d_array<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> materials: array<MaterialEntry>;

fn computeApproxLod(uv: vec2<f32>, texSize: vec2<f32>) -> f32 {
  let ddx = dpdx(uv) * texSize;
  let ddy = dpdy(uv) * texSize;
  let d   = max(dot(ddx, ddx), dot(ddy, ddy));
  return 0.5 * log2(max(d, 0.0001));
}

@fragment
fn main(
  @location(0)                     uv:    vec2<f32>,
  @location(1) @interpolate(flat)  matId: u32,
) -> @location(0) vec4<f32> {
  let mat   = materials[matId];
  let layer = i32(mat.layerIndex);

  // textureSample требует однородного потока управления; ветвь residentMip < tierLodMinClamp
  // неоднородна (значения буфера хранения различаются для каждого фрагмента/инстанса).
  // Всегда используем textureSampleLevel с явным LOD = max(approxLod, residentMip).
  // lodMinClamp сэмплера обеспечивает аппаратный нижний предел безопасности поверх этого.
  let sz    = vec2<f32>(textureDimensions(tex, 0).xy);
  let lod   = max(computeApproxLod(uv, sz), f32(mat.residentMip));
  let color = textureSampleLevel(tex, samp, uv, layer, lod);

  return color;
}
`;

// ---- Пайплайн и группа привязки ----------------------------------------------------------------------------------------------------------

device.pushErrorScope("validation");
const pipeline = device.createRenderPipeline({
  label: "demo17-pipeline",
  layout: "auto",
  vertex: {
    module: device.createShaderModule({ label: "demo17-vert", code: VERT_WGSL }),
    entryPoint: "main",
  },
  fragment: {
    module: device.createShaderModule({ label: "demo17-frag", code: FRAG_WGSL }),
    entryPoint: "main",
    targets: [{ format }],
  },
  primitive: { topology: "triangle-list" },
});
void device.popErrorScope().then((e) => {
  if (e) console.error("[demo17] Pipeline validation error:", e);
});

let bindGroup: GPUBindGroup | null = null;

function getBindGroup(): GPUBindGroup {
  if (bindGroup) return bindGroup;
  bindGroup = device.createBindGroup({
    label: "demo17-bg",
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: texView },
      { binding: 1, resource: getSampler() },
      { binding: 2, resource: { buffer: matBuf } },
    ],
  });
  return bindGroup;
}

// ---- Наложение --------------------------------------------------------------------------------------------------------------------------------------

const elGClamp = document.getElementById("g-clamp")!;
const elCols = [0, 1, 2, 3].map((i) => document.getElementById(`col${i}`)!);

function updateOverlay(tierClamp: number): void {
  elGClamp.textContent = String(tierClamp);

  for (let i = 0; i < 4; i++) {
    const rm = residentMip[i]!;
    const path = rm < tierClamp ? "SampleLevel (per-tex clamp active)" : "SampleLevel (aligned w/ tier)";
    const pathClass = rm < tierClamp ? "path-lod" : "path-hw";

    const col = elCols[i]!;
    // Сохраняем заголовок h4.
    const h4 = col.querySelector("h4")!;
    col.innerHTML = "";
    col.appendChild(h4);

    const addRow = (label: string, val: string, cls?: string): void => {
      const d = document.createElement("div");
      d.textContent = `${label}: ${val}`;
      if (cls) d.className = cls;
      col.appendChild(d);
    };

    addRow("residentMip", String(rm));
    addRow("tierLodMinClamp", String(tierClamp));
    addRow(
      "mipColor",
      rm < MIP_COLORS.length ? `mip${rm}` : "?",
    );
    addRow("path", path, pathClass);
    if (STREAM_INTERVALS[i] === Infinity) {
      addRow("rate", "static");
    } else {
      const mipsStreamed = (MIP_COUNT - 1) - rm;
      addRow("rate", `${STREAM_INTERVALS[i]}ms/mip (${mipsStreamed} done)`);
    }
  }
}

// ---- Цикл рендеринга ------------------------------------------------------------------------------------------------------------------------------

let lastTime = performance.now();

function frame(now: number): void {
  const dt = now - lastTime;
  lastTime = now;

  // ---- Потоковая передача мипов для каждой текстуры по индивидуальным таймерам ------------------------------------
  let anyUploaded = false;
  for (let i = 0; i < 4; i++) {
    const interval = STREAM_INTERVALS[i]!;
    if (interval === Infinity) continue;
    if (residentMip[i]! <= 0) continue; // уже на самом детальном мипе

    lastUploadTime[i] = (lastUploadTime[i] ?? 0) + dt;
    if (lastUploadTime[i]! >= interval) {
      lastUploadTime[i] = 0;
      const nextMip = residentMip[i]! - 1; // следующий более детальный мип
      uploadMip(i, nextMip);
      residentMip[i] = nextMip;
      anyUploaded = true;
    }
  }

  if (anyUploaded) updateMaterialBuffer();

  const tierClamp = computeTierLodMinClamp();
  updateOverlay(tierClamp);

  // ---- Рендеринг на GPU --------------------------------------------------------------------------------------------------------------------
  const bg = getBindGroup();
  const encoder = device.createCommandEncoder({ label: "demo17-enc" });
  const pass = encoder.beginRenderPass({
    label: "demo17-pass",
    colorAttachments: [{
      view: ctx.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store",
      clearValue: { r: 0.08, g: 0.08, b: 0.12, a: 1 },
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.draw(6, 4); // 6 вершин, 4 инстанса
  pass.end();
  device.queue.submit([encoder.finish()]);

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

} // end main()

main().catch((e) => { console.error(e); });
