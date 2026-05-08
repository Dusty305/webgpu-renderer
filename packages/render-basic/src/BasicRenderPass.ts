import type {
  IRenderPass,
  RenderPassInitContext,
  FrameContext,
  ResourceRegistry,
} from "@webgpu-streaming/gpu-types";
export const MATERIAL_BIND_GROUP_KEY = "texture-streaming:materialBindGroup";

/** Шаг буфера юниформ на объект (должен быть >= minUniformBufferOffsetAlignment = 256). */
const OBJ_STRIDE = 256;
/** Максимальное количество объектов сцены на кадр. */
const MAX_OBJECTS = 256;
/** Размер буфера юниформ сцены в байтах. */
const SCENE_UBO_SIZE = 128;

// ---- WGSL ------------------------------------------------------------------------------------------------------------------------------------------

const VERT_WGSL = /* wgsl */ `
struct SceneUniforms {
  viewProj:   mat4x4<f32>,  // offset 0
  cameraPos:  vec4<f32>,    // offset 64
  lightDir:   vec4<f32>,    // offset 80
  lightColor: vec4<f32>,    // offset 96
}
struct ObjectUniforms {
  model:      mat4x4<f32>,
  materialId: u32,
  _pad0: u32, _pad1: u32, _pad2: u32,
}

@group(1) @binding(0) var<uniform> scene: SceneUniforms;
@group(2) @binding(0) var<uniform> obj:   ObjectUniforms;

struct VSIn {
  @location(0) position: vec3<f32>,
  @location(1) normal:   vec3<f32>,
  @location(2) uv:       vec2<f32>,
}
struct VSOut {
  @builtin(position)               clipPos:    vec4<f32>,
  @location(0)                     worldPos:   vec3<f32>,
  @location(1)                     worldNorm:  vec3<f32>,
  @location(2)                     uv:         vec2<f32>,
  @location(3) @interpolate(flat)  materialId: u32,
}

@vertex fn vs_main(in: VSIn) -> VSOut {
  let wp = (obj.model * vec4<f32>(in.position, 1.0)).xyz;
  let wn = normalize((obj.model * vec4<f32>(in.normal, 0.0)).xyz);
  var o: VSOut;
  o.clipPos    = scene.viewProj * vec4<f32>(wp, 1.0);
  o.worldPos   = wp;
  o.worldNorm  = wn;
  o.uv         = in.uv;
  o.materialId = obj.materialId;
  return o;
}`;

const FRAG_WGSL = /* wgsl */ `
struct SceneUniforms {
  viewProj:   mat4x4<f32>,
  cameraPos:  vec4<f32>,
  lightDir:   vec4<f32>,
  lightColor: vec4<f32>,
}
struct MaterialEntry {
  tierIndex:       u32,
  layerIndex:      u32,
  residentMip:     u32,
  // Консервативный tierLodMinClamp для уровня - max(residentMip) по всем текстурам уровня.
  tierLodMinClamp: u32,
}

@group(0) @binding(0) var tier0Tex:  texture_2d_array<f32>;
@group(0) @binding(1) var tier1Tex:  texture_2d_array<f32>;
@group(0) @binding(2) var tier2Tex:  texture_2d_array<f32>;
@group(0) @binding(3) var tier0Samp: sampler;
@group(0) @binding(4) var tier1Samp: sampler;
@group(0) @binding(5) var tier2Samp: sampler;
@group(0) @binding(6) var<storage, read> materials: array<MaterialEntry>;

@group(1) @binding(0) var<uniform> scene: SceneUniforms;

// dpdx/dpdy требуют однородного потока управления (анализ однородности WGSL).
// Производные UV вычисляются один раз в fs_main до любых неоднородных ветвлений,
// затем передаются в вспомогательную функцию, вычисляющую итоговый LOD для каждого уровня.
fn lodFromDerivatives(ddx_uv: vec2<f32>, ddy_uv: vec2<f32>, texSize: vec2<f32>, residentMip: u32) -> f32 {
  let ddx = ddx_uv * texSize;
  let ddy = ddy_uv * texSize;
  let d   = max(dot(ddx, ddx), dot(ddy, ddy));
  return max(0.5 * log2(max(d, 0.0001)), f32(residentMip));
}

@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let mat   = materials[in.materialId];
  let layer = i32(mat.layerIndex);

  // Производные вычисляются здесь в однородном потоке управления, до неоднородного
  // ветвления по mat.tierIndex (значение из storage-буфера, неоднородное для каждого вызова).
  let ddx_uv = dpdx(in.uv);
  let ddy_uv = dpdy(in.uv);

  var albedo: vec4<f32>;
  if mat.tierIndex == 0u {
    let sz  = vec2<f32>(textureDimensions(tier0Tex, 0).xy);
    albedo = textureSampleLevel(tier0Tex, tier0Samp, in.uv, layer, lodFromDerivatives(ddx_uv, ddy_uv, sz, mat.residentMip));
  } else if mat.tierIndex == 1u {
    let sz  = vec2<f32>(textureDimensions(tier1Tex, 0).xy);
    albedo = textureSampleLevel(tier1Tex, tier1Samp, in.uv, layer, lodFromDerivatives(ddx_uv, ddy_uv, sz, mat.residentMip));
  } else {
    let sz  = vec2<f32>(textureDimensions(tier2Tex, 0).xy);
    albedo = textureSampleLevel(tier2Tex, tier2Samp, in.uv, layer, lodFromDerivatives(ddx_uv, ddy_uv, sz, mat.residentMip));
  }

  // Диффузное освещение Ламберта + окружающий свет
  let N      = normalize(in.worldNorm);
  let L      = normalize(-scene.lightDir.xyz);
  let NdotL  = max(dot(N, L), 0.0);
  let diff   = albedo.rgb * scene.lightColor.rgb * NdotL;
  let amb    = albedo.rgb * 0.12;

  return vec4<f32>(diff + amb, 1.0);
}

// Требуется в стадии фрагмента для ссылки на структуру
struct VSOut {
  @builtin(position)               clipPos:    vec4<f32>,
  @location(0)                     worldPos:   vec3<f32>,
  @location(1)                     worldNorm:  vec3<f32>,
  @location(2)                     uv:         vec2<f32>,
  @location(3) @interpolate(flat)  materialId: u32,
}`;

// ---- Типы ------------------------------------------------------------------------------------------------------------------------------------------

interface MeshGPU {
  vbo: GPUBuffer;
  ibo: GPUBuffer;
  indexCount: number;
}

/**
 * Упрощённый форвард-рендерер PBR (диффузное освещение Ламберта, без IBL).
 *
 * Группа 0: bind group материала (из реестра TextureStreamingManager, или резервный белый).
 * Группа 1: юниформы сцены (viewProj, lightDir, lightColor).
 * Группа 2: юниформы на объект с динамическим смещением (матрица модели, materialId).
 *
 * Использование:
 *   1. Вызвать registerMesh() для загрузки геометрии.
 *   2. Вызвать addObject() для связи ID узлов сцены с мешами.
 *   3. Передать в WebGPUElement.addRenderPass().
 *   4. Заполнить узлы sceneGraph; BasicRenderPass отрисовывает их каждый кадр.
 */
export class BasicRenderPass implements IRenderPass {
  readonly name = "basic-render-pass";

  private _device: GPUDevice | null = null;
  private _registry: ResourceRegistry | null = null;
  private _pipeline: GPURenderPipeline | null = null;

  private readonly _meshes = new Map<string, MeshGPU>();
  private readonly _objects = new Map<string, string>(); // nodeId → meshId

  // Буфер юниформ сцены (группа 1)
  private _sceneUbo: GPUBuffer | null = null;
  private _sceneBg: GPUBindGroup | null = null;

  // Буфер юниформ на объект (группа 2, динамическое смещение)
  private _objUbo: GPUBuffer | null = null;
  private _objBg: GPUBindGroup | null = null;

  // Резервная белая bind group материала (группа 0, используется если в реестре нет записи)
  private _fallbackBg: GPUBindGroup | null = null;
  private _fallbackTex: GPUTexture | null = null;
  private _fallbackMatBuf: GPUBuffer | null = null;

  private _lightDir = new Float32Array([0.5773, -0.5773, -0.5773]);
  private _lightColor = new Float32Array([1.2, 1.1, 1.0]);
  private _aspect = 1;

  // Переиспользуемые буферы для загрузки UBO на кадр - выделяются один раз, не на каждый кадр.
  private readonly _sceneUboData = new Float32Array(SCENE_UBO_SIZE / 4);
  private readonly _objUboData   = new Float32Array(MAX_OBJECTS * OBJ_STRIDE / 4);

  /**
   * Загрузить данные вершин/индексов для меша.
   * Должен вызываться после initialize().
   *
   * @param meshId - Уникальный идентификатор меша.
   * @param vertices - Чередующиеся данные: position(vec3f), normal(vec3f), uv(vec2f) - 32 байта/вершина.
   * @param indices  - Индексы списка треугольников (Uint16Array).
   */
  registerMesh(meshId: string, vertices: Float32Array<ArrayBuffer>, indices: Uint16Array<ArrayBuffer>): void {
    if (!this._device) throw new Error("[BasicRenderPass] registerMesh вызван до initialize()");

    // Удалить старые буферы при повторной регистрации.
    const old = this._meshes.get(meshId);
    if (old) { old.vbo.destroy(); old.ibo.destroy(); }

    const device = this._device;
    device.pushErrorScope("out-of-memory");
    const vbo = device.createBuffer({
      label: `vbo-${meshId}`,
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[BasicRenderPass] OOM vbo:", e); });

    device.pushErrorScope("out-of-memory");
    const ibo = device.createBuffer({
      label: `ibo-${meshId}`,
      size: Math.ceil(indices.byteLength / 4) * 4, // выравнивание по 4 байтам
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[BasicRenderPass] OOM ibo:", e); });

    device.queue.writeBuffer(vbo, 0, vertices);
    device.queue.writeBuffer(ibo, 0, indices);

    this._meshes.set(meshId, { vbo, ibo, indexCount: indices.length });
  }

  /**
   * Связать ID узла сцены с зарегистрированным мешем.
   * Узел должен существовать в SceneGraph, чтобы быть отрисованным.
   */
  addObject(nodeId: string, meshId: string): void {
    this._objects.set(nodeId, meshId);
  }

  /** Удалить объект по ID узла. */
  removeObject(nodeId: string): void {
    this._objects.delete(nodeId);
  }

  async initialize(ctx: RenderPassInitContext): Promise<void> {
    const { device, presentationFormat } = ctx;
    this._device = device;
    this._registry = ctx.registry;

    // ---- Явные layouts bind group --------------------------------------------------------------------------------------------
    // Группа 0: bind group материала (точно совпадает с layout BindGroupManager)
    const matBgl = device.createBindGroupLayout({
      label: "basic-mat-bgl",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      ],
    });

    // Группа 1: юниформы сцены
    const sceneBgl = device.createBindGroupLayout({
      label: "basic-scene-bgl",
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform", minBindingSize: SCENE_UBO_SIZE },
      }],
    });

    // Группа 2: юниформы на объект с динамическим смещением
    const objBgl = device.createBindGroupLayout({
      label: "basic-obj-bgl",
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 80 },
      }],
    });

    // ---- Пайплайн ----------------------------------------------------------------------------------------------------------------------------
    const layout = device.createPipelineLayout({
      label: "basic-pipeline-layout",
      bindGroupLayouts: [matBgl, sceneBgl, objBgl],
    });

    device.pushErrorScope("validation");
    this._pipeline = device.createRenderPipeline({
      label: "basic-render-pipeline",
      layout,
      vertex: {
        module: device.createShaderModule({ label: "basic-vert", code: VERT_WGSL }),
        entryPoint: "vs_main",
        buffers: [{
          arrayStride: 32,
          attributes: [
            { shaderLocation: 0, offset: 0,  format: "float32x3" }, // позиция
            { shaderLocation: 1, offset: 12, format: "float32x3" }, // нормаль
            { shaderLocation: 2, offset: 24, format: "float32x2" }, // uv
          ],
        }],
      },
      fragment: {
        module: device.createShaderModule({ label: "basic-frag", code: FRAG_WGSL }),
        entryPoint: "fs_main",
        targets: [{ format: presentationFormat }],
      },
      primitive: { topology: "triangle-list", cullMode: "back" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });
    void device.popErrorScope().then((e) => {
      if (e) console.error("[BasicRenderPass] Ошибка валидации пайплайна:", e);
    });

    // ---- UBO сцены (группа 1) --------------------------------------------------------------------------------------------------
    device.pushErrorScope("out-of-memory");
    this._sceneUbo = device.createBuffer({
      label: "scene-ubo",
      size: SCENE_UBO_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[BasicRenderPass] OOM UBO сцены:", e); });

    this._sceneBg = device.createBindGroup({
      label: "scene-bg",
      layout: sceneBgl,
      entries: [{ binding: 0, resource: { buffer: this._sceneUbo } }],
    });

    // ---- UBO на объект (группа 2, динамическое смещение) --------------------------------------------
    device.pushErrorScope("out-of-memory");
    this._objUbo = device.createBuffer({
      label: "obj-ubo",
      size: MAX_OBJECTS * OBJ_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[BasicRenderPass] OOM UBO объекта:", e); });

    this._objBg = device.createBindGroup({
      label: "obj-bg",
      layout: objBgl,
      entries: [{ binding: 0, resource: { buffer: this._objUbo, size: OBJ_STRIDE } }],
    });

    // ---- Резервная белая bind group (группа 0) ----------------------------------------------------------------
    device.pushErrorScope("out-of-memory");
    this._fallbackTex = device.createTexture({
      label: "fallback-white-tex",
      size: [1, 1, 1],
      format: "rgba8unorm",
      dimension: "2d",
      mipLevelCount: 1,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    void device.popErrorScope().then((e) => { if (e) console.error("[BasicRenderPass] OOM резервная текстура:", e); });
    const fallbackTex = this._fallbackTex;
    device.queue.writeTexture(
      { texture: fallbackTex },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4, rowsPerImage: 1 },
      [1, 1, 1]
    );
    const fallbackView = fallbackTex.createView({ dimension: "2d-array" });
    const fallbackSampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

    this._fallbackMatBuf = device.createBuffer({
      label: "fallback-mat-buf",
      size: 16, // одна запись MaterialEntry: tierIndex=0, layerIndex=0, residentMip=0, _pad=0
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this._fallbackMatBuf, 0, new Uint32Array([0, 0, 0, 0]));

    this._fallbackBg = device.createBindGroup({
      label: "fallback-mat-bg",
      layout: matBgl,
      entries: [
        { binding: 0, resource: fallbackView },
        { binding: 1, resource: fallbackView },
        { binding: 2, resource: fallbackView },
        { binding: 3, resource: fallbackSampler },
        { binding: 4, resource: fallbackSampler },
        { binding: 5, resource: fallbackSampler },
        { binding: 6, resource: { buffer: this._fallbackMatBuf } },
      ],
    });

  }

  execute(ctx: FrameContext): void {
    if (!this._pipeline || !this._sceneUbo || !this._sceneBg || !this._objUbo || !this._objBg) return;

    const device = ctx.device;
    const nodes  = ctx.scene.nodes.filter((n) => n.visible);
    if (nodes.length === 0) return;

    // ---- Обновить UBO сцены ------------------------------------------------------------------------------------------------------
    const sceneData = this._sceneUboData;
    sceneData.set(ctx.camera.viewProjectionMatrix, 0);          // viewProj: [0..15]
    sceneData[16] = ctx.camera.position[0] ?? 0;                // cameraPos.x
    sceneData[17] = ctx.camera.position[1] ?? 0;
    sceneData[18] = ctx.camera.position[2] ?? 0;
    sceneData[19] = 0;
    sceneData[20] = this._lightDir[0] ?? 0;                     // lightDir
    sceneData[21] = this._lightDir[1] ?? 0;
    sceneData[22] = this._lightDir[2] ?? 0;
    sceneData[23] = 0;
    sceneData[24] = this._lightColor[0] ?? 1;                   // lightColor
    sceneData[25] = this._lightColor[1] ?? 1;
    sceneData[26] = this._lightColor[2] ?? 1;
    sceneData[27] = 0;
    device.queue.writeBuffer(this._sceneUbo, 0, sceneData);

    // ---- Обновить UBO на объект ----------------------------------------------------------------------------------------------
    const objData = this._objUboData;
    let drawCount = 0;
    const drawNodes: typeof nodes = [];

    for (const node of nodes) {
      if (drawCount >= MAX_OBJECTS) break;
      const meshId = this._objects.get(node.id);
      if (!meshId || !this._meshes.has(meshId)) continue;

      const base = drawCount * (OBJ_STRIDE / 4);
      objData.set(node.worldTransform, base);                    // model[0..15]
      const matId = new Uint32Array(objData.buffer, (base + 16) * 4, 4);
      matId[0] = node.materialId;
      matId[1] = 0; matId[2] = 0; matId[3] = 0;

      drawNodes.push(node);
      drawCount++;
    }
    device.queue.writeBuffer(this._objUbo, 0, objData);

    // ---- Bind group материала (стриминг или резервная) ------------------------------------------------
    const matBg = this._registry?.request<GPUBindGroup>(MATERIAL_BIND_GROUP_KEY)
                  ?? this._fallbackBg!;

    // ---- Проход рендеринга --------------------------------------------------------------------------------------------------------
    const pass = ctx.encoder.beginRenderPass({
      label: "basic-render-pass",
      colorAttachments: [{
        view: ctx.colorAttachment,
        loadOp: "load",
        storeOp: "store",
      }],
      depthStencilAttachment: {
        view: ctx.depthAttachment,
        depthLoadOp: "load",
        depthStoreOp: "store",
      },
    });

    pass.setPipeline(this._pipeline);
    pass.setBindGroup(0, matBg);
    pass.setBindGroup(1, this._sceneBg);

    for (let i = 0; i < drawNodes.length; i++) {
      const node   = drawNodes[i]!;
      const meshId = this._objects.get(node.id)!;
      const mesh   = this._meshes.get(meshId)!;
      pass.setBindGroup(2, this._objBg, [i * OBJ_STRIDE]);
      pass.setVertexBuffer(0, mesh.vbo);
      pass.setIndexBuffer(mesh.ibo, "uint16");
      pass.drawIndexed(mesh.indexCount);
    }

    pass.end();
  }

  onResize(width: number, height: number): void {
    this._aspect = height > 0 ? width / height : 1;
  }

  destroy(): void {
    for (const mesh of this._meshes.values()) {
      mesh.vbo.destroy();
      mesh.ibo.destroy();
    }
    this._meshes.clear();
    this._objects.clear();
    this._sceneUbo?.destroy();
    this._objUbo?.destroy();
    this._fallbackTex?.destroy();
    this._fallbackMatBuf?.destroy();
    this._sceneUbo = null;
    this._objUbo = null;
    this._fallbackTex = null;
    this._fallbackMatBuf = null;
    this._pipeline = null;
    this._sceneBg = null;
    this._objBg = null;
    this._fallbackBg = null;
    this._registry = null;
    this._device = null;
  }
}
