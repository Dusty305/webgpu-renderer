import { MATERIAL_ENTRY_WGSL } from "@webgpu-streaming/gpu-types";

// materialId sentinel: object uses baseColor (no texture)
export const NO_TEXTURE_MATERIAL_ID = 0xffffffff;

export const BUILTIN_WGSL = /* wgsl */ `
${MATERIAL_ENTRY_WGSL}

struct SceneUniforms {
  viewProj:   mat4x4<f32>,
  cameraPos:  vec4<f32>,
  lightDir:   vec4<f32>,
  lightColor: vec4<f32>,
}

// mat4(64) + baseColor(16) + materialId(4) + pad(12) = 96 bytes
struct ObjectUniforms {
  model:      mat4x4<f32>,
  baseColor:  vec4<f32>,
  materialId: u32,
  _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(0) var<uniform> obj:   ObjectUniforms;

// Group 2: streaming bind group (populated by TextureStreamingManager via registry)
@group(2) @binding(0) var tier0: texture_2d_array<f32>;
@group(2) @binding(1) var tier1: texture_2d_array<f32>;
@group(2) @binding(2) var tier2: texture_2d_array<f32>;
@group(2) @binding(3) var samp0: sampler;
@group(2) @binding(4) var samp1: sampler;
@group(2) @binding(5) var samp2: sampler;
@group(2) @binding(6) var<storage, read> materials: array<MaterialEntry>;

struct VSIn {
  @location(0) position: vec3<f32>,
  @location(1) normal:   vec3<f32>,
  @location(2) uv:       vec2<f32>,
}
struct VSOut {
  @builtin(position) clipPos:   vec4<f32>,
  @location(0)       worldNorm: vec3<f32>,
  @location(1)       uv:        vec2<f32>,
}

@vertex fn vs_main(in: VSIn) -> VSOut {
  var o: VSOut;
  o.clipPos   = scene.viewProj * obj.model * vec4<f32>(in.position, 1.0);
  o.worldNorm = normalize((obj.model * vec4<f32>(in.normal, 0.0)).xyz);
  o.uv        = in.uv;
  return o;
}

fn sampleAlbedo(matId: u32, uv: vec2<f32>) -> vec4<f32> {
  let m   = materials[matId];
  let lod = f32(m.residentMip);
  if (m.tierIndex == 0u) {
    return textureSampleLevel(tier0, samp0, uv, m.layerIndex, lod);
  } else if (m.tierIndex == 1u) {
    return textureSampleLevel(tier1, samp1, uv, m.layerIndex, lod);
  }
  return textureSampleLevel(tier2, samp2, uv, m.layerIndex, lod);
}

@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N     = normalize(in.worldNorm);
  let L     = normalize(-scene.lightDir.xyz);
  let NdotL = max(dot(N, L), 0.0);
  var albedo: vec3<f32>;
  if (obj.materialId == ${NO_TEXTURE_MATERIAL_ID}u) {
    albedo = obj.baseColor.rgb;
  } else {
    albedo = sampleAlbedo(obj.materialId, in.uv).rgb;
  }
  let diff = albedo * scene.lightColor.rgb * NdotL;
  let amb  = albedo * 0.15;
  return vec4<f32>(diff + amb, 1.0);
}
`;
