export const BUILTIN_WGSL = /* wgsl */ `
struct SceneUniforms {
  viewProj:   mat4x4<f32>,
  cameraPos:  vec4<f32>,
  lightDir:   vec4<f32>,
  lightColor: vec4<f32>,
}
struct ObjectUniforms {
  model:      mat4x4<f32>,
  baseColor:  vec4<f32>,
  useTexture: u32,
}

@group(0) @binding(0) var<uniform> scene: SceneUniforms;
@group(1) @binding(0) var<uniform> obj: ObjectUniforms;
@group(2) @binding(0) var albedoTex:  texture_2d<f32>;
@group(2) @binding(1) var albedoSamp: sampler;

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

@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  let N     = normalize(in.worldNorm);
  let L     = normalize(-scene.lightDir.xyz);
  let NdotL = max(dot(N, L), 0.0);
  var albedo: vec3<f32>;
  if (obj.useTexture == 1u) {
    albedo = textureSample(albedoTex, albedoSamp, in.uv).rgb;
  } else {
    albedo = obj.baseColor.rgb;
  }
  let diff = albedo * scene.lightColor.rgb * NdotL;
  let amb  = albedo * 0.15;
  return vec4<f32>(diff + amb, 1.0);
}
`;
