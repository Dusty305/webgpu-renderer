import { defineConfig } from "tsup";

export default defineConfig([
  {
    entry: ["src/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    sourcemap: true,
    clean: true,
    treeshake: true,
    splitting: false,
    target: "es2022",
    tsconfig: "tsconfig.json",
    external: ["@webgpu-streaming/core", "@webgpu-streaming/gpu-types", "@webgpu-streaming/render-basic", "@webgpu-streaming/texture-streaming"],
  },
  {
    entry: { "webgpu-streaming": "src/index.ts" },
    format: ["iife"],
    globalName: "WebGPUStreaming",
    sourcemap: true,
    treeshake: true,
    splitting: false,
    target: "es2022",
    tsconfig: "tsconfig.json",
    noExternal: ["@webgpu-streaming/core", "@webgpu-streaming/gpu-types", "@webgpu-streaming/render-basic", "@webgpu-streaming/texture-streaming"],
    outDir: "dist",
  },
]);
