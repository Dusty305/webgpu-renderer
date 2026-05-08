import { defineConfig } from "vite";
import { resolve } from "path";
import { fileURLToPath } from "url";

const __dirname = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  root:      resolve(__dirname, "src/20-showcase"),
  publicDir: resolve(__dirname, "public"),
  base:      "./",
  build: {
    outDir:     resolve(__dirname, "../../docs/demo"),
    emptyOutDir: true,
    rollupOptions: {
      input: { index: resolve(__dirname, "src/20-showcase/index.html") },
    },
  },
  plugins: [
    {
      name: "remove-crossorigin",
      transformIndexHtml(html: string) {
        return html.replace(/<script ([^>]*)crossorigin([^>]*)>/g, "<script $1$2>");
      },
    },
  ],
  resolve: {
    alias: {
      "@webgpu-streaming/gpu-types":        resolve(__dirname, "../gpu-types/src/index.ts"),
      "@webgpu-streaming/core":             resolve(__dirname, "../core/src/_internal.ts"),
      "@webgpu-streaming/render-basic":     resolve(__dirname, "../render-basic/src/_internal.ts"),
      "@webgpu-streaming/texture-streaming":resolve(__dirname, "../texture-streaming/src/_internal.ts"),
    },
  },
});
