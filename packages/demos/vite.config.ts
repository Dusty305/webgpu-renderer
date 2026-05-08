import { defineConfig } from "vite";
import { resolve } from "path";
import { readdirSync, existsSync, createReadStream, statSync } from "fs";
import { extname } from "path";
import { fileURLToPath } from "url";

// Аналог __dirname для ESM
const __dirname = fileURLToPath(new URL(".", import.meta.url));

// Автоматически обнаруживает все точки входа демо.
function discoverEntries(): Record<string, string> {
  const srcDir = resolve(__dirname, "src");
  const entries: Record<string, string> = {};

  if (!existsSync(srcDir)) return entries;

  for (const dir of readdirSync(srcDir)) {
    const htmlPath = resolve(srcDir, dir, "index.html");
    if (existsSync(htmlPath)) {
      entries[dir] = htmlPath;
    }
  }
  return entries;
}

export default defineConfig({
  root: resolve(__dirname, "src"),
  publicDir: resolve(__dirname, "public"),
  plugins: [
    {
      name: "trailing-slash-redirect",
      configureServer(server) {
        server.middlewares.use((req, _res, next) => {
          // Перенаправляет пути без слеша в конце, чтобы /05 и /05/ оба вели к index.html.
          if (req.url && !req.url.endsWith("/") && !req.url.includes(".")) {
            req.url = req.url + "/";
          }
          next();
        });
      },
    },
    {
      // Обслуживает packages/demos/assets/ по URL /assets/ для загрузки glTF-сцен.
      name: "serve-demo-assets",
      configureServer(server) {
        const assetsRoot = resolve(__dirname, "assets");
        const MIME: Record<string, string> = {
          ".gltf": "application/json",
          ".bin":  "application/octet-stream",
          ".jpg":  "image/jpeg",
          ".jpeg": "image/jpeg",
          ".png":  "image/png",
          ".ktx2": "image/ktx2",
        };
        server.middlewares.use("/assets", (req, res, next) => {
          const rel = (req.url ?? "/").replace(/\?.*$/, "");
          const filePath = resolve(assetsRoot, "." + rel);
          if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
            next(); return;
          }
          const mime = MIME[extname(filePath).toLowerCase()] ?? "application/octet-stream";
          res.setHeader("Content-Type", mime);
          res.setHeader("Cache-Control", "no-cache");
          createReadStream(filePath).pipe(res);
        });
      },
    },
  ],
  build: {
    outDir: resolve(__dirname, "dist"),
    rollupOptions: {
      input: discoverEntries(),
    },
  },
  resolve: {
    alias: {
      "@webgpu-streaming/gpu-types": resolve(__dirname, "../gpu-types/src/index.ts"),
      // Демо 01–15 используют внутренние классы; ссылаемся на _internal.ts,
      // чтобы публичный index.ts экспортировал только потребительский API.
      "@webgpu-streaming/core": resolve(__dirname, "../core/src/_internal.ts"),
      "@webgpu-streaming/render-basic": resolve(__dirname, "../render-basic/src/_internal.ts"),
      "@webgpu-streaming/texture-streaming": resolve(__dirname, "../texture-streaming/src/_internal.ts"),
    },
  },
});
