/**
 * @webgpu-streaming/core - Публичный API
 *
 * Два варианта интеграции:
 *  1. Custom Element: импортировать этот модуль (побочный эффект регистрирует <webgpu-canvas-streaming>),
 *     затем использовать элемент в HTML.
 *  2. Программный: вызвать createRenderer() с существующим <canvas>.
 */

// ---- Custom Element ------------------------------------------------------------------------------------------------------------------------
export { WebGPUCanvasElement } from "./WebGPUCanvasPublic.js";

// ---- Загрузчики форматов (разбор → GeometryDescriptor[]) --------------------------------------------
export { detectFormat } from "./loaders/FormatDetector.js";
export { parseOBJ, parseOBJBuffer } from "./loaders/OBJLoader.js";
export { parseSTL } from "./loaders/STLLoader.js";
export type { ParsedModel, ParsedMesh } from "./loaders/OBJLoader.js";

// ---- Программный API ----------------------------------------------------------------------------------------------------------------------
export { createRenderer } from "./api.js";

// ---- Публичные типы (реэкспортируются для удобства потребителей) ------------------------------
export type {
  TextureOptions,
  TextureHandle,
  GeometryDescriptor,
  MaterialDescriptor,
  MeshHandle,
  CameraOptions,
  StreamingStats,
  CreateRendererOptions,
  Renderer,
} from "@webgpu-streaming/gpu-types";
