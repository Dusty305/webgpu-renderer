/** Реэкспортирует все внутренние классы для использования демонстрациями и внутренними инструментами. */
export { DeviceManager } from "./DeviceManager.js";
export { RenderLoop } from "./RenderLoop.js";
export type { FrameCallback } from "./RenderLoop.js";
export { ResourceRegistryImpl } from "./ResourceRegistryImpl.js";
export { PluginHost } from "./PluginHost.js";
export { SceneGraph } from "./SceneGraph.js";
export { CameraController } from "./CameraController.js";
export { WebGPUElement } from "./WebGPUElement.js";
export type { WebGPUReadyDetail } from "./WebGPUElement.js";

// Публичный API (также доступен через потребительский index.ts)
export { WebGPUCanvasElement } from "./WebGPUCanvasPublic.js";
export { createRenderer } from "./api.js";

// Загрузчики форматов
export { detectFormat } from "./loaders/FormatDetector.js";
export { parseOBJ, parseOBJBuffer } from "./loaders/OBJLoader.js";
export { parseSTL } from "./loaders/STLLoader.js";
export type { ParsedModel, ParsedMesh } from "./loaders/OBJLoader.js";
