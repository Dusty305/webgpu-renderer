/**
 * @webgpu-streaming/render-basic - Публичный API
 *
 * Необязательно: вызвать configureRenderer() до createRenderer() для переопределения настроек освещения.
 */
export { configureRenderer, getRendererDefaults } from "./configure.js";
export type { RendererConfig } from "./configure.js";
