/**
 * Необязательное переопределение настроек рендеринга BasicRenderPass по умолчанию.
 * Вызывается до createRenderer() для изменения настроек освещения и т.д.
 */
export interface RendererConfig {
  /** Направление направленного света [x, y, z]. По умолчанию: [0.5773, -0.5773, -0.5773] */
  lightDir?: [number, number, number];
  /** Цвет направленного света [r, g, b]. По умолчанию: [1.2, 1.1, 1.0] */
  lightColor?: [number, number, number];
  /** Множитель интенсивности окружающего освещения. По умолчанию: 0.12 */
  ambientIntensity?: number;
}

let _config: RendererConfig = {};

/**
 * Переопределить конфигурацию прохода рендеринга по умолчанию для всех последующих создаваемых рендереров.
 *
 * @example
 * configureRenderer({ lightDir: [0, -1, 0], lightColor: [1, 1, 1] });
 */
export function configureRenderer(config: RendererConfig): void {
  _config = { ..._config, ...config };
}

/** Получить текущую глобальную конфигурацию рендерера. */
export function getRendererDefaults(): Readonly<RendererConfig> {
  return _config;
}
