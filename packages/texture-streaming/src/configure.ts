/**
 * Необязательное переопределение глобальных настроек потоковой передачи по умолчанию.
 * Вызовите до createRenderer(), чтобы изменить значения по умолчанию для всех новых рендереров.
 */
export interface StreamingConfig {
  /** Бюджет видеопамяти по умолчанию в МБ. По умолчанию: 256 */
  defaultMemoryBudget?: number;
  /** Максимальный объём загрузки за кадр по умолчанию в МБ. По умолчанию: 8 */
  defaultFrameUploadCap?: number;
  /** Тиры разрешений по умолчанию. По умолчанию: [512, 1024, 2048] */
  defaultTextureTiers?: number[];
}

let _config: StreamingConfig = {};

/**
 * Переопределить конфигурацию потоковой передачи по умолчанию для всех последующих рендереров.
 *
 * @example
 * configureStreaming({ defaultMemoryBudget: 512, defaultFrameUploadCap: 16 });
 */
export function configureStreaming(config: StreamingConfig): void {
  _config = { ..._config, ...config };
}

/** Прочитать текущие глобальные настройки потоковой передачи по умолчанию. */
export function getStreamingDefaults(): Readonly<StreamingConfig> {
  return _config;
}
