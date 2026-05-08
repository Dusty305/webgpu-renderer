/**
 * Публичные типы пакетов @webgpu-streaming.
 * Это единственные типы, необходимые потребителям для взаимодействия с библиотекой.
 */

/** Параметры-подсказки при регистрации текстуры для потоковой загрузки. */
export interface TextureOptions {
  /** Подсказка для выбора уровня разрешения. Определяется автоматически, если не задано. */
  resolution?: 512 | 1024 | 2048 | 4096;
  /** Подсказка по использованию текстуры. Влияет на выбор формата сжатия. По умолчанию: "color" */
  usage?: "color" | "normal" | "orm";
  /** Смещение приоритета: чем выше, тем раньше будет загружена потоком. По умолчанию: 0 */
  priority?: number;
}

/** Непрозрачный дескриптор текстуры, зарегистрированной в системе потоковой загрузки. */
export interface TextureHandle {
  readonly id: string;
  readonly width: number;
  readonly height: number;
  readonly mipLevels: number;
  /** Текущий резидентный уровень мипа (0 = наиболее детальный, mipLevels-1 = наименее детальный). */
  readonly residentMip: number;
  destroy(): void;
}

/** Сырые данные геометрии меша. */
export interface GeometryDescriptor {
  positions: Float32Array;
  normals?: Float32Array;
  uvs?: Float32Array;
  indices?: Uint16Array | Uint32Array;
}

/** Свойства материала для меша. */
export interface MaterialDescriptor {
  /** Строка цвета в формате hex (например, "#ff6600") или TextureHandle. */
  baseColor?: string | TextureHandle;
  normalMap?: TextureHandle;
  metallicRoughness?: TextureHandle | { metallic: number; roughness: number };
  emissive?: string | TextureHandle;
}

/** Непрозрачный дескриптор меша, добавленного в рендерер. */
export interface MeshHandle {
  readonly id: string;
  /** Заменить полное мировое преобразование (матрица 4x4, хранящаяся по столбцам). */
  setTransform(matrix: Float32Array): void;
  setPosition(x: number, y: number, z: number): void;
  setRotation(x: number, y: number, z: number): void;
  setScale(x: number, y: number, z: number): void;
  readonly visible: boolean;
  setVisible(v: boolean): void;
  destroy(): void;
}

/** Параметры позиционирования камеры. */
export interface CameraOptions {
  position?: [number, number, number];
  target?: [number, number, number];
  fov?: number;
  near?: number;
  far?: number;
}

/** Статистика рендеринга и потоковой загрузки в реальном времени. */
export interface StreamingStats {
  memoryUsedMB: number;
  memoryBudgetMB: number;
  texturesLoaded: number;
  texturesTotal: number;
  /** Карта от уровня мипа к числу текстур на этом уровне. */
  residentMipDistribution: Record<number, number>;
  fps: number;
  frameTimeP99Ms: number;
}

/** Параметры для createRenderer(). */
export interface CreateRendererOptions {
  /** Элемент холста для рендеринга. */
  canvas: HTMLCanvasElement;
  /** Бюджет памяти GPU для текстур в МБ. По умолчанию: 256 */
  memoryBudget?: number;
  /** Максимальный объём данных текстур, загружаемых за кадр, в МБ. По умолчанию: 8 */
  frameUploadCap?: number;
  /** Уровни разрешения в пикселях. По умолчанию: [512, 1024, 2048] */
  textureTiers?: number[];
  /** Максимальное число слоёв массива текстур на уровень. По умолчанию: 64 */
  maxLayersPerTier?: number;
  powerPreference?: "high-performance" | "low-power";
}

/** Программный API рендерера, возвращаемый createRenderer(). */
export interface Renderer {
  /** Загрузить сцену glTF по URL или из ArrayBuffer. */
  loadScene(source: string | ArrayBuffer): Promise<void>;
  /** Зарегистрировать текстуру для потоковой загрузки мипов. Возвращает дескриптор для использования в материалах. */
  loadTexture(source: string | ArrayBuffer, options?: TextureOptions): Promise<TextureHandle>;
  /** Добавить меш в сцену. Возвращает дескриптор для обновления или удаления позднее. */
  addMesh(geometry: GeometryDescriptor, material: MaterialDescriptor): MeshHandle;
  /** Удалить ранее добавленный меш. */
  removeMesh(handle: MeshHandle): void;
  /** Обновить позицию камеры, точку цели и проекцию. */
  setCamera(options: CameraOptions): void;
  /** Вернуть текущую статистику рендеринга и потоковой загрузки. */
  getStats(): StreamingStats;
  /** Уничтожить все GPU-ресурсы и остановить цикл рендеринга. */
  dispose(): void;
  /** Базовый GPUDevice (для продвинутого использования плагинами). */
  readonly device: GPUDevice;
  /** True, когда рендерер полностью инициализирован и выполняет рендеринг. */
  readonly ready: boolean;
}
