/**
 * Состояние камеры, передаваемое плагинам каждый кадр.
 */
export interface CameraState {
  /** Матрица вида 4x4, хранящаяся по столбцам */
  viewMatrix: Float32Array;
  /** Матрица проекции 4x4, хранящаяся по столбцам */
  projectionMatrix: Float32Array;
  /** Матрица вид-проекция 4x4, хранящаяся по столбцам */
  viewProjectionMatrix: Float32Array;
  /** Позиция камеры в мировом пространстве */
  position: Float32Array;
  /** Вертикальное поле зрения в радианах */
  fovY: number;
  /** Расстояние до ближней плоскости отсечения */
  near: number;
  /** Расстояние до дальней плоскости отсечения */
  far: number;
  /** Ширина области просмотра в пикселях (реальный размер пикселей холста, не CSS-размер) */
  viewportWidth: number;
  /** Высота области просмотра в пикселях (реальный размер пикселей холста, не CSS-размер) */
  viewportHeight: number;
}

/**
 * Узел единственного меша в графе сцены, доступный плагинам (только для чтения).
 */
export interface SceneNode {
  readonly id: string;
  /** Мировое преобразование 4x4, хранящееся по столбцам */
  readonly worldTransform: Float32Array;
  /** Ограничивающая сфера: [cx, cy, cz, radius] */
  readonly boundingSphere: Float32Array;
  /** Индекс идентификатора материала в буфере материалов */
  readonly materialId: number;
  /** Был ли этот узел отрисован в предыдущем кадре */
  readonly visible: boolean;
}

/**
 * Проекция графа сцены только для чтения, передаваемая плагинам каждый кадр.
 */
export interface SceneGraphReadView {
  readonly nodes: readonly SceneNode[];
}

/**
 * Контекстный объект, передаваемый в IResourceManager.prepareFrame и IRenderPass.execute.
 */
export interface FrameContext {
  readonly device: GPUDevice;
  readonly encoder: GPUCommandEncoder;
  readonly camera: CameraState;
  readonly scene: SceneGraphReadView;
  /** Монотонно возрастающий счётчик кадров */
  readonly frameIndex: number;
  /** Количество секунд, прошедших с предыдущего кадра */
  readonly deltaTime: number;
  /** Текущий вид текстуры холста для цветового буфера */
  readonly colorAttachment: GPUTextureView;
  /** Текущий вид текстуры глубины для буфера глубины */
  readonly depthAttachment: GPUTextureView;
}
