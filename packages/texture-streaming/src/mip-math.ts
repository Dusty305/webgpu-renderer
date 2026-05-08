/**
 * Утилиты для чистых математических вычислений уровней мипмапов.
 * Без GPU-состояния - безопасно тестировать в Node.
 */

/**
 * Вычислить желаемый уровень мипмапа для текстуры на основе параметров просмотра.
 * @param distance - Расстояние от камеры до центра объекта (мировые единицы)
 * @param texWidth - Базовая ширина текстуры в текселях
 * @param screenHeight - Высота области просмотра в пикселях
 * @param fovY - Вертикальное поле зрения в радианах
 * @returns Целочисленный уровень мипмапа (0 = полное разрешение, больше = грубее)
 */
export function computeDesiredMip(
  distance: number,
  texWidth: number,
  screenHeight: number,
  fovY: number
): number {
  if (distance <= 0 || texWidth <= 0 || screenHeight <= 0) return 0;
  // λ = log2(distance × texWidth / (screenHeight × tan(fovY/2)))
  // Вывод: на расстоянии d объект размером 1 мировая единица занимает
  //   screenHeight / (2d × tan(fovY/2)) пикселей. Текстура имеет texWidth текселей.
  // Нам нужно 1 тексель ≈ 1 пиксель → mip = log2(texWidth / projectedPixels).
  const projectedPixels = screenHeight / (2 * distance * Math.tan(fovY / 2));
  if (projectedPixels <= 0) return 0;
  return Math.max(0, Math.floor(Math.log2(texWidth / projectedPixels)));
}

/**
 * Вычислить долю экрана, занятую ограничивающей сферой.
 * @param boundingSphere - [cx, cy, cz, radius] в мировом пространстве
 * @param viewProjection - Матрица вид-проекция 4x4 в столбцово-мажорном порядке
 * @param viewportWidth - Ширина области просмотра в пикселях
 * @param viewportHeight - Высота области просмотра в пикселях
 * @returns Доля покрытия в диапазоне [0, 1]
 */
export function computeScreenCoverage(
  boundingSphere: Float32Array,
  viewProjection: Float32Array,
  viewportWidth: number,
  viewportHeight: number
): number {
  const cx = boundingSphere[0] ?? 0;
  const cy = boundingSphere[1] ?? 0;
  const cz = boundingSphere[2] ?? 0;
  const r  = boundingSphere[3] ?? 0;

  // Преобразовать центр в пространство отсечения.
  const w =
    viewProjection[3]! * cx +
    viewProjection[7]! * cy +
    viewProjection[11]! * cz +
    viewProjection[15]!;

  if (w <= 0) return 0;

  // Проецировать радиус: аппроксимация с использованием масштаба по оси X.
  const scaleX = viewProjection[0]!;
  const projectedRadius = (r * scaleX) / w;
  const ndcArea = Math.PI * projectedRadius * projectedRadius;
  const screenArea = viewportWidth * viewportHeight;
  const ndcScreenArea = 4; // NDC-квадрат [-1,1]×[-1,1] имеет площадь 4

  return Math.min(1, (ndcArea / ndcScreenArea) * screenArea / (viewportWidth * viewportHeight));
}

/**
 * Вычислить размер в байтах одного уровня мипмапа для заданного формата.
 * @param width - Базовая ширина текстуры
 * @param height - Базовая высота текстуры
 * @param format - Строка GPUTextureFormat
 * @param mipLevel - Индекс уровня мипмапа (0 = полное разрешение)
 * @returns Размер этого уровня мипмапа в байтах
 */
export function mipSizeBytes(
  width: number,
  height: number,
  format: string,
  mipLevel: number
): number {
  const mipWidth  = Math.max(1, width  >> mipLevel);
  const mipHeight = Math.max(1, height >> mipLevel);
  return mipWidth * mipHeight * bytesPerTexel(format);
}

/**
 * Возвращает байт на тексель (или на блок для форматов со сжатием).
 * Блочно-сжатые форматы используют блоки 4x4.
 */
function bytesPerTexel(format: string): number {
  switch (format) {
    case "rgba8unorm":
    case "rgba8unorm-srgb":
    case "bgra8unorm":
    case "bgra8unorm-srgb":
      return 4;
    case "rgba16float":
      return 8;
    case "rgba32float":
      return 16;
    case "bc7-rgba-unorm":
    case "bc7-rgba-unorm-srgb":
    case "bc1-rgba-unorm":
    case "bc1-rgba-unorm-srgb":
      return 1; // 16 байт на блок 4x4 → 1 байт на тексель (приблизительно)
    case "astc-4x4-unorm":
    case "astc-4x4-unorm-srgb":
      return 1;
    default:
      return 4;
  }
}
