/**
 * QualityMetrics - считывание с GPU, PSNR и SSIM для сравнения бенчмарков.
 *
 * Все тяжёлые вычисления выполняются на CPU без использования внешних библиотек.
 *
 * Использование:
 *   1. Вызовите captureFrame() после стабилизации рендера каждой конфигурации,
 *      чтобы получить Uint8Array пикселей RGBA8 в разрешении canvas.
 *   2. Сохраните снимок конфигурации A как эталон.
 *   3. Вызовите computePSNR(ref, test) и computeSSIM(ref, test, w, h) для
 *      каждой другой конфигурации при той же позиции камеры.
 */

// ---- Считывание с GPU ------------------------------------------------------------------------------------------------------------------------------------

/**
 * Рендерит один кадр во внеэкранную текстуру rgba8unorm, затем копирует на CPU.
 *
 * @param device   - Активный GPUDevice.
 * @param width    - Ширина захвата в пикселях.
 * @param height   - Высота захвата в пикселях.
 * @param renderFn - Коллбэк, записывающий команды рисования в переданный
 *                   энкодер с целевым `colorView`.
 * @returns Пиксельные данные RGBA8 (width × height × 4 байта, row-major).
 */
export async function captureFrame(
  device: GPUDevice,
  width:  number,
  height: number,
  renderFn: (encoder: GPUCommandEncoder, colorView: GPUTextureView) => void,
): Promise<Uint8Array> {
  // Внеэкранный целевой буфер рендера.
  const renderTex = device.createTexture({
    label:  "capture-render",
    size:   [width, height, 1],
    format: "rgba8unorm",
    usage:  GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  // Промежуточный буфер - bytesPerRow должен быть выровнен до 256.
  const bytesPerRow    = Math.ceil((width * 4) / 256) * 256;
  const stagingBytes   = bytesPerRow * height;
  const stagingBuffer  = device.createBuffer({
    label:  "capture-staging",
    size:   stagingBytes,
    usage:  GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const encoder   = device.createCommandEncoder({ label: "capture-enc" });
  const colorView = renderTex.createView();

  renderFn(encoder, colorView);

  encoder.copyTextureToBuffer(
    { texture: renderTex },
    { buffer: stagingBuffer, bytesPerRow, rowsPerImage: height },
    [width, height, 1],
  );
  device.queue.submit([encoder.finish()]);

  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(stagingBuffer.getMappedRange());

  // Убираем шаг: копируем только активные пиксели (отбрасываем столбцы-паддинг).
  const pixels = new Uint8Array(width * height * 4);
  const rowBytes = width * 4;
  for (let row = 0; row < height; row++) {
    pixels.set(
      mapped.subarray(row * bytesPerRow, row * bytesPerRow + rowBytes),
      row * rowBytes,
    );
  }

  stagingBuffer.unmap();
  stagingBuffer.destroy();
  renderTex.destroy();

  return pixels;
}

// ---- PSNR (пиковое отношение сигнал/шум) --------------------------------------------------------------------------------------------------------------------------------------------

/**
 * Вычисляет пиковое отношение сигнал/шум между двумя изображениями RGBA8.
 * Учитываются только каналы R, G, B; альфа-канал игнорируется.
 *
 * Формула: PSNR = 10 × log₁₀(255² / MSE)
 *   где MSE = среднеквадратичная ошибка по значениям каналов по всем пикселям.
 *
 * @returns PSNR в дБ, или Infinity если изображения идентичны.
 */
export function computePSNR(ref: Uint8Array, test: Uint8Array): number {
  if (ref.length !== test.length) throw new Error("computePSNR: size mismatch");
  const n = ref.length / 4; // количество пикселей
  let sumSq = 0;
  for (let i = 0; i < ref.length; i++) {
    if ((i & 3) === 3) continue; // пропускаем альфа-канал
    const d = (ref[i]! - test[i]!);
    sumSq += d * d;
  }
  const mse = sumSq / (n * 3); // 3 канала на пиксель
  if (mse === 0) return Infinity;
  return 10 * Math.log10((255 * 255) / mse);
}

// ---- SSIM (структурное сходство) --------------------------------------------------------------------------------------------------------------------------------------------

const SSIM_C1 = (0.01 * 255) ** 2; // константа стабилизации L=255, k=0.01
const SSIM_C2 = (0.03 * 255) ** 2;
const SSIM_BLOCK = 8;               // размер блока 8×8

/**
 * Вычисляет упрощённый блочный SSIM между двумя изображениями RGBA8.
 * Использует яркость Y = 0.299R + 0.587G + 0.114B на пиксель (ITU-R BT.601).
 * Каждый блок 8×8 даёт одно значение SSIM; все блоки усредняются.
 *
 * @param ref    - Эталонные пиксели RGBA8.
 * @param test   - Тестируемые пиксели RGBA8.
 * @param width  - Ширина изображения в пикселях.
 * @param height - Высота изображения в пикселях.
 * @returns SSIM в [0, 1]; 1.0 = идентичны.
 */
export function computeSSIM(
  ref:    Uint8Array,
  test:   Uint8Array,
  width:  number,
  height: number,
): number {
  const B   = SSIM_BLOCK;
  const bx  = Math.floor(width  / B);
  const by  = Math.floor(height / B);
  if (bx === 0 || by === 0) return 1;

  let totalSSIM = 0;
  let blockCount = 0;

  for (let by_ = 0; by_ < by; by_++) {
    for (let bx_ = 0; bx_ < bx; bx_++) {
      const x0 = bx_ * B;
      const y0 = by_ * B;

      let sumX = 0, sumY = 0, sumXX = 0, sumYY = 0, sumXY = 0;
      const count = B * B;

      for (let py = y0; py < y0 + B; py++) {
        for (let px = x0; px < x0 + B; px++) {
          const idx = (py * width + px) * 4;
          const lx = luminance(ref[idx]!, ref[idx+1]!, ref[idx+2]!);
          const ly = luminance(test[idx]!, test[idx+1]!, test[idx+2]!);
          sumX  += lx;
          sumY  += ly;
          sumXX += lx * lx;
          sumYY += ly * ly;
          sumXY += lx * ly;
        }
      }

      const muX  = sumX  / count;
      const muY  = sumY  / count;
      const sigX = sumXX / count - muX * muX;
      const sigY = sumYY / count - muY * muY;
      const sigXY = sumXY / count - muX * muY;

      const num = (2 * muX * muY + SSIM_C1) * (2 * sigXY + SSIM_C2);
      const den = (muX*muX + muY*muY + SSIM_C1) * (sigX + sigY + SSIM_C2);
      totalSSIM += num / den;
      blockCount++;
    }
  }

  return blockCount > 0 ? totalSSIM / blockCount : 1;
}

function luminance(r: number, g: number, b: number): number {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

// ---- Позиции камеры ----------------------------------------------------------------------------------------------------------------------------

/** Фиксированная позиция камеры для воспроизводимых измерений качества. */
export interface CameraPose {
  name:     string;
  position: Float32Array;
}

/**
 * Три канонические позиции измерения для сетки тайлов 5×5,
 * центрированной в начале координат с шагом 1.5 (пресет стресс-теста "small").
 */
export const MEASUREMENT_POSES: CameraPose[] = [
  {
    name:     "overview",
    position: new Float32Array([0, 0, 10]),  // далеко - вся сетка в кадре
  },
  {
    name:     "closeup",
    position: new Float32Array([0, 0, 2]),   // очень близко - один тайл заполняет кадр
  },
  {
    name:     "midrange",
    position: new Float32Array([0, 0, 5]),   // средняя дистанция - зона перехода LOD
  },
];
