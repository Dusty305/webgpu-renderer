import type { MaterialEntry } from "@webgpu-streaming/gpu-types";

/** Размер одной записи MaterialEntry в байтах (4 поля u32 = 16 байт). */
const ENTRY_STRIDE = 16;
// Поля: tierIndex, layerIndex, residentMip, tierLodMinClamp

/**
 * Управляет GPU-буфером хранилища со структурами MaterialEntry.
 * Каждый индекс материала соответствует одной записи, описывающей тир, слой и резидентный мипмап.
 */
export class MaterialBufferWriter {
  private readonly _buffer: GPUBuffer;
  private readonly _cpuData: Uint32Array<ArrayBuffer>;
  private _dirty = false;

  constructor(
    private readonly _device: GPUDevice,
    private readonly _maxMaterials: number = 1024
  ) {
    const byteSize = _maxMaterials * ENTRY_STRIDE;
    _device.pushErrorScope("out-of-memory");
    this._buffer = _device.createBuffer({
      label: "material-buffer",
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    void _device.popErrorScope().then((err) => {
      if (err) console.error("[MaterialBufferWriter] Нехватка памяти:", err);
    });
    this._cpuData = new Uint32Array(new ArrayBuffer(byteSize));
  }

  get buffer(): GPUBuffer {
    return this._buffer;
  }

  /**
   * Обновить запись MaterialEntry для слота материала.
   */
  write(materialId: number, entry: MaterialEntry): void {
    if (materialId < 0 || materialId >= this._maxMaterials) {
      console.warn(`[MaterialBufferWriter] materialId ${materialId} вне допустимого диапазона`);
      return;
    }
    const base = materialId * 4;
    this._cpuData[base]     = entry.tierIndex;
    this._cpuData[base + 1] = entry.layerIndex;
    this._cpuData[base + 2] = entry.residentMip;
    this._cpuData[base + 3] = entry.tierLodMinClamp;
    this._dirty = true;
  }

  /**
   * Загрузить изменённые CPU-данные в GPU-буфер.
   * Вызывается один раз за кадр, до выполнения проходов рендеринга.
   */
  flush(): void {
    if (!this._dirty) return;
    this._device.queue.writeBuffer(this._buffer, 0, this._cpuData);
    this._dirty = false;
  }

  destroy(): void {
    this._buffer.destroy();
  }
}
