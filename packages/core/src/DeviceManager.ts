/**
 * Управляет жизненным циклом GPUAdapter и GPUDevice.
 * Обрабатывает выбор адаптера, создание устройства, определение возможностей и потерю устройства.
 */
export class DeviceManager {
  private _adapter: GPUAdapter | null = null;
  private _device: GPUDevice | null = null;
  private _adapterInfo: GPUAdapterInfo | null = null;
  private _lostPromise: Promise<void> | null = null;

  get adapter(): GPUAdapter | null {
    return this._adapter;
  }

  get device(): GPUDevice | null {
    return this._device;
  }

  /** Синхронная информация об адаптере (производитель, архитектура, описание). */
  get adapterInfo(): GPUAdapterInfo | null {
    return this._adapterInfo;
  }

  /** Разрешается при потере устройства. */
  get lostPromise(): Promise<void> | null {
    return this._lostPromise;
  }

  /**
   * Запросить адаптер и устройство. Логирует доступные возможности и лимиты.
   * @throws если WebGPU недоступен или создание устройства завершилось ошибкой.
   */
  async initialize(): Promise<void> {
    if (!navigator.gpu) {
      throw new Error("WebGPU недоступен в этом браузере.");
    }

    this._adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!this._adapter) {
      throw new Error("Подходящий GPUAdapter не найден.");
    }

    this._adapterInfo = this._adapter.info ?? null;

    const optionalFeatures: GPUFeatureName[] = [
      "texture-compression-bc",
      "texture-compression-astc",
      "texture-compression-etc2",
      "timestamp-query",
    ];

    const supportedFeatures = optionalFeatures.filter((f) =>
      this._adapter!.features.has(f)
    );

    console.info("[DeviceManager] Адаптер:", this._adapterInfo?.description ?? this._adapterInfo?.vendor ?? "неизвестно");
    console.info("[DeviceManager] Поддерживаемые дополнительные возможности:", supportedFeatures);

    this._device = await this._adapter.requestDevice({
      requiredFeatures: supportedFeatures as GPUFeatureName[],
      label: "webgpu-streaming-device",
    });

    this._lostPromise = this._device.lost.then((info) => {
      console.error("[DeviceManager] Устройство потеряно:", info.reason, info.message);
    });

    this._device.addEventListener("uncapturederror", (event) => {
      console.error("[DeviceManager] Неперехваченная ошибка:", (event as GPUUncapturedErrorEvent).error);
    });
  }

  /** Освободить устройство. */
  destroy(): void {
    this._device?.destroy();
    this._device = null;
    this._adapter = null;
    this._adapterInfo = null;
  }
}
