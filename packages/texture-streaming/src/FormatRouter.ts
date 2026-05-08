/**
 * Направляет загрузки текстур в наилучший аппаратно-поддерживаемый формат сжатия GPU
 * на основе доступных возможностей устройства.
 */
export class FormatRouter {
  private readonly _features: ReadonlySet<string>;

  constructor(features: GPUSupportedFeatures) {
    this._features = features as unknown as ReadonlySet<string>;
  }

  /**
   * Выбрать наилучший GPUTextureFormat для заданного способа использования текстуры.
   * Возвращает RGBA8, если сжатие недоступно.
   */
  selectFormat(usage: "color" | "normal" | "orm"): GPUTextureFormat {
    if (usage === "color") {
      if (this._features.has("texture-compression-bc")) return "bc7-rgba-unorm-srgb";
      if (this._features.has("texture-compression-astc")) return "astc-4x4-unorm-srgb";
      return "rgba8unorm-srgb";
    }

    if (usage === "normal") {
      if (this._features.has("texture-compression-bc")) return "bc5-rg-unorm";
      if (this._features.has("texture-compression-etc2")) return "eac-rg11unorm";
      return "rg8unorm";
    }

    // orm (окклюзия/шероховатость/металличность)
    if (this._features.has("texture-compression-bc")) return "bc7-rgba-unorm";
    if (this._features.has("texture-compression-astc")) return "astc-4x4-unorm";
    return "rgba8unorm";
  }
}
