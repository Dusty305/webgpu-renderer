import type { ResourceRegistry } from "@webgpu-streaming/gpu-types";

export const MATERIAL_BIND_GROUP_KEY = "texture-streaming:materialBindGroup";

/**
 * Создаёт группу привязки материалов для системы потоковой передачи текстур и управляет ею.
 * Пересоздаёт группу привязки при изменении массивов тиров, сэмплеров или буфера материалов.
 * Регистрирует результат в ResourceRegistry для использования в проходах рендеринга.
 */
export class BindGroupManager {
  private _bindGroup: GPUBindGroup | null = null;
  private _layout: GPUBindGroupLayout | null = null;

  constructor(
    private readonly _device: GPUDevice,
    private readonly _registry: ResourceRegistry
  ) {}

  /**
   * Создать или пересоздать макет группы привязки.
   * Вызывается один раз при инициализации.
   */
  createLayout(): GPUBindGroupLayout {
    this._layout = this._device.createBindGroupLayout({
      label: "material-bind-group-layout",
      entries: [
        // Массив текстур тира 0
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
        // Массив текстур тира 1
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
        // Массив текстур тира 2
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "2d-array" } },
        // Сэмплер тира 0
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        // Сэмплер тира 1
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        // Сэмплер тира 2
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        // Буфер хранилища материалов
        {
          binding: 6,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    });
    return this._layout;
  }

  /** Макет группы привязки, доступен после вызова createLayout(). */
  get layout(): GPUBindGroupLayout | null {
    return this._layout;
  }

  /**
   * Создать новую группу привязки и зарегистрировать её в ResourceRegistry.
   */
  rebuild(
    tierViews: [GPUTextureView, GPUTextureView, GPUTextureView],
    samplers: [GPUSampler, GPUSampler, GPUSampler],
    materialBuffer: GPUBuffer
  ): GPUBindGroup {
    if (!this._layout) {
      throw new Error("[BindGroupManager] createLayout() должен быть вызван до rebuild()");
    }

    this._bindGroup = this._device.createBindGroup({
      label: "material-bind-group",
      layout: this._layout,
      entries: [
        { binding: 0, resource: tierViews[0] },
        { binding: 1, resource: tierViews[1] },
        { binding: 2, resource: tierViews[2] },
        { binding: 3, resource: samplers[0] },
        { binding: 4, resource: samplers[1] },
        { binding: 5, resource: samplers[2] },
        { binding: 6, resource: { buffer: materialBuffer } },
      ],
    });

    this._registry.register(MATERIAL_BIND_GROUP_KEY, this._bindGroup, {
      label: "material-bind-group",
    });

    return this._bindGroup;
  }

  destroy(): void {
    this._registry.deregister(MATERIAL_BIND_GROUP_KEY);
    this._bindGroup = null;
    this._layout = null;
  }
}
