# Справочник API

Полный справочник API по всем публичным типам и методам.

---

## `createRenderer(options: CreateRendererOptions): Promise<Renderer>`

Создаёт WebGPU-рендерер для существующего элемента `<canvas>`. Настраивает GPU-устройство, буфер глубины, конвейер рендеринга и бюджет потоковой передачи текстур за один вызов.

```ts
import { createRenderer } from "webgpu-streaming";

const renderer = await createRenderer({
  canvas: document.querySelector("canvas")!,
  memoryBudget: 256,
  powerPreference: "high-performance",
});
```

---

## `CreateRendererOptions`

```ts
interface CreateRendererOptions {
  canvas: HTMLCanvasElement;
  memoryBudget?: number;       // default: 256 MB
  frameUploadCap?: number;     // default: 8 MB/frame
  textureTiers?: number[];     // default: [512, 1024, 2048]
  maxLayersPerTier?: number;   // default: 64
  powerPreference?: "high-performance" | "low-power"; // default: "high-performance"
}
```

---

## `Renderer`

```ts
interface Renderer {
  loadScene(source?: string | ArrayBuffer): Promise<void>;
  loadTexture(source: string | ArrayBuffer, options?: TextureOptions): Promise<TextureHandle>;
  addMesh(geometry: GeometryDescriptor, material: MaterialDescriptor): MeshHandle;
  removeMesh(handle: MeshHandle): void;
  setCamera(options: CameraOptions): void;
  getStats(): StreamingStats;
  dispose(): void;
  readonly device: GPUDevice;
  readonly ready: boolean;
}
```

### `renderer.loadScene(source?)`

Загрузить сцену из URL-строки или `ArrayBuffer`. Если `source` не указан, создаётся процедурный демонстрационный куб. В будущих версиях будет поддержка glTF.

```ts
await renderer.loadScene(); // процедурный куб
await renderer.loadScene("/models/sponza.gltf"); // пока не полностью поддерживается
```

### `renderer.loadTexture(source, options?)`

Загрузить текстуру из URL или ArrayBuffer. Поддерживает PNG, JPEG и KTX2 (со сжатием Basis Universal).

```ts
const tex = await renderer.loadTexture("/tex/diffuse.png", { usage: "color" });
const mesh = renderer.addMesh(geo, { baseColor: tex });
```

### `renderer.addMesh(geometry, material)`

Добавить меш в сцену. Возвращает `MeshHandle` для дальнейших манипуляций.

```ts
const mesh = renderer.addMesh(
  { positions, normals, uvs },
  { baseColor: "#e88033", metallicRoughness: { metallic: 0, roughness: 0.8 } }
);
mesh.setPosition(0, 1, 0);
```

### `renderer.removeMesh(handle)`

Удалить меш из сцены и освободить его GPU-ресурсы. После этого вызова `handle` не должен использоваться.

### `renderer.setCamera(options)`

Обновить параметры камеры. Все поля необязательны - пропущенные поля сохраняют предыдущие значения.

```ts
renderer.setCamera({ position: [3, 2, 3], target: [0, 0, 0], fov: 60 });
```

### `renderer.getStats()`

Возвращает снимок `StreamingStats` для текущего кадра.

### `renderer.dispose()`

Уничтожить все GPU-ресурсы, созданные этим рендерером. Элемент canvas не удаляется из DOM.

---

## `GeometryDescriptor`

```ts
interface GeometryDescriptor {
  positions: Float32Array;          // xyz triples, one per vertex
  normals?: Float32Array;           // xyz triples, one per vertex
  uvs?: Float32Array;               // uv pairs, one per vertex
  indices?: Uint16Array | Uint32Array; // triangle list indices
}
```

---

## `MaterialDescriptor`

```ts
interface MaterialDescriptor {
  baseColor?: string | TextureHandle;
  normalMap?: TextureHandle;
  metallicRoughness?: TextureHandle | { metallic: number; roughness: number };
  emissive?: string | TextureHandle;
}
```

Строки цвета принимают любое значение CSS-цвета (`"#ff8800"`, `"rgb(200,100,50)"`, именованные цвета).

---

## `MeshHandle`

```ts
interface MeshHandle {
  readonly id: string;
  setTransform(matrix: Float32Array): void;
  setPosition(x: number, y: number, z: number): void;
  setRotation(x: number, y: number, z: number): void;
  setScale(x: number, y: number, z: number): void;
  readonly visible: boolean;
  setVisible(v: boolean): void;
  destroy(): void;
}
```

`setTransform` принимает матрицу 4×4 по столбцам в виде `Float32Array(16)`.

`setRotation` принимает углы Эйлера в радианах (порядок XYZ).

---

## `TextureHandle`

```ts
interface TextureHandle {
  readonly id: string;
  readonly width: number;
  readonly height: number;
  readonly mipLevels: number;
  readonly residentMip: number; // 0 = самый детальный (полное разрешение), mipLevels-1 = самый грубый
  destroy(): void;
}
```

---

## `TextureOptions`

```ts
interface TextureOptions {
  resolution?: 512 | 1024 | 2048 | 4096;
  usage?: "color" | "normal" | "orm";
  priority?: number; // Выше = загружается первым. По умолчанию: 0
}
```

---

## `CameraOptions`

```ts
interface CameraOptions {
    position?: [number, number, number];
    target?: [number, number, number];
    fov?: number;   // вертикальное поле зрения в градусах
    near?: number;  // ближняя плоскость отсечения. По умолчанию: 0.1
    far?: number;   // дальняя плоскость отсечения. По умолчанию: 1000
}
```

---

## `StreamingStats`

```ts
interface StreamingStats {
  memoryUsedMB: number;
  memoryBudgetMB: number;
  texturesLoaded: number;
  texturesTotal: number;
  residentMipDistribution: Record<number, number>;
  fps: number;
  frameTimeP99Ms: number;
}
```

---

## `WebGPUCanvasElement`

Класс пользовательского элемента `<webgpu-canvas-streaming>`. 
Регистрируется автоматически при импорте.

```ts
import "webgpu-streaming"; // регистрирует <webgpu-canvas-streaming>
// или
import { WebGPUCanvasElement } from "webgpu-streaming";
```

### Свойства элемента (зеркало атрибутов)

| Свойство | Тип | Атрибут |
|----------|-----|---------|
| `memoryBudget` | `number` | `memory-budget` |
| `frameUploadCap` | `number` | `frame-upload-cap` |
| `textureTiers` | `string` | `texture-tiers` |
| `maxLayersPerTier` | `number` | `max-layers-per-tier` |
| `showStats` | `boolean` | `show-stats` |
| `cameraMode` | `string` | `camera-mode` |
| `powerPreference` | `string` | `power-preference` |
| `ready` | `boolean` | - (только для чтения) |

### Методы элемента

Те же сигнатуры, что у `Renderer`: `loadScene()`, `loadTexture()`, `addMesh()`, `removeMesh()`, `setCamera()`, `getStats()`, `reset()`.

---

## `configureStreaming(config)`

Переопределить глобальные параметры потоковой передачи по умолчанию перед созданием любого рендерера.

```ts
import { configureStreaming } from "webgpu-streaming";

configureStreaming({
  defaultMemoryBudget: 512,
  defaultFrameUploadCap: 16,
  defaultTextureTiers: [256, 512, 1024, 2048],
});
```

---

## `configureRenderer(config)`

Переопределить глобальные параметры рендерера по умолчанию (освещение).

```ts
import { configureRenderer } from "webgpu-streaming";

configureRenderer({
  lightDir: [1, 2, 1],
  lightColor: [1.0, 0.95, 0.85],
  ambientIntensity: 0.15,
});
```
