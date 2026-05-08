# webgpu-streaming

> WebGPU-рендерер с потоковой передачей мип-уровней и управлением бюджетом памяти. Работает как Web-компонент или через программный API - без Three.js, без Babylon.js, без WebGL.

## Демо

[Демонстрационный пример](https://dusty305.github.io/webgpu-renderer/) позволяет загружать встроенные процедурные сцены или перетаскивайте собственные файлы `.glb`, `.obj`.
Требуется поддержка WebGPU. 

В качестве примера можно использовать [данную модель](https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj).

## Быстрый старт

```bash
npm install webgpu-streaming
```

### Пользовательский элемент

```html
<webgpu-canvas-streaming memory-budget="256" show-stats></webgpu-canvas-streaming>
<script type="module">
  import "webgpu-streaming";
  const el = document.querySelector("webgpu-canvas-streaming");
  el.addEventListener("gpu-ready", async () => {
    await el.loadScene();
    el.setCamera({ position: [0, 2, 5], target: [0, 0, 0] });
  });
</script>
```

### Программный API

```ts
import { createRenderer } from "webgpu-streaming";

const renderer = await createRenderer({
  canvas: document.querySelector("canvas"),
  memoryBudget: 256,
});

await renderer.loadScene();
renderer.setCamera({ position: [0, 2, 5], target: [0, 0, 0] });

const mesh = renderer.addMesh(geometry, { baseColor: "#e88033" });
mesh.setPosition(1, 0, 0);
```

---

## Конфигурация

### Атрибуты `<webgpu-canvas-streaming>`

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| `memory-budget` | `number` | `256` | Бюджет памяти GPU для текстур в МБ |
| `frame-upload-cap` | `number` | `8` | Максимальный объём загружаемых данных текстур за кадр в МБ |
| `texture-tiers` | `string` | `"512,1024,2048"` | Размеры уровней разрешения через запятую |
| `max-layers-per-tier` | `number` | `64` | Максимальное количество слоёв на массив текстур |
| `show-stats` | `boolean` | `false` | Показывать оверлей статистики потоковой передачи на холсте |
| `camera-mode` | `string` | `"orbit"` | Управление камерой: `"orbit"` \| `"fly"` \| `"none"` |
| `power-preference` | `string` | `"high-performance"` | Предпочтение энергопотребления GPU |

---

## События

| Событие | Данные | Когда |
|---------|--------|-------|
| `gpu-ready` | `{ adapter, features, limits }` | GPU-устройство получено и элемент выполняет рендеринг |
| `gpu-lost` | `{ reason }` | GPU-устройство потеряно |
| `streaming-stats` | `StreamingStats` | Генерируется каждый кадр во время рендеринга |
| `gpu-error` | `{ message, type }` | Неустранимая ошибка инициализации |

---

## Справочник API

### `createRenderer(options): Promise<Renderer>`

Создаёт полностью настроенный рендерер для существующего элемента `<canvas>`.

**Параметры (`CreateRendererOptions`):**

| Поле | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `canvas` | `HTMLCanvasElement` | - | Обязательно. Целевой элемент canvas |
| `memoryBudget` | `number` | `256` | Бюджет памяти текстур в МБ |
| `frameUploadCap` | `number` | `8` | Максимальный объём загрузки за кадр в МБ |
| `textureTiers` | `number[]` | `[512,1024,2048]` | Уровни разрешения |
| `maxLayersPerTier` | `number` | `64` | Максимальное количество слоёв массива на уровень |
| `powerPreference` | `string` | `"high-performance"` | Предпочтение энергопотребления GPU |

### `Renderer`

| Метод/Свойство | Описание |
|----------------|----------|
| `loadScene(source?)` | Загрузить сцену из URL или ArrayBuffer. Если не указано - создаётся демонстрационный куб. |
| `loadTexture(source, options?)` | Загрузить текстуру. Возвращает `TextureHandle`. |
| `addMesh(geometry, material)` | Добавить меш. Возвращает `MeshHandle`. |
| `removeMesh(handle)` | Удалить меш и освободить его GPU-ресурсы. |
| `setCamera(options)` | Обновить позицию камеры, цель, fov, near, far. |
| `getStats()` | Возвращает `StreamingStats`. |
| `dispose()` | Уничтожить все GPU-ресурсы. |
| `device` | Базовый `GPUDevice`. |
| `ready` | `true` после инициализации рендерера. |

### `MeshHandle`

| Метод/Свойство | Описание |
|----------------|----------|
| `id` | Уникальный строковый идентификатор |
| `setTransform(matrix)` | Установить мировое преобразование (column-major Float32Array[16]) |
| `setPosition(x, y, z)` | Установить мировую позицию |
| `setRotation(x, y, z)` | Установить поворот Эйлера в радианах |
| `setScale(x, y, z)` | Установить масштаб |
| `visible` | Текущая видимость |
| `setVisible(v)` | Переключить видимость |
| `destroy()` | Удалить из сцены и освободить GPU-ресурсы |

### `TextureHandle`

| Свойство | Описание |
|----------|----------|
| `id` | Уникальный строковый идентификатор |
| `width`, `height`, `mipLevels` | Размеры текстуры |
| `residentMip` | Текущий наиболее детальный резидентный мип-уровень (0 = полное разрешение) |
| `destroy()` | Освободить память текстуры |

### `StreamingStats`

| Поле | Тип | Описание |
|------|-----|----------|
| `memoryUsedMB` | `number` | Текущий объём используемой памяти GPU для текстур |
| `memoryBudgetMB` | `number` | Настроенный бюджет |
| `texturesLoaded` | `number` | Текстуры, полностью резидентные на желаемом мип-уровне |
| `texturesTotal` | `number` | Общее количество зарегистрированных текстур |
| `residentMipDistribution` | `Record<number, number>` | Количество текстур на каждом резидентном мип-уровне |
| `fps` | `number` | Количество кадров в секунду (скользящее среднее) |
| `frameTimeP99Ms` | `number` | 99-й перцентиль времени кадра в мс |

---

## Поддержка браузеров

| Браузер | Минимальная версия | Примечания |
|---------|--------------------|-----------|
| Chrome | 120+ | Полная поддержка |
| Firefox | Nightly (флаг WebGPU) | Включить `dom.webgpu.enabled` в `about:config` |
| Safari | 18+ | Требуется macOS Sequoia / iOS 18 |

---

## Принцип работы

**Потоковая передача мип-уровней** - вместо загрузки текстур в полном разрешении сразу, библиотека загружает текстуры по одному мип-уровню за раз. 
Сэмплеры всегда используют `lodMinClamp = 0`; вместо этого шейдер вызывает `textureSampleLevel` с резидентным мип-уровнем 
для каждого материала, считанным из GPU-буфера хранилища. Это предотвращает обращение к неинициализированным 
данным мип-уровней без необходимости создавать варианты сэмплеров для каждой текстуры. 
По мере освобождения пропускной способности более детальные мип-уровни передаются через кольцевой 
буфер промежуточного хранения и копируются в массивы текстур GPU.

**Управление бюджетом** - настраиваемый байтовый бюджет контролирует суммарную память GPU для текстур. 
Очередь приоритетов ранжирует ожидающие загрузки мип-уровней по срочности в экранном пространстве 
(охват экрана объектом × дефицит мип-уровней). Когда новая загрузка превысила бы бюджет, 
политика вытеснения LRU удаляет наименее недавно использованные детальные мип-уровни в первую очередь, 
всегда оставляя в памяти как минимум два наиболее грубых уровня.

**Массивы текстур** -  текстуры хранятся в объектах `texture_2d_array` по уровням разрешения: 512×512, 1024×1024, 2048×2048. 
Каждый уровень имеет настраиваемое максимальное количество слоёв, инициализируемое 
один раз при запуске. Метаданные материалов, хранящиеся в GPU-буфере хранилища, отображают материалы на 
их уровень + слот.

---

## Расширенное использование

### Текстуры KTX2

```ts
const handle = await renderer.loadTexture("/textures/diffuse.ktx2", {
  usage: "color",
  priority: 1,
});
renderer.addMesh(geometry, { baseColor: handle });
```

### Приоритет потоковой передачи

```ts
await renderer.loadTexture(src, {
  priority: 2, // Выше = передаётся первой. По умолчанию: 0
});
```

### Интеграция с React

```tsx
import { useRef, useEffect, useState } from "react";
import { createRenderer } from "webgpu-streaming";
import type { Renderer } from "webgpu-streaming";

function useWebGPU(options?: { memoryBudget?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [renderer, setRenderer] = useState<Renderer | null>(null);

  useEffect(() => {
    let r: Renderer;
    createRenderer({ canvas: canvasRef.current!, ...options })
      .then(created => { r = created; setRenderer(created); });
    return () => r?.dispose();
  }, []);

  return { canvasRef, renderer };
}
```

Смотрите [`examples/react/`](examples/react/) для полного примера.

### Интеграция с Vue

```ts
import { ref, onMounted, onUnmounted } from "vue";
import { createRenderer } from "webgpu-streaming";

function useWebGPU(canvasRef, options?) {
  const renderer = ref(null);
  onMounted(async () => {
    renderer.value = await createRenderer({ canvas: canvasRef.value, ...options });
  });
  onUnmounted(() => renderer.value?.dispose());
  return { renderer };
}
```

Смотрите [`examples/vue/`](examples/vue/) для полного примера.

### Несколько экземпляров

Каждый элемент `<webgpu-canvas-streaming>` и каждый вызов `createRenderer()` создают независимое GPU-устройство и набор ресурсов. Глобального состояния нет.

### Прямой доступ к GPUDevice

```ts
const renderer = await createRenderer({ canvas, memoryBudget: 256 });
const device = renderer.device; // GPUDevice - используйте для пользовательских проходов
```

---
