# Архитектура

Внутренняя архитектура для разработчиков. Смотрите [CONTRIBUTING.md](../CONTRIBUTING.md) для инструкций по установке.

---

## DAG зависимостей пакетов

```
webgpu-streaming (унифицированный пакет)
    ------ @webgpu-streaming/core        (Custom Element + createRenderer)
    |       ----- @webgpu-streaming/gpu-types
    ------ @webgpu-streaming/render-basic (BasicRenderPass + configureRenderer)
    |       ----- @webgpu-streaming/gpu-types
    ----- @webgpu-streaming/texture-streaming (TextureStreamingManager + configureStreaming)
            ----- @webgpu-streaming/gpu-types
```

`@webgpu-streaming/core` зависит **только** от `gpu-types` - он 
НЕ импортирует из `render-basic` или `texture-streaming`. 
Публичная функция `createRenderer()` связывает всё воедино во время выполнения 
через интерфейс плагинов.

---

## Поток данных - выполнение кадра

```
RenderLoop.tick()
    |
    ---- manager.prepareFrame()   <- TextureStreamingManager
    |     ------ Оценка ожидающих загрузки мип-уровней (срочность в экранном пространстве)
    |     ------ Вытеснение мип-уровней по алгоритму LRU при превышении бюджета
    |     ------ Загрузка N МБ данных мип-уровней через StagingRingBuffer
    |
    --- pass.execute()           <- BasicRenderPass / BuiltinRenderPass
          ------ Начало рендер-прохода (очистка цвета + глубины)
          ------ Установка пайплайна, связывание групп (сценарные Uniform'ы, Uniform'ы объектов)
          ------ Рисование с индексацией (по одному вызову отрисовки на каждый видимый меш)
```

---

## Интерфейс плагинов

Все расширения рендеринга реализуют один из двух интерфейсов из `@webgpu-streaming/gpu-types`:

```ts
interface IResourceManager {
  initialize(device: GPUDevice, registry: ResourceRegistry, format: GPUTextureFormat): Promise<void>;
  prepareFrame(ctx: FrameContext): void;
  destroy(): void;
}

interface IRenderPass {
  initialize(device: GPUDevice, registry: ResourceRegistry, format: GPUTextureFormat): Promise<void>;
  execute(ctx: FrameContext): void;
  onResize(width: number, height: number): void;
  destroy(): void;
}
```

`PluginHost` (в `@webgpu-streaming/core`) вызывает менеджеры перед проходами в каждом кадре. 
Порядок регистрации определяет порядок выполнения. Освобождение ресурсов происходит в порядке, 
обратном регистрации.

---

## Подсистема потоковой передачи текстур

```
TextureStreamingManager (IResourceManager)
    ------ TierAllocator        - 3 текстурных 2D-массива (уровни: 512/1024/2048), список свободных блоков на уровень
    ------ MaterialBufferWriter - GPU-буфер хранилища: для каждого материала {tier, layer, lodMin, ...}
    ------ MipPriorityQueue     - двоичная max-куча, оценка по (экранное покрытие × дефицит мип-уровня)
    ------ BudgetTracker        - побайтовый учет, настраиваемый лимит сверху
    ------ LRUEvictionPolicy    - временные метки доступа к мип-уровням, вытеснение сначала мелких деталей
    ------ StagingRingBuffer    - 4 кольцевых буфера с MAP_WRITE|COPY_SRC, асинхронное переотображение после отправки
    ------ KTX2Parser           - побайтовые смещения мип-уровней, извлечение DFD
    ------ FormatRouter         - определение возможностей устройства → выбор формата BC7/ASTC/ETC2
    ------ TranscodePipeline    - пул Web Worker + basis_transcoder.wasm
    ------ SamplerManager       - GPUSampler для каждого уровня с отслеживанием lodMinClamp
    ------ BindGroupManager     - пересоздание групп связывания при изменении семплера/массива, подписка через registry
```

**Ключевой инвариант:** сэмплеры всегда используют `lodMinClamp = 0`. Вместо этого шейдер 
вызывает `textureSampleLevel(tex, samp, uv, layer, max(approxLod, residentMip))`, 
где `residentMip` - наиболее детальный резидентный уровень для каждой текстуры, 
считанный из GPU-буфера хранилища (`MaterialEntry`). Это гарантирует, что GPU никогда не обращается 
к нулевым мип-данным без необходимости создавать варианты сэмплеров для каждой текстуры. `SamplerManager` 
отслеживает `tierLodMinClamp` только для учета - это не обеспечивается GPU.

---

## ResourceRegistry

Именованный канал публикации/подписки для совместного использования GPU-ресурсов между плагинами 
без прямых зависимостей импорта. Пример: `TextureStreamingManager` публикует 
`"materialBindGroup"` и `"materialBuffer"` после каждого кадра; `BasicRenderPass` 
запрашивает их в каждом кадре.

```ts
registry.register("materialBindGroup", bindGroup);
const bg = registry.request<GPUBindGroup>("materialBindGroup");
registry.onChange("materialBindGroup", (bg) => { /* re-bind */ });
```

---

## SceneGraph

Минимальный плоский словарь узлов `{id, materialId, worldTransform, boundingSphere, visible}`. 
Предоставляется проходам рендеринга в виде снимка только для чтения (`SceneGraphReadView`) для 
предотвращения случайной мутации во время выполнения кадра.

---

## Публичный и внутренний API

Каждый пакет имеет две точки входа:

- `src/index.ts` - **только публичный API**. Экспортируется из собранного пакета.
- `src/_internal.ts` - реэкспортирует всё, включая внутренние классы. Используется 
псевдонимами `packages/demos/vite.config.ts`, чтобы демонстрации при `npm run dev` продолжали работать без изменений.

Конфигурация Vite для демонстраций (`packages/demos/vite.config.ts`) сопоставляет каждый пакет 
с его файлом `_internal.ts`. Собранный `dist/index.js` содержит только публичный API.

---

## Жизненный цикл GPU-ресурсов

Каждое выделение GPU-ресурсов следует этому паттерну:

```ts
device.pushErrorScope("out-of-memory");
const texture = device.createTexture(descriptor);
const error = await device.popErrorScope();
if (error) { /* правильная обработка OOM */ }
```

Каждый `createTexture` / `createBuffer` имеет соответствующий вызов `.destroy()` в методе `destroy()` владеющего класса. `PluginHost.destroy()` вызывает плагины в порядке, обратном регистрации, гарантируя, что зависимые ресурсы (например, группы привязки, ссылающиеся на текстуры) уничтожаются раньше самих текстур.
