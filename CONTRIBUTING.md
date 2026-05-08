# Участие в разработке

## Установка

```bash
git clone https://github.com/Dusty305/webgpu-renderer
cd webgpu-renderer
npm install
npm run build
```

## Запуск демонстраций

```bash
npm run dev
# Откройте http://localhost:5173
# Перейдите в /src/01-device-init/ - /src/16-public-api/
```

## Запуск тестов

```bash
npm test
# Запускает Vitest во всех пакетах
```

## Сборка пакетов

```bash
npm run build
# Сначала собирает gpu-types, затем все остальные пакеты параллельно
```

## Добавление плагина прохода рендеринга

1. Создайте класс, реализующий `IRenderPass` из `@webgpu-streaming/gpu-types`.
2. Инициализируйте GPU-ресурсы в `initialize()`. Оборачивайте каждое выделение в pushErrorScope.
3. Реализуйте `execute(ctx: FrameContext)` - создайте и отправьте `GPURenderPassEncoder`.
4. Реализуйте `onResize(w, h)` - пересоздайте все зависящие от размера ресурсы (текстуры глубины и т.д.).
5. Реализуйте `destroy()` - вызовите `.destroy()` для каждого созданного вами GPU-ресурса.
6. Зарегистрируйте с помощью `pluginHost.registerRenderPass(myPass)` перед вызовом `initialize()`.
7. Используйте `registry.request<T>(name)` для получения ресурсов, опубликованных менеджерами.

## Добавление плагина менеджера ресурсов

Тот же жизненный цикл, но реализуйте `IResourceManager`. Вызывайте `registry.register(name, resource)` для публикации GPU-ресурсов, которые могут потреблять проходы рендеринга.

