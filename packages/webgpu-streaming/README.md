# webgpu-streaming

WebGPU-рендерер с бюджетно-управляемой потоковой передачей мип-карт. Используется как Web Component или через программный API.
Это унифицированный пакет. Он реэкспортирует всё из четырёх пакетов с областью видимости:

```bash
# Всё в одном
npm install webgpu-streaming

# Или установите только то, что нужно
npm install @webgpu-streaming/core @webgpu-streaming/texture-streaming
```

## CDN / Script Tag

Предсобранный IIFE-бандл доступен по пути `dist/webgpu-streaming.global.js`:

```html
<script src="node_modules/webgpu-streaming/dist/webgpu-streaming.global.js"></script>
<script>
  const { createRenderer } = WebGPUStreaming;
</script>
```
