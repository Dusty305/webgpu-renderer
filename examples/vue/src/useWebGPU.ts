import { ref, onMounted, onUnmounted } from "vue";
import type { Ref } from "vue";
import { createRenderer } from "webgpu-streaming";
import type { Renderer } from "webgpu-streaming";

export function useWebGPU(
  canvasRef: Ref<HTMLCanvasElement | null>,
  options?: { memoryBudget?: number; frameUploadCap?: number }
) {
  const renderer = ref<Renderer | null>(null);
  const error = ref<string | null>(null);

  onMounted(async () => {
    if (!canvasRef.value) return;
    try {
      renderer.value = await createRenderer({ canvas: canvasRef.value, ...options });
    } catch (e) {
      error.value = String(e);
    }
  });

  onUnmounted(() => {
    renderer.value?.dispose();
    renderer.value = null;
  });

  return { renderer, error };
}
