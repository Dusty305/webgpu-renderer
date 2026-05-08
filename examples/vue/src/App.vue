<template>
  <div :style="containerStyle">
    <canvas ref="canvasEl" :width="720" :height="480" :style="canvasStyle" />
    <div :style="statsStyle">
      <template v-if="error">Ошибка: {{ error }}</template>
      <template v-else-if="stats">
        fps: {{ stats.fps.toFixed(1) }}&nbsp;&nbsp;mem: {{ stats.memoryUsedMB.toFixed(1) }} / {{ stats.memoryBudgetMB }} МБ
      </template>
      <template v-else>Инициализация…</template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onUnmounted } from "vue";
import { useWebGPU } from "./useWebGPU";
import type { StreamingStats } from "webgpu-streaming";

const canvasEl = ref<HTMLCanvasElement | null>(null);
const { renderer, error } = useWebGPU(canvasEl, { memoryBudget: 128 });
const stats = ref<StreamingStats | null>(null);

let statsInterval: ReturnType<typeof setInterval> | null = null;

watch(renderer, async (r) => {
  if (!r) return;
  await r.loadScene();
  r.setCamera({ position: [0, 2, 5], target: [0, 0, 0], fov: 60 });

  const s = 0.7;
  const positions = new Float32Array([
    -s,-s, s,  s,-s, s,  s, s, s,  -s,-s, s,  s, s, s,  -s, s, s,
     s,-s,-s, -s,-s,-s, -s, s,-s,   s,-s,-s, -s, s,-s,   s, s,-s,
    -s, s, s,  s, s, s,  s, s,-s,  -s, s, s,  s, s,-s,  -s, s,-s,
    -s,-s,-s,  s,-s,-s,  s,-s, s,  -s,-s,-s,  s,-s, s,  -s,-s, s,
     s,-s, s,  s,-s,-s,  s, s,-s,   s,-s, s,  s, s,-s,   s, s, s,
    -s,-s,-s, -s,-s, s, -s, s, s,  -s,-s,-s, -s, s, s,  -s, s,-s,
  ]);
  const normals = new Float32Array([
    ...(Array(6).fill([0,0,1]).flat() as number[]),
    ...(Array(6).fill([0,0,-1]).flat() as number[]),
    ...(Array(6).fill([0,1,0]).flat() as number[]),
    ...(Array(6).fill([0,-1,0]).flat() as number[]),
    ...(Array(6).fill([1,0,0]).flat() as number[]),
    ...(Array(6).fill([-1,0,0]).flat() as number[]),
  ]);
  r.addMesh({ positions, normals }, { baseColor: "#6adf6a" });

  statsInterval = setInterval(() => { stats.value = r.getStats(); }, 500);
});

onUnmounted(() => {
  if (statsInterval !== null) clearInterval(statsInterval);
});

const containerStyle = {
  background: "#0e0e0e", color: "#ddd", fontFamily: "sans-serif",
  display: "flex", flexDirection: "column" as const, alignItems: "center",
  justifyContent: "center", height: "100dvh", gap: "12px",
};
const canvasStyle = { display: "block", borderRadius: "8px" };
const statsStyle = { fontSize: ".75rem", opacity: .55, fontVariantNumeric: "tabular-nums" };
</script>
