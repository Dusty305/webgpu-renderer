import { createRenderer } from "webgpu-streaming";
import type { Renderer } from "webgpu-streaming";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const statsEl = document.getElementById("stats")!;

let renderer: Renderer | null = null;

async function init() {
  renderer = await createRenderer({
    canvas,
    memoryBudget: 128,
    frameUploadCap: 8,
  });

  await renderer.loadScene();
  renderer.setCamera({ position: [0, 2, 5], target: [0, 0, 0], fov: 60 });

  // Добавляем несколько цветных мешей
  const colors = ["#e8803a", "#4a9eff", "#6adf6a", "#df6adf"];
  const offsets = [[-1.5, 0, 0], [1.5, 0, 0], [0, 0, -1.5], [0, 0, 1.5]] as const;
  for (let i = 0; i < 4; i++) {
    const geo = makeCube(0.5);
    const mesh = renderer.addMesh(geo, { baseColor: colors[i]! });
    const t = new Float32Array(16);
    mat4Identity(t);
    t[12] = offsets[i]![0];
    t[13] = offsets[i]![1];
    t[14] = offsets[i]![2];
    mesh.setTransform(t);
  }

  statsEl.textContent = "Рендеринг - используйте мышь для вращения камеры";
  updateStats();
}

function updateStats() {
  if (!renderer) return;
  const s = renderer.getStats();
  statsEl.textContent =
    `fps: ${s.fps.toFixed(1)}  frame p99: ${s.frameTimeP99Ms.toFixed(1)} ms  ` +
    `mem: ${s.memoryUsedMB.toFixed(1)} / ${s.memoryBudgetMB} MB`;
  requestAnimationFrame(updateStats);
}

function makeCube(h: number) {
  const s = h;
  const positions = new Float32Array([
    -s,-s, s,  s,-s, s,  s, s, s,  -s,-s, s,  s, s, s,  -s, s, s,
     s,-s,-s, -s,-s,-s, -s, s,-s,   s,-s,-s, -s, s,-s,   s, s,-s,
    -s, s, s,  s, s, s,  s, s,-s,  -s, s, s,  s, s,-s,  -s, s,-s,
    -s,-s,-s,  s,-s,-s,  s,-s, s,  -s,-s,-s,  s,-s, s,  -s,-s, s,
     s,-s, s,  s,-s,-s,  s, s,-s,   s,-s, s,  s, s,-s,   s, s, s,
    -s,-s,-s, -s,-s, s, -s, s, s,  -s,-s,-s, -s, s, s,  -s, s,-s,
  ]);
  const normals = new Float32Array([
    ...Array(6).fill([0,0,1]).flat() as number[],
    ...Array(6).fill([0,0,-1]).flat() as number[],
    ...Array(6).fill([0,1,0]).flat() as number[],
    ...Array(6).fill([0,-1,0]).flat() as number[],
    ...Array(6).fill([1,0,0]).flat() as number[],
    ...Array(6).fill([-1,0,0]).flat() as number[],
  ]);
  return { positions, normals };
}

function mat4Identity(m: Float32Array) {
  m.fill(0);
  m[0] = m[5] = m[10] = m[15] = 1;
}

init().catch((e: unknown) => {
  statsEl.textContent = `Ошибка: ${String(e)}`;
  console.error(e);
});
