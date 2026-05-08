/**
 * Демо 11 - Полная сцена + тестирование жизненного цикла
 *
 * Интегрирует: WebGPUElement + CameraController + TextureStreamingManager +
 * BasicRenderPass + SceneGraph.
 *
 * Сцена: сетка кубов 4×5 (20 объектов), каждый со своей потоковой текстурой.
 *
 * Тесты жизненного цикла (кнопки на боковой панели):
 *   Remove/Re-add Component - тестирует disconnectedCallback / connectedCallback
 *   Simulate Device Loss    - тестирует обработчик device.lost
 *   Recover Device          - тестирует WebGPUElement.recover()
 *   Add 2nd Instance        - тестирует два независимых canvas на одной странице
 */
import "@webgpu-streaming/core";
import { CameraController } from "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import { BasicRenderPass } from "@webgpu-streaming/render-basic";
import {
  TextureStreamingManager,
  parseKTX2,
} from "@webgpu-streaming/texture-streaming";
import { createOverlay, FpsTracker } from "../shared/overlay.js";
import { generateColorKtx2, hsvToRgb } from "../10/generateColorKtx2.js";
import { generateCubeMesh, mat4Translation } from "./cubeMesh.js";

const COLS        = 4;
const ROWS        = 5;
const NUM_OBJECTS = COLS * ROWS; // 20
const TEX_SIZE    = 512;
const SPACING     = 1.6;

// ---- Оверлей ------------------------------------------------------------------------------------------------------------------------------------

const overlay = createOverlay("11 - Full Scene");
const fps     = new FpsTracker();
overlay.set("Status", "Initializing…");

// ---- Element references ----------------------------------------------------------------------------------------------------------------

const scene  = document.getElementById("scene")!;
const status = document.getElementById("status-bar")!;

const btnRemove  = document.getElementById("btn-remove")  as HTMLButtonElement;
const btnAdd     = document.getElementById("btn-add")     as HTMLButtonElement;
const btnLoss    = document.getElementById("btn-loss")    as HTMLButtonElement;
const btnRecover = document.getElementById("btn-recover") as HTMLButtonElement;
const btnSecond  = document.getElementById("btn-second")  as HTMLButtonElement;
const secondContainer = document.getElementById("second-instance-container") as HTMLElement;

type WebGPUCanvas = HTMLElement & {
  clearColor: GPUColorDict;
  canvasElement: HTMLCanvasElement;
  addRenderPass:       (p: unknown) => Promise<void>;
  addResourceManager:  (m: unknown) => Promise<void>;
  sceneGraph:          { addNode(id: string, matId: number, wt: Float32Array, bs: Float32Array): void } | null;
  setCameraController: (c: unknown) => void;
  device:              GPUDevice | null;
  recover:             () => Promise<void>;
};

let el = document.getElementById("canvas") as WebGPUCanvas;

// ---- Scene setup helper ----------------------------------------------------------------------------------------------------------------

async function setupScene(canvas: WebGPUCanvas): Promise<void> {
  canvas.clearColor = { r: 0.10, g: 0.10, b: 0.12, a: 1.0 };

  const camera = new CameraController(canvas.canvasElement);
  canvas.setCameraController(camera);

  const streaming = new TextureStreamingManager({
    budgetBytes:       128 * 1024 * 1024,
    frameUploadBudget:   8 * 1024 * 1024,
  });
  await canvas.addResourceManager(streaming);

  const renderPass = new BasicRenderPass();
  await canvas.addRenderPass(renderPass);

  const { vertices, indices } = generateCubeMesh();
  renderPass.registerMesh("cube", vertices, indices);

  const bsRadius    = 0.87;
  const gridOffsetX = -((COLS - 1) * SPACING) / 2;
  const gridOffsetY = -((ROWS - 1) * SPACING) / 2;

  for (let i = 0; i < NUM_OBJECTS; i++) {
    const col = i % COLS;
    const row = Math.floor(i / COLS);
    const x   = gridOffsetX + col * SPACING;
    const y   = gridOffsetY + row * SPACING;

    const hue        = i / NUM_OBJECTS;
    const [r, g, b]  = hsvToRgb(hue, 1, 1);
    const ktx2       = generateColorKtx2(TEX_SIZE, r, g, b);
    const parsed     = parseKTX2(ktx2);
    const bs         = new Float32Array([x, y, 0, bsRadius]);

    streaming.registerTexture(`tex-${i}`, parsed, ktx2, i, bs);
    canvas.sceneGraph?.addNode(`node-${i}`, i, mat4Translation(x, y, 0), bs);
    renderPass.addObject(`node-${i}`, "cube");
  }

  overlay.set("Status", "Running");
  status.textContent = "Status: running";

  // Обновление статистики только для основного canvas.
  if (canvas === el) {
    function tick() {
      fps.tick();
      overlay.set("FPS", fps.fps);
      const bt = streaming.budgetTracker;
      if (bt) {
        overlay.set("GPU memory", `${(bt.totalUsed / 1e6).toFixed(1)} / ${(bt.budget / 1e6).toFixed(0)} MB`);
      }
      overlay.set("Uploads/frame",   streaming.uploadsLastFrame);
      overlay.set("Evictions/frame", streaming.evictionsLastFrame);
      let fully = 0;
      for (const entry of streaming.entries.values()) {
        if (entry.residentMip === 0) fully++;
      }
      overlay.set("Fully loaded", `${fully} / ${NUM_OBJECTS}`);
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }
}

// ---- Main canvas init --------------------------------------------------------------------------------------------------------------------

el.addEventListener("webgpu-error", (e) => {
  overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`);
  status.textContent = "Status: error";
});

el.addEventListener("webgpu-lost", (e) => {
  overlay.set("Status", `Lost: ${(e as CustomEvent<string>).detail}`);
  status.textContent = "Status: device lost";
  btnLoss.disabled    = true;
  btnRecover.disabled = false;
});

el.addEventListener("webgpu-ready", async () => {
  await setupScene(el);
  btnLoss.disabled    = false;
  btnRecover.disabled = true;
});

// ---- Lifecycle button handlers --------------------------------------------------------------------------------------------------

btnRemove.addEventListener("click", () => {
  scene.removeChild(el);
  btnRemove.disabled = true;
  btnAdd.disabled    = false;
  btnLoss.disabled   = true;
  status.textContent = "Status: removed from DOM";
  overlay.set("Status", "Removed from DOM");
});

btnAdd.addEventListener("click", () => {
  scene.appendChild(el);
  btnAdd.disabled    = true;
  btnRemove.disabled = false;
  status.textContent = "Status: re-added, waiting…";
  overlay.set("Status", "Re-added, reinitializing…");
  // webgpu-ready снова сработает из connectedCallback
});

btnLoss.addEventListener("click", () => {
  // Уничтожение устройства запускает промис device.lost.
  el.device?.destroy();
  status.textContent = "Status: loss triggered…";
  btnLoss.disabled = true;
});

btnRecover.addEventListener("click", async () => {
  btnRecover.disabled = true;
  status.textContent  = "Status: recovering…";
  overlay.set("Status", "Recovering…");
  await el.recover();
  // webgpu-ready срабатывает внутри recover()
});

// ---- Second instance ----------------------------------------------------------------------------------------------------------------------

btnSecond.addEventListener("click", async () => {
  if (secondContainer.style.display === "block") return;

  const canvas2 = document.createElement("webgpu-canvas") as WebGPUCanvas;
  secondContainer.appendChild(canvas2);
  secondContainer.style.display = "block";
  btnSecond.disabled = true;
  btnSecond.textContent = "2nd Instance Active";

  canvas2.addEventListener("webgpu-ready", async () => {
    canvas2.clearColor = { r: 0.05, g: 0.08, b: 0.12, a: 1.0 };
    const camera2 = new CameraController(canvas2.canvasElement);
    canvas2.setCameraController(camera2);

    const streaming2 = new TextureStreamingManager({ budgetBytes: 32 * 1024 * 1024 });
    await canvas2.addResourceManager(streaming2);

    const pass2 = new BasicRenderPass();
    await canvas2.addRenderPass(pass2);

    const { vertices, indices } = generateCubeMesh();
    pass2.registerMesh("cube", vertices, indices);

    // Сетка 2×2 для второго экземпляра.
    for (let i = 0; i < 4; i++) {
      const x = (i % 2) * 1.4 - 0.7;
      const y = Math.floor(i / 2) * 1.4 - 0.7;
      const hue = (i + 15) / 20;
      const [r, g, b] = hsvToRgb(hue, 1, 1);
      const ktx2  = generateColorKtx2(512, r, g, b);
      const parsed = parseKTX2(ktx2);
      const bs = new Float32Array([x, y, 0, 0.87]);
      streaming2.registerTexture(`tex2-${i}`, parsed, ktx2, i, bs);
      canvas2.sceneGraph?.addNode(`n2-${i}`, i, mat4Translation(x, y, 0), bs);
      pass2.addObject(`n2-${i}`, "cube");
    }
  });
});
