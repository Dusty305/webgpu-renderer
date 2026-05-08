import "@webgpu-streaming/core";
import type { WebGPUReadyDetail } from "@webgpu-streaming/core";
import { createOverlay, FpsTracker } from "../shared/overlay.js";

const overlay = createOverlay("01 - Device Init");
const fps = new FpsTracker();

overlay.set("Status", "Initializing…");

const el = document.getElementById("canvas") as HTMLElement & {
  clearColor: GPUColorDict;
};

el.addEventListener("webgpu-error", (e) => {
  overlay.set("Status", `ERROR: ${(e as CustomEvent<string>).detail}`);
});

el.addEventListener("webgpu-ready", (e) => {
  const { device, adapterInfo } = (e as CustomEvent<WebGPUReadyDetail>).detail;

  overlay.set("Status", "Running");
  overlay.set("Vendor", adapterInfo?.vendor ?? "-");
  overlay.set("Architecture", adapterInfo?.architecture ?? "-");
  overlay.set("Description", adapterInfo?.description ?? "-");

  const compression: string[] = [];
  if (device.features.has("texture-compression-bc"))   compression.push("BC");
  if (device.features.has("texture-compression-astc")) compression.push("ASTC");
  if (device.features.has("texture-compression-etc2")) compression.push("ETC2");
  overlay.set("Compression", compression.join(", ") || "none");
  overlay.set("Timestamp query", device.features.has("timestamp-query") ? "yes" : "no");
  overlay.set("Max tex dim", device.limits.maxTextureDimension2D.toLocaleString());
  overlay.set("Max buffer size", `${(device.limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB`);

  // Циклически изменяем оттенок и обновляем clearColor элемента каждый кадр.
  let hue = 0;
  function frame() {
    hue = (hue + 0.4) % 360;
    el.clearColor = hslToGPUColor(hue, 0.6, 0.15);
    fps.tick();
    overlay.set("FPS", fps.fps);
    overlay.set("Hue", `${Math.round(hue)}°`);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
});

/** Преобразует HSL в GPUColorDict (линейное приближение). */
function hslToGPUColor(h: number, s: number, l: number): GPUColorDict {
  const a = s * Math.min(l, 1 - l);
  function f(n: number): number {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  }
  return { r: f(0), g: f(8), b: f(4), a: 1.0 };
}
