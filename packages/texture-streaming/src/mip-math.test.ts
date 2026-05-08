import { describe, it, expect } from "vitest";
import { computeDesiredMip, mipSizeBytes } from "./mip-math.js";

describe("computeDesiredMip", () => {
  const fovY = Math.PI / 4; // 45°
  const screenHeight = 1080;
  const texWidth = 1024;

  it("возвращает 0, когда объект заполняет экран", () => {
    // Очень близко: проецируемые тексели ≈ texWidth → мип 0
    const mip = computeDesiredMip(0.1, texWidth, screenHeight, fovY);
    expect(mip).toBe(0);
  });

  it("возвращает более высокий мип при большем расстоянии", () => {
    // При fovY=π/4, screenHeight=1080, texWidth=1024:
    // projectedPixels при d=0.1 → 1080/(2×0.1×0.414) ≈ 13034 → мип=0 (увеличение)
    // projectedPixels при d=10  → 1080/(2×10×0.414)  ≈ 130   → мип=floor(log2(1024/130))≈2
    const near = computeDesiredMip(0.1, texWidth, screenHeight, fovY);
    const far  = computeDesiredMip(10,  texWidth, screenHeight, fovY);
    expect(far).toBeGreaterThan(near);
  });

  it("мип увеличивается примерно на 1 при удвоении расстояния (свойство log2)", () => {
    const d = 2; // в диапазоне, где мипы ненулевые
    const m1 = computeDesiredMip(d,     texWidth, screenHeight, fovY);
    const m2 = computeDesiredMip(d * 2, texWidth, screenHeight, fovY);
    // Каждое удвоение расстояния → ровно 1 более грубый мип (log2 точен)
    expect(m2 - m1).toBe(1);
  });

  it("ограничивает до 0 при некорректных входных данных", () => {
    expect(computeDesiredMip(0, 1024, 1080, fovY)).toBe(0);
    expect(computeDesiredMip(-5, 1024, 1080, fovY)).toBe(0);
    expect(computeDesiredMip(10, 0, 1080, fovY)).toBe(0);
  });

  it("более широкий fovY даёт более грубый мип на том же расстоянии (больше мира видно)", () => {
    const narrow = computeDesiredMip(10, texWidth, screenHeight, Math.PI / 8);
    const wide   = computeDesiredMip(10, texWidth, screenHeight, Math.PI / 2);
    expect(wide).toBeGreaterThanOrEqual(narrow);
  });
});

describe("чувствительность computeDesiredMip к высоте экрана", () => {
  const distance = 10;
  const texWidth = 2048;
  const fov = Math.PI / 3; // 60°

  it("возвращает более грубый мип при меньшей высоте экрана", () => {
    const mip720  = computeDesiredMip(distance, texWidth, 720,  fov);
    const mip1080 = computeDesiredMip(distance, texWidth, 1080, fov);
    const mip2160 = computeDesiredMip(distance, texWidth, 2160, fov);

    // Меньший экран → меньше пикселей → более грубый мип (большее число)
    expect(mip720).toBeGreaterThan(mip1080);
    expect(mip1080).toBeGreaterThan(mip2160);
  });

  it("разница примерно в 1 уровень мипа между 720p и 1080p", () => {
    const mip720  = computeDesiredMip(distance, texWidth, 720,  fov);
    const mip1080 = computeDesiredMip(distance, texWidth, 1080, fov);
    const diff = mip720 - mip1080;
    // log2(1080/720) ≈ 0.58, поэтому разница равна 0 или 1 после floor()
    expect(diff).toBeGreaterThanOrEqual(0);
    expect(diff).toBeLessThanOrEqual(1);
  });
});

describe("mipSizeBytes", () => {
  it("rgba8unorm полное разрешение 256×256 = 256КБ", () => {
    expect(mipSizeBytes(256, 256, "rgba8unorm", 0)).toBe(256 * 256 * 4);
  });

  it("уровень мипа 1 составляет четверть от мипа 0", () => {
    const m0 = mipSizeBytes(256, 256, "rgba8unorm", 0);
    const m1 = mipSizeBytes(256, 256, "rgba8unorm", 1);
    expect(m1).toBe(m0 / 4);
  });

  it("ограничивает размеры мипа до 1", () => {
    // текстура 4×4 на мипе 4 → 1×1
    expect(mipSizeBytes(4, 4, "rgba8unorm", 4)).toBe(1 * 1 * 4);
  });

  it("rgba16float - 8 байт на тексель", () => {
    expect(mipSizeBytes(64, 64, "rgba16float", 0)).toBe(64 * 64 * 8);
  });
});
