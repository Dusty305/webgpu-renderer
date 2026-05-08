import { describe, it, expect } from "vitest";
import { holmBonferroniCorrection } from "./WilcoxonTest.js";

describe("holmBonferroniCorrection", () => {
  it("корректно корректирует пороги для 3 тестов", () => {
    const input = [
      { testId: "a", pValue: 0.01 },
      { testId: "b", pValue: 0.04 },
      { testId: "c", pValue: 0.06 },
    ];
    const result = holmBonferroniCorrection(input, 0.05);
    // Отсортировано: a(0.01), b(0.04), c(0.06)
    // k=1: порог = 0.05/3 ≈ 0.0167. 0.01 < 0.0167 → значимо
    // k=2: порог = 0.05/2 = 0.025.  0.04 > 0.025 → НЕ значимо
    // k=3: пошаговое снижение → НЕ значимо (так как k=2 не прошло)
    expect(result.find(r => r.testId === "a")?.significant).toBe(true);
    expect(result.find(r => r.testId === "b")?.significant).toBe(false);
    expect(result.find(r => r.testId === "c")?.significant).toBe(false);
  });

  it("все значимы, если все p-значения значительно ниже порога", () => {
    const input = [
      { testId: "x", pValue: 0.001 },
      { testId: "y", pValue: 0.002 },
    ];
    const result = holmBonferroniCorrection(input, 0.05);
    expect(result.every(r => r.significant)).toBe(true);
  });

  it("возвращает пустой массив для пустого входа", () => {
    expect(holmBonferroniCorrection([])).toEqual([]);
  });

  it("скорректированная альфа возрастает с рангом", () => {
    const input = [
      { testId: "a", pValue: 0.001 },
      { testId: "b", pValue: 0.002 },
      { testId: "c", pValue: 0.003 },
    ];
    const result = holmBonferroniCorrection(input, 0.05);
    const sorted = [...result].sort((a, b) => a.pValue - b.pValue);
    // Скорректированная альфа должна монотонно возрастать при переборе отсортированных тестов
    for (let i = 1; i < sorted.length; i++) {
      expect(sorted[i]!.holmAdjustedAlpha).toBeGreaterThanOrEqual(sorted[i - 1]!.holmAdjustedAlpha);
    }
  });

  it("одиночный тест использует альфа напрямую", () => {
    const result = holmBonferroniCorrection([{ testId: "only", pValue: 0.03 }], 0.05);
    expect(result[0]?.holmAdjustedAlpha).toBeCloseTo(0.05);
    expect(result[0]?.significant).toBe(true);
  });

  it("пошаговое снижение останавливается на первом незначимом", () => {
    // Отсортировано по p-значению: a(0.001), c(0.002), b(0.1)
    // k=1: альфа=0.05/3≈0.0167 → a(0.001) значимо
    // k=2: альфа=0.05/2=0.025  → c(0.002) значимо
    // k=3: альфа=0.05/1=0.05   → b(0.1) НЕ значимо (пошаговое снижение остановилось)
    const input = [
      { testId: "a", pValue: 0.001 },
      { testId: "b", pValue: 0.1 },
      { testId: "c", pValue: 0.002 },
    ];
    const result = holmBonferroniCorrection(input, 0.05);
    const byId = Object.fromEntries(result.map(r => [r.testId, r]));
    expect(byId["a"]?.significant).toBe(true);
    expect(byId["c"]?.significant).toBe(true);
    expect(byId["b"]?.significant).toBe(false);
  });
});
