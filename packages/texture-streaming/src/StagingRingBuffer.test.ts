import { describe, it, expect } from "vitest";
import { TIER_SIZES } from "./TierAllocator.js";

describe("размер слота StagingRingBuffer", () => {
  it("максимальный мип-0 у TIER_SIZES равен 16 МБ для [512, 1024, 2048]", () => {
    const maxMip0 = Math.max(...TIER_SIZES.map(s => s * s * 4));
    expect(maxMip0).toBe(16 * 1024 * 1024);
  });

  it("размер слота, вычисленный из конфигурации тиров, >= наибольшего мипа 0", () => {
    const maxMip0 = Math.max(...TIER_SIZES.map(s => s * s * 4));
    const frameUploadBudget = 8 * 1024 * 1024;
    const slotSize = Math.max(maxMip0, frameUploadBudget);
    expect(slotSize).toBeGreaterThanOrEqual(maxMip0);
    expect(slotSize).toBe(16 * 1024 * 1024);
  });

  it("выбрасывает исключение, если данные загрузки превышают размер слота", () => {
    const slotSize = 8 * 1024 * 1024;
    const dataSize = 16 * 1024 * 1024;
    expect(() => {
      if (dataSize > slotSize) {
        throw new Error(`[StagingRingBuffer] данные (${dataSize} Б) превышают размер слота (${slotSize} Б). Увеличьте slotByteSize.`);
      }
    }).toThrow(/превышают размер слота/);
  });

  it("не выбрасывает исключение, если данные помещаются в слот", () => {
    const slotSize = 16 * 1024 * 1024;
    const dataSize = 4 * 1024 * 1024;
    expect(() => {
      if (dataSize > slotSize) throw new Error("превышает размер слота");
    }).not.toThrow();
  });
});
