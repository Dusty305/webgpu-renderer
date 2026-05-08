import { describe, it, expect, beforeEach } from "vitest";
import { LRUEvictionPolicy } from "./LRUEvictionPolicy.js";

describe("LRUEvictionPolicy", () => {
  let lru: LRUEvictionPolicy;

  beforeEach(() => {
    lru = new LRUEvictionPolicy();
  });

  it("touch() записывает временны́е метки доступа", () => {
    lru.registerTexture("tex0", 8);
    lru.recordSize("tex0", 2, 1024);
    lru.touch("tex0", 2, 10);
    lru.touch("tex0", 2, 20);
    // Должен выбираться как кандидат для вытеснения (кадр 20, не 10)
    const evicted = lru.selectEvictions(1);
    expect(evicted).toHaveLength(1);
    expect(evicted[0]!.lastAccessFrame).toBe(20);
  });

  it("selectEvictions() возвращает наименее недавно использованные записи первыми", () => {
    lru.registerTexture("tex0", 8);
    lru.registerTexture("tex1", 8);
    lru.recordSize("tex0", 2, 1024);
    lru.recordSize("tex1", 2, 1024);
    lru.touch("tex0", 2, 5);   // старее
    lru.touch("tex1", 2, 50);  // новее

    const evicted = lru.selectEvictions(1024);
    expect(evicted[0]!.textureId).toBe("tex0");
  });

  it("вытеснение соблюдает порядок «сначала самый детализированный» внутри текстуры", () => {
    lru.registerTexture("tex0", 8);
    // Регистрируем мип 0 (наиболее детализированный) и мип 1, одного возраста
    lru.recordSize("tex0", 0, 512);
    lru.recordSize("tex0", 1, 256);
    lru.touch("tex0", 0, 10);
    lru.touch("tex0", 1, 10); // тот же кадр

    const evicted = lru.selectEvictions(10000);
    const mips = evicted.map((c) => c.mipLevel);
    // мип 0 (наиболее детализированный) должен идти перед мипом 1
    expect(mips.indexOf(0)).toBeLessThan(mips.indexOf(1));
  });

  it("никогда не вытесняет 2 наиболее грубых уровня мипов", () => {
    lru.registerTexture("tex0", 8); // totalMips=8 → наиболее грубые 2: мип 6 и 7
    for (let m = 0; m < 8; m++) {
      lru.recordSize("tex0", m, 100);
      lru.touch("tex0", m, 1);
    }

    const evicted = lru.selectEvictions(100_000);
    const mips = evicted.map((c) => c.mipLevel);
    expect(mips).not.toContain(6);
    expect(mips).not.toContain(7);
  });

  it("selectEvictions() возвращает достаточно записей чтобы покрыть bytesNeeded", () => {
    lru.registerTexture("tex0", 8);
    for (let m = 0; m < 6; m++) {
      lru.recordSize("tex0", m, 1000);
      lru.touch("tex0", m, m + 1);
    }

    const evicted = lru.selectEvictions(3000);
    const freed = evicted.reduce((sum, c) => sum + c.bytes, 0);
    expect(freed).toBeGreaterThanOrEqual(3000);
  });

  it("вытесненные записи не появляются в последующих вызовах selectEvictions()", () => {
    lru.registerTexture("tex0", 8);
    lru.recordSize("tex0", 0, 512);
    lru.touch("tex0", 0, 1);

    const first = lru.selectEvictions(512);
    expect(first).toHaveLength(1);

    // Симулируем вытеснение вызовом forget()
    lru.forget("tex0", 0);

    const second = lru.selectEvictions(512);
    expect(second.find((c) => c.textureId === "tex0" && c.mipLevel === 0)).toBeUndefined();
  });

  it("touch() для существующей записи обновляет метку времени без дублирования", () => {
    lru.registerTexture("tex0", 8);
    lru.recordSize("tex0", 2, 200);
    lru.touch("tex0", 2, 1);
    lru.touch("tex0", 2, 99); // обновляем ту же запись

    const evicted = lru.selectEvictions(200);
    const matches = evicted.filter((c) => c.textureId === "tex0" && c.mipLevel === 2);
    expect(matches).toHaveLength(1);
    expect(matches[0]!.lastAccessFrame).toBe(99);
  });

  it("не вытесняет когда нет кандидатов (все наиболее грубые)", () => {
    lru.registerTexture("tex0", 2); // только 2 мипа - оба наиболее грубые, защищены
    lru.recordSize("tex0", 0, 1024);
    lru.recordSize("tex0", 1, 512);
    lru.touch("tex0", 0, 1);
    lru.touch("tex0", 1, 1);

    const evicted = lru.selectEvictions(100_000);
    expect(evicted).toHaveLength(0);
  });
});
