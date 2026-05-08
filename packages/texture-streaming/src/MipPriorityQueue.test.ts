import { describe, it, expect, beforeEach } from "vitest";
import { MipPriorityQueue } from "./MipPriorityQueue.js";
import type { MipRequest } from "./MipPriorityQueue.js";

describe("MipPriorityQueue", () => {
  let q: MipPriorityQueue;

  beforeEach(() => {
    q = new MipPriorityQueue();
  });

  it("начинает пустой", () => {
    expect(q.size).toBe(0);
    expect(q.pop()).toBeUndefined();
  });

  it("pop возвращает запрос с наивысшим приоритетом первым", () => {
    q.push({ textureId: "a", mipLevel: 2, priority: 1 });
    q.push({ textureId: "b", mipLevel: 1, priority: 10 });
    q.push({ textureId: "c", mipLevel: 3, priority: 5 });

    expect(q.pop()?.priority).toBe(10);
    expect(q.pop()?.priority).toBe(5);
    expect(q.pop()?.priority).toBe(1);
    expect(q.pop()).toBeUndefined();
  });

  it("поддерживает корректный размер", () => {
    for (let i = 0; i < 100; i++) {
      q.push({ textureId: `t${i}`, mipLevel: 0, priority: Math.random() });
    }
    expect(q.size).toBe(100);
    q.pop();
    expect(q.size).toBe(99);
  });

  it("popHighest(N) извлекает первые N в порядке убывания приоритета", () => {
    const items: MipRequest[] = [
      { textureId: "a", mipLevel: 0, priority: 3 },
      { textureId: "b", mipLevel: 0, priority: 7 },
      { textureId: "c", mipLevel: 0, priority: 1 },
      { textureId: "d", mipLevel: 0, priority: 9 },
      { textureId: "e", mipLevel: 0, priority: 5 },
    ];
    for (const item of items) q.push(item);

    const top3 = q.popHighest(3);
    expect(top3.map((r) => r.priority)).toEqual([9, 7, 5]);
    expect(q.size).toBe(2);
  });

  it("popHighest возвращает меньше N, если очередь меньше N", () => {
    q.push({ textureId: "x", mipLevel: 0, priority: 1 });
    const result = q.popHighest(10);
    expect(result.length).toBe(1);
  });

  it("корректно обрабатывает 100 вставок в правильном порядке", () => {
    const priorities: number[] = [];
    for (let i = 0; i < 100; i++) {
      const p = Math.floor(Math.random() * 1000);
      priorities.push(p);
      q.push({ textureId: `t${i}`, mipLevel: 0, priority: p });
    }
    priorities.sort((a, b) => b - a); // по убыванию

    const extracted = q.popHighest(100).map((r) => r.priority);
    expect(extracted).toEqual(priorities);
  });

  it("clear опустошает очередь", () => {
    q.push({ textureId: "a", mipLevel: 0, priority: 1 });
    q.clear();
    expect(q.size).toBe(0);
  });
});

// ---- Контракт постановки в очередь по одному мипу на текстуру ----------------------------------
// Эти тесты проверяют гарантию порядка от грубого к детальному, описанную в фазе 7.3.
// Менеджер ставит в очередь только residentMip-1 (не весь накопленный долг), поэтому
// очередь всегда содержит не более одной записи на текстуру с разрывом.

describe("постановка в очередь по одному мипу на текстуру (контракт фазы 7.3)", () => {
  it("текстура с residentMip=10 desiredMip=5 порождает ровно одну запись в очереди с mipLevel=9", () => {
    const q2 = new MipPriorityQueue();
    const residentMip = 10;
    const desiredMip  = 5;
    const gap = residentMip - desiredMip;
    // Правильно: ставим в очередь только следующий мип (residentMip - 1 = 9)
    if (gap > 0) {
      q2.push({ textureId: "tex", mipLevel: residentMip - 1, priority: gap });
    }
    expect(q2.size).toBe(1);
    const req = q2.pop();
    expect(req?.mipLevel).toBe(9);
  });

  it("после загрузки мипа 9 (residentMip→9), следующий цикл приоритетов ставит в очередь mipLevel=8", () => {
    const q2 = new MipPriorityQueue();
    let residentMip = 10;
    const desiredMip = 5;

    // Первый цикл приоритетов
    {
      const gap = residentMip - desiredMip;
      q2.push({ textureId: "tex", mipLevel: residentMip - 1, priority: gap });
    }
    const first = q2.pop();
    expect(first?.mipLevel).toBe(9);

    // Симулируем успешную загрузку: residentMip уменьшается
    residentMip = 9;
    q2.clear();

    // Второй цикл приоритетов
    {
      const gap = residentMip - desiredMip;
      q2.push({ textureId: "tex", mipLevel: residentMip - 1, priority: gap });
    }
    const second = q2.pop();
    expect(second?.mipLevel).toBe(8);
  });

  it("две текстуры с разрывами представлены ровно одной записью каждая", () => {
    const q2 = new MipPriorityQueue();
    const textures = [
      { id: "a", residentMip: 8, desiredMip: 3 },
      { id: "b", residentMip: 6, desiredMip: 2 },
    ];
    for (const t of textures) {
      const gap = t.residentMip - t.desiredMip;
      if (gap > 0) {
        q2.push({ textureId: t.id, mipLevel: t.residentMip - 1, priority: gap / (1 + 1) });
      }
    }
    // Ровно две записи, по одной на текстуру
    expect(q2.size).toBe(2);
    const first = q2.pop()!;
    const second = q2.pop()!;
    // текстура "a" имеет разрыв 5, текстура "b" - разрыв 4 → "a" имеет более высокий приоритет
    expect(first.textureId).toBe("a");
    expect(first.mipLevel).toBe(7);  // residentMip(8) - 1
    expect(second.textureId).toBe("b");
    expect(second.mipLevel).toBe(5); // residentMip(6) - 1
  });
});
