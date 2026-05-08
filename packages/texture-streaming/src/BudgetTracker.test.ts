import { describe, it, expect, beforeEach } from "vitest";
import { BudgetTracker } from "./BudgetTracker.js";

describe("BudgetTracker", () => {
  let tracker: BudgetTracker;
  const BUDGET = 100 * 1024 * 1024; // 100 МБ

  beforeEach(() => {
    tracker = new BudgetTracker(BUDGET);
  });

  it("начинает с нулевым использованием", () => {
    expect(tracker.totalUsed).toBe(0);
    expect(tracker.utilization).toBe(0);
    expect(tracker.budget).toBe(BUDGET);
  });

  it("canUpload возвращает true когда бюджет не исчерпан", () => {
    expect(tracker.canUpload(1024)).toBe(true);
  });

  it("canUpload возвращает false когда бюджет превышен", () => {
    expect(tracker.canUpload(BUDGET + 1)).toBe(false);
  });

  it("recordUpload увеличивает totalUsed", () => {
    tracker.recordUpload("tex1", 0, 1024);
    expect(tracker.totalUsed).toBe(1024);
  });

  it("recordUpload заменяет предыдущий размер для того же ключа", () => {
    tracker.recordUpload("tex1", 0, 1024);
    tracker.recordUpload("tex1", 0, 2048); // перезапись
    expect(tracker.totalUsed).toBe(2048);
  });

  it("recordEviction уменьшает totalUsed", () => {
    tracker.recordUpload("tex1", 0, 4096);
    tracker.recordEviction("tex1", 0);
    expect(tracker.totalUsed).toBe(0);
  });

  it("recordEviction для неизвестного ключа не производит изменений", () => {
    tracker.recordEviction("nonexistent", 0);
    expect(tracker.totalUsed).toBe(0);
  });

  it("utilization отражает долю от бюджета", () => {
    tracker.recordUpload("tex1", 0, BUDGET / 2);
    expect(tracker.utilization).toBeCloseTo(0.5);
  });

  it("canUpload учитывает уже загруженные байты", () => {
    tracker.recordUpload("tex1", 0, BUDGET - 100);
    expect(tracker.canUpload(50)).toBe(true);
    expect(tracker.canUpload(200)).toBe(false);
  });

  it("бюджет можно изменить во время выполнения", () => {
    tracker.recordUpload("tex1", 0, 50 * 1024 * 1024);
    tracker.budget = 40 * 1024 * 1024;
    expect(tracker.canUpload(1)).toBe(false);
  });

  it("destroy сбрасывает состояние", () => {
    tracker.recordUpload("tex1", 0, 1024);
    tracker.destroy();
    expect(tracker.totalUsed).toBe(0);
  });

  it("несколько текстур накапливаются независимо", () => {
    tracker.recordUpload("a", 0, 100);
    tracker.recordUpload("a", 1, 200);
    tracker.recordUpload("b", 0, 300);
    expect(tracker.totalUsed).toBe(600);
    tracker.recordEviction("a", 0);
    expect(tracker.totalUsed).toBe(500);
  });
});
