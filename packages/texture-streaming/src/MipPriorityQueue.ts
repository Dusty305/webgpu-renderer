/** Запрос на загрузку определённого уровня мипмапа. */
export interface MipRequest {
  textureId: string;
  mipLevel: number;
  /** Выше = более срочно */
  priority: number;
}

/**
 * Бинарная max-куча запросов на загрузку мипмапов, упорядоченных по приоритету.
 * Приоритет = (desiredMip - residentMip) × screenCoverage
 */
export class MipPriorityQueue {
  private _heap: MipRequest[] = [];

  get size(): number {
    return this._heap.length;
  }

  /** Добавить новый запрос мипмапа в очередь. */
  push(request: MipRequest): void {
    this._heap.push(request);
    this._bubbleUp(this._heap.length - 1);
  }

  /** Удалить и вернуть запрос с наивысшим приоритетом. */
  pop(): MipRequest | undefined {
    if (this._heap.length === 0) return undefined;
    const top = this._heap[0]!;
    const last = this._heap.pop()!;
    if (this._heap.length > 0) {
      this._heap[0] = last;
      this._sinkDown(0);
    }
    return top;
  }

  /** Извлечь `count` наиболее приоритетных запросов. */
  popHighest(count: number): MipRequest[] {
    const results: MipRequest[] = [];
    for (let i = 0; i < count && this._heap.length > 0; i++) {
      results.push(this.pop()!);
    }
    return results;
  }

  /** Очистить все ожидающие запросы. */
  clear(): void {
    this._heap = [];
  }

  private _bubbleUp(i: number): void {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this._heap[i]!.priority > this._heap[parent]!.priority) {
        this._swap(i, parent);
        i = parent;
      } else {
        break;
      }
    }
  }

  private _sinkDown(i: number): void {
    const n = this._heap.length;
    while (true) {
      let largest = i;
      const l = 2 * i + 1;
      const r = 2 * i + 2;
      if (l < n && this._heap[l]!.priority > this._heap[largest]!.priority) largest = l;
      if (r < n && this._heap[r]!.priority > this._heap[largest]!.priority) largest = r;
      if (largest === i) break;
      this._swap(i, largest);
      i = largest;
    }
  }

  private _swap(a: number, b: number): void {
    const tmp = this._heap[a]!;
    this._heap[a] = this._heap[b]!;
    this._heap[b] = tmp;
  }
}
