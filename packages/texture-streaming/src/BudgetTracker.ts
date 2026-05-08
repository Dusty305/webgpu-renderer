/**
 * Отслеживает использование видеопамяти текстурами относительно заданного бюджета.
 */
export class BudgetTracker {
  private _used = 0;
  private readonly _perTextureMip = new Map<string, number>();

  constructor(private _budget: number = 256 * 1024 * 1024) {}

  get budget(): number {
    return this._budget;
  }

  set budget(value: number) {
    this._budget = value;
  }

  get totalUsed(): number {
    return this._used;
  }

  get utilization(): number {
    return this._budget > 0 ? this._used / this._budget : 0;
  }

  /** Возвращает true, если загрузка ещё `bytes` байт не превысит бюджет. */
  canUpload(bytes: number): boolean {
    return this._used + bytes <= this._budget;
  }

  /** Зафиксировать загрузку уровня мипмапа. */
  recordUpload(textureId: string, mipLevel: number, bytes: number): void {
    const key = `${textureId}:${mipLevel}`;
    const existing = this._perTextureMip.get(key) ?? 0;
    this._used += bytes - existing;
    this._perTextureMip.set(key, bytes);
  }

  /** Зафиксировать вытеснение уровня мипмапа. */
  recordEviction(textureId: string, mipLevel: number): void {
    const key = `${textureId}:${mipLevel}`;
    const existing = this._perTextureMip.get(key);
    if (existing !== undefined) {
      this._used -= existing;
      this._perTextureMip.delete(key);
    }
  }

  destroy(): void {
    this._perTextureMip.clear();
    this._used = 0;
  }
}
