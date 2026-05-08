/** Сигнатура колбэка для обновлений каждого кадра. */
export type FrameCallback = (deltaTime: number, frameIndex: number) => void;

/**
 * Цикл рендеринга на основе requestAnimationFrame с отслеживанием времени дельта
 * и монотонным счётчиком кадров.
 */
export class RenderLoop {
  private _callbacks: FrameCallback[] = [];
  private _rafHandle = 0;
  private _running = false;
  private _frameIndex = 0;
  private _lastTimestamp = 0;

  get isRunning(): boolean {
    return this._running;
  }

  get frameIndex(): number {
    return this._frameIndex;
  }

  /** Зарегистрировать колбэк, вызываемый каждый кадр. Возвращает функцию отписки. */
  addCallback(cb: FrameCallback): () => void {
    this._callbacks.push(cb);
    return () => {
      this._callbacks = this._callbacks.filter((c) => c !== cb);
    };
  }

  /** Запустить цикл рендеринга. */
  start(): void {
    if (this._running) return;
    this._running = true;
    this._lastTimestamp = performance.now();
    this._tick(this._lastTimestamp);
  }

  /** Остановить цикл рендеринга. */
  stop(): void {
    this._running = false;
    if (this._rafHandle !== 0) {
      cancelAnimationFrame(this._rafHandle);
      this._rafHandle = 0;
    }
  }

  private _tick = (timestamp: number): void => {
    if (!this._running) return;

    const deltaTime = Math.min((timestamp - this._lastTimestamp) / 1000, 0.1);
    this._lastTimestamp = timestamp;

    for (const cb of this._callbacks) {
      cb(deltaTime, this._frameIndex);
    }

    this._frameIndex = (this._frameIndex + 1) >>> 0;
    this._rafHandle = requestAnimationFrame(this._tick);
  };

  destroy(): void {
    this.stop();
    this._callbacks = [];
  }
}
