/**
 * Общая утилита оверлея для страниц демонстраций.
 * Создаёт фиксированный HUD-оверлей и предоставляет вспомогательные функции обновления статистики.
 */
export function createOverlay(title: string): {
  el: HTMLElement;
  set: (key: string, value: string | number) => void;
} {
  const el = document.createElement("div");
  el.style.cssText = [
    "position:fixed",
    "top:12px",
    "left:12px",
    "background:rgba(0,0,0,0.65)",
    "color:#e0e0e0",
    "font:13px/1.6 monospace",
    "padding:10px 14px",
    "border-radius:6px",
    "pointer-events:none",
    "z-index:999",
    "min-width:220px",
  ].join(";");

  const titleEl = document.createElement("div");
  titleEl.style.cssText = "color:#fff;font-weight:bold;margin-bottom:6px;border-bottom:1px solid #555;padding-bottom:4px;";
  titleEl.textContent = title;
  el.appendChild(titleEl);

  document.body.appendChild(el);

  const rows = new Map<string, HTMLElement>();

  function set(key: string, value: string | number): void {
    let row = rows.get(key);
    if (!row) {
      row = document.createElement("div");
      el.appendChild(row);
      rows.set(key, row);
    }
    row.textContent = `${key}: ${value}`;
  }

  return { el, set };
}

/** Простой счётчик FPS. Вызывайте tick() каждый кадр; читайте свойство fps. */
export class FpsTracker {
  private _frames = 0;
  private _last = performance.now();
  fps = 0;

  tick(): void {
    this._frames++;
    const now = performance.now();
    if (now - this._last >= 500) {
      this.fps = Math.round(this._frames * 1000 / (now - this._last));
      this._frames = 0;
      this._last = now;
    }
  }
}
