/**
 * Точка входа Web Worker для транскодирования WASM Basis Universal.
 * Полная реализация добавлена в фазе 2.6.
 *
 * Протокол сообщений (фаза 2.6):
 *   ВХОДЯЩИЕ:  { type: "transcode", id: number, ktx2Slice: ArrayBuffer, targetFormat: string, mipLevel: number }
 *   ИСХОДЯЩИЕ: { type: "result", id: number, data: ArrayBuffer, width: number, height: number }
 *            | { type: "error", id: number, message: string }
 */

self.addEventListener("message", () => {
  // Заглушка - транскодирование ещё не реализовано.
  throw new Error("[transcode-worker] Ещё не реализовано - см. фазу 2.6.");
});
