import { useEffect, useRef, useState } from "react";
import { createRenderer } from "webgpu-streaming";
import type { Renderer } from "webgpu-streaming";

export function useWebGPU(options?: { memoryBudget?: number; frameUploadCap?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [renderer, setRenderer] = useState<Renderer | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let r: Renderer | undefined;
    let cancelled = false;

    (async () => {
      if (!canvasRef.current) return;
      try {
        r = await createRenderer({ canvas: canvasRef.current, ...options });
        if (!cancelled) setRenderer(r);
      } catch (e) {
        if (!cancelled) setError(String(e));
      }
    })();

    return () => {
      cancelled = true;
      r?.dispose();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { canvasRef, renderer, error };
}
