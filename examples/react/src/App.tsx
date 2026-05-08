import { useEffect, useState } from "react";
import { useWebGPU } from "./useWebGPU";
import type { StreamingStats } from "webgpu-streaming";

export function App() {
  const { canvasRef, renderer, error } = useWebGPU({ memoryBudget: 128 });
  const [stats, setStats] = useState<StreamingStats | null>(null);

  useEffect(() => {
    if (!renderer) return;

    renderer.loadScene().then(() => {
      renderer.setCamera({ position: [0, 2, 5], target: [0, 0, 0], fov: 60 });
      const s = 0.7;
      const positions = new Float32Array([
        -s,-s, s,  s,-s, s,  s, s, s,  -s,-s, s,  s, s, s,  -s, s, s,
         s,-s,-s, -s,-s,-s, -s, s,-s,   s,-s,-s, -s, s,-s,   s, s,-s,
        -s, s, s,  s, s, s,  s, s,-s,  -s, s, s,  s, s,-s,  -s, s,-s,
        -s,-s,-s,  s,-s,-s,  s,-s, s,  -s,-s,-s,  s,-s, s,  -s,-s, s,
         s,-s, s,  s,-s,-s,  s, s,-s,   s,-s, s,  s, s,-s,   s, s, s,
        -s,-s,-s, -s,-s, s, -s, s, s,  -s,-s,-s, -s, s, s,  -s, s,-s,
      ]);
      const normals = new Float32Array([
        ...Array(6).fill([0,0,1]).flat() as number[],
        ...Array(6).fill([0,0,-1]).flat() as number[],
        ...Array(6).fill([0,1,0]).flat() as number[],
        ...Array(6).fill([0,-1,0]).flat() as number[],
        ...Array(6).fill([1,0,0]).flat() as number[],
        ...Array(6).fill([-1,0,0]).flat() as number[],
      ]);
      renderer.addMesh({ positions, normals }, { baseColor: "#4a9eff" });
    });

    const id = setInterval(() => setStats(renderer.getStats()), 500);
    return () => clearInterval(id);
  }, [renderer]);

  return (
    <div style={{ background: "#0e0e0e", color: "#ddd", fontFamily: "sans-serif", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100dvh", gap: 12 }}>
      <canvas ref={canvasRef} width={720} height={480} style={{ display: "block", borderRadius: 8 }} />
      <div style={{ fontSize: ".75rem", opacity: .55, fontVariantNumeric: "tabular-nums" }}>
        {error
          ? `Ошибка: ${error}`
          : stats
            ? `fps: ${stats.fps.toFixed(1)}  mem: ${stats.memoryUsedMB.toFixed(1)} / ${stats.memoryBudgetMB} MB`
            : "Инициализация…"}
      </div>
    </div>
  );
}
