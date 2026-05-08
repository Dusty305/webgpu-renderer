/**
 * Demo 16 - Public API Surface (Phase 6.1)
 *
 * Demonstrates both integration paths:
 *  1. Custom Element (<webgpu-canvas-streaming>) with HTML attributes + loadScene()
 *  2. Programmatic API (createRenderer) with an existing <canvas> + addMesh()
 */

// Import from the internal alias so Vite resolves correctly in dev.
// In a published package, this would just be:
//   import { WebGPUCanvasElement, createRenderer } from "webgpu-streaming";
import { WebGPUCanvasElement, createRenderer } from "@webgpu-streaming/core";
import type { StreamingStats } from "@webgpu-streaming/core";

// ---- Side-effect import ensures the custom element is registered ------------------------------
// WebGPUCanvasElement is already registered as "webgpu-canvas-streaming"
// by WebGPUCanvasPublic.ts when the module is loaded.
void WebGPUCanvasElement; // reference to prevent tree-shaking

// ---- Helper --------------------------------------------------------------------------------------------------------------------------------------

function renderStats(el: HTMLElement, stats: StreamingStats): void {
  el.innerHTML = `<pre>` +
    `<span class="key">fps           </span><span class="val">${stats.fps.toFixed(1)}</span>\n` +
    `<span class="key">frameTimeP99  </span><span class="val">${stats.frameTimeP99Ms.toFixed(2)} ms</span>\n` +
    `<span class="key">memoryUsed    </span><span class="val">${stats.memoryUsedMB.toFixed(1)} / ${stats.memoryBudgetMB.toFixed(0)} MB</span>\n` +
    `<span class="key">textures      </span><span class="val">${stats.texturesLoaded} / ${stats.texturesTotal}</span>` +
    `</pre>`;
}

// ---- 1. Custom Element approach --------------------------------------------------------------------------------------------------

const elCanvas = document.querySelector<WebGPUCanvasElement>("webgpu-canvas-streaming");
const elStats  = document.getElementById("el-stats")!;

if (elCanvas) {
  elCanvas.addEventListener("gpu-ready", async () => {
    console.log("[Demo16] Custom Element gpu-ready - calling loadScene()");
    await elCanvas.loadScene();

    // Verify addMesh works too - add a second, smaller cube offset to the right
    const mesh = elCanvas.addMesh(
      {
        positions: new Float32Array([
          // Simple tetrahedron (4 verts, 4 triangles)
          0, 1, 0,   -1,-1, 1,   1,-1, 1,
          0, 1, 0,    1,-1, 1,   1,-1,-1,
          0, 1, 0,    1,-1,-1,  -1,-1,-1,
          0, 1, 0,   -1,-1,-1,  -1,-1, 1,
        ]),
        normals: new Float32Array([
          0,.7,.7, 0,.7,.7, 0,.7,.7,
          .7,.7,0, .7,.7,0, .7,.7,0,
          0,.7,-.7, 0,.7,-.7, 0,.7,-.7,
          -.7,.7,0, -.7,.7,0, -.7,.7,0,
        ]),
        indices: new Uint16Array([0,1,2, 3,4,5, 6,7,8, 9,10,11]),
      },
      { baseColor: "#33aaff" }
    );
    mesh.setPosition(2.5, 0, 0);
    mesh.setScale(0.6, 0.6, 0.6);
  });

  elCanvas.addEventListener("gpu-error", (e) => {
    elStats.textContent = `gpu-error: ${(e as CustomEvent<string>).detail}`;
  });

  elCanvas.addEventListener("streaming-stats", (e) => {
    renderStats(elStats, (e as CustomEvent<StreamingStats>).detail);
  });
}

// ---- 2. Programmatic API approach --------------------------------------------------------------------------------------------

const pgmCanvas = document.getElementById("pgm-canvas") as HTMLCanvasElement;
const pgmStats  = document.getElementById("pgm-stats")!;

(async () => {
  let renderer;
  try {
    renderer = await createRenderer({
      canvas: pgmCanvas,
      memoryBudget: 128,
      frameUploadCap: 8,
    });
  } catch (err) {
    pgmStats.textContent = `Error: ${String(err)}`;
    return;
  }

  console.log("[Demo16] Programmatic renderer ready:", renderer.ready);

  // Set camera position
  renderer.setCamera({ position: [3, 2, 5], target: [0, 0, 0] });

  // Add several meshes with different colors
  const colors = ["#e84040", "#40e860", "#4080e8", "#e8c040", "#e040e0"];
  const meshes = colors.map((color, i) => {
    const angle = (i / colors.length) * Math.PI * 2;
    const r = 2;
    const m = renderer.addMesh(
      makeSphere(0.4, 12, 8),
      { baseColor: color }
    );
    m.setPosition(Math.cos(angle) * r, 0, Math.sin(angle) * r);
    return m;
  });

  // Add a central cube
  await renderer.loadScene();

  // Animate
  let t = 0;
  function tick() {
    t += 0.01;
    meshes.forEach((m, i) => {
      const angle = (i / meshes.length) * Math.PI * 2 + t;
      m.setPosition(Math.cos(angle) * 2, Math.sin(t * 0.7 + i) * 0.3, Math.sin(angle) * 2);
    });
    renderStats(pgmStats, renderer.getStats());
    requestAnimationFrame(tick);
  }
  tick();
})();

// ---- Sphere geometry helper --------------------------------------------------------------------------------------------------------

function makeSphere(
  radius: number,
  latSegments: number,
  lonSegments: number
): { positions: Float32Array; normals: Float32Array; uvs: Float32Array; indices: Uint16Array } {
  const positions: number[] = [];
  const normals:   number[] = [];
  const uvs:       number[] = [];
  const indices:   number[] = [];

  for (let lat = 0; lat <= latSegments; lat++) {
    const theta = (lat / latSegments) * Math.PI;
    const sinT = Math.sin(theta);
    const cosT = Math.cos(theta);
    for (let lon = 0; lon <= lonSegments; lon++) {
      const phi  = (lon / lonSegments) * Math.PI * 2;
      const nx = Math.cos(phi) * sinT;
      const ny = cosT;
      const nz = Math.sin(phi) * sinT;
      positions.push(nx * radius, ny * radius, nz * radius);
      normals.push(nx, ny, nz);
      uvs.push(lon / lonSegments, lat / latSegments);
    }
  }

  for (let lat = 0; lat < latSegments; lat++) {
    for (let lon = 0; lon < lonSegments; lon++) {
      const a = lat * (lonSegments + 1) + lon;
      const b = a + lonSegments + 1;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals:   new Float32Array(normals),
    uvs:       new Float32Array(uvs),
    indices:   new Uint16Array(indices),
  };
}
