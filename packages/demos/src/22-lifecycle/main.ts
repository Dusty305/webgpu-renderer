/**
 * Демо 22 — Жизненный цикл веб-компонента <webgpu-canvas-streaming>
 *
 * Проверяет:
 *   • Корректность освобождения GPU-ресурсов при disconnectedCallback()
 *   • Отсутствие прогрессивных утечек при 10 циклах создания/удаления
 *   • Независимость нескольких экземпляров на одной странице
 *   • Устойчивость к изменению размера (ResizeObserver)
 *   • Поведение при потере GPU-устройства (событие webgpu-lost)
 */

import { WebGPUCanvasElement } from "@webgpu-streaming/core";
void WebGPUCanvasElement; // ensure custom element is registered

// ── helpers ───────────────────────────────────────────────────────────────────

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`#${id} not found`);
  return el;
}

function delay(ms: number): Promise<void> {
  return new Promise(r => setTimeout(r, ms));
}

/** Creates a <webgpu-canvas-streaming> element and appends it to container.
 *  Resolves when gpu-ready fires (before loadScene). */
function spawnElement(
  container: HTMLElement,
  id?: string,
): Promise<WebGPUCanvasElement> {
  return new Promise((resolve, reject) => {
    const el = document.createElement("webgpu-canvas-streaming") as WebGPUCanvasElement;
    el.setAttribute("memory-budget", "64");
    el.setAttribute("frame-upload-cap", "4");
    el.setAttribute("camera-mode", "orbit");
    if (id) el.id = id;
    el.style.cssText = "display:block;width:100%;height:100%;";

    const timer = setTimeout(() => {
      reject(new Error("gpu-ready timeout (15 s)"));
    }, 15_000);

    el.addEventListener("gpu-ready", () => {
      clearTimeout(timer);
      resolve(el);
    }, { once: true });

    el.addEventListener("webgpu-error", (ev: Event) => {
      clearTimeout(timer);
      reject(new Error((ev as CustomEvent<{ message: string }>).detail?.message ?? "WebGPU error"));
    }, { once: true });

    container.innerHTML = "";
    container.appendChild(el);
  });
}

/** Removes element from DOM, returns after pauseMs for async cleanup. */
async function killElement(el: WebGPUCanvasElement, pauseMs = 2000): Promise<void> {
  el.remove();
  await delay(pauseMs);
}

// ── global error guard ────────────────────────────────────────────────────────

if (!navigator.gpu) {
  const box = $("error-box");
  box.style.display = "block";
  box.textContent = "WebGPU недоступен в этом браузере. Откройте в Chrome 113+ с включённым WebGPU.";
}

// ═══════════════════════════════════════════════════════════════════════════════
// РАЗДЕЛ 1: основной тест жизненного цикла
// ═══════════════════════════════════════════════════════════════════════════════

const lcHost      = $("lc-canvas-host");
const lcLabel     = $("lc-canvas-label");
const lcPlaceholder = $("lc-placeholder");

const btnStep1    = $("btn-step1")  as HTMLButtonElement;
const btnCreate   = $("btn-create") as HTMLButtonElement;
const btnRemove   = $("btn-remove") as HTMLButtonElement;
const btnCycles   = $("btn-cycles") as HTMLButtonElement;
const btnStep2    = $("btn-step2")  as HTMLButtonElement;
const btnStep3    = $("btn-step3")  as HTMLButtonElement;
const btnStep4    = $("btn-step4")  as HTMLButtonElement;

const inpBaseline     = $("inp-baseline")     as HTMLInputElement;
const inpAfterCreate  = $("inp-after-create") as HTMLInputElement;
const inpAfterRemove  = $("inp-after-remove") as HTMLInputElement;
const inpAfterCycles  = $("inp-after-cycles") as HTMLInputElement;

const chipStep1   = $("chip-step1");
const chipStep2   = $("chip-step2");
const chipStep3   = $("chip-step3");
const chipStep4   = $("chip-step4");

const cyclesProgressWrap = $("cycles-progress-wrap") as HTMLDivElement;
const cyclesBar   = $("cycles-bar")   as HTMLDivElement;
const cyclesLabel = $("cycles-label") as HTMLDivElement;

const rBaseline    = $("r-baseline");
const rAfterCreate = $("r-after-create");
const rAfterRemove = $("r-after-remove");
const rAfterCycles = $("r-after-cycles");
const rDelta       = $("r-delta");
const verdictText  = $("verdict-text");
const resultsSection = $("results-section") as HTMLDivElement;

let lcEl: WebGPUCanvasElement | null = null;
let measurements = { baseline: NaN, afterCreate: NaN, afterRemove: NaN, afterCycles: NaN };

function setStep(step: 1 | 2 | 3 | 4): void {
  for (let i = 1; i <= 4; i++) {
    const el = $(`step${i}`);
    el.className = "step" + (i === step ? " active" : i < step ? " done" : "");
  }
}

function setChip(chip: HTMLElement, text: string, kind: "ok" | "busy" | "warn" | "error" | ""): void {
  chip.textContent = text;
  chip.className = "chip" + (kind ? ` ${kind}` : "");
}

function updateTable(): void {
  const { baseline, afterCreate, afterRemove, afterCycles } = measurements;
  const fmt = (v: number) => isNaN(v) ? "—" : `${v.toFixed(1)} МБ`;

  rBaseline.textContent    = fmt(baseline);
  rAfterCreate.textContent = fmt(afterCreate);
  rAfterRemove.textContent = fmt(afterRemove);
  rAfterCycles.textContent = fmt(afterCycles);
  rBaseline.className    = isNaN(baseline) ? "pending" : "";
  rAfterCreate.className = isNaN(afterCreate) ? "pending" : "";
  rAfterRemove.className = isNaN(afterRemove) ? "pending" : "";
  rAfterCycles.className = isNaN(afterCycles) ? "pending" : "";

  if (!isNaN(baseline) && !isNaN(afterCycles)) {
    const delta = afterCycles - baseline;
    rDelta.textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} МБ`;
    rDelta.className = "";
    const THRESHOLD = 5; // МБ
    if (Math.abs(delta) <= THRESHOLD) {
      verdictText.textContent =
        `✓ Дельта ${delta.toFixed(1)} МБ — в пределах погрешности (±${THRESHOLD} МБ). ` +
        `disconnectedCallback() корректно освобождает GPU-ресурсы.`;
      verdictText.style.color = "#5c5";
    } else {
      verdictText.textContent =
        `⚠ Дельта ${delta.toFixed(1)} МБ превышает порог ${THRESHOLD} МБ — ` +
        `возможна утечка GPU-памяти.`;
      verdictText.style.color = "#fa0";
    }
  }

  resultsSection.style.display = "block";
}

// ── Шаг 1: базовый уровень ────────────────────────────────────────────────────

btnStep1.addEventListener("click", () => {
  const v = parseFloat(inpBaseline.value);
  if (isNaN(v)) { setChip(chipStep1, "введите значение", "warn"); return; }
  measurements.baseline = v;
  setChip(chipStep1, `${v.toFixed(1)} МБ`, "ok");
  setStep(2);
  btnCreate.disabled = false;
  updateTable();
});

// ── Шаг 2: создать элемент ────────────────────────────────────────────────────

btnCreate.addEventListener("click", async () => {
  btnCreate.disabled = true;
  setChip(chipStep2, "инициализация…", "busy");
  lcLabel.textContent = "инициализация…";
  lcPlaceholder.style.display = "none";

  try {
    lcEl = await spawnElement(lcHost);
    setChip(chipStep2, "gpu-ready, загрузка сцены…", "busy");
    lcLabel.textContent = "загрузка сцены…";

    await lcEl.loadScene();

    // wait a couple of seconds for streaming to settle
    await delay(2000);

    setChip(chipStep2, "готов — снимите показание", "ok");
    lcLabel.textContent = "работает";
    inpAfterCreate.disabled = false;
    btnStep2.disabled = false;
  } catch (e) {
    setChip(chipStep2, String(e), "error");
    lcPlaceholder.style.display = "flex";
    lcLabel.textContent = "ошибка";
    btnCreate.disabled = false;
  }
});

btnStep2.addEventListener("click", () => {
  const v = parseFloat(inpAfterCreate.value);
  if (isNaN(v)) { setChip(chipStep2, "введите значение", "warn"); return; }
  measurements.afterCreate = v;
  setChip(chipStep2, `${v.toFixed(1)} МБ`, "ok");
  setStep(3);
  btnRemove.disabled = false;
  updateTable();
});

// ── Шаг 3: удалить элемент ────────────────────────────────────────────────────

btnRemove.addEventListener("click", async () => {
  if (!lcEl) return;
  btnRemove.disabled = true;
  setChip(chipStep3, "удаление…", "busy");
  lcLabel.textContent = "удаление из DOM…";

  await killElement(lcEl, 2000);
  lcEl = null;

  lcPlaceholder.innerHTML = "— элемент удалён из DOM —";
  lcPlaceholder.style.display = "flex";
  lcLabel.textContent = "нет элемента";

  setChip(chipStep3, "удалён, снимите показание", "ok");
  inpAfterRemove.disabled = false;
  btnStep3.disabled = false;
});

btnStep3.addEventListener("click", () => {
  const v = parseFloat(inpAfterRemove.value);
  if (isNaN(v)) { setChip(chipStep3, "введите значение", "warn"); return; }
  measurements.afterRemove = v;
  setChip(chipStep3, `${v.toFixed(1)} МБ`, "ok");
  setStep(4);
  btnCycles.disabled = false;
  updateTable();
});

// ── Шаг 4: 10 циклов ─────────────────────────────────────────────────────────

const TOTAL_CYCLES = 10;

btnCycles.addEventListener("click", async () => {
  btnCycles.disabled = true;
  setChip(chipStep4, "выполняется…", "busy");
  cyclesProgressWrap.style.display = "block";
  cyclesLabel.style.display = "block";

  for (let i = 0; i < TOTAL_CYCLES; i++) {
    cyclesLabel.textContent = `Цикл ${i + 1} / ${TOTAL_CYCLES} — создание…`;
    lcPlaceholder.innerHTML = `Цикл ${i + 1} / ${TOTAL_CYCLES}`;
    lcPlaceholder.style.display = "none";

    try {
      const el = await spawnElement(lcHost);
      cyclesLabel.textContent = `Цикл ${i + 1} / ${TOTAL_CYCLES} — загрузка…`;
      await el.loadScene();
      await delay(2000);
      cyclesLabel.textContent = `Цикл ${i + 1} / ${TOTAL_CYCLES} — удаление…`;
      await killElement(el, 2000);
    } catch (e) {
      console.warn(`[Demo22] цикл ${i + 1} ошибка:`, e);
    }

    const pct = ((i + 1) / TOTAL_CYCLES) * 100;
    cyclesBar.style.width = `${pct}%`;
    cyclesLabel.textContent = `Выполнено ${i + 1} / ${TOTAL_CYCLES}`;
  }

  lcPlaceholder.innerHTML = "— 10 циклов завершены —";
  lcPlaceholder.style.display = "flex";
  lcLabel.textContent = "нет элемента";

  setChip(chipStep4, "10 циклов завершены — снимите показание", "ok");
  inpAfterCycles.disabled = false;
  btnStep4.disabled = false;
});

btnStep4.addEventListener("click", () => {
  const v = parseFloat(inpAfterCycles.value);
  if (isNaN(v)) { setChip(chipStep4, "введите значение", "warn"); return; }
  measurements.afterCycles = v;
  setChip(chipStep4, `${v.toFixed(1)} МБ`, "ok");
  $("step4").className = "step done";
  updateTable();
});

// ═══════════════════════════════════════════════════════════════════════════════
// РАЗДЕЛ 2: множественные экземпляры
// ═══════════════════════════════════════════════════════════════════════════════

const miHostA   = $("mi-canvas-a");
const miHostB   = $("mi-canvas-b");
const miStatusA = $("mi-status-a");
const miStatusB = $("mi-status-b");
const btnMiCreate   = $("btn-mi-create")   as HTMLButtonElement;
const btnMiRemoveA  = $("btn-mi-remove-a") as HTMLButtonElement;
const btnMiRemoveB  = $("btn-mi-remove-b") as HTMLButtonElement;
const btnMiReset    = $("btn-mi-reset")    as HTMLButtonElement;

let miElA: WebGPUCanvasElement | null = null;
let miElB: WebGPUCanvasElement | null = null;

btnMiCreate.addEventListener("click", async () => {
  btnMiCreate.disabled = true;
  miStatusA.textContent = "инициализация…";
  miStatusB.textContent = "инициализация…";

  try {
    [miElA, miElB] = await Promise.all([
      spawnElement(miHostA, "mi-el-a"),
      spawnElement(miHostB, "mi-el-b"),
    ]);
    await Promise.all([miElA.loadScene(), miElB.loadScene()]);
    miStatusA.textContent = "работает ✓";
    miStatusB.textContent = "работает ✓";
    btnMiRemoveA.disabled = false;
    btnMiRemoveB.disabled = false;
  } catch (e) {
    miStatusA.textContent = `ошибка: ${e}`;
    miStatusB.textContent = `ошибка: ${e}`;
    btnMiCreate.disabled = false;
  }
});

btnMiRemoveA.addEventListener("click", async () => {
  if (!miElA) return;
  btnMiRemoveA.disabled = true;
  miStatusA.textContent = "удаление…";
  await killElement(miElA, 500);
  miElA = null;
  miHostA.innerHTML = `<div class="canvas-placeholder" style="height:100%;width:100%;">— удалён —</div>`;
  miStatusA.textContent = "удалён из DOM";
  // проверяем, что B продолжает работать
  if (miElB?.isConnected) {
    miStatusB.textContent = "продолжает работать ✓";
  }
});

btnMiRemoveB.addEventListener("click", async () => {
  if (!miElB) return;
  btnMiRemoveB.disabled = true;
  miStatusB.textContent = "удаление…";
  await killElement(miElB, 500);
  miElB = null;
  miHostB.innerHTML = `<div class="canvas-placeholder" style="height:100%;width:100%;">— удалён —</div>`;
  miStatusB.textContent = "удалён из DOM";
  if (miElA?.isConnected) {
    miStatusA.textContent = "продолжает работать ✓";
  }
});

btnMiReset.addEventListener("click", () => {
  miElA?.remove(); miElA = null;
  miElB?.remove(); miElB = null;
  miHostA.innerHTML = `<div class="canvas-placeholder" style="height:100%;width:100%;">— ожидание —</div>`;
  miHostB.innerHTML = `<div class="canvas-placeholder" style="height:100%;width:100%;">— ожидание —</div>`;
  miStatusA.textContent = "не создан";
  miStatusB.textContent = "не создан";
  btnMiCreate.disabled = false;
  btnMiRemoveA.disabled = true;
  btnMiRemoveB.disabled = true;
});

// ═══════════════════════════════════════════════════════════════════════════════
// РАЗДЕЛ 3: изменение размера
// ═══════════════════════════════════════════════════════════════════════════════

const rsHost      = $("rs-canvas-host");
const rsLog       = $("rs-log");
const btnRsCreate = $("btn-rs-create")     as HTMLButtonElement;
const btnRsRemove = $("btn-rs-remove")     as HTMLButtonElement;
const btnRsFull   = $("btn-rs-fullscreen") as HTMLButtonElement;
const rsWidthSlider  = $("rs-width")       as HTMLInputElement;
const rsHeightSlider = $("rs-height")      as HTMLInputElement;
const rsWidthVal     = $("rs-width-val");
const rsHeightVal    = $("rs-height-val");

let rsEl: WebGPUCanvasElement | null = null;
let rsLogCount = 0;

function rsLogEntry(msg: string, kind: "ok" | "warn" | "" = ""): void {
  rsLogCount++;
  const line = document.createElement("div");
  if (kind) line.className = `log-${kind}`;
  line.textContent = `[${rsLogCount}] ${msg}`;
  rsLog.appendChild(line);
  rsLog.scrollTop = rsLog.scrollHeight;
}

function applyRsSize(): void {
  rsHost.style.width  = rsWidthSlider.value + "px";
  rsHost.style.height = rsHeightSlider.value + "px";
  rsWidthVal.textContent  = rsWidthSlider.value + "px";
  rsHeightVal.textContent = rsHeightSlider.value + "px";
  rsLogEntry(`resize → ${rsWidthSlider.value}×${rsHeightSlider.value} px`);
}

rsWidthSlider.addEventListener("input", applyRsSize);
rsHeightSlider.addEventListener("input", applyRsSize);

btnRsCreate.addEventListener("click", async () => {
  btnRsCreate.disabled = true;
  rsLog.textContent = "";
  rsLogEntry("создание элемента…");
  try {
    rsEl = await spawnElement(rsHost);
    rsLogEntry("gpu-ready", "ok");
    await rsEl.loadScene();
    rsLogEntry("сцена загружена", "ok");
    rsWidthSlider.disabled  = false;
    rsHeightSlider.disabled = false;
    btnRsFull.disabled = false;
    btnRsRemove.disabled = false;

    // listen for future resize events from the element
    rsEl.addEventListener("streaming-stats", () => {
      // no-op: just verifying element stays alive
    });
  } catch (e) {
    rsLogEntry(`ошибка: ${e}`, "warn");
    btnRsCreate.disabled = false;
  }
});

btnRsRemove.addEventListener("click", async () => {
  if (!rsEl) return;
  btnRsRemove.disabled = true;
  rsWidthSlider.disabled = true;
  rsHeightSlider.disabled = true;
  btnRsFull.disabled = true;
  rsLogEntry("удаление из DOM…");
  await killElement(rsEl, 500);
  rsEl = null;
  rsHost.innerHTML = `<div class="canvas-placeholder" style="height:100%;width:100%;">— удалён —</div>`;
  rsLogEntry("удалён", "ok");
  btnRsCreate.disabled = false;
});

btnRsFull.addEventListener("click", () => {
  if (!rsEl) return;
  rsLogEntry("запрос fullscreen…");
  void rsEl.requestFullscreen().then(() => {
    rsLogEntry("fullscreen активен", "ok");
  }).catch((e: unknown) => {
    rsLogEntry(`fullscreen отклонён: ${e}`, "warn");
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// РАЗДЕЛ 4: потеря устройства
// ═══════════════════════════════════════════════════════════════════════════════

const dlHost      = $("dl-canvas-host");
const dlLog       = $("dl-log");
const dlStatus    = $("dl-status");
const btnDlCreate   = $("btn-dl-create")   as HTMLButtonElement;
const btnDlRemove   = $("btn-dl-remove")   as HTMLButtonElement;
const btnDlSimulate = $("btn-dl-simulate") as HTMLButtonElement;

let dlEl: WebGPUCanvasElement | null = null;
let dlLogCount = 0;

function dlLogEntry(msg: string, kind: "ok" | "warn" | "err" | "" = ""): void {
  dlLogCount++;
  const line = document.createElement("div");
  if (kind) line.className = `log-${kind}`;
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  dlLog.appendChild(line);
  dlLog.scrollTop = dlLog.scrollHeight;
}

btnDlCreate.addEventListener("click", async () => {
  btnDlCreate.disabled = true;
  dlLog.textContent = "";
  dlStatus.textContent = "инициализация…";
  dlLogEntry("создание элемента…");

  try {
    dlEl = await spawnElement(dlHost, "dl-el");
    dlLogEntry("gpu-ready", "ok");

    dlEl.addEventListener("webgpu-lost", (ev: Event) => {
      const reason = (ev as CustomEvent<{ reason?: string }>).detail?.reason ?? "unknown";
      dlLogEntry(`webgpu-lost: reason="${reason}"`, "warn");
      dlStatus.textContent = `устройство потеряно (${reason}) — отображается последний кадр`;
    });

    dlEl.addEventListener("streaming-stats", () => {
      /* render loop still running — no-op */
    });

    await dlEl.loadScene();
    dlLogEntry("сцена загружена", "ok");
    dlStatus.textContent = "работает — ожидание потери устройства";
    btnDlRemove.disabled = false;
    btnDlSimulate.disabled = false;
  } catch (e) {
    dlLogEntry(`ошибка: ${e}`, "err");
    dlStatus.textContent = `ошибка: ${e}`;
    btnDlCreate.disabled = false;
  }
});

btnDlRemove.addEventListener("click", async () => {
  if (!dlEl) return;
  btnDlRemove.disabled = true;
  btnDlSimulate.disabled = true;
  dlLogEntry("удаление из DOM…");
  await killElement(dlEl, 500);
  dlEl = null;
  dlHost.innerHTML = `<div class="canvas-placeholder" style="height:100%;width:100%;">— удалён —</div>`;
  dlLogEntry("удалён", "ok");
  dlStatus.textContent = "удалён из DOM";
  btnDlCreate.disabled = false;
});

btnDlSimulate.addEventListener("click", () => {
  if (!dlEl) return;
  dlLogEntry("симуляция webgpu-lost (soft)…", "warn");
  dlEl.dispatchEvent(new CustomEvent("webgpu-lost", {
    detail: { reason: "simulated" },
    bubbles: false,
    composed: false,
  }));
});
