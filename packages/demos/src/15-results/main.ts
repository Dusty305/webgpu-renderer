/**
 * Демо 15 - Сводка результатов бенчмарков (фаза 5.3)
 *
 * Загружает JSON-экспорты, произведённые демо 14, и отрисовывает:
 *  - Таблица 1: агрегированная статистика по конфигам
 *  - График 1: столбчатый график пиковой памяти GPU с планками погрешности стд. откл.
 *  - График 2: ящичные диаграммы времени кадра (требует сырой JSON с покадровыми данными)
 *  - График 3: диаграмма рассеяния качество vs. память (ключевой рисунок тезиса)
 *  - Таблица 2: результаты парных тестов Уилкоксона
 *  - Таблица 3: сырые данные по запускам (сворачиваемая; требует сырой JSON)
 *
 * Без библиотеки графиков - все графики отрисованы через Canvas 2D.
 */

import type {
  SummaryExport,
  RawExport,
  AggregateStats,
  RunData,
  DynamicCameraStats,
} from "../14-full-benchmark/main.js";
import type { WilcoxonResult } from "../shared/WilcoxonTest.js";

// ---- Состояние ------------------------------------------------------------------------------------------------------------------------------------------

let summary:    SummaryExport | null = null;
let raw:        RawExport     | null = null;
let summary720: SummaryExport | null = null;

// ---- Вспомогательные функции для графиков --------------------------------------------------------------------------------------------------------------------------

const CONFIG_COLORS: Record<string, string> = {
  A: "#f88", B: "#fa8", C: "#ff8", D: "#8f8", E: "#8af",
};

function configColor(id: string): string {
  return CONFIG_COLORS[id] ?? "#ccc";
}

/** Размеры области графика с отступами в масштабе Canvas. */
interface ChartArea {
  canvas: HTMLCanvasElement;
  ctx:    CanvasRenderingContext2D;
  left:   number;
  top:    number;
  right:  number;
  bottom: number;
  width:  number;   // right - left
  height: number;   // bottom - top
  /** Отображает значение данных в пиксельную координату x в области графика. */
  scaleX: (v: number, domainMin: number, domainMax: number) => number;
  /** Отображает значение данных в пиксельную координату y в области графика (инвертировано). */
  scaleY: (v: number, domainMin: number, domainMax: number) => number;
}

function makeChartArea(
  canvas: HTMLCanvasElement,
  padLeft = 68, padTop = 28, padRight = 28, padBottom = 52,
): ChartArea {
  const ctx = canvas.getContext("2d")!;
  const left   = padLeft;
  const top    = padTop;
  const right  = canvas.width  - padRight;
  const bottom = canvas.height - padBottom;
  const width  = right  - left;
  const height = bottom - top;
  return {
    canvas, ctx, left, top, right, bottom, width, height,
    scaleX(v, dMin, dMax) {
      return left + ((v - dMin) / (dMax - dMin)) * width;
    },
    scaleY(v, dMin, dMax) {
      return bottom - ((v - dMin) / (dMax - dMin)) * height;
    },
  };
}

function clearCanvas(c: HTMLCanvasElement): void {
  const ctx = c.getContext("2d")!;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, c.width, c.height);
}

/** Отрисовывает горизонтальные линии сетки и подпись оси Y. */
function drawYGrid(
  ca: ChartArea,
  ticks: number[],
  domainMin: number,
  domainMax: number,
  axisLabel: string,
): void {
  const { ctx } = ca;
  ctx.strokeStyle = "#2a2a2a";
  ctx.lineWidth   = 1;
  ctx.font        = "10px monospace";
  ctx.fillStyle   = "#666";
  ctx.textAlign   = "right";
  ctx.textBaseline = "middle";

  for (const t of ticks) {
    const y = ca.scaleY(t, domainMin, domainMax);
    ctx.beginPath(); ctx.moveTo(ca.left, y); ctx.lineTo(ca.right, y); ctx.stroke();
    ctx.fillText(t.toFixed(t < 10 ? 1 : 0), ca.left - 6, y);
  }

  // Подпись оси (повёрнутая)
  ctx.save();
  ctx.translate(12, (ca.top + ca.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(axisLabel, 0, 0);
  ctx.restore();
}

/** Красивые значения делений, покрывающие [min, max] примерно с 5 шагами. */
function niceTicks(min: number, max: number, count = 5): number[] {
  const range  = max - min;
  const raw    = range / (count - 1);
  const mag    = Math.pow(10, Math.floor(Math.log10(raw)));
  const step   = [1, 2, 2.5, 5, 10].map(f => f * mag).find(s => s >= raw) ?? mag;
  const start  = Math.floor(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 0.01; v += step) {
    if (v >= min - step * 0.01) ticks.push(parseFloat(v.toFixed(8)));
  }
  return ticks;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(Math.ceil(sorted.length * p) - 1, sorted.length - 1);
  return sorted[idx] ?? 0;
}

// ---- График 1: Столбчатый график памяти --------------------------------------------------------------------------------------------------

function drawMemoryChart(configs: AggregateStats[]): void {
  const canvas = document.getElementById("chart-memory") as HTMLCanvasElement;
  clearCanvas(canvas);
  if (configs.length === 0) return;

  const ca = makeChartArea(canvas, 68, 28, 20, 52);
  const { ctx } = ca;

  const maxVal  = Math.max(...configs.map(c => c.peakMemoryMB.mean + c.peakMemoryMB.std));
  const ticks   = niceTicks(0, maxVal * 1.1);
  const domainMax = ticks[ticks.length - 1] ?? maxVal * 1.1;

  drawYGrid(ca, ticks, 0, domainMax, "Peak GPU Memory (MB)");

  // Отрисовываем рамку осей
  ctx.strokeStyle = "#444";
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(ca.left, ca.top); ctx.lineTo(ca.left, ca.bottom);
  ctx.lineTo(ca.right, ca.bottom);
  ctx.stroke();

  const n       = configs.length;
  const barW    = (ca.width / n) * 0.55;
  const spacing = ca.width / n;

  ctx.font        = "11px monospace";
  ctx.textBaseline = "top";

  for (let i = 0; i < n; i++) {
    const cfg  = configs[i]!;
    const cx   = ca.left + spacing * i + spacing / 2;
    const barX = cx - barW / 2;
    const barH = (cfg.peakMemoryMB.mean / domainMax) * ca.height;
    const barY = ca.bottom - barH;

    // Столбец
    ctx.fillStyle = configColor(cfg.config);
    ctx.globalAlpha = 0.85;
    ctx.fillRect(barX, barY, barW, barH);
    ctx.globalAlpha = 1.0;

    // Планка погрешности стандартного отклонения
    if (cfg.peakMemoryMB.std > 0) {
      const eTop = ca.scaleY(cfg.peakMemoryMB.mean + cfg.peakMemoryMB.std, 0, domainMax);
      const eBot = ca.scaleY(cfg.peakMemoryMB.mean - cfg.peakMemoryMB.std, 0, domainMax);
      const capW = barW * 0.25;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth   = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, eTop); ctx.lineTo(cx, eBot);
      ctx.moveTo(cx - capW, eTop); ctx.lineTo(cx + capW, eTop);
      ctx.moveTo(cx - capW, eBot); ctx.lineTo(cx + capW, eBot);
      ctx.stroke();
    }

    // Подпись значения над столбцом
    ctx.fillStyle    = "#ccc";
    ctx.textAlign    = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(cfg.peakMemoryMB.mean.toFixed(1), cx, barY - 3);

    // Подпись конфига снизу
    ctx.textBaseline = "top";
    ctx.font = "bold 12px monospace";
    ctx.fillStyle = configColor(cfg.config);
    ctx.fillText(cfg.config, cx - (ctx.measureText(cfg.config).width / 2),
                 ca.bottom + 8);
    ctx.font = "10px monospace";
    ctx.fillStyle = "#666";
    ctx.fillText(cfg.config === "B" ? "÷4 proj." : "", cx, ca.bottom + 22);
  }

  // Легенда: индикатор стандартного отклонения
  ctx.fillStyle    = "#666";
  ctx.font         = "10px monospace";
  ctx.textAlign    = "left";
  ctx.textBaseline = "top";
  ctx.fillText("Error bars: ±1 std dev", ca.left, ca.top + 4);
}

// ---- График 2: Ящичные диаграммы времени кадра ------------------------------------------------------------------------------------------

interface BoxStats {
  p10: number; p25: number; p50: number; p75: number; p90: number; p99: number;
}

function computeBox(values: number[]): BoxStats {
  const s = [...values].sort((a, b) => a - b);
  return {
    p10: percentile(s, 0.10), p25: percentile(s, 0.25), p50: percentile(s, 0.50),
    p75: percentile(s, 0.75), p90: percentile(s, 0.90), p99: percentile(s, 0.99),
  };
}

function drawFrameTimeChart(
  configs:    AggregateStats[],
  rawByConfig: Map<string, RunData[]> | null,
): void {
  const canvas = document.getElementById("chart-frametimes") as HTMLCanvasElement;
  clearCanvas(canvas);
  if (configs.length === 0) return;

  const ca = makeChartArea(canvas, 68, 28, 20, 52);
  const { ctx } = ca;

  // Собираем статистику ящика для каждого конфига
  const boxes: { id: string; box: BoxStats }[] = [];
  for (const cfg of configs) {
    let values: number[];
    if (rawByConfig) {
      const runs = rawByConfig.get(cfg.config) ?? [];
      values = runs.flatMap(r => r.frameTimesMs);
    } else {
      // Запасной вариант: синтезируем из агрегатов по запускам (приблизительно)
      values = [
        cfg.frameTimeMedian.mean - cfg.frameTimeMedian.std,
        cfg.frameTimeMedian.mean,
        cfg.frameTimeMedian.mean + cfg.frameTimeMedian.std,
        cfg.frameTimeP95.mean,
        cfg.frameTimeP99.mean,
      ];
    }
    boxes.push({ id: cfg.config, box: computeBox(values) });
  }

  const maxVal  = Math.max(...boxes.map(b => b.box.p99)) * 1.05;
  const ticks   = niceTicks(0, maxVal);
  const domainMax = ticks[ticks.length - 1] ?? maxVal;

  drawYGrid(ca, ticks, 0, domainMax, "Frame Time (ms)");

  ctx.strokeStyle = "#444";
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(ca.left, ca.top); ctx.lineTo(ca.left, ca.bottom);
  ctx.lineTo(ca.right, ca.bottom);
  ctx.stroke();

  const n       = boxes.length;
  const boxW    = (ca.width / n) * 0.45;
  const spacing = ca.width / n;

  for (let i = 0; i < n; i++) {
    const { id, box } = boxes[i]!;
    const cx  = ca.left + spacing * i + spacing / 2;
    const col = configColor(id);

    const yP10 = ca.scaleY(box.p10, 0, domainMax);
    const yP25 = ca.scaleY(box.p25, 0, domainMax);
    const yP50 = ca.scaleY(box.p50, 0, domainMax);
    const yP75 = ca.scaleY(box.p75, 0, domainMax);
    const yP90 = ca.scaleY(box.p90, 0, domainMax);
    const yP99 = ca.scaleY(box.p99, 0, domainMax);

    // Ус (P10 → P90)
    ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.moveTo(cx, yP10); ctx.lineTo(cx, yP90); ctx.stroke();
    // Засечки усов
    const capW = boxW * 0.3;
    ctx.beginPath();
    ctx.moveTo(cx - capW, yP10); ctx.lineTo(cx + capW, yP10);
    ctx.moveTo(cx - capW, yP90); ctx.lineTo(cx + capW, yP90);
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Ящик IQR (P25 → P75)
    ctx.fillStyle   = col; ctx.globalAlpha = 0.25;
    ctx.fillRect(cx - boxW / 2, yP75, boxW, yP25 - yP75);
    ctx.globalAlpha = 1.0;
    ctx.strokeStyle = col; ctx.lineWidth = 1.5;
    ctx.strokeRect(cx - boxW / 2, yP75, boxW, yP25 - yP75);

    // Линия медианы
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx - boxW / 2, yP50); ctx.lineTo(cx + boxW / 2, yP50); ctx.stroke();

    // Точка P99
    ctx.fillStyle   = col;
    ctx.beginPath(); ctx.arc(cx, yP99, 3, 0, 2 * Math.PI); ctx.fill();

    // Подпись конфига
    ctx.fillStyle = col; ctx.font = "bold 12px monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(id, cx, ca.bottom + 8);
  }

  ctx.fillStyle = "#666"; ctx.font = "10px monospace";
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText("Box: P25–P75  |  line: median  |  whiskers: P10/P90  |  dot: P99",
               ca.left, ca.top + 4);
}

// ---- График 3: Диаграмма рассеяния качество vs. память ------------------------------------------------------------------------------

function drawScatterChart(configs: AggregateStats[]): void {
  const canvas = document.getElementById("chart-scatter") as HTMLCanvasElement;
  clearCanvas(canvas);
  if (configs.length === 0) return;

  const ca = makeChartArea(canvas, 68, 36, 40, 52);
  const { ctx } = ca;

  // Ограничиваем PSNR: ∞ дБ отображается как 100
  const psnrOf = (c: AggregateStats) => Math.min(c.psnrDb.overview.mean, 100);

  const memMax  = Math.max(...configs.map(c => c.peakMemoryMB.mean)) * 1.15;
  const psnrMax = 105;
  const psnrMin = Math.max(0, Math.min(...configs.map(psnrOf)) - 5);

  const memTicks  = niceTicks(0, memMax);
  const psnrTicks = niceTicks(psnrMin, psnrMax);
  const domainMemMax  = memTicks[memTicks.length  - 1] ?? memMax;
  const domainPsnrMax = psnrTicks[psnrTicks.length - 1] ?? psnrMax;

  // Сетка
  ctx.strokeStyle = "#2a2a2a"; ctx.lineWidth = 1;
  for (const t of psnrTicks) {
    const y = ca.scaleY(t, psnrMin, domainPsnrMax);
    ctx.beginPath(); ctx.moveTo(ca.left, y); ctx.lineTo(ca.right, y); ctx.stroke();
  }
  for (const t of memTicks) {
    const x = ca.scaleX(t, 0, domainMemMax);
    ctx.beginPath(); ctx.moveTo(x, ca.top); ctx.lineTo(x, ca.bottom); ctx.stroke();
  }

  // Эталонная линия на 30 дБ (минимально допустимый порог качества)
  const y30 = ca.scaleY(30, psnrMin, domainPsnrMax);
  if (y30 >= ca.top && y30 <= ca.bottom) {
    ctx.strokeStyle = "#444"; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(ca.left, y30); ctx.lineTo(ca.right, y30); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#555"; ctx.font = "10px monospace";
    ctx.textAlign = "right"; ctx.textBaseline = "bottom";
    ctx.fillText("30 dB min", ca.right - 2, y30 - 3);
  }

  // Оси
  ctx.strokeStyle = "#444"; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ca.left, ca.top); ctx.lineTo(ca.left, ca.bottom);
  ctx.lineTo(ca.right, ca.bottom);
  ctx.stroke();

  // Деления + подписи оси Y
  ctx.fillStyle = "#666"; ctx.font = "10px monospace";
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (const t of psnrTicks) {
    ctx.fillText(t === 100 ? "∞" : String(t), ca.left - 5, ca.scaleY(t, psnrMin, domainPsnrMax));
  }

  // Деления + подписи оси X
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (const t of memTicks) {
    ctx.fillText(String(t), ca.scaleX(t, 0, domainMemMax), ca.bottom + 5);
  }

  // Подписи осей
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText("Peak GPU Memory (MB)", (ca.left + ca.right) / 2, ca.bottom + 22);
  ctx.save();
  ctx.translate(12, (ca.top + ca.bottom) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText("PSNR Overview (dB)", 0, 0);
  ctx.restore();

  // Эллипсы погрешности (стд. откл.), затем точки
  for (const cfg of configs) {
    const x = ca.scaleX(cfg.peakMemoryMB.mean, 0, domainMemMax);
    const y = ca.scaleY(psnrOf(cfg), psnrMin, domainPsnrMax);
    const rx = Math.max(2, (cfg.peakMemoryMB.std / domainMemMax) * ca.width);
    const ry = Math.max(2, (cfg.psnrDb.overview.std / (domainPsnrMax - psnrMin)) * ca.height);
    const col = configColor(cfg.config);

    // Эллипс стандартного отклонения
    ctx.fillStyle = col; ctx.globalAlpha = 0.12;
    ctx.beginPath(); ctx.ellipse(x, y, rx, ry, 0, 0, 2 * Math.PI); ctx.fill();
    ctx.globalAlpha = 1.0;

    // Точка
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = "#111"; ctx.lineWidth = 1.5;
    ctx.stroke();

    // Смещение подписи для предотвращения перекрытия
    const labelOffset = cfg.config === "A" ? [-12, -14] :
                        cfg.config === "B" ? [10, -14]  :
                        cfg.config === "C" ? [-12,  10] :
                        cfg.config === "D" ? [10,  10]  : [10, -14];
    ctx.fillStyle = col; ctx.font = "bold 11px monospace";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(cfg.config, x + labelOffset[0]!, y + labelOffset[1]!);
  }

  // Заголовок
  ctx.fillStyle = "#888"; ctx.font = "11px monospace";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText("Quality vs. Memory Trade-off  (ellipses = ±1σ)", (ca.left + ca.right) / 2, ca.top + 4);
}

// ---- Таблица 1 --------------------------------------------------------------------------------------------------------------------------------------

function ss(s: { mean: number; std: number }): string {
  return `${s.mean.toFixed(2)} ±${s.std.toFixed(2)}`;
}
function ps(v: number): string { return v >= 100 ? "∞" : v.toFixed(1); }

function renderTable1(configs: AggregateStats[]): void {
  const tbody = document.getElementById("t1-body")!;
  tbody.innerHTML = "";
  for (const cfg of configs) {
    const cls  = `cfg-${cfg.config.toLowerCase()}`;
    const conv = cfg.convergenceMs ? ss(cfg.convergenceMs) : "-";
    const tr   = document.createElement("tr");
    const flPct = cfg.fullyLoadedPct
      ? (cfg.fullyLoadedPct.mean * 100).toFixed(1) + "%"
      : "-";
    tr.innerHTML = `
      <td class="${cls}"><strong>${cfg.config}</strong></td>
      <td>${ss(cfg.peakMemoryMB)}</td>
      <td>${ss(cfg.frameTimeMedian)}</td>
      <td>${ss(cfg.frameTimeP95)}</td>
      <td>${ss(cfg.frameTimeP99)}</td>
      <td>${conv}</td>
      <td title="overview / closeup / midrange">
        ${ps(cfg.psnrDb.overview.mean)} / ${ps(cfg.psnrDb.closeup.mean)} / ${ps(cfg.psnrDb.midrange.mean)}
      </td>
      <td title="overview / closeup / midrange">
        ${cfg.ssim.overview.mean.toFixed(3)} / ${cfg.ssim.closeup.mean.toFixed(3)} / ${cfg.ssim.midrange.mean.toFixed(3)}
      </td>
      <td>${ss(cfg.uploadCount)}</td>
      <td>${ss(cfg.evictionCount)}</td>
      <td>${flPct}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ---- Таблица 2 --------------------------------------------------------------------------------------------------------------------------------------

function renderTable2(tests: WilcoxonResult[]): void {
  const tbody = document.getElementById("t2-body")!;
  tbody.innerHTML = "";
  for (const t of tests) {
    const sig = t.significant;
    const pStr = t.pValue < 0.001 ? "<0.001" : t.pValue.toFixed(3);
    const tr   = document.createElement("tr");
    tr.innerHTML = `
      <td>${t.metric}</td>
      <td>${t.configA} vs ${t.configB}</td>
      <td>${t.n}</td>
      <td>${t.W.toFixed(0)}</td>
      <td>${t.z.toFixed(2)}</td>
      <td class="${sig ? "sig-yes" : "sig-no"}">${pStr}</td>
      <td class="${sig ? "sig-yes" : "sig-no"}">${sig ? "✓" : "✗"}</td>
      <td>${t.direction}</td>
      <td>${t.effectSize.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ---- Таблица 3 --------------------------------------------------------------------------------------------------------------------------------------

function renderTable3(runs: RunData[]): void {
  const tbody = document.getElementById("t3-body")!;
  tbody.innerHTML = "";
  for (const r of runs) {
    const cls  = `cfg-${r.config.toLowerCase()}`;
    const s    = [...r.frameTimesMs].sort((a, b) => a - b);
    const med  = percentile(s, 0.5).toFixed(2);
    const p99  = percentile(s, 0.99).toFixed(2);
    const conv = r.convergenceMs !== null ? r.convergenceMs.toFixed(0) : "-";
    const tr   = document.createElement("tr");
    tr.innerHTML = `
      <td class="${cls}">${r.config}</td>
      <td>${r.runIndex}</td>
      <td>${r.peakGPUMemoryMB.toFixed(1)}</td>
      <td>${med}</td>
      <td>${p99}</td>
      <td>${conv}</td>
      <td>${ps(r.qualityPsnrDb.overview)}</td>
      <td>${ps(r.qualityPsnrDb.closeup)}</td>
      <td>${ps(r.qualityPsnrDb.midrange)}</td>
      <td>${r.qualitySsim.overview.toFixed(3)}</td>
      <td>${r.uploadCount}</td>
      <td>${r.evictionCount}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ---- Отображение окружения --------------------------------------------------------------------------------------------------------------

function showEnv(env: SummaryExport["environment"]): void {
  const box = document.getElementById("env-box")!;
  box.style.display = "block";

  // Строка сводки
  const summary = document.createElement("div");
  summary.textContent =
    `GPU: ${env.gpu.description}  |  Vendor: ${env.gpu.vendor}  |  ` +
    `Scene: "${env.scenePreset}" (${env.runsPerConfig} runs, ${env.effectiveRunsPerConfig} effective)  |  ` +
    `Date: ${env.timestamp.slice(0, 10)}`;
  box.innerHTML = "";
  box.appendChild(summary);

  // Сворачиваемый блок деталей
  const details = document.createElement("details");
  details.style.cssText = "margin-top:6px";
  const sumEl = document.createElement("summary");
  sumEl.textContent = "GPU limits & features";
  sumEl.style.cssText = "cursor:pointer;color:#888;font-size:11px;user-select:none";
  details.appendChild(sumEl);

  // Флаги возможностей
  if (env.gpu.features.length > 0) {
    const ftitle = document.createElement("div");
    ftitle.style.cssText = "margin-top:6px;font-size:11px;color:#666";
    ftitle.textContent   = "Features:";
    details.appendChild(ftitle);
    const flist = document.createElement("div");
    flist.style.cssText  = "font-size:11px;color:#4d4;margin-bottom:6px;line-height:1.8";
    flist.textContent    = env.gpu.features.join("  ·  ") || "(none)";
    details.appendChild(flist);
  }

  // Таблица лимитов
  const limitsEntries = Object.entries(env.gpu.limits);
  if (limitsEntries.length > 0) {
    const ltitle = document.createElement("div");
    ltitle.style.cssText = "font-size:11px;color:#666;margin-bottom:4px";
    ltitle.textContent   = "Limits:";
    details.appendChild(ltitle);

    const tbl = document.createElement("table");
    tbl.style.cssText = "font-size:11px;width:auto;margin:0";
    const thead = tbl.createTHead();
    const hrow  = thead.insertRow();
    ["Limit", "Value"].forEach(h => {
      const th = document.createElement("th");
      th.textContent = h;
      th.style.cssText = "background:#222;color:#666;padding:3px 12px 3px 0;text-align:left;font-weight:normal";
      hrow.appendChild(th);
    });
    const tbody = tbl.createTBody();
    for (const [key, val] of limitsEntries.sort(([a], [b]) => a.localeCompare(b))) {
      const row = tbody.insertRow();
      const tdKey = row.insertCell();
      tdKey.textContent   = key;
      tdKey.style.cssText = "padding:2px 16px 2px 0;color:#888;white-space:nowrap";
      const tdVal = row.insertCell();
      tdVal.textContent   = val.toLocaleString();
      tdVal.style.cssText = "padding:2px 0;color:#ccc;text-align:right;white-space:nowrap";
    }
    details.appendChild(tbl);
  }

  // Информация о браузере / экране
  const meta = document.createElement("div");
  meta.style.cssText = "margin-top:6px;font-size:11px;color:#555;line-height:1.6";
  meta.textContent   =
    `Browser: ${env.userAgent}  |  ` +
    `Screen: ${env.screen.width}×${env.screen.height} @${env.screen.devicePixelRatio}x  |  ` +
    `Canvas: ${env.canvas.width}×${env.canvas.height}`;
  details.appendChild(meta);

  box.appendChild(details);
}

// ---- График 4: Временная шкала вытеснений во время орбиты камеры --------------------------------------------------------

function drawEvictionChart(rawByConfig: Map<string, RunData[]> | null): void {
  const canvas = document.getElementById("chart-evictions") as HTMLCanvasElement;
  clearCanvas(canvas);

  const ca = makeChartArea(canvas, 68, 28, 20, 52);
  const { ctx } = ca;

  if (!rawByConfig) {
    ctx.fillStyle = "#444";
    ctx.font = "12px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Load raw JSON to enable this chart", (ca.left + ca.right) / 2, (ca.top + ca.bottom) / 2);
    return;
  }

  // Строим усреднённые покадровые массивы вытеснений для каждого конфига.
  type FrameData = { id: string; frames: number[] };
  const series: FrameData[] = [];
  for (const [configId, runs] of rawByConfig) {
    const validRuns = runs.filter(r => r.dynamicCamera?.perFrameEvictions);
    if (validRuns.length === 0) continue;
    const nFrames = validRuns[0]!.dynamicCamera.perFrameEvictions.length;
    const avg = new Array<number>(nFrames).fill(0);
    for (const run of validRuns) {
      for (let f = 0; f < nFrames; f++) {
        avg[f]! += (run.dynamicCamera.perFrameEvictions[f] ?? 0) / validRuns.length;
      }
    }
    series.push({ id: configId, frames: avg });
  }

  if (series.length === 0) {
    ctx.fillStyle = "#444";
    ctx.font = "12px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("No dynamic camera data in loaded raw JSON", (ca.left + ca.right) / 2, (ca.top + ca.bottom) / 2);
    return;
  }

  const nFrames = series[0]!.frames.length;
  const maxEvict = Math.max(1, ...series.flatMap(s => s.frames));
  const ticks = niceTicks(0, maxEvict * 1.1);
  const domainMax = ticks[ticks.length - 1] ?? maxEvict * 1.1;

  drawYGrid(ca, ticks, 0, domainMax, "Evictions / frame (avg)");

  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ca.left, ca.top); ctx.lineTo(ca.left, ca.bottom);
  ctx.lineTo(ca.right, ca.bottom);
  ctx.stroke();

  // Подписи оси X
  ctx.fillStyle = "#666"; ctx.font = "10px monospace";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let f = 0; f <= nFrames; f += 60) {
    const x = ca.left + (f / nFrames) * ca.width;
    ctx.fillText(String(f), x, ca.bottom + 5);
  }
  ctx.fillText("Frame (orbit)", (ca.left + ca.right) / 2, ca.bottom + 22);

  // Линии для каждого конфига
  for (const { id, frames } of series) {
    const col = configColor(id);
    ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (let f = 0; f < frames.length; f++) {
      const x = ca.left + (f / (nFrames - 1)) * ca.width;
      const y = ca.scaleY(frames[f] ?? 0, 0, domainMax);
      if (f === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Подпись легенды у правого края
    const lastX = ca.right + 4;
    const lastY = ca.scaleY(frames[frames.length - 1] ?? 0, 0, domainMax);
    ctx.fillStyle = col; ctx.font = "bold 11px monospace";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(id, lastX, lastY);
  }

  ctx.fillStyle = "#666"; ctx.font = "10px monospace";
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText("Avg over effective runs  |  non-zero = eviction active",
               ca.left, ca.top + 4);
}

// ---- Таблица 2b: сводка орбиты динамической камеры ------------------------------------------------------------------------

function renderDynamicCameraTable(configs: AggregateStats[]): void {
  const tbody = document.getElementById("t2b-body")!;
  tbody.innerHTML = "";
  let hasData = false;
  for (const cfg of configs) {
    if (!cfg.dynamicCamera) continue;
    hasData = true;
    const dc = cfg.dynamicCamera;
    const cls = `cfg-${cfg.config.toLowerCase()}`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="${cls}"><strong>${cfg.config}</strong></td>
      <td>${ss(dc.totalEvictions)}</td>
      <td>${ss(dc.totalUploads)}</td>
      <td>${ss(dc.peakMemoryMB)}</td>
      <td>${ss(dc.frameTimeMedianMs)}</td>
      <td>${ss(dc.frameTimeP99Ms)}</td>
      <td>${dc.finalPsnrDb.mean >= 100 ? "∞" : dc.finalPsnrDb.mean.toFixed(1)} ±${dc.finalPsnrDb.std.toFixed(1)}</td>
      <td>${dc.finalSsim.mean.toFixed(3)} ±${dc.finalSsim.std.toFixed(3)}</td>
    `;
    tbody.appendChild(tr);
  }
  if (!hasData) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8" class="placeholder">No dynamic camera data in loaded JSON</td>`;
    tbody.appendChild(tr);
  }
}

// ---- График сравнения разрешений ----------------------------------------------------------------------------------------------

function drawResolutionChart(
  configs1080: AggregateStats[],
  configs720:  AggregateStats[],
): void {
  const canvas = document.getElementById("chart-resolution") as HTMLCanvasElement;
  const ca = makeChartArea(canvas, 68, 28, 28, 52);
  const { ctx } = ca;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (configs720.length === 0) {
    ctx.fillStyle = "#444";
    ctx.font = "12px monospace";
    ctx.fillText("Load 720p summary JSON to enable this chart", ca.left + 20, (ca.top + ca.bottom) / 2);
    return;
  }

  const cfgIds = configs1080.map(c => c.config);
  const n = cfgIds.length;
  const barW = 20;
  const groupW = (ca.width) / n;
  const maxMem = Math.max(
    ...configs1080.map(c => c.peakMemoryMB.mean + c.peakMemoryMB.std),
    ...configs720.map(c => c.peakMemoryMB.mean + c.peakMemoryMB.std),
  ) * 1.15;

  ctx.font = "10px monospace";

  for (let i = 0; i < n; i++) {
    const cx = ca.left + groupW * (i + 0.5);
    const c1080 = configs1080[i]!;
    const c720  = configs720.find(c => c.config === cfgIds[i]) ?? null;

    // Столбец 1080p (слева)
    {
      const x = cx - barW - 2;
      const h = (c1080.peakMemoryMB.mean / maxMem) * ca.height;
      const y = ca.bottom - h;
      ctx.fillStyle = configColor(c1080.config);
      ctx.fillRect(x, y, barW, h);
      // Планка погрешности стандартного отклонения
      const errH = (c1080.peakMemoryMB.std / maxMem) * ca.height;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x + barW / 2, y - errH);
      ctx.lineTo(x + barW / 2, y + errH);
      ctx.stroke();
    }

    // Столбец 720p (справа, светлее)
    if (c720) {
      const x = cx + 2;
      const h = (c720.peakMemoryMB.mean / maxMem) * ca.height;
      const y = ca.bottom - h;
      ctx.fillStyle = configColor(c720.config);
      ctx.globalAlpha = 0.5;
      ctx.fillRect(x, y, barW, h);
      ctx.globalAlpha = 1.0;
      ctx.strokeStyle = "#888";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, barW, h);
    }

    // Подпись
    ctx.fillStyle = "#aaa";
    ctx.textAlign = "center";
    ctx.fillText(cfgIds[i]!, cx, ca.bottom + 16);
  }

  // Ось Y
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(ca.left, ca.top);
  ctx.lineTo(ca.left, ca.bottom);
  ctx.lineTo(ca.right, ca.bottom);
  ctx.stroke();

  for (let v = 0; v <= maxMem; v += Math.ceil(maxMem / 5 / 50) * 50) {
    const y = ca.scaleY(v, 0, maxMem);
    ctx.strokeStyle = "#222";
    ctx.beginPath(); ctx.moveTo(ca.left, y); ctx.lineTo(ca.right, y); ctx.stroke();
    ctx.fillStyle = "#666"; ctx.textAlign = "right";
    ctx.fillText(String(Math.round(v)), ca.left - 6, y + 4);
  }

  // Легенда
  ctx.textAlign = "left";
  ctx.fillStyle = "#ccc"; ctx.fillRect(ca.right - 110, ca.top + 6, 10, 10);
  ctx.fillStyle = "#888"; ctx.fillText("1080p", ca.right - 97, ca.top + 15);
  ctx.fillStyle = "#888"; ctx.fillRect(ca.right - 110, ca.top + 22, 10, 10);
  ctx.strokeStyle = "#888"; ctx.strokeRect(ca.right - 110, ca.top + 22, 10, 10);
  ctx.fillStyle = "#888"; ctx.fillText("720p", ca.right - 97, ca.top + 31);
}

// ---- Пайплайн рендеринга ----------------------------------------------------------------------------------------------------------------------

function renderAll(): void {
  if (!summary) return;

  showEnv(summary.environment);
  renderTable1(summary.configs);
  renderTable2(summary.pairwiseTests ?? []);
  drawMemoryChart(summary.configs);
  drawScatterChart(summary.configs);

  // Строим rawByConfig, если доступны сырые данные
  let rawByConfig: Map<string, RunData[]> | null = null;
  if (raw) {
    rawByConfig = new Map();
    for (const r of raw.runs) {
      let arr = rawByConfig.get(r.config);
      if (!arr) { arr = []; rawByConfig.set(r.config, arr); }
      arr.push(r);
    }
    renderTable3(raw.runs);
  }

  drawFrameTimeChart(summary.configs, rawByConfig);
  drawEvictionChart(rawByConfig);
  renderDynamicCameraTable(summary.configs);
  drawResolutionChart(summary.configs, summary720?.configs ?? []);
}

// ---- Загрузка файлов ----------------------------------------------------------------------------------------------------------------------------

function parseJson<T>(text: string): T | null {
  try { return JSON.parse(text) as T; } catch { return null; }
}

const statusEl = document.getElementById("load-status")!;

function setStatus(msg: string): void { statusEl.textContent = msg; }

document.getElementById("summary-file")!.addEventListener("change", async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  setStatus(`Loading ${file.name}…`);
  const text = await file.text();
  const parsed = parseJson<SummaryExport>(text);
  if (!parsed || !Array.isArray(parsed.configs)) {
    setStatus("Error: not a valid summary JSON (expected { configs, environment, pairwiseTests })");
    return;
  }
  summary = parsed;
  setStatus(`Loaded summary: ${parsed.configs.length} configs, scene "${parsed.environment.scenePreset}"`);
  renderAll();
});

document.getElementById("raw-file")!.addEventListener("change", async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  setStatus(`Loading ${file.name}…`);
  const text = await file.text();
  const parsed = parseJson<RawExport>(text);
  if (!parsed || !Array.isArray(parsed.runs)) {
    setStatus("Error: not a valid raw JSON (expected { runs, environment })");
    return;
  }
  raw = parsed;
  setStatus(`Loaded raw: ${parsed.runs.length} runs`);
  renderAll();
});

document.getElementById("summary-file-720")!.addEventListener("change", async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  setStatus(`Loading ${file.name}…`);
  const text = await file.text();
  const parsed = parseJson<SummaryExport>(text);
  if (!parsed || !Array.isArray(parsed.configs)) {
    setStatus("Error: not a valid 720p summary JSON");
    return;
  }
  summary720 = parsed;
  setStatus(`Loaded 720p summary: ${parsed.configs.length} configs, canvas ${parsed.environment.canvas.width}×${parsed.environment.canvas.height}`);
  if (summary) renderAll();
});

// ---- Экспорт PNG (экспортируется глобально для обработчиков onclick в HTML) ----------------------------------

(window as Record<string, unknown>)["saveChart"] = function saveChart(
  canvasId: string,
  filename: string,
): void {
  const canvas = document.getElementById(canvasId) as HTMLCanvasElement | null;
  if (!canvas) return;
  const a = document.createElement("a");
  a.href     = canvas.toDataURL("image/png");
  a.download = filename;
  a.click();
};
