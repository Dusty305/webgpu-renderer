/**
 * Критерий знаковых рангов Вилкоксона
 *
 * Непараметрическое попарное сравнение для парных наблюдений.
 * Использует нормальное приближение (подходит для N ≥ 20).
 *
 * Источник: Wilcoxon (1945); нормальное приближение из Zar (1999)
 * «Biostatistical Analysis», §8.4. Функция CDF по Abramowitz & Stegun 26.2.17.
 */

// ── Типы ─────────────────────────────────────────────────────────────────────

/** Результат одного попарного сравнения по критерию знаковых рангов Вилкоксона. */
export interface WilcoxonResult {
  /** Читаемая метка тестируемой метрики. */
  metric:      string;
  /** Идентификатор первой конфигурации («эталон»). */
  configA:     string;
  /** Идентификатор второй конфигурации («претендент»). */
  configB:     string;
  /** Статистика W Вилкоксона (min из W+ и W-). */
  W:           number;
  /** Z-оценка из нормального приближения. */
  z:           number;
  /** Двусторонний p-уровень значимости. */
  pValue:      number;
  /** true, когда pValue < 0.05. */
  significant: boolean;
  /** Какой конфиг имеет меньшие значения или «нет разницы». */
  direction:   string;
  /** Размер эффекта r = |z| / sqrt(N). */
  effectSize:  number;
  /** Эффективный N после удаления нулевых разностей. */
  n:           number;
}

// ── Нормальная функция распределения ─────────────────────────────────────────────────────────────────

/**
 * Стандартная нормальная CDF P(Z ≤ x) по Abramowitz & Stegun 26.2.17.
 * Максимальная абсолютная погрешность: 7.5×10⁻⁸.
 */
function normalCdf(x: number): number {
  const P  =  0.2316419;
  const a1 =  0.319381530;
  const a2 = -0.356563782;
  const a3 =  1.781477937;
  const a4 = -1.821255978;
  const a5 =  1.330274429;

  const t     = 1 / (1 + P * Math.abs(x));
  const poly  = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
  const phi   = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
  const upper = phi * poly;   // P(Z > |x|)

  return x >= 0 ? 1 - upper : upper;
}

// ── Основной алгоритм ────────────────────────────────────────────────────────────

/**
 * Критерий знаковых рангов Вилкоксона (двусторонний) для парных выборок.
 *
 * @param metric  - Читаемая метка метрики (например, "frameTimeP99")
 * @param cfgA    - Метка первого конфига (эталон)
 * @param cfgB    - Метка второго конфига (претендент)
 * @param xA      - Наблюдения по прогонам для конфига A (длина N)
 * @param xB      - Наблюдения по прогонам для конфига B (длина N, парные с xA)
 *
 * Гипотеза: H₀: xA и xB происходят из одного распределения.
 *
 * Использует нормальное приближение с поправкой на непрерывность при N ≥ 10.
 * Возвращает вырожденный результат «нет теста» при N < 10 (недостаточно данных).
 */
export function wilcoxon(
  metric: string,
  cfgA:   string,
  cfgB:   string,
  xA:     number[],
  xB:     number[],
): WilcoxonResult {
  if (xA.length !== xB.length || xA.length === 0) {
    return noTest(metric, cfgA, cfgB, "array length mismatch or empty");
  }

  // Шаг 1: вычислить знаковые разности d[i] = xA[i] - xB[i]
  const diffs: { d: number; absD: number }[] = [];
  for (let i = 0; i < xA.length; i++) {
    const d = (xA[i] ?? 0) - (xB[i] ?? 0);
    if (d !== 0) diffs.push({ d, absD: Math.abs(d) });
  }

  const N = diffs.length;
  if (N < 10) {
    return noTest(metric, cfgA, cfgB, `only ${N} non-tied pairs`);
  }

  // Шаг 2: ранжировать |d[i]| со средним рангом для связок
  diffs.sort((a, b) => a.absD - b.absD);
  const ranks = new Array<number>(N).fill(0);
  let i = 0;
  while (i < N) {
    let j = i;
    while (j < N - 1 && (diffs[j + 1]?.absD ?? 0) === (diffs[j]?.absD ?? 0)) j++;
    const avgRank = (i + j) / 2 + 1;   // с единицы, усредняется по группе связок
    for (let k = i; k <= j; k++) ranks[k] = avgRank;
    i = j + 1;
  }

  // Шаг 3: суммы W+ и W-
  let wPlus = 0, wMinus = 0;
  for (let k = 0; k < N; k++) {
    if ((diffs[k]?.d ?? 0) > 0) wPlus  += ranks[k] ?? 0;
    else                         wMinus += ranks[k] ?? 0;
  }
  const W = Math.min(wPlus, wMinus);

  // Шаг 4: нормальное приближение (с поправкой на непрерывность 0.5)
  const mu     = N * (N + 1) / 4;
  const sigma  = Math.sqrt(N * (N + 1) * (2 * N + 1) / 24);
  // W ≤ mu всегда (минимальная статистика), применяем верхнюю поправку непрерывности
  const z      = sigma > 0 ? (W + 0.5 - mu) / sigma : 0;
  const pValue = 2 * normalCdf(z);          // двусторонний; z ≤ 0, поэтому CDF(z) < 0.5

  const significant = pValue < 0.05;
  const effectSize  = sigma > 0 ? Math.abs(z) / Math.sqrt(N) : 0;

  // Шаг 5: направление (у какого конфига меньше значений?)
  // wPlus = Σrank где xA > xB  →  высокий wPlus = A больше = B меньше
  // wMinus = Σrank где xA < xB →  высокий wMinus = A меньше = A меньше
  let direction: string;
  if (!significant) {
    direction = "no difference";
  } else if (wPlus > wMinus) {
    direction = `${cfgB} lower`;
  } else {
    direction = `${cfgA} lower`;
  }

  return { metric, configA: cfgA, configB: cfgB, W, z, pValue, significant, direction, effectSize, n: N };
}

// ── Вспомогательные функции ───────────────────────────────────────────────────────

function noTest(metric: string, cfgA: string, cfgB: string, reason: string): WilcoxonResult {
  return {
    metric, configA: cfgA, configB: cfgB,
    W: 0, z: 0, pValue: 1,
    significant: false,
    direction: `no test (${reason})`,
    effectSize: 0, n: 0,
  };
}

/**
 * Извлекает времена P99 кадра по прогонам из сырых данных, сгруппированных по конфигу.
 * runsByConfig: Map<configId, RunData[]>
 */
export function extractP99s(runs: { frameTimesMs: number[] }[]): number[] {
  return runs.map(r => {
    const s = [...r.frameTimesMs].sort((a, b) => a - b);
    const idx = Math.min(Math.ceil(s.length * 0.99) - 1, s.length - 1);
    return s[idx] ?? 0;
  });
}

/**
 * Запускает все 11 попарных тестов Вилкоксона, указанных в CLAUDE.md §5.2.
 *
 * @param runsByConfig  Map<configId, RunData[]>
 */
export function runPairwiseTests(
  runsByConfig: Map<string, {
    frameTimesMs: number[];
    peakGPUMemoryMB: number;
    convergenceMs: number | null;
    qualityPsnrDb: { overview: number; closeup: number; midrange: number };
  }[]>,
): WilcoxonResult[] {
  const get = (id: string) => runsByConfig.get(id) ?? [];

  const A = get("A"), C = get("C"), E = get("E");

  // P99 времён кадра по прогонам
  const p99 = (runs: typeof A) => extractP99s(runs);
  // Пиковая память по прогонам
  const mem = (runs: typeof A) => runs.map(r => r.peakGPUMemoryMB);
  // Сходимость по прогонам (только ненулевые - должны быть парными с соответствующим конфигом)
  const conv = (a: typeof A, b: typeof A): [number[], number[]] => {
    const pairs: [number, number][] = [];
    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) {
      const va = a[i]?.convergenceMs, vb = b[i]?.convergenceMs;
      if (va !== null && va !== undefined && vb !== null && vb !== undefined) {
        pairs.push([va, vb]);
      }
    }
    return [pairs.map(p => p[0]), pairs.map(p => p[1])];
  };
  const psnr = (runs: typeof A, pose: "overview" | "closeup" | "midrange") =>
    runs.map(r => r.qualityPsnrDb[pose]);

  const [convC, convE] = conv(C, E);

  return [
    wilcoxon("frameTimeP99",        "A", "C", p99(A), p99(C)),
    wilcoxon("frameTimeP99",        "A", "E", p99(A), p99(E)),
    wilcoxon("peakMemoryMB",        "A", "C", mem(A), mem(C)),
    wilcoxon("peakMemoryMB",        "A", "E", mem(A), mem(E)),
    wilcoxon("convergenceMs",       "C", "E", convC,  convE),
    wilcoxon("psnrDb_overview",     "A", "C", psnr(A, "overview"),  psnr(C, "overview")),
    wilcoxon("psnrDb_closeup",      "A", "C", psnr(A, "closeup"),   psnr(C, "closeup")),
    wilcoxon("psnrDb_midrange",     "A", "C", psnr(A, "midrange"),  psnr(C, "midrange")),
    wilcoxon("psnrDb_overview",     "A", "E", psnr(A, "overview"),  psnr(E, "overview")),
    wilcoxon("psnrDb_closeup",      "A", "E", psnr(A, "closeup"),   psnr(E, "closeup")),
    wilcoxon("psnrDb_midrange",     "A", "E", psnr(A, "midrange"),  psnr(E, "midrange")),
  ];
}

// ── Поправка Холма–Бонферрони ────────────────────────────────────────────────

/** Одна запись в скорректированном выводе Холма–Бонферрони. */
export interface HolmResult {
  testId:          string;
  pValue:          number;
  holmAdjustedAlpha: number;
  /** Значимо после пошаговой поправки Холма–Бонферрони. */
  significant:     boolean;
}

/**
 * Пошаговая поправка Холма–Бонферрони для множественных сравнений.
 * Контролирует семейный уровень ошибки на уровне `alpha` по всем `m` тестам.
 *
 * Алгоритм:
 *   1. Отсортировать тесты по возрастанию p-значения.
 *   2. Для ранга k (с единицы): порог = alpha / (m − k + 1).
 *   3. Пошаговое снижение: как только тест не проходит, все остальные тоже.
 *
 * Мощнее простого Бонферрони, т.к. пороги растут при переборе тестов.
 *
 * @param pValues - Массив {testId, pValue} из попарных тестов.
 * @param alpha   - Семейный уровень значимости (по умолчанию 0.05).
 */
export function holmBonferroniCorrection(
  pValues: Array<{ testId: string; pValue: number }>,
  alpha: number = 0.05,
): HolmResult[] {
  const m = pValues.length;
  if (m === 0) return [];

  const sorted = [...pValues].sort((a, b) => a.pValue - b.pValue);

  const result: HolmResult[] = sorted.map((item, index) => {
    const k = index + 1;
    const holmAdjustedAlpha = alpha / (m - k + 1);
    return { testId: item.testId, pValue: item.pValue, holmAdjustedAlpha, significant: item.pValue < holmAdjustedAlpha };
  });

  // Пошаговое снижение: как только любой тест не проходит, все последующие тоже незначимы.
  let foundNonSig = false;
  for (const item of result) {
    if (!item.significant) foundNonSig = true;
    if (foundNonSig) item.significant = false;
  }

  return result;
}
