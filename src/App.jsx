import { useState, useMemo, useCallback, useEffect } from "react";
import * as XLSX from "xlsx";
import Papa from "papaparse";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer,
  LineChart, Line, CartesianGrid, PieChart, Pie, ReferenceLine
} from "recharts";
import {
  RefreshCw, CheckCircle, XCircle, Layers, TrendingUp, Info,
  FileSpreadsheet, Filter, Calendar, Tag, BarChart3, ChevronDown, ChevronUp,
  Plus, Trash2, Zap, Brain, AlertCircle
} from "lucide-react";

// ─────────────────────────────────────────────
//  DESIGN TOKENS
// ─────────────────────────────────────────────
const T = {
  bg: "#0D1117", bgCard: "#161B22", bgSurface: "#1C2330",
  bgInput: "#0D1117",
  accent: "#58A6FF", border: "#30363D",
  text: "#E6EDF3", textMuted: "#8B949E", textDim: "#484F58",
  green: "#3FB950", red: "#F85149", orange: "#D29922",
  yellow: "#E3B341", blue: "#58A6FF", purple: "#BC8CFF",
  pink: "#FF7EB3", cyan: "#39D0D8",
  font: "'JetBrains Mono',monospace",
  fontSans: "'DM Sans',sans-serif",
  r: "6px", rLg: "12px"
};
const crdS = { background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: T.rLg, padding: "18px" };
const lbS = { fontSize: "10px", fontFamily: T.font, color: T.textMuted, textTransform: "uppercase", letterSpacing: ".08em", fontWeight: 500, display: "flex", alignItems: "center" };

// ─────────────────────────────────────────────
//  STAT HELPERS
// ─────────────────────────────────────────────
function mean(arr) { return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0; }
function std(arr) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}
function median(arr) {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
function calcADI(series) {
  const pos = series.map((v, i) => v > 0 ? i : null).filter(i => i !== null);
  if (pos.length < 2) return series.length;
  const gaps = []; for (let i = 1; i < pos.length; i++) gaps.push(pos[i] - pos[i - 1]);
  return mean(gaps);
}
function calcCV2(series) {
  const nz = series.filter(v => v > 0);
  if (nz.length < 2) return Infinity;
  const m = mean(nz), s = std(nz);
  return m === 0 ? Infinity : (s / m) ** 2;
}
function classify(adi, cv2) {
  if (adi < 1.32 && cv2 < 0.49) return "Smooth";
  if (adi < 1.32) return "Erratic";
  if (cv2 < 0.49) return "Intermittent";
  return "Lumpy";
}
function backtestMAPE(series, w = 4) {
  if (series.length < 8) return null;
  const trainEnd = Math.max(w, Math.floor(series.length * 0.7));
  const apes = [];
  for (let t = trainEnd; t < series.length; t++) {
    const pred = mean(series.slice(Math.max(0, t - w), t));
    const actual = series[t];
    if (actual !== 0) apes.push(Math.abs((actual - pred) / actual));
  }
  return apes.length ? mean(apes) * 100 : null;
}
function forecastabilityScore(medCV, pctSmooth, mape) {
  const cvScore = Math.max(0, Math.min(100, 100 - medCV * 100));
  const smoothScore = pctSmooth * 100;
  const mapeScore = mape != null ? Math.max(0, Math.min(100, 100 - mape)) : 50;
  return Math.round(cvScore * 0.35 + smoothScore * 0.40 + mapeScore * 0.25);
}
function qualityLabel(score) { return score >= 65 ? "GOOD" : score >= 40 ? "FAIR" : "POOR"; }
function qualityColor(score) { return score >= 65 ? T.green : score >= 40 ? T.orange : T.red; }
function allSubsets(cols) {
  const result = [];
  for (let mask = 1; mask < (1 << cols.length); mask++)
    result.push(cols.filter((_, i) => mask & (1 << i)));
  return result.sort((a, b) => a.length - b.length || a.join().localeCompare(b.join()));
}
function parseCurrency(v) {
  if (typeof v === "number") return v;
  if (typeof v === "string") { const n = parseFloat(v.replace(/[$,\s]/g, "")); return isNaN(n) ? null : n; }
  return null;
}
function parseDate(v) {
  if (!v) return null;
  if (v instanceof Date) return isNaN(v) ? null : v.toISOString();
  const d = new Date(v); if (!isNaN(d)) return d.toISOString();
  return null;
}

// ─────────────────────────────────────────────
//  GBM ENGINE (decision-stump gradient boosting)
// ─────────────────────────────────────────────
const ML = (() => {
  function bestSplit(X, residuals, fi) {
    const n = X.length; if (n < 4) return null;
    const vals = X.map((r, i) => ({ v: r[fi], r: residuals[i] })).sort((a, b) => a.v - b.v);
    let bestGain = -Infinity, bestThresh = 0, bL = 0, bR = 0;
    let lSum = 0, lN = 0, total = residuals.reduce((a, b) => a + b, 0);
    for (let i = 0; i < n - 1; i++) {
      lSum += vals[i].r; lN++;
      const rSum = total - lSum, rN = n - lN;
      if (rN < 1) continue;
      const gain = lSum * lSum / lN + rSum * rSum / rN;
      if (gain > bestGain) { bestGain = gain; bestThresh = (vals[i].v + vals[i + 1].v) / 2; bL = lSum / lN; bR = rSum / rN; }
    }
    return { fi, threshold: bestThresh, leftVal: bL, rightVal: bR, gain: bestGain };
  }
  function fitGBM(X, y, nTrees = 40, lr = 0.1) {
    const n = X.length, nF = X[0].length, base = mean(y);
    let preds = new Array(n).fill(base); const trees = [];
    for (let t = 0; t < nTrees; t++) {
      const res = y.map((yi, i) => yi - preds[i]);
      let best = null;
      for (let f = 0; f < nF; f++) { const s = bestSplit(X, res, f); if (s && (!best || s.gain > best.gain)) best = s; }
      if (!best) break;
      trees.push({ ...best, lr });
      for (let i = 0; i < n; i++) preds[i] += lr * (X[i][best.fi] <= best.threshold ? best.leftVal : best.rightVal);
    }
    return { base, trees };
  }
  function predictGBM(model, X) {
    return X.map(row => { let p = model.base; model.trees.forEach(t => { p += t.lr * (row[t.fi] <= t.threshold ? t.leftVal : t.rightVal); }); return p; });
  }
  function rmse(actual, pred) { let s = 0; for (let i = 0; i < actual.length; i++) s += (actual[i] - pred[i]) ** 2; return Math.sqrt(s / actual.length); }
  function walkForward(X, y, featIdx, nFolds = 4, minTrain = 12) {
    const n = X.length, step = Math.max(1, Math.floor((n - minTrain) / nFolds)), testSize = step;
    const rmses = [];
    for (let fold = 0; fold < nFolds; fold++) {
      const trainEnd = minTrain + fold * step, testEnd = Math.min(trainEnd + testSize, n);
      if (trainEnd >= n || testEnd <= trainEnd) continue;
      const Xtr = X.slice(0, trainEnd).map(r => featIdx.map(f => r[f]));
      const ytr = y.slice(0, trainEnd);
      const Xte = X.slice(trainEnd, testEnd).map(r => featIdx.map(f => r[f]));
      const yte = y.slice(trainEnd, testEnd);
      if (Xtr.length < 10 || Xte.length < 3) continue;
      rmses.push(rmse(yte, predictGBM(fitGBM(Xtr, ytr, 30, 0.1), Xte)));
    }
    return rmses.length ? mean(rmses) : Infinity;
  }
  return { fitGBM, predictGBM, rmse, walkForward };
})();

// Compute signal lift for one aligned demand+signal series pair
function computeSignalLift(demandVals, signalVals) {
  const n = demandVals.length;
  if (n < 16) return null;
  const maxL = 4;
  const Xrows = [], y = [];
  for (let i = maxL; i < n; i++) {
    const row = [];
    for (let l = 1; l <= maxL; l++) row.push(demandVals[i - l]); // target lags
    for (let l = 1; l <= maxL; l++) row.push(signalVals[i - l]);  // signal lags
    Xrows.push(row);
    y.push(demandVals[i]);
  }
  const targetIdx = [0, 1, 2, 3];
  const allIdx = [0, 1, 2, 3, 4, 5, 6, 7];
  const baseRMSE = ML.walkForward(Xrows, y, targetIdx, 4, 12);
  const withSignalRMSE = ML.walkForward(Xrows, y, allIdx, 4, 12);
  const lift = baseRMSE === Infinity ? null : (baseRMSE - withSignalRMSE) / baseRMSE * 100;
  return { baseRMSE, withSignalRMSE, lift };
}

// ─────────────────────────────────────────────
//  ANALYSIS ENGINE
// ─────────────────────────────────────────────
const CLS_COLORS = { Smooth: T.green, Erratic: T.orange, Intermittent: T.yellow, Lumpy: T.red };

function analyzeLevel(rows, targetCol, dateCol, grainCols, filterCol, filterVal) {
  let filtered = rows;
  if (filterCol && filterVal) filtered = rows.filter(r => String(r[filterCol]) === String(filterVal));

  const levels = [{ key: "Total", cols: [] }];
  if (grainCols.length > 0) allSubsets(grainCols).forEach(s => levels.push({ key: s.join(" + "), cols: s }));

  return levels.map(level => {
    const seriesMap = {};
    filtered.forEach(row => {
      const gk = level.cols.length ? level.cols.map(c => String(row[c] ?? "")).join(" | ") : "__total__";
      const date = parseDate(row[dateCol]), qty = parseCurrency(row[targetCol]);
      if (date == null || qty == null) return;
      if (!seriesMap[gk]) seriesMap[gk] = {};
      seriesMap[gk][date] = (seriesMap[gk][date] || 0) + qty;
    });

    const seriesArrays = Object.entries(seriesMap).map(([key, dm]) => {
      const sorted = Object.entries(dm).sort(([a], [b]) => a < b ? -1 : 1);
      return { key, values: sorted.map(([, v]) => v), dates: sorted.map(([d]) => d) };
    });

    const seriesMetrics = seriesArrays.map(s => {
      const adi = calcADI(s.values), cv2 = calcCV2(s.values);
      const cv = cv2 === Infinity ? 9999 : Math.sqrt(cv2);
      return { key: s.key, n: s.values.length, adi, cv2, cv, cls: classify(adi, cv2), mape: backtestMAPE(s.values), values: s.values, dates: s.dates };
    });

    const valid = seriesMetrics.filter(s => s.n >= 6);
    if (!valid.length) return { level: level.key, cols: level.cols, numSeries: seriesArrays.length, skipped: true, score: 0, adjustedScore: 0, seriesMetrics: [] };

    const cvs = valid.map(s => s.cv).filter(v => v < 9999);
    const medCV = cvs.length ? median(cvs) : 1;
    const pctSmooth = valid.filter(s => s.cls === "Smooth").length / valid.length;
    const mapes = valid.map(s => s.mape).filter(v => v != null);
    const avgMAPE = mapes.length ? mean(mapes) : null;
    const score = forecastabilityScore(medCV, pctSmooth, avgMAPE);
    const clsCounts = { Smooth: 0, Erratic: 0, Intermittent: 0, Lumpy: 0 };
    valid.forEach(s => { clsCounts[s.cls] = (clsCounts[s.cls] || 0) + 1; });

    return {
      level: level.key, cols: level.cols,
      numSeries: seriesArrays.length, avgN: Math.round(mean(valid.map(s => s.n))),
      medianCV: +medCV.toFixed(3), pctSmooth: +(pctSmooth * 100).toFixed(1),
      avgMAPE: avgMAPE != null ? +avgMAPE.toFixed(1) : null,
      score, adjustedScore: score, quality: qualityLabel(score), clsCounts,
      seriesMetrics: valid.sort((a, b) => b.n - a.n),
      signalLifts: null, bestLift: null, bestLiftLabel: null,
      skipped: false
    };
  }).sort((a, b) => b.score - a.score);
}

// Run signal lift for all levels × all signals (called on-demand)
function computeAllSignalLifts(results, signals) {
  return results.map(levelResult => {
    if (levelResult.skipped || !signals.length) return levelResult;

    const liftsBySignal = signals.map(sig => {
      // Build signal series map at this level's grain
      const sigMap = {};
      sig.rows.forEach(row => {
        const gk = levelResult.cols.length
          ? levelResult.cols.map(dc => {
              const sc = sig.grainMapping[dc];
              return sc ? String(row[sc] ?? "") : "";
            }).join(" | ")
          : "__total__";
        const date = parseDate(row[sig.dateCol]);
        const val = parseCurrency(row[sig.valueCol]);
        if (date == null || val == null) return;
        if (!sigMap[gk]) sigMap[gk] = {};
        sigMap[gk][date] = (sigMap[gk][date] || 0) + val;
      });

      // For each demand series at this level, find matching signal series and compute lift
      const lifts = [];
      levelResult.seriesMetrics.forEach(demSeries => {
        const sigSeries = sigMap[demSeries.key];
        if (!sigSeries) return;
        // Align on dates
        const demDates = demSeries.dates;
        const aligned = demDates.map((d, i) => ({
          demVal: demSeries.values[i],
          sigVal: sigSeries[d] ?? null
        })).filter(p => p.sigVal !== null);
        if (aligned.length < 16) return;
        const lift = computeSignalLift(aligned.map(p => p.demVal), aligned.map(p => p.sigVal));
        if (lift && lift.lift !== null) lifts.push(lift.lift);
      });

      const avgLift = lifts.length ? mean(lifts) : null;
      return { label: sig.label, avgLift };
    });

    const validLifts = liftsBySignal.filter(l => l.avgLift !== null);
    const bestLiftEntry = validLifts.length ? validLifts.reduce((b, c) => c.avgLift > b.avgLift ? c : b) : null;
    const bestLift = bestLiftEntry ? +bestLiftEntry.avgLift.toFixed(1) : null;
    const bestLiftLabel = bestLiftEntry ? bestLiftEntry.label : null;

    // Adjusted score: blend demand score (80%) with signal lift bonus (20%)
    const liftScore = bestLift != null ? Math.min(100, Math.max(0, bestLift * 4)) : null;
    const adjustedScore = liftScore != null
      ? Math.round(levelResult.score * 0.80 + liftScore * 0.20)
      : levelResult.score;

    return {
      ...levelResult,
      signalLifts: liftsBySignal,
      bestLift,
      bestLiftLabel,
      adjustedScore,
      quality: qualityLabel(adjustedScore)
    };
  }).sort((a, b) => b.adjustedScore - a.adjustedScore);
}

// ─────────────────────────────────────────────
//  UI COMPONENTS
// ─────────────────────────────────────────────
function Badge({ text, color }) {
  return (
    <span style={{ display: "inline-block", padding: "2px 7px", borderRadius: "4px", background: color + "20", color, fontFamily: T.font, fontSize: "9px", fontWeight: 700, textTransform: "uppercase", letterSpacing: ".06em" }}>
      {text}
    </span>
  );
}
function Sel({ label, value, options, onChange, width = "180px" }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
      {label && <div style={{ ...lbS, display: "block" }}>{label}</div>}
      <select value={value} onChange={e => onChange(e.target.value)} style={{ background: T.bgInput, border: `1px solid ${T.border}`, borderRadius: T.r, color: T.text, fontFamily: T.fontSans, fontSize: "12px", padding: "6px 10px", width, cursor: "pointer", outline: "none" }}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}
function MultiCheck({ options, selected, onToggle }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
      {options.map(o => {
        const active = selected.includes(o.value);
        return (
          <button key={o.value} onClick={() => onToggle(o.value)} style={{ padding: "4px 10px", borderRadius: "6px", cursor: "pointer", border: `1px solid ${active ? T.accent : T.border}`, background: active ? T.accent + "20" : "transparent", color: active ? T.accent : T.textMuted, fontFamily: T.fontSans, fontSize: "11px", fontWeight: active ? 600 : 400 }}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

// Reusable file parse helper
function parseFile(file, onSuccess, onError) {
  const ext = file.name.split(".").pop().toLowerCase();
  if (["csv", "txt"].includes(ext)) {
    Papa.parse(file, {
      header: true, skipEmptyLines: true,
      complete: r => r.data.length ? onSuccess(r.data) : onError("File has no rows"),
      error: e => onError(e.message)
    });
  } else if (["xlsx", "xls", "xlsm"].includes(ext)) {
    const reader = new FileReader();
    reader.onload = e => {
      try {
        const wb = XLSX.read(e.target.result, { type: "array" });
        const rows = XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { defval: "" });
        rows.length ? onSuccess(rows) : onError("Sheet has no rows");
      } catch (err) { onError(err.message); }
    };
    reader.readAsArrayBuffer(file);
  } else { onError("Please upload .xlsx, .xls, or .csv"); }
}

// ─────────────────────────────────────────────
//  SCREEN 1: UPLOAD
// ─────────────────────────────────────────────
function UploadScreen({ onData }) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);
  const [signalFiles, setSignalFiles] = useState([]); // [{rows, name}]
  const [sigDragging, setSigDragging] = useState(false);
  const [signalError, setSignalError] = useState(null);

  function handleDemandFile(file) {
    setError(null);
    parseFile(file, rows => onData(rows, file.name, signalFiles), e => setError(e));
  }
  function handleSignalFile(file) {
    setSignalError(null);
    parseFile(file, rows => setSignalFiles(prev => [...prev, { rows, name: file.name }]), e => setSignalError(`Could not read "${file.name}": ${e}`));
  }
  function removeSignal(i) { setSignalFiles(prev => prev.filter((_, idx) => idx !== i)); }

  return (
    <div style={{ minHeight: "100vh", background: T.bg, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: T.fontSans }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');*{box-sizing:border-box;margin:0;padding:0}select option{background:${T.bgCard};color:${T.text}}`}</style>
      <div style={{ maxWidth: 560, width: "100%", padding: "0 24px" }}>
        <div style={{ textAlign: "center", marginBottom: 32 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 8 }}>
            <Layers size={22} style={{ color: T.accent }} />
            <span style={{ fontSize: 22, fontWeight: 700, color: T.text }}>Right Level to Forecast</span>
          </div>
          <p style={{ color: T.textMuted, fontSize: 13, lineHeight: 1.6 }}>
            Upload historical demand data — and optionally signal files — to find the optimal aggregation level for forecasting.
          </p>
        </div>

        {/* Demand upload */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ ...lbS, marginBottom: 8 }}>1. Demand Data <span style={{ color: T.red, marginLeft: 4 }}>*</span></div>
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={e => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleDemandFile(f); }}
            onClick={() => document.getElementById("demand-input").click()}
            style={{ border: `2px dashed ${dragging ? T.accent : T.border}`, borderRadius: T.rLg, padding: "32px", cursor: "pointer", background: dragging ? T.accent + "08" : T.bgCard, textAlign: "center", transition: "all .15s" }}
          >
            <FileSpreadsheet size={28} style={{ color: dragging ? T.accent : T.textMuted, marginBottom: 8 }} />
            <div style={{ color: T.text, fontSize: 13, fontWeight: 600, marginBottom: 4 }}>Drop demand file or click to browse</div>
            <div style={{ color: T.textMuted, fontSize: 11 }}>.xlsx · .xls · .csv</div>
            <input id="demand-input" type="file" accept=".xlsx,.xls,.csv" style={{ display: "none" }}
              onChange={e => e.target.files[0] && handleDemandFile(e.target.files[0])} />
          </div>
          {error && <div style={{ marginTop: 8, color: T.red, fontSize: 11, fontFamily: T.font, display: "flex", alignItems: "center", gap: 6 }}><XCircle size={11} />{error}</div>}
        </div>

        {/* Signal uploads */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ ...lbS, marginBottom: 8 }}>2. Signal Files <span style={{ color: T.textDim, marginLeft: 4 }}>(optional — Nielsen POS, MSA shipments, Offtake, etc.)</span></div>
          {signalFiles.map((sf, i) => (
            <div key={sf.name} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 12px", background: T.bgSurface, borderRadius: T.r, border: `1px solid ${T.border}`, marginBottom: 6 }}>
              <FileSpreadsheet size={12} style={{ color: T.accent }} />
              <span style={{ fontSize: 12, color: T.text, flex: 1 }}>{sf.name}</span>
              <span style={{ fontSize: 11, color: T.textMuted }}>{sf.rows.length.toLocaleString()} rows</span>
              <button onClick={() => removeSignal(i)} style={{ background: "none", border: "none", cursor: "pointer", color: T.textMuted, padding: 2 }}><Trash2 size={12} /></button>
            </div>
          ))}
          {signalFiles.length < 3 && (
            <div
              onDragOver={e => { e.preventDefault(); setSigDragging(true); }}
              onDragLeave={() => setSigDragging(false)}
              onDrop={e => { e.preventDefault(); setSigDragging(false); const f = e.dataTransfer.files[0]; if (f) handleSignalFile(f); }}
              onClick={() => document.getElementById("signal-input").click()}
              style={{ border: `1px dashed ${sigDragging ? T.accent : T.border}`, borderRadius: T.r, padding: "12px", cursor: "pointer", background: sigDragging ? T.accent + "06" : "transparent", display: "flex", alignItems: "center", gap: 8, transition: "all .15s" }}
            >
              <Plus size={14} style={{ color: T.textMuted }} />
              <span style={{ fontSize: 12, color: T.textMuted }}>Add signal file ({signalFiles.length}/3)</span>
              <input id="signal-input" type="file" accept=".xlsx,.xls,.csv" style={{ display: "none" }}
                onChange={e => { if (e.target.files[0]) { handleSignalFile(e.target.files[0]); e.target.value = ""; } }} />
            </div>
          )}
          {signalError && <div style={{ marginTop: 8, color: T.red, fontSize: 11, fontFamily: T.font, display: "flex", alignItems: "center", gap: 6 }}><XCircle size={11} />{signalError}</div>}
        </div>

        <div style={{ padding: "12px 16px", background: T.bgSurface, borderRadius: T.r, border: `1px solid ${T.border}` }}>
          <div style={{ ...lbS, marginBottom: 6 }}><Info size={10} style={{ marginRight: 4 }} />Signal file format</div>
          <div style={{ fontFamily: T.font, fontSize: "10px", color: T.textMuted, lineHeight: 1.8 }}>
            Columns: Part · Customer · Site · Date/WeekIndex · Value (ShippedUnits, OfftakeUnits, Projected_Total_POS_Units, etc.)
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
//  SCREEN 2: CONFIGURE
// ─────────────────────────────────────────────
function SignalConfig({ sig, index, demandGrainCols, onChange }) {
  const cols = Object.keys(sig.rows[0] || {});
  const colOpts = cols.map(c => ({ value: c, label: c }));

  function detectSigDate() { return cols.find(c => /date|week|start|period|time/i.test(c)) || cols[0]; }
  function detectSigValue() {
    const numericCols = cols.filter(c => {
      const vals = sig.rows.slice(0, 10).map(r => parseCurrency(r[c]));
      return vals.filter(v => v != null).length >= 5;
    });
    return numericCols.find(c => /unit|qty|vol|ship|sale|pos|offtake|project/i.test(c)) || numericCols[numericCols.length - 1] || cols[0];
  }
  function autoMapGrain() {
    const mapping = {};
    demandGrainCols.forEach(dc => {
      const match = cols.find(sc => sc.toLowerCase() === dc.toLowerCase() || dc.toLowerCase().includes(sc.toLowerCase()) || sc.toLowerCase().includes(dc.toLowerCase().split(" ")[0]));
      if (match) mapping[dc] = match;
    });
    return mapping;
  }

  const [dateCol, setDateCol] = useState(() => detectSigDate());
  const [valueCol, setValueCol] = useState(() => detectSigValue());
  const [grainMapping, setGrainMapping] = useState(() => autoMapGrain());
  const [label, setLabel] = useState(sig.name.replace(/\.[^.]+$/, ""));

  useEffect(() => {
    onChange(index, { ...sig, dateCol, valueCol, grainMapping, label });
  }, [dateCol, valueCol, grainMapping, label]);

  function updateMapping(demCol, sigCol) {
    setGrainMapping(prev => ({ ...prev, [demCol]: sigCol }));
  }

  return (
    <div style={{ ...crdS, borderColor: T.accent + "40" }}>
      <div style={{ ...lbS, marginBottom: 12 }}>
        <Zap size={11} style={{ marginRight: 4, color: T.accent }} />
        Signal {index + 1} — <input value={label} onChange={e => setLabel(e.target.value)} style={{ background: "transparent", border: "none", outline: "none", color: T.accent, fontFamily: T.font, fontSize: "10px", fontWeight: 700, marginLeft: 6, width: 160 }} />
      </div>
      <div style={{ fontSize: 11, color: T.textMuted, marginBottom: 12 }}>{sig.name} · {sig.rows.length.toLocaleString()} rows</div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 12 }}>
        <Sel label="Date / Week column" value={dateCol} options={colOpts} onChange={setDateCol} width="180px" />
        <Sel label="Signal value column" value={valueCol} options={colOpts} onChange={setValueCol} width="220px" />
      </div>
      {demandGrainCols.length > 0 && (
        <div>
          <div style={{ ...lbS, marginBottom: 8 }}>Map demand grain → signal column</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 8 }}>
            {demandGrainCols.map(dc => (
              <div key={dc} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontFamily: T.font, fontSize: 10, color: T.text, width: 100, flexShrink: 0 }}>{dc}</span>
                <span style={{ color: T.textDim, fontSize: 10 }}>→</span>
                <select value={grainMapping[dc] || ""} onChange={e => updateMapping(dc, e.target.value)} style={{ background: T.bgInput, border: `1px solid ${T.border}`, borderRadius: T.r, color: T.text, fontFamily: T.fontSans, fontSize: "11px", padding: "4px 8px", flex: 1, outline: "none" }}>
                  <option value="">— none —</option>
                  {cols.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ConfigScreen({ rows, fileName, signalFiles, onRun, onBack }) {
  const cols = Object.keys(rows[0] || {});

  function detectDate() { return cols.find(c => /date|week|period|time|day|month/i.test(c)) || cols[0]; }
  function detectTarget() {
    const numeric = cols.filter(c => { const vals = rows.slice(0, 20).map(r => parseCurrency(r[c])); return vals.filter(v => v != null).length >= 10; });
    return numeric.find(c => /qty|quant|demand|sales|volume|amount/i.test(c)) || numeric[0] || cols[0];
  }
  function detectGrains() {
    return cols.filter(c => { const vals = new Set(rows.slice(0, 200).map(r => r[c])); return vals.size >= 2 && vals.size <= 200 && !/date|time|week|period|qty|quant|demand|sales|cost|price|amount/i.test(c); });
  }
  function detectFilterCol() {
    return cols.find(c => { const vals = new Set(rows.map(r => r[c])); return vals.size <= 10 && /header|category|type|class|flag|kind/i.test(c); }) || "";
  }

  const [dateCol, setDateCol] = useState(() => detectDate());
  const [targetCol, setTargetCol] = useState(() => detectTarget());
  const [grainCols, setGrainCols] = useState(() => detectGrains());
  const [filterCol, setFilterCol] = useState(() => detectFilterCol());
  const [filterVal, setFilterVal] = useState("");
  const [configuredSignals, setConfiguredSignals] = useState(() => signalFiles.map(sf => ({ ...sf, dateCol: "", valueCol: "", grainMapping: {}, label: sf.name.replace(/\.[^.]+$/, "") })));

  const filterVals = useMemo(() => filterCol ? [...new Set(rows.map(r => String(r[filterCol])))].sort() : [], [filterCol, rows]);
  useEffect(() => { if (filterVals.length && !filterVal) setFilterVal(filterVals[0]); }, [filterVals]);

  const colOptions = cols.map(c => ({ value: c, label: c }));
  const effectiveRows = useMemo(() => filterCol && filterVal ? rows.filter(r => String(r[filterCol]) === filterVal).length : rows.length, [rows, filterCol, filterVal]);

  function updateSignal(i, updated) {
    setConfiguredSignals(prev => prev.map((s, idx) => idx === i ? updated : s));
  }

  function handleRun() {
    const signals = configuredSignals.filter(s => s.dateCol && s.valueCol).map(s => ({
      rows: s.rows, dateCol: s.dateCol, valueCol: s.valueCol,
      grainMapping: s.grainMapping, label: s.label
    }));
    onRun({ rows, dateCol, targetCol, grainCols, filterCol, filterVal, fileName, signals });
  }

  return (
    <div style={{ minHeight: "100vh", background: T.bg, fontFamily: T.fontSans, padding: "32px 24px" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');*{box-sizing:border-box;margin:0;padding:0}select option{background:${T.bgCard};color:${T.text}}`}</style>
      <div style={{ maxWidth: 760, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 24 }}>
          <Layers size={18} style={{ color: T.accent }} />
          <span style={{ fontSize: 16, fontWeight: 700, color: T.text }}>Right Level to Forecast</span>
          <span style={{ color: T.textMuted, fontSize: 12 }}>— Configure</span>
          <button onClick={onBack} style={{ marginLeft: "auto", background: "transparent", border: `1px solid ${T.border}`, borderRadius: T.r, color: T.textMuted, cursor: "pointer", padding: "4px 10px", fontSize: 11, fontFamily: T.fontSans, display: "flex", alignItems: "center", gap: 4 }}><RefreshCw size={11} /> Back</button>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 24, padding: "10px 14px", background: T.bgSurface, borderRadius: T.r, border: `1px solid ${T.border}` }}>
          <FileSpreadsheet size={14} style={{ color: T.accent }} />
          <span style={{ color: T.text, fontSize: 12, fontWeight: 600 }}>{fileName}</span>
          <span style={{ color: T.textMuted, fontSize: 11 }}>— {rows.length.toLocaleString()} rows · {cols.length} columns</span>
          {configuredSignals.length > 0 && <span style={{ marginLeft: 8, color: T.cyan, fontSize: 11 }}>+ {configuredSignals.length} signal file{configuredSignals.length > 1 ? "s" : ""}</span>}
        </div>

        <div style={{ display: "grid", gap: 16 }}>
          {/* Filter */}
          <div style={crdS}>
            <div style={{ ...lbS, marginBottom: 12 }}><Filter size={11} style={{ marginRight: 4 }} />Row Filter <span style={{ color: T.textDim, marginLeft: 4 }}>(optional)</span></div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <Sel label="Filter column" value={filterCol} width="200px" options={[{ value: "", label: "— none —" }, ...colOptions]} onChange={v => { setFilterCol(v); setFilterVal(""); }} />
              {filterCol && filterVals.length > 0 && <Sel label="Filter value" value={filterVal} width="200px" options={filterVals.map(v => ({ value: v, label: v }))} onChange={setFilterVal} />}
            </div>
            {filterCol && filterVal && <div style={{ marginTop: 10, fontSize: 11, color: T.textMuted, display: "flex", alignItems: "center", gap: 4 }}><CheckCircle size={11} style={{ color: T.green }} />{effectiveRows.toLocaleString()} rows match</div>}
          </div>

          {/* Date + Target */}
          <div style={crdS}>
            <div style={{ ...lbS, marginBottom: 12 }}><Calendar size={11} style={{ marginRight: 4 }} />Date & Target Column</div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <Sel label="Date column" value={dateCol} options={colOptions} onChange={setDateCol} />
              <Sel label="Target (value to forecast)" value={targetCol} options={colOptions} onChange={setTargetCol} width="220px" />
            </div>
          </div>

          {/* Grain columns */}
          <div style={crdS}>
            <div style={{ ...lbS, marginBottom: 10 }}><Tag size={11} style={{ marginRight: 4 }} />Grain / Dimension Columns</div>
            <div style={{ fontSize: 12, color: T.textMuted, marginBottom: 12, lineHeight: 1.6 }}>Select dimensions that define your hierarchy. Every combination will be tested.</div>
            <MultiCheck options={cols.filter(c => c !== dateCol && c !== targetCol && c !== filterCol).map(c => ({ value: c, label: c }))} selected={grainCols} onToggle={col => setGrainCols(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col])} />
            {grainCols.length > 0 && <div style={{ marginTop: 10, fontSize: 11, color: T.textMuted, display: "flex", alignItems: "center", gap: 4 }}><Info size={11} />Will test {2 ** grainCols.length} levels</div>}
          </div>

          {/* Signal configurations */}
          {configuredSignals.map((sig, i) => (
            <SignalConfig key={sig.name} sig={sig} index={i} demandGrainCols={grainCols} onChange={updateSignal} />
          ))}

          <button onClick={handleRun} disabled={!dateCol || !targetCol} style={{ background: T.accent, color: "#fff", border: "none", borderRadius: T.r, padding: "12px 24px", cursor: "pointer", fontFamily: T.fontSans, fontSize: 14, fontWeight: 700, display: "flex", alignItems: "center", gap: 8, justifyContent: "center", opacity: (!dateCol || !targetCol) ? 0.5 : 1 }}>
            <BarChart3 size={15} /> Analyze Forecast Levels
          </button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
//  SCREEN 3: DASHBOARD
// ─────────────────────────────────────────────
function Dashboard({ config, onReset }) {
  const { rows, dateCol, targetCol, grainCols, filterCol, filterVal, fileName, signals } = config;
  const hasSignals = signals && signals.length > 0;

  const baseResults = useMemo(
    () => analyzeLevel(rows, targetCol, dateCol, grainCols, filterCol, filterVal),
    [rows, targetCol, dateCol, grainCols, filterCol, filterVal]
  );

  const [signalResults, setSignalResults] = useState(null);
  const [signalRunning, setSignalRunning] = useState(false);
  const [signalError, setSignalError] = useState(null);

  const results = signalResults || baseResults;
  const sortField = signalResults ? "adjustedScore" : "score";

  const best = results.find(r => !r.skipped);
  const [selected, setSelected] = useState(null);
  const [sortKey, setSortKey] = useState(sortField);
  const [sortDir, setSortDir] = useState("desc");

  const sorted = useMemo(() => {
    return [...results].sort((a, b) => {
      const av = a[sortKey] ?? (sortDir === "desc" ? -Infinity : Infinity);
      const bv = b[sortKey] ?? (sortDir === "desc" ? -Infinity : Infinity);
      return sortDir === "desc" ? bv - av : av - bv;
    });
  }, [results, sortKey, sortDir]);

  function toggleSort(key) {
    if (sortKey === key) setSortDir(d => d === "desc" ? "asc" : "desc");
    else { setSortKey(key); setSortDir("desc"); }
  }
  function SortIcon({ k }) {
    if (sortKey !== k) return null;
    return sortDir === "desc" ? <ChevronDown size={10} /> : <ChevronUp size={10} />;
  }

  const runSignalAnalysis = useCallback(() => {
    if (!hasSignals || signalRunning) return;
    setSignalRunning(true);
    setSignalError(null);
    setTimeout(() => {
      try { setSignalResults(computeAllSignalLifts(baseResults, signals)); }
      catch (e) { console.error(e); setSignalError(`Signal analysis failed: ${e.message}`); }
      setSignalRunning(false);
    }, 50);
  }, [baseResults, signals, hasSignals, signalRunning]);

  const drillSeries = useMemo(() => {
    if (!selected || selected.skipped || !selected.seriesMetrics.length) return null;
    const top = selected.seriesMetrics[0];
    return { ...top, chartData: top.values.map((v, i) => ({ i, actual: v })) };
  }, [selected]);

  const barData = results.filter(r => !r.skipped).map(r => ({
    name: r.level.length > 22 ? r.level.slice(0, 20) + "…" : r.level,
    score: signalResults ? r.adjustedScore : r.score,
    color: qualityColor(signalResults ? r.adjustedScore : r.score)
  }));

  const thS = k => ({ padding: "6px 10px", textAlign: "right", cursor: "pointer", color: sortKey === k ? T.accent : T.textMuted, fontFamily: T.font, fontSize: "9px", textTransform: "uppercase", letterSpacing: ".06em", userSelect: "none", whiteSpace: "nowrap", background: T.bgSurface });
  const tdS = { padding: "7px 10px", textAlign: "right", fontFamily: T.font, fontSize: "11px", color: T.text, borderTop: `1px solid ${T.border}20` };

  const displayScore = r => signalResults ? r.adjustedScore : r.score;
  const displayBest = best ? displayScore(best) : 0;

  return (
    <div style={{ minHeight: "100vh", background: T.bg, fontFamily: T.fontSans }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');*{box-sizing:border-box;margin:0;padding:0}::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:${T.bg}}::-webkit-scrollbar-thumb{background:${T.border};border-radius:3px}select option{background:${T.bgCard};color:${T.text}}`}</style>

      {/* Header */}
      <div style={{ background: T.bgCard, borderBottom: `1px solid ${T.border}`, padding: "10px 22px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Layers size={15} style={{ color: T.accent }} />
          <span style={{ fontSize: 14, fontWeight: 700 }}>Right Level to Forecast</span>
          <span style={{ color: T.textMuted, fontSize: 11, marginLeft: 4 }}>— {fileName}</span>
          {hasSignals && <span style={{ color: T.cyan, fontSize: 11 }}>· {signals.length} signal{signals.length > 1 ? "s" : ""} loaded</span>}
        </div>
        <button onClick={onReset} style={{ background: T.bgInput, border: `1px solid ${T.border}`, borderRadius: "6px", padding: "5px 10px", cursor: "pointer", color: T.textMuted, fontFamily: T.fontSans, fontSize: "11px", display: "flex", alignItems: "center", gap: 4 }}><RefreshCw size={11} /> New File</button>
      </div>

      <div style={{ padding: "20px 22px", display: "flex", flexDirection: "column", gap: 16 }}>

        {/* Signal analysis CTA */}
        {hasSignals && !signalResults && (
          <div style={{ background: T.cyan + "10", border: `1px solid ${T.cyan}30`, borderRadius: T.rLg, padding: "14px 18px", display: "flex", alignItems: "center", gap: 12 }}>
            <Brain size={18} style={{ color: T.cyan, flexShrink: 0 }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: T.text, marginBottom: 2 }}>Signal Lift Analysis ready</div>
              <div style={{ fontSize: 11, color: T.textMuted }}>{signals.map(s => s.label).join(", ")} · GBM walk-forward validation per level</div>
            </div>
            <button onClick={runSignalAnalysis} disabled={signalRunning} style={{ background: T.cyan, color: "#000", border: "none", borderRadius: T.r, padding: "8px 16px", cursor: signalRunning ? "not-allowed" : "pointer", fontFamily: T.fontSans, fontSize: 12, fontWeight: 700, display: "flex", alignItems: "center", gap: 6, opacity: signalRunning ? 0.7 : 1 }}>
              {signalRunning ? <><RefreshCw size={12} style={{ animation: "spin 1s linear infinite" }} /> Running…</> : <><Zap size={12} /> Run Signal Analysis</>}
            </button>
          </div>
        )}
        {signalError && (
          <div style={{ background: T.red + "10", border: `1px solid ${T.red}40`, borderRadius: T.rLg, padding: "12px 16px", display: "flex", alignItems: "center", gap: 10 }}>
            <AlertCircle size={15} style={{ color: T.red, flexShrink: 0 }} />
            <span style={{ fontSize: 12, color: T.red }}>{signalError}</span>
          </div>
        )}

        {/* Recommendation banner */}
        {best && (
          <div style={{ background: qualityColor(displayBest) + "10", border: `1px solid ${qualityColor(displayBest)}40`, borderRadius: T.rLg, padding: "14px 18px", display: "flex", alignItems: "flex-start", gap: 12 }}>
            <TrendingUp size={18} style={{ color: qualityColor(displayBest), marginTop: 1, flexShrink: 0 }} />
            <div>
              <div style={{ fontSize: 13, fontWeight: 700, color: T.text, marginBottom: 4 }}>
                Recommended level: <span style={{ color: qualityColor(displayBest) }}>{best.level}</span>
              </div>
              <div style={{ fontSize: 11, color: T.textMuted, lineHeight: 1.7 }}>
                {signalResults ? "Adjusted Score" : "Forecastability Score"} <b style={{ color: T.text }}>{displayBest}/100</b> ·{" "}
                {best.numSeries} series · Median CV {best.medianCV} · {best.pctSmooth}% smooth
                {best.avgMAPE != null && ` · MAPE ${best.avgMAPE}%`}
                {best.bestLift != null && ` · Best signal lift: `}
                {best.bestLift != null && <b style={{ color: T.cyan }}>{best.bestLift > 0 ? "+" : ""}{best.bestLift}% ({best.bestLiftLabel})</b>}
              </div>
              {signalResults && best.bestLift != null && best.bestLift > 5 && (
                <div style={{ marginTop: 6, fontSize: 11, color: T.cyan }}>
                  <CheckCircle size={11} style={{ marginRight: 4, verticalAlign: "middle" }} />
                  Signal provides meaningful lift at this level — forecasting with {best.bestLiftLabel} is recommended.
                </div>
              )}
              {signalResults && best.bestLift != null && best.bestLift <= 0 && (
                <div style={{ marginTop: 6, fontSize: 11, color: T.orange }}>
                  <AlertCircle size={11} style={{ marginRight: 4, verticalAlign: "middle" }} />
                  Signals do not improve accuracy at this level. Consider forecasting at a different level where lift is higher.
                </div>
              )}
            </div>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 16, alignItems: "start" }}>
          {/* Table */}
          <div style={crdS}>
            <div style={{ ...lbS, marginBottom: 10 }}><BarChart3 size={11} style={{ marginRight: 4 }} />Forecast Level Rankings — click a row to drill down</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ borderCollapse: "collapse", width: "100%", minWidth: hasSignals ? 680 : 560 }}>
                <thead>
                  <tr>
                    <th style={{ ...thS("level"), textAlign: "left" }} onClick={() => toggleSort("level")}>Level <SortIcon k="level" /></th>
                    <th style={thS("numSeries")} onClick={() => toggleSort("numSeries")}># Series <SortIcon k="numSeries" /></th>
                    <th style={thS("avgN")} onClick={() => toggleSort("avgN")}>Avg N <SortIcon k="avgN" /></th>
                    <th style={thS("medianCV")} onClick={() => toggleSort("medianCV")}>Median CV <SortIcon k="medianCV" /></th>
                    <th style={thS("pctSmooth")} onClick={() => toggleSort("pctSmooth")}>% Smooth <SortIcon k="pctSmooth" /></th>
                    <th style={thS("avgMAPE")} onClick={() => toggleSort("avgMAPE")}>MAPE% <SortIcon k="avgMAPE" /></th>
                    {signalResults && <th style={thS("bestLift")} onClick={() => toggleSort("bestLift")}>Best Signal Lift <SortIcon k="bestLift" /></th>}
                    <th style={thS(signalResults ? "adjustedScore" : "score")} onClick={() => toggleSort(signalResults ? "adjustedScore" : "score")}>{signalResults ? "Adj. Score" : "Score"} <SortIcon k={signalResults ? "adjustedScore" : "score"} /></th>
                    <th style={{ ...thS("quality"), textAlign: "center" }}>Quality</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((r, i) => {
                    const sel = selected?.level === r.level;
                    const sc = displayScore(r);
                    return (
                      <tr key={r.level} onClick={() => setSelected(sel ? null : r)} style={{ background: sel ? T.accent + "12" : i % 2 === 0 ? "transparent" : T.bgSurface + "60", cursor: "pointer" }}>
                        <td style={{ ...tdS, textAlign: "left", color: sel ? T.accent : r.level === "Total" ? T.purple : T.text, fontWeight: sel ? 600 : 400 }}>{r.level}</td>
                        <td style={tdS}>{r.skipped ? "—" : r.numSeries}</td>
                        <td style={tdS}>{r.skipped ? "—" : r.avgN}</td>
                        <td style={{ ...tdS, color: r.skipped ? T.textDim : r.medianCV < 0.3 ? T.green : r.medianCV < 0.6 ? T.orange : T.red }}>{r.skipped ? "—" : r.medianCV}</td>
                        <td style={{ ...tdS, color: r.skipped ? T.textDim : r.pctSmooth >= 70 ? T.green : r.pctSmooth >= 40 ? T.orange : T.red }}>{r.skipped ? "—" : `${r.pctSmooth}%`}</td>
                        <td style={tdS}>{r.skipped ? <Badge text="Too few rows" color={T.textDim} /> : r.avgMAPE != null ? `${r.avgMAPE}%` : "—"}</td>
                        {signalResults && (
                          <td style={{ ...tdS, color: r.bestLift == null ? T.textDim : r.bestLift > 5 ? T.green : r.bestLift > 0 ? T.orange : T.red }}>
                            {r.bestLift == null ? "—" : `${r.bestLift > 0 ? "+" : ""}${r.bestLift}%`}
                            {r.bestLiftLabel && <span style={{ color: T.textDim, fontSize: 9, marginLeft: 4 }}>{r.bestLiftLabel}</span>}
                          </td>
                        )}
                        <td style={{ ...tdS, color: r.skipped ? T.textDim : qualityColor(sc), fontWeight: 700 }}>{r.skipped ? "—" : sc}</td>
                        <td style={{ ...tdS, textAlign: "center" }}><Badge text={r.skipped ? "SKIP" : r.quality} color={r.skipped ? T.textDim : qualityColor(sc)} /></td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Right column */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={crdS}>
              <div style={{ ...lbS, marginBottom: 12 }}>{signalResults ? "Adjusted Score" : "Forecastability Score"} by Level</div>
              <ResponsiveContainer width="100%" height={Math.max(160, barData.length * 32)}>
                <BarChart data={barData} layout="vertical" margin={{ left: 0, right: 16, top: 0, bottom: 0 }}>
                  <XAxis type="number" domain={[0, 100]} tick={{ fontFamily: T.font, fontSize: 9, fill: T.textMuted }} />
                  <YAxis type="category" dataKey="name" width={120} tick={{ fontFamily: T.font, fontSize: 9, fill: T.textMuted }} />
                  <Tooltip contentStyle={{ background: T.bgCard, border: `1px solid ${T.border}`, borderRadius: T.r, fontFamily: T.font, fontSize: 11 }} formatter={v => [`${v} / 100`, "Score"]} />
                  <Bar dataKey="score" radius={3}>{barData.map((e) => <Cell key={e.name} fill={e.color} />)}</Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={{ ...crdS, padding: "12px 16px" }}>
              <div style={{ ...lbS, marginBottom: 8 }}>Score Guide</div>
              {[["GOOD", "≥ 65", T.green, "Stable + forecastable"], ["FAIR", "40–64", T.orange, "Moderate — monitor"], ["POOR", "< 40", T.red, "High noise"]].map(([q, r, c, d]) => (
                <div key={q} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                  <Badge text={q} color={c} /><span style={{ fontFamily: T.font, fontSize: 10, color: T.textMuted }}>{r} — {d}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Drill-down */}
        {selected && !selected.skipped && (
          <div style={{ display: "grid", gridTemplateColumns: signalResults && selected.signalLifts ? "260px 1fr 1fr" : "260px 1fr", gap: 16 }}>

            {/* Classification pie */}
            <div style={crdS}>
              <div style={{ ...lbS, marginBottom: 10 }}>Demand Classification — <span style={{ color: T.accent }}>{selected.level}</span></div>
              <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                <ResponsiveContainer width={110} height={110}>
                  <PieChart>
                    <Pie data={Object.entries(selected.clsCounts).filter(([, v]) => v > 0).map(([name, value]) => ({ name, value }))} cx="50%" cy="50%" outerRadius={50} dataKey="value">
                      {Object.entries(selected.clsCounts).filter(([, v]) => v > 0).map(([cls]) => <Cell key={cls} fill={CLS_COLORS[cls]} />)}
                    </Pie>
                    <Tooltip contentStyle={{ background: T.bgCard, border: `1px solid ${T.border}`, fontFamily: T.font, fontSize: 11 }} />
                  </PieChart>
                </ResponsiveContainer>
                <div>
                  {Object.entries(selected.clsCounts).filter(([, v]) => v > 0).map(([cls, n]) => (
                    <div key={cls} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 5 }}>
                      <div style={{ width: 8, height: 8, borderRadius: "50%", background: CLS_COLORS[cls] }} />
                      <span style={{ fontFamily: T.font, fontSize: 10, color: T.text }}>{cls}</span>
                      <span style={{ fontFamily: T.font, fontSize: 10, color: T.textMuted }}>{n}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ marginTop: 10, fontSize: 10, color: T.textDim, lineHeight: 1.7, fontFamily: T.font }}>
                Smooth: ADI&lt;1.32 &amp; CV²&lt;0.49<br />Erratic: ADI&lt;1.32 &amp; CV²≥0.49<br />Intermittent: ADI≥1.32 &amp; CV²&lt;0.49<br />Lumpy: ADI≥1.32 &amp; CV²≥0.49
              </div>
            </div>

            {/* Sample series */}
            <div style={crdS}>
              <div style={{ ...lbS, marginBottom: 10 }}>Top-Volume Series — <span style={{ color: T.accent }}>{selected.level}</span></div>
              {drillSeries ? (
                <>
                  <div style={{ fontSize: 11, color: T.textMuted, marginBottom: 8, display: "flex", gap: 10, flexWrap: "wrap" }}>
                    <b style={{ color: T.text }}>{drillSeries.key}</b>
                    <span>N={drillSeries.n}</span>
                    <span>CV={drillSeries.cv < 9999 ? drillSeries.cv.toFixed(3) : "∞"}</span>
                    <Badge text={drillSeries.cls} color={CLS_COLORS[drillSeries.cls]} />
                    {drillSeries.mape != null && <span>MAPE={drillSeries.mape.toFixed(1)}%</span>}
                  </div>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={drillSeries.chartData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke={T.border} strokeDasharray="3 3" />
                      <XAxis dataKey="i" tick={{ fontFamily: T.font, fontSize: 9, fill: T.textMuted }} interval={Math.max(0, Math.floor(drillSeries.chartData.length / 8) - 1)} />
                      <YAxis tick={{ fontFamily: T.font, fontSize: 9, fill: T.textMuted }} width={40} />
                      <Tooltip contentStyle={{ background: T.bgCard, border: `1px solid ${T.border}`, fontFamily: T.font, fontSize: 11 }} formatter={v => [v?.toFixed ? v.toFixed(1) : v, targetCol]} />
                      <Line type="monotone" dataKey="actual" stroke={T.accent} dot={false} strokeWidth={1.5} />
                    </LineChart>
                  </ResponsiveContainer>
                </>
              ) : <div style={{ color: T.textMuted, fontSize: 12, textAlign: "center", padding: "40px 0" }}>No series data</div>}
            </div>

            {/* Signal lift bar chart */}
            {signalResults && selected.signalLifts && (
              <div style={crdS}>
                <div style={{ ...lbS, marginBottom: 10 }}>Signal Lift at this Level — <span style={{ color: T.accent }}>{selected.level}</span></div>
                {selected.signalLifts.some(s => s.avgLift !== null) ? (
                  <>
                    <ResponsiveContainer width="100%" height={150}>
                      <BarChart data={selected.signalLifts.filter(s => s.avgLift !== null).map(s => ({ name: s.label, lift: +s.avgLift.toFixed(1) }))} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                        <CartesianGrid stroke={T.border} strokeDasharray="3 3" />
                        <XAxis dataKey="name" tick={{ fontFamily: T.font, fontSize: 9, fill: T.textMuted }} />
                        <YAxis tick={{ fontFamily: T.font, fontSize: 9, fill: T.textMuted }} width={36} tickFormatter={v => `${v}%`} />
                        <Tooltip contentStyle={{ background: T.bgCard, border: `1px solid ${T.border}`, fontFamily: T.font, fontSize: 11 }} formatter={v => [`${v > 0 ? "+" : ""}${v}%`, "RMSE lift"]} />
                        <ReferenceLine y={0} stroke={T.border} />
                        <Bar dataKey="lift" radius={3}>
                          {selected.signalLifts.filter(s => s.avgLift !== null).map((s) => <Cell key={s.label} fill={s.avgLift > 5 ? T.green : s.avgLift > 0 ? T.orange : T.red} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                    <div style={{ marginTop: 8, fontSize: 10, color: T.textMuted, lineHeight: 1.7, fontFamily: T.font }}>
                      Lift = % reduction in forecast RMSE when signal is added. Positive = signal helps. Negative = signal adds noise at this level.
                    </div>
                  </>
                ) : <div style={{ color: T.textMuted, fontSize: 12, textAlign: "center", padding: "40px 0" }}>Insufficient aligned data for signal lift calculation</div>}
              </div>
            )}
          </div>
        )}

        {/* Methodology */}
        <div style={{ ...crdS, padding: "12px 16px" }}>
          <div style={{ ...lbS, marginBottom: 8 }}><Info size={10} style={{ marginRight: 4 }} />How scores are calculated</div>
          <div style={{ fontSize: 11, color: T.textMuted, lineHeight: 1.8 }}>
            <b style={{ color: T.text }}>Median CV (35%)</b> — Coefficient of variation. Lower = more stable demand.<br />
            <b style={{ color: T.text }}>% Smooth series (40%)</b> — Syntetos-Boylan matrix: ADI &lt; 1.32 &amp; CV² &lt; 0.49. Higher = better.<br />
            <b style={{ color: T.text }}>Backtest MAPE (25%)</b> — 4-week moving average holdout (70/30 split).<br />
            {hasSignals && <><b style={{ color: T.cyan }}>Signal Lift (blended 20%)</b> — GBM walk-forward: % RMSE reduction when signal lags are added to target lags. Adjusts base score: Adjusted = base × 80% + lift_score × 20%.</>}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
//  ROOT APP
// ─────────────────────────────────────────────
export default function App() {
  const [screen, setScreen] = useState("upload");
  const [rows, setRows] = useState(null);
  const [fileName, setFileName] = useState("");
  const [signalFiles, setSignalFiles] = useState([]);
  const [runConfig, setRunConfig] = useState(null);

  function handleData(data, name, sigs) { setRows(data); setFileName(name); setSignalFiles(sigs || []); setScreen("config"); }
  function handleRun(cfg) { setRunConfig(cfg); setScreen("results"); }
  function handleReset() { setRows(null); setFileName(""); setSignalFiles([]); setRunConfig(null); setScreen("upload"); }

  if (screen === "upload") return <UploadScreen onData={handleData} />;
  if (screen === "config") return <ConfigScreen rows={rows} fileName={fileName} signalFiles={signalFiles} onRun={handleRun} onBack={handleReset} />;
  if (screen === "results") return <Dashboard config={runConfig} onReset={handleReset} />;
  return null;
}
