'use strict';

/* ============================================================
 * Modèle A — dashboard front-end
 * Talks to the FastAPI server defined in scripts/modelA/api.py
 * ============================================================ */

const SETTINGS_KEY = 'modelA.settings.v1';
const HISTORY_KEY = 'modelA.history.v1';
const HISTORY_MAX = 12;

const DEFAULT_SETTINGS = {
  apiUrl: '',          // empty = same-origin (relative)
  autoPredict: false,
  ecgGrid: true,
};

const LEAD_COLORS = [
  '#5cc8ff', '#4f6bff', '#a78bfa', '#f472b6',
  '#fbbf24', '#a3e635', '#34d399', '#22d3ee',
  '#c084fc', '#fb7185', '#fb923c', '#facc15',
];

const CLASS_NAMES = ['normal', 'suspect', 'critique'];

/* ---------- DOM ---------- */
const $ = (id) => document.getElementById(id);
const dom = {
  healthPill: $('healthPill'),
  healthText: $('healthText'),
  settingsBtn: $('settingsBtn'),
  reloadSamples: $('reloadSamples'),
  predictBtn: $('predictBtn'),
  sampleList: $('sampleList'),
  sampleEmpty: $('sampleEmpty'),
  filterChips: document.querySelectorAll('.chip'),
  featureForm: $('featureForm'),
  featureDialog: $('featureDialog'),
  featureSubmitBtn: $('featureSubmitBtn'),
  openFeatureFormBtn: $('openFeatureFormBtn'),
  closeFeatureForm: $('closeFeatureForm'),
  cancelFeatureForm: $('cancelFeatureForm'),
  canvas: $('ecgCanvas'),
  canvasWrap: $('canvasWrap'),
  canvasEmpty: $('canvasEmpty'),
  canvasSkeleton: $('canvasSkeleton'),
  legend: $('leadsLegend'),
  viewerMeta: $('viewerMeta'),
  resultPanel: $('resultPanel'),
  resultEmpty: $('resultEmpty'),
  resultBody: $('resultBody'),
  resultLabel: $('resultLabel'),
  resultMatch: $('resultMatch'),
  resultTimestamp: $('resultTimestamp'),
  ringFill: $('ringFill'),
  ringLabel: $('ringLabel'),
  riskFill: $('riskFill'),
  riskValue: $('riskValue'),
  probNormal: $('probNormal'),
  probSuspect: $('probSuspect'),
  probCritique: $('probCritique'),
  probNormalVal: $('probNormalVal'),
  probSuspectVal: $('probSuspectVal'),
  probCritiqueVal: $('probCritiqueVal'),
  metaLatency: $('metaLatency'),
  historyList: $('historyList'),
  historyEmpty: $('historyEmpty'),
  clearHistory: $('clearHistory'),
  settingsDialog: $('settingsDialog'),
  closeSettings: $('closeSettings'),
  apiUrlInput: $('apiUrlInput'),
  autoPredict: $('autoPredict'),
  ecgGrid: $('ecgGrid'),
  resetSettings: $('resetSettings'),
  toastStack: $('toastStack'),
};

/* ---------- State ---------- */
const state = {
  settings: loadSettings(),
  leads: [],
  samples: [],          // full list from /samples
  filter: 'all',
  currentSample: null,  // {ecg_id, filename_lr, signal, true_label, ...}
  currentResult: null,
  history: loadHistory(),
};

/* ============================================================
 *  Settings (persisted in localStorage)
 * ============================================================ */
function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS };
    return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
  } catch { return { ...DEFAULT_SETTINGS }; }
}
function saveSettings() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
}

/* ============================================================
 *  Networking — all requests go through `api()`
 * ============================================================ */
function apiUrl(path) {
  const base = (state.settings.apiUrl || '').replace(/\/+$/, '');
  return base + path;
}

async function api(path, options = {}) {
  const res = await fetch(apiUrl(path), {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    let detail = '';
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch { detail = await res.text().catch(() => ''); }
    const err = new Error(`${res.status} ${res.statusText} — ${detail}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/* ============================================================
 *  Toast notifications
 * ============================================================ */
function toast(message, kind = 'info', timeout = 3800) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.innerHTML = `<span>${escapeHtml(message)}</span>`;
  const close = document.createElement('button');
  close.className = 'toast-close';
  close.setAttribute('aria-label', 'Fermer');
  close.textContent = '×';
  close.onclick = () => el.remove();
  el.appendChild(close);
  dom.toastStack.appendChild(el);
  if (timeout > 0) setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 240); }, timeout);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

/* ============================================================
 *  Health
 * ============================================================ */
async function loadHealth() {
  setHealth('connexion…', 'pending');
  try {
    const h = await api('/health');
    state.leads = h.lead_names || [];
    renderLegend();
    if (h.model_loaded) {
      setHealth(`modèle prêt · ${h.device}`, 'ok');
    } else {
      setHealth('poids aléatoires (modèle non entraîné)', 'error');
      toast('Le modèle entraîné n\'est pas chargé — les prédictions seront aléatoires.', 'warn', 6000);
    }
  } catch (err) {
    setHealth('API hors ligne', 'error');
    toast(`Impossible de joindre l'API : ${err.message}`, 'error', 6000);
  }
}
function setHealth(text, kind) {
  dom.healthText.textContent = text;
  dom.healthPill.classList.remove('ok', 'error');
  if (kind === 'ok') dom.healthPill.classList.add('ok');
  if (kind === 'error') dom.healthPill.classList.add('error');
}

/* ============================================================
 *  Sample list
 * ============================================================ */
function renderSampleListSkeleton(n = 6) {
  dom.sampleList.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const li = document.createElement('li');
    li.className = 'skel skel-row';
    dom.sampleList.appendChild(li);
  }
  dom.sampleEmpty.classList.add('hidden');
}

function renderSampleList() {
  dom.sampleList.innerHTML = '';
  const filtered = state.filter === 'all'
    ? state.samples
    : state.samples.filter(s => s.true_label === state.filter);

  if (filtered.length === 0) {
    dom.sampleEmpty.classList.remove('hidden');
    return;
  }
  dom.sampleEmpty.classList.add('hidden');

  for (const s of filtered) {
    const li = document.createElement('li');
    li.className = 'sample-item';
    li.dataset.ecgId = s.ecg_id;
    if (state.currentSample && state.currentSample.ecg_id === s.ecg_id) {
      li.classList.add('active');
    }
    li.setAttribute('role', 'option');
    li.tabIndex = 0;
    li.innerHTML = `
      <span class="sample-id">#${s.ecg_id}</span>
      <span class="sample-tag ${s.true_label}">${s.true_label}</span>
    `;
    li.addEventListener('click', () => loadSample(s.ecg_id));
    li.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); loadSample(s.ecg_id); }
    });
    dom.sampleList.appendChild(li);
  }
}

async function loadSamples() {
  renderSampleListSkeleton();
  try {
    const items = await api('/samples?limit=100');
    state.samples = items;
    renderSampleList();
    if (items.length > 0 && !state.currentSample) {
      await loadSample(items[0].ecg_id);
    }
  } catch (err) {
    dom.sampleList.innerHTML = '';
    dom.sampleEmpty.classList.remove('hidden');
    dom.sampleEmpty.textContent = `Échec du chargement : ${err.message}`;
    toast(`Liste indisponible : ${err.message}`, 'error');
  }
}

async function loadSample(ecgId) {
  dom.canvasSkeleton.classList.remove('hidden');
  dom.canvasEmpty.classList.add('hidden');
  try {
    const sample = await api(`/samples/${ecgId}`);
    state.currentSample = sample;
    state.currentResult = null;
    state.leads = sample.leads || state.leads;
    renderLegend();
    updateViewerMeta();
    drawECG(sample.signal);
    highlightActiveSample();
    resetResultPanel();
    dom.predictBtn.disabled = false;
    if (state.settings.autoPredict) predictCurrent();
  } catch (err) {
    toast(`Échec du chargement : ${err.message}`, 'error');
    dom.canvasEmpty.classList.remove('hidden');
  } finally {
    dom.canvasSkeleton.classList.add('hidden');
  }
}

function highlightActiveSample() {
  for (const el of dom.sampleList.querySelectorAll('.sample-item')) {
    el.classList.toggle(
      'active',
      state.currentSample && Number(el.dataset.ecgId) === state.currentSample.ecg_id
    );
  }
}

function updateViewerMeta() {
  const s = state.currentSample;
  if (!s) { dom.viewerMeta.textContent = 'Sélectionnez un ECG pour commencer.'; return; }
  const trueLbl = s.true_label
    ? `réel : <span class="sample-tag ${s.true_label}">${s.true_label}</span>`
    : '';
  const file = s.filename_lr || s.source || 'fichier importé';
  dom.viewerMeta.innerHTML = `<span class="mono">${escapeHtml(file)}</span> · ${s.sampling_rate} Hz · 12 leads ${trueLbl}`;
}

/* ============================================================
 *  ECG drawing (HiDPI)
 * ============================================================ */
let canvasCtx = null;
function ensureCtx() {
  if (!canvasCtx) canvasCtx = dom.canvas.getContext('2d');
  return canvasCtx;
}

function resizeCanvas() {
  const rect = dom.canvasWrap.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  dom.canvas.width = Math.floor(rect.width * dpr);
  dom.canvas.height = Math.floor(rect.height * dpr);
  const ctx = ensureCtx();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (state.currentSample) drawECG(state.currentSample.signal);
}

function drawECG(signal) {
  if (!signal || !signal.length) return;
  const ctx = ensureCtx();
  const rect = dom.canvasWrap.getBoundingClientRect();
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  if (state.settings.ecgGrid) drawGrid(ctx, W, H);

  const rows = 6, cols = 2;
  const rowH = H / rows;
  const colW = W / cols;

  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  ctx.lineWidth = 1;
  for (let r = 1; r < rows; r++) {
    ctx.beginPath(); ctx.moveTo(0, r * rowH); ctx.lineTo(W, r * rowH); ctx.stroke();
  }
  ctx.beginPath(); ctx.moveTo(colW, 0); ctx.lineTo(colW, H); ctx.stroke();

  const N = signal[0].length;
  signal.forEach((lead, i) => {
    const col = Math.floor(i / rows);
    const row = i % rows;
    const x0 = col * colW + 10;
    const y0 = row * rowH;
    const usableW = colW - 20;
    const center = y0 + rowH / 2;

    let min = Infinity, max = -Infinity;
    for (let k = 0; k < N; k++) { const v = lead[k]; if (v < min) min = v; if (v > max) max = v; }
    const range = Math.max(1e-3, max - min);
    const scale = (rowH * 0.42) / range;
    const mid = (min + max) / 2;

    // Lead label
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillText(state.leads[i] || `L${i+1}`, x0 + 2, y0 + 14);

    // Signal trace
    ctx.beginPath();
    ctx.lineWidth = 1.3;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.strokeStyle = LEAD_COLORS[i % LEAD_COLORS.length];
    ctx.shadowColor = LEAD_COLORS[i % LEAD_COLORS.length];
    ctx.shadowBlur = 4;
    for (let k = 0; k < N; k++) {
      const x = x0 + (k / (N - 1)) * usableW;
      const y = center - (lead[k] - mid) * scale;
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  });
}

function drawGrid(ctx, W, H) {
  // Subtle ECG-paper-like grid (1mm minor / 5mm major).
  // We render relative to width: ~50 minor squares horizontally.
  const minor = W / 50;
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(239, 68, 68, 0.05)';
  for (let x = 0; x < W; x += minor) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y < H; y += minor) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  ctx.strokeStyle = 'rgba(239, 68, 68, 0.12)';
  for (let x = 0; x < W; x += minor * 5) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
  for (let y = 0; y < H; y += minor * 5) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
}

function renderLegend() {
  dom.legend.innerHTML = '';
  state.leads.forEach((lead, i) => {
    const li = document.createElement('li');
    const color = LEAD_COLORS[i % LEAD_COLORS.length];
    li.innerHTML = `<span class="dot" style="background:${color}"></span>${lead}`;
    dom.legend.appendChild(li);
  });
}

/* ============================================================
 *  Predict
 * ============================================================ */
async function predictCurrent() {
  if (!state.currentSample) {
    toast('Aucun ECG chargé.', 'warn');
    return;
  }
  setBtnLoading(dom.predictBtn, true);
  try {
    const t0 = performance.now();
    const res = await api('/predict', {
      method: 'POST',
      body: JSON.stringify({
        signal: state.currentSample.signal,
        sampling_rate: state.currentSample.sampling_rate || 100,
      }),
    });
    res._roundTripMs = performance.now() - t0;
    state.currentResult = res;
    renderResult(res);
    pushHistory(res);
  } catch (err) {
    toast(`Échec de la prédiction : ${err.message}`, 'error', 5500);
    console.error(err);
  } finally {
    setBtnLoading(dom.predictBtn, false);
  }
}

function setBtnLoading(btn, loading) {
  btn.classList.toggle('loading', loading);
  btn.disabled = loading;
}

function resetResultPanel() {
  dom.resultPanel.dataset.class = '';
  dom.resultBody.classList.add('hidden');
  dom.resultEmpty.classList.remove('hidden');
  dom.resultTimestamp.textContent = '—';
  dom.metaLatency.textContent = '—';
}

function renderResult(res) {
  dom.resultEmpty.classList.add('hidden');
  dom.resultBody.classList.remove('hidden');

  dom.resultPanel.dataset.class = res.label;
  dom.resultLabel.textContent = res.label;
  dom.resultTimestamp.textContent = formatTime(res.timestamp);
  dom.metaLatency.textContent = `${res.latency_ms.toFixed(1)} ms`;

  // True-vs-predicted badge
  const trueLbl = state.currentSample && state.currentSample.true_label;
  dom.resultMatch.className = 'result-match';
  if (trueLbl) {
    const ok = trueLbl === res.label;
    dom.resultMatch.classList.add(ok ? 'match-ok' : 'match-wrong');
    dom.resultMatch.innerHTML = ok
      ? `✓ correspond au label réel`
      : `✗ label réel : <strong>${escapeHtml(trueLbl)}</strong>`;
  } else {
    dom.resultMatch.innerHTML = '';
  }

  // Risk bar
  const risk = clamp01(res.risk_score);
  dom.riskFill.style.width = (risk * 100).toFixed(1) + '%';
  dom.riskValue.textContent = risk.toFixed(3);

  // Confidence ring (top class probability)
  const topProb = res.probabilities[res.label] ?? 0;
  const circumference = 176; // 2*pi*28
  dom.ringFill.style.strokeDashoffset = (circumference * (1 - topProb)).toFixed(1);
  dom.ringLabel.textContent = `${Math.round(topProb * 100)}%`;

  // Per-class bars
  setProb('Normal', res.probabilities.normal ?? 0);
  setProb('Suspect', res.probabilities.suspect ?? 0);
  setProb('Critique', res.probabilities.critique ?? 0);
}

function setProb(name, val) {
  const pct = (val * 100);
  dom['prob' + name].style.width = pct.toFixed(1) + '%';
  dom['prob' + name + 'Val'].textContent = pct.toFixed(1) + '%';
}

function clamp01(v) { return Math.max(0, Math.min(1, Number(v) || 0)); }

function formatTime(iso) {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour12: false });
  } catch { return iso; }
}

/* ============================================================
 *  History (last N predictions, persisted)
 * ============================================================ */
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]'); }
  catch { return []; }
}
function saveHistory() { localStorage.setItem(HISTORY_KEY, JSON.stringify(state.history)); }

function pushHistory(res) {
  const s = state.currentSample || {};
  state.history.unshift({
    ecg_id: s.ecg_id,
    filename: s.filename_lr || s.source || 'import',
    label: res.label,
    risk: res.risk_score,
    true_label: s.true_label || null,
    timestamp: res.timestamp,
  });
  state.history = state.history.slice(0, HISTORY_MAX);
  saveHistory();
  renderHistory();
}

function renderHistory() {
  dom.historyList.innerHTML = '';
  if (state.history.length === 0) {
    dom.historyEmpty.classList.remove('hidden');
    return;
  }
  dom.historyEmpty.classList.add('hidden');
  for (const h of state.history) {
    const li = document.createElement('li');
    li.className = 'hist-item';
    li.title = `Rouvrir #${h.ecg_id}`;
    li.innerHTML = `
      <span class="hist-time">${formatTime(h.timestamp)}</span>
      <span class="hist-id">${h.ecg_id ? '#' + h.ecg_id : 'import'}</span>
      <span class="hist-tag ${h.label}">${h.label}</span>
    `;
    if (h.ecg_id) li.addEventListener('click', () => loadSample(h.ecg_id));
    dom.historyList.appendChild(li);
  }
}

function clearHistory() {
  state.history = [];
  saveHistory();
  renderHistory();
  toast('Historique vidé.', 'info', 2000);
}

/* ============================================================
 *  Feature form — manual clinical measurement entry
 * ============================================================ */
const FEATURE_FIELDS = [
  'hr_mean', 'hr_std', 'sdnn', 'rmssd', 'pnn50',
  'qrs_width_mean', 'qrs_width_std',
  'pr_interval_mean', 'qt_interval_mean',
  'st_level_mean', 't_amplitude_mean', 'n_beats',
];

async function predictFromFeatures(e) {
  e.preventDefault();
  const data = {};
  for (const name of FEATURE_FIELDS) {
    const input = dom.featureForm.elements[name];
    const val = parseFloat(input.value);
    if (input.value.trim() === '' || !isFinite(val)) {
      toast(`Champ manquant ou invalide : ${name}`, 'warn', 4000);
      input.focus();
      return;
    }
    data[name] = val;
  }
  setBtnLoading(dom.featureSubmitBtn, true);
  try {
    const res = await api('/predict_features', {
      method: 'POST',
      body: JSON.stringify(data),
    });
    state.currentResult = res;
    state.currentSample = { source: 'saisie manuelle' };
    renderResult(res);
    pushHistory(res);
    closeFeatureFormDialog();
  } catch (err) {
    toast(`Échec de la prédiction : ${err.message}`, 'error', 5500);
    console.error(err);
  } finally {
    setBtnLoading(dom.featureSubmitBtn, false);
  }
}

function openFeatureFormDialog() {
  if (typeof dom.featureDialog.showModal === 'function') {
    dom.featureDialog.showModal();
  } else {
    dom.featureDialog.setAttribute('open', '');
  }
}

function closeFeatureFormDialog() {
  if (typeof dom.featureDialog.close === 'function') dom.featureDialog.close();
  else dom.featureDialog.removeAttribute('open');
}

/* ============================================================
 *  Settings dialog
 * ============================================================ */
function openSettings() {
  dom.apiUrlInput.value = state.settings.apiUrl;
  dom.autoPredict.checked = state.settings.autoPredict;
  dom.ecgGrid.checked = state.settings.ecgGrid;
  if (typeof dom.settingsDialog.showModal === 'function') {
    dom.settingsDialog.showModal();
  } else {
    dom.settingsDialog.setAttribute('open', '');
  }
}

function closeSettings() {
  if (typeof dom.settingsDialog.close === 'function') dom.settingsDialog.close();
  else dom.settingsDialog.removeAttribute('open');
}

async function commitSettings() {
  const prev = { ...state.settings };
  state.settings.apiUrl = dom.apiUrlInput.value.trim();
  state.settings.autoPredict = dom.autoPredict.checked;
  state.settings.ecgGrid = dom.ecgGrid.checked;
  saveSettings();
  if (prev.apiUrl !== state.settings.apiUrl) {
    await loadHealth();
    await loadSamples();
  } else if (state.currentSample) {
    drawECG(state.currentSample.signal);
  }
  toast('Paramètres enregistrés.', 'success', 2200);
}

function resetSettings() {
  state.settings = { ...DEFAULT_SETTINGS };
  saveSettings();
  dom.apiUrlInput.value = state.settings.apiUrl;
  dom.autoPredict.checked = state.settings.autoPredict;
  dom.ecgGrid.checked = state.settings.ecgGrid;
  toast('Paramètres réinitialisés.', 'info', 2000);
}

/* ============================================================
 *  Event wiring
 * ============================================================ */
function attachEvents() {
  dom.reloadSamples.addEventListener('click', () => loadSamples());
  dom.predictBtn.addEventListener('click', () => predictCurrent());
  dom.clearHistory.addEventListener('click', clearHistory);
  dom.settingsBtn.addEventListener('click', openSettings);
  dom.closeSettings.addEventListener('click', closeSettings);
  dom.resetSettings.addEventListener('click', resetSettings);
  dom.settingsDialog.addEventListener('close', () => {
    if (dom.settingsDialog.returnValue === 'default') return;
    commitSettings();
  });

  // Filter chips
  for (const chip of dom.filterChips) {
    chip.addEventListener('click', () => {
      for (const c of dom.filterChips) c.classList.remove('active');
      chip.classList.add('active');
      state.filter = chip.dataset.filter;
      renderSampleList();
    });
  }
  document.querySelector('.chip[data-filter="all"]').classList.add('active');

  // Feature form modal
  dom.openFeatureFormBtn.addEventListener('click', openFeatureFormDialog);
  dom.closeFeatureForm.addEventListener('click', closeFeatureFormDialog);
  dom.cancelFeatureForm.addEventListener('click', closeFeatureFormDialog);
  dom.featureForm.addEventListener('submit', predictFromFeatures);

  // Keyboard shortcuts
  document.addEventListener('keydown', e => {
    if (e.target.matches('input, textarea, [contenteditable="true"]')) return;
    if (dom.settingsDialog.open || dom.featureDialog.open) return;
    if (e.key === ' ' && !dom.predictBtn.disabled) {
      e.preventDefault(); predictCurrent();
    } else if (e.key === 'r' || e.key === 'R') {
      loadSamples();
    } else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      navigateSamples(e.key === 'ArrowRight' ? 1 : -1);
    }
  });

  // Resize handling
  let resizeRaf = null;
  window.addEventListener('resize', () => {
    if (resizeRaf) cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(resizeCanvas);
  });
}

function navigateSamples(dir) {
  const visible = Array.from(dom.sampleList.querySelectorAll('.sample-item'));
  if (visible.length === 0) return;
  const activeIdx = visible.findIndex(el => el.classList.contains('active'));
  let next = activeIdx + dir;
  if (next < 0) next = visible.length - 1;
  if (next >= visible.length) next = 0;
  loadSample(Number(visible[next].dataset.ecgId));
}

/* ============================================================
 *  Init
 * ============================================================ */
(async function init() {
  attachEvents();
  renderHistory();
  resizeCanvas();
  await loadHealth();
  await loadSamples();
})();
