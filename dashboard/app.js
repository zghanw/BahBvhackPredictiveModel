/* app.js — Dashboard logic */

// Derive API base from current page location so it works on any host/port.
// When opened as a file (file://) fall back to localhost:8000.
const API = (() => {
  if (location.protocol === 'file:') return 'http://localhost:8000';
  return `${location.protocol}//${location.hostname}:${location.port || 8000}`;
})();
const WS_BASE = API.replace(/^http/, 'ws');

let activeEngineId = null;
let activeSubset   = 'FD001';
let wsConn         = null;
let rulChart       = null;
let attnChart      = null;
let fleetData      = [];
let alertLog       = [];
let lastStatus     = null;

// Values fetched from /api/model/metadata — no hardcoding
let anomalyThreshold = 30;   // updated on loadMeta()
let rulCap           = 125;  // updated on loadMeta()

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  checkApi();
  loadFleet();

  document.getElementById('subsetSelect').addEventListener('change', e => {
    activeSubset = e.target.value;
    activeEngineId = null;
    lastStatus = null;
    resetDetail();
    loadFleet();
    loadMeta();
  });

  document.getElementById('fleetSearch').addEventListener('input', e => {
    renderFleet(e.target.value.trim());
  });

  document.getElementById('btnSimulate').addEventListener('click', startSimulation);
  document.getElementById('btnStop').addEventListener('click', stopSimulation);
});

// ── API health ────────────────────────────────────────────────────────────────
async function checkApi() {
  const badge = document.getElementById('apiStatus');
  try {
    const r = await fetch(`${API}/`);
    if (r.ok) {
      badge.textContent = '● Connected';
      badge.className = 'api-badge api-ok';
    } else throw new Error();
  } catch {
    badge.textContent = '● API Offline';
    badge.className = 'api-badge api-error';
  }
}

// ── Model metadata ────────────────────────────────────────────────────────────
async function loadMeta() {
  try {
    const r = await fetch(`${API}/api/model/metadata?subset=${activeSubset}`);
    const d = await r.json();

    // Update module-level config from server — single source of truth
    if (d.anomaly_threshold != null) anomalyThreshold = d.anomaly_threshold;
    if (d.rul_cap           != null) rulCap           = d.rul_cap;

    document.getElementById('metaArch').textContent      = `Arch: ${d.arch?.toUpperCase() ?? '—'}`;
    document.getElementById('metaRmse').textContent      = `RMSE: ${d.rmse ?? '—'} cycles`;
    document.getElementById('metaMae').textContent       = `MAE: ${d.mae ?? '—'} cycles`;
    document.getElementById('metaWindow').textContent    = `Window: ${d.window_size ?? '—'} cycles`;
    document.getElementById('metaFeatures').textContent  = `Features: ${d.feature_count ?? '—'}`;
    document.getElementById('metaThreshold').textContent = `Threshold: ${anomalyThreshold} cycles`;

    // Warn if the loaded checkpoint wasn't trained on this subset
    const archEl = document.getElementById('metaArch');
    if (d.checkpoint_exists && !d.checkpoint_subset_specific) {
      archEl.textContent += ' ⚠ generic checkpoint';
      archEl.style.color = 'var(--yellow)';
      addAlert('impaired', `No checkpoint trained for ${d.subset} — using generic model. Predictions may be inaccurate. Train with: python -m src.train --subset ${d.subset}`);
    } else {
      archEl.style.color = '';
    }

    // Update chart threshold label now that we have the real value
    if (rulChart) {
      rulChart.data.datasets[1].label = `Anomaly Threshold (${anomalyThreshold})`;
      rulChart.update('none');
    }
  } catch { /* silent — badge already shows offline */ }
}

// ── Fleet ─────────────────────────────────────────────────────────────────────
async function loadFleet() {
  document.getElementById('fleetList').innerHTML = '<div class="fleet-loading">Loading fleet…</div>';
  try {
    const r = await fetch(`${API}/api/fleet?subset=${activeSubset}&max_engines=50`);
    const d = await r.json();
    fleetData = d.engines ?? [];
    document.getElementById('engineCount').textContent = `${fleetData.length} engines`;
    renderFleet('');
    loadMeta();
  } catch {
    document.getElementById('fleetList').innerHTML =
      '<div class="fleet-loading" style="color:#ef4444">Failed to load fleet. Is the API running?</div>';
  }
}

function renderFleet(filter) {
  const list = document.getElementById('fleetList');
  const items = filter
    ? fleetData.filter(e => String(e.engine_id).includes(filter))
    : fleetData;

  if (!items.length) {
    list.innerHTML = '<div class="fleet-loading">No engines found.</div>';
    return;
  }

  list.innerHTML = items.map(e => {
    const cls    = rulClass(e.predicted_rul);
    const active = e.engine_id === activeEngineId ? ' active' : '';
    return `
      <div class="fleet-item${active}" onclick="selectEngine(${e.engine_id})">
        <div class="fleet-item-left">
          <span class="fleet-engine-id">Engine #${e.engine_id}</span>
          <span class="fleet-cycle">Cycle ${e.last_cycle}</span>
        </div>
        <span class="fleet-rul ${cls}">${e.predicted_rul} cyc</span>
      </div>`;
  }).join('');
}

function rulClass(rul) {
  if (rul <= anomalyThreshold)       return 'impaired';
  if (rul <= anomalyThreshold * 2)   return 'warning';
  return 'healthy';
}

// ── Engine selection ──────────────────────────────────────────────────────────
async function selectEngine(id) {
  stopSimulation();
  activeEngineId = id;
  lastStatus = null;
  alertLog = [];
  renderFleet(document.getElementById('fleetSearch').value.trim());

  document.getElementById('detailTitle').textContent    = `Engine #${id}`;
  document.getElementById('detailSubtitle').textContent = `${activeSubset} · Loading full history…`;
  document.getElementById('btnSimulate').disabled = false;

  // Clear chart immediately so stale data from the previous engine doesn't linger
  rulChart.data.labels = [];
  rulChart.data.datasets.forEach(d => d.data = []);
  rulChart.update('none');
  attnChart.data.labels = [];
  attnChart.data.datasets[0].data = [];
  attnChart.update('none');

  try {
    const r = await fetch(`${API}/api/predict/${id}?subset=${activeSubset}`);
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    renderHistory(d.cycles);
    document.getElementById('detailSubtitle').textContent =
      `${activeSubset} · ${d.arch?.toUpperCase()} · ${d.cycles.length} cycles loaded`;
    addAlert('info', `Engine #${id} history loaded (${d.cycles.length} cycles)`);
  } catch (e) {
    document.getElementById('detailSubtitle').textContent = `Error: ${e.message}`;
  }
}

// ── Render full history ───────────────────────────────────────────────────────
function renderHistory(cycles) {
  if (!cycles.length) return;
  const last = cycles[cycles.length - 1];
  updateKpis(last, cycles);
  updateRulChart(cycles);
  updateAttnChart(last.attention_weights ?? []);
  updateSensors(last.attention_weights ?? [], last.top_sensors ?? []);
}

// ── KPI Cards ─────────────────────────────────────────────────────────────────
function updateKpis(record, allCycles) {
  const rul    = record.predicted_rul;
  const status = record.status;
  const cycle  = record.cycle;
  const pct    = Math.round((1 - rul / rulCap) * 100);

  document.querySelector('#kpiRul .kpi-value').textContent      = rul.toFixed(1);
  document.querySelector('#kpiCycle .kpi-value').textContent    = cycle;
  document.querySelector('#kpiProgress .kpi-value').textContent = `${Math.min(pct, 100)}%`;

  const statusEl = document.getElementById('kpiStatus');
  statusEl.querySelector('.kpi-value').textContent = status;
  statusEl.className = `kpi-card ${status.toLowerCase()}`;

  if (status !== lastStatus) {
    if (lastStatus !== null) {
      addAlert(status.toLowerCase(), `Engine #${activeEngineId} → ${status.toUpperCase()} at cycle ${cycle}`);
    }
    lastStatus = status;
  }
}

// ── RUL Chart ─────────────────────────────────────────────────────────────────
function initCharts() {
  const rulCtx  = document.getElementById('rulChart').getContext('2d');
  const attnCtx = document.getElementById('attnChart').getContext('2d');

  rulChart = new Chart(rulCtx, {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Predicted RUL', data: [], borderColor: '#3b82f6',
        backgroundColor: 'rgba(59,130,246,.08)', borderWidth: 2,
        pointRadius: 0, pointHoverRadius: 5, pointHoverBackgroundColor: '#3b82f6',
        tension: 0.3, fill: true },
      { label: `Anomaly Threshold (${anomalyThreshold})`, data: [],
        borderColor: '#ef4444', borderWidth: 1.5, borderDash: [6, 4],
        pointRadius: 0, pointHoverRadius: 0, fill: false },
    ]},
    options: {
      responsive: true, maintainAspectRatio: true,
      animation: { duration: 200 },
      interaction: {
        mode: 'index',        // snap tooltip to nearest x index across all datasets
        intersect: false,     // trigger even when not directly over a point
      },
      plugins: {
        legend: { labels: { color: '#8892b0', font: { size: 11 } } },
        tooltip: {
          backgroundColor: '#1a1d27',
          borderColor: '#2e3250',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#8892b0',
          padding: 10,
          callbacks: {
            title: items => `Cycle ${items[0].label}`,
            label: item => {
              if (item.datasetIndex === 1) return null; // hide threshold line from tooltip
              const rul = item.parsed.y;
              const status = rul <= anomalyThreshold ? '🔴 Impaired' : '🟢 Healthy';
              return [`  RUL: ${rul.toFixed(1)} cycles`, `  Status: ${status}`];
            },
          },
        },
        // Vertical crosshair line drawn via a custom inline plugin
        crosshairLine: {},
      },
      scales: {
        x: { ticks: { color: '#8892b0', maxTicksLimit: 12 }, grid: { color: '#2e3250' } },
        y: { ticks: { color: '#8892b0' }, grid: { color: '#2e3250' },
             title: { display: true, text: 'RUL (cycles)', color: '#8892b0', font: { size: 11 } } },
      },
    },
    plugins: [{
      // Inline plugin: draws a vertical line at the hovered x position
      id: 'crosshairLine',
      afterDraw(chart) {
        if (!chart.tooltip?._active?.length) return;
        const ctx = chart.ctx;
        const x   = chart.tooltip._active[0].element.x;
        const top = chart.chartArea.top;
        const bot = chart.chartArea.bottom;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bot);
        ctx.lineWidth   = 1;
        ctx.strokeStyle = 'rgba(226,232,240,0.25)';
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.restore();
      },
    }],
  });

  attnChart = new Chart(attnCtx, {
    type: 'bar',
    data: { labels: [], datasets: [{
      label: 'Attention Weight', data: [],
      backgroundColor: 'rgba(59,130,246,.6)', borderColor: '#3b82f6', borderWidth: 1,
    }]},
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 150 },
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#8892b0', font: { size: 10 }, maxTicksLimit: 10 }, grid: { color: '#2e3250' } },
        y: { ticks: { color: '#8892b0', font: { size: 10 } }, grid: { color: '#2e3250' } },
      },
    },
  });
}

function updateRulChart(cycles) {
  // Destroy and replace data arrays entirely so Chart.js always re-renders
  rulChart.data.labels           = [];
  rulChart.data.datasets[0].data = [];
  rulChart.data.datasets[1].data = [];
  rulChart.update('none');

  rulChart.data.labels           = cycles.map(c => c.cycle);
  rulChart.data.datasets[0].data = cycles.map(c => c.predicted_rul);
  rulChart.data.datasets[1].data = cycles.map(() => anomalyThreshold);
  rulChart.update();
}

function updateAttnChart(weights) {
  if (!weights || !weights.length) {
    attnChart.data.labels = [];
    attnChart.data.datasets[0].data = [];
    attnChart.update();
    return;
  }
  attnChart.data.labels = weights.map((_, i) => `t-${weights.length - i}`);
  attnChart.data.datasets[0].data = weights;
  attnChart.update();
}

// ── Sensor Insights ───────────────────────────────────────────────────────────
const SENSOR_LABELS = {
  sensor_02: 'LPC Outlet Temp',       sensor_03: 'HPC Outlet Temp',
  sensor_04: 'LPT Outlet Temp',       sensor_06: 'Total HPC Outlet Pressure',
  sensor_07: 'Fan Inlet Pressure',    sensor_08: 'Bypass Duct Pressure',
  sensor_09: 'HPC Outlet Pressure',   sensor_11: 'HPC Outlet Temp (core)',
  sensor_12: 'LPT Outlet Pressure',   sensor_13: 'Corrected Fan Speed',
  sensor_14: 'Corrected Core Speed',  sensor_15: 'BPR',
  sensor_17: 'Bleed Enthalpy',        sensor_20: 'HPT Coolant Bleed',
  sensor_21: 'LPT Coolant Bleed',
};

/**
 * Render sensor importance bars.
 * Uses top_sensors list from the API (computed server-side from attention weights).
 * Falls back to showing the list without weights if attention is unavailable.
 */
function updateSensors(attnWeights, topSensors) {
  const el = document.getElementById('sensorList');

  if (!topSensors || !topSensors.length) {
    el.innerHTML = '<div class="sensor-placeholder">No sensor data available</div>';
    return;
  }

  // Assign uniform bars when we have no attention weights (model without attention layer)
  const hasAttn = attnWeights && attnWeights.length > 0;
  const count   = topSensors.length;

  el.innerHTML = topSensors.map((sensorId, i) => {
    // Descending importance: first sensor gets 100%, rest scale down proportionally
    const pct   = hasAttn ? Math.round(100 * (count - i) / count) : Math.round(100 / (i + 1));
    const label = SENSOR_LABELS[sensorId] ?? sensorId;
    return `
      <div class="sensor-item">
        <span style="min-width:80px;color:#e2e8f0">${sensorId}</span>
        <div class="sensor-bar-wrap"><div class="sensor-bar" style="width:${pct}%"></div></div>
        <span class="sensor-pct">${pct}%</span>
      </div>
      <div style="font-size:10px;color:#8892b0;padding:0 4px 4px">${label}</div>`;
  }).join('');
}

// ── Alert Log ─────────────────────────────────────────────────────────────────
function addAlert(type, msg) {
  alertLog.unshift({ type, msg, time: new Date().toLocaleTimeString() });
  if (alertLog.length > 50) alertLog.pop();
  renderAlerts();
}

function renderAlerts() {
  const el = document.getElementById('alertLog');
  if (!alertLog.length) {
    el.innerHTML = '<div class="alert-empty">No alerts yet.</div>';
    return;
  }
  el.innerHTML = alertLog.map(a =>
    `<div class="alert-item ${a.type}"><strong>${a.time}</strong> — ${a.msg}</div>`
  ).join('');
}

// ── WebSocket Simulation ──────────────────────────────────────────────────────
function startSimulation() {
  if (!activeEngineId) return;
  stopSimulation();

  document.getElementById('btnSimulate').disabled = true;
  document.getElementById('btnStop').disabled = false;
  addAlert('info', `Live simulation started for Engine #${activeEngineId}`);

  // Reset chart for live mode
  rulChart.data.labels = [];
  rulChart.data.datasets.forEach(d => d.data = []);
  rulChart.update();

  const wsUrl = `${WS_BASE}/ws/simulate/${activeEngineId}?subset=${activeSubset}&delay=0.2`;
  wsConn = new WebSocket(wsUrl);

  wsConn.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.done)  { stopSimulation(); addAlert('info', 'Simulation complete.'); return; }
    if (d.error) { addAlert('impaired', d.error); stopSimulation(); return; }

    rulChart.data.labels.push(d.cycle);
    rulChart.data.datasets[0].data.push(d.predicted_rul);
    rulChart.data.datasets[1].data.push(anomalyThreshold);
    rulChart.update('none');

    const syntheticCycles = rulChart.data.labels.map((cyc, i) => ({
      cycle: cyc, predicted_rul: rulChart.data.datasets[0].data[i],
    }));
    updateKpis(d, syntheticCycles);
    updateAttnChart(d.attention_weights ?? []);
    updateSensors(d.attention_weights ?? [], d.top_sensors ?? []);
  };

  wsConn.onerror = () => addAlert('impaired', 'WebSocket error — check API server.');
  wsConn.onclose = () => {
    document.getElementById('btnSimulate').disabled = false;
    document.getElementById('btnStop').disabled = true;
  };
}

function stopSimulation() {
  if (wsConn) { wsConn.close(); wsConn = null; }
  document.getElementById('btnSimulate').disabled = activeEngineId === null;
  document.getElementById('btnStop').disabled = true;
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetDetail() {
  document.getElementById('detailTitle').textContent    = 'Select an engine';
  document.getElementById('detailSubtitle').textContent = 'Click any engine in the fleet list to inspect it';
  document.getElementById('btnSimulate').disabled = true;
  document.getElementById('btnStop').disabled = true;
  document.querySelector('#kpiRul .kpi-value').textContent      = '—';
  document.querySelector('#kpiCycle .kpi-value').textContent    = '—';
  document.querySelector('#kpiProgress .kpi-value').textContent = '—';
  const statusEl = document.getElementById('kpiStatus');
  statusEl.querySelector('.kpi-value').textContent = '—';
  statusEl.className = 'kpi-card';
  rulChart.data.labels = [];
  rulChart.data.datasets.forEach(d => d.data = []);
  rulChart.update();
  attnChart.data.labels = [];
  attnChart.data.datasets[0].data = [];
  attnChart.update();
  document.getElementById('sensorList').innerHTML =
    '<div class="sensor-placeholder">Run inference to see sensor insights</div>';
  document.getElementById('alertLog').innerHTML = '';
  alertLog = [];
}
