import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_DATA_FILE = "viz_data_human.json"
FALLBACK_DATA_FILE = "viz_data.json"

out_file = sys.argv[2] if len(sys.argv) > 2 else "viz.html"
data_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_FILE
input_path = DATA_DIR / data_file
if not input_path.exists() and data_file == DEFAULT_DATA_FILE:
    input_path = DATA_DIR / FALLBACK_DATA_FILE
if not input_path.exists():
    raise FileNotFoundError(f"Could not find visualization data file: {input_path}")

with input_path.open(encoding="utf-8") as f:
    data = json.load(f)

# Hand-picked categorical palette - distinct hues, tuned for visibility on a
# near-black canvas rather than a default d3-category scheme.
PALETTE = [
    "#5ec8f8", "#f2795a", "#7ee081", "#e6c65c", "#c084f5",
    "#f06fa0", "#4fd6c4", "#f5a35c", "#8a9bff", "#b6e05a",
    "#e85c6b", "#5cc9a7", "#d9a4f5", "#f5e05c", "#6fa8f0",
    "#f58c6f", "#a3e05c", "#5cf5d0",
]
domain_color = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(data["domains"])}

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EPR Clause Similarity Explorer</title>
<style>
  :root {
    --bg: #0b0d12;
    --panel-bg: #12151c;
    --border: #262b36;
    --text: #dde3ee;
    --text-dim: #7c869c;
    --accent: #5ec8f8;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; width: 100%; height: 100%;
    background: var(--bg); color: var(--text);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    overflow: hidden;
  }
  #app { display: flex; width: 100%; height: 100%; }
  #canvas-wrap { position: relative; flex: 1; min-width: 0; }
  canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }

  #legend {
    position: absolute; top: 16px; left: 16px;
    background: rgba(18,21,28,0.88);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    max-height: calc(100% - 32px);
    overflow-y: auto;
    font-size: 12px;
    max-width: 260px;
  }
  #legend h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text-dim); margin: 0 0 8px 0; font-weight: 600;
  }
  .legend-row {
    display: flex; align-items: center; gap: 7px;
    padding: 3px 4px; border-radius: 4px; cursor: pointer;
    user-select: none;
  }
  .legend-row:hover { background: rgba(255,255,255,0.06); }
  .legend-row.dimmed { opacity: 0.35; }
  .swatch { width: 9px; height: 9px; border-radius: 2px; flex-shrink: 0; }
  .legend-count { color: var(--text-dim); margin-left: auto; font-variant-numeric: tabular-nums; }

  #hint {
    position: absolute; bottom: 14px; left: 16px;
    font-size: 11px; color: var(--text-dim);
    background: rgba(18,21,28,0.8); padding: 5px 9px; border-radius: 6px;
    border: 1px solid var(--border);
  }

  #panel {
    width: 380px; flex-shrink: 0;
    background: var(--panel-bg);
    border-left: 1px solid var(--border);
    padding: 20px;
    overflow-y: auto;
  }
  #panel .empty { color: var(--text-dim); font-size: 13px; margin-top: 40px; text-align: center; }
  .field-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.07em;
    color: var(--text-dim); margin: 16px 0 4px 0; font-weight: 600;
  }
  .field-label:first-child { margin-top: 0; }
  #formula-box {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 12.5px; line-height: 1.5; color: var(--accent);
    background: #0e1016; border: 1px solid var(--border); border-radius: 6px;
    padding: 10px 12px; word-break: break-word; white-space: pre-wrap;
  }
  .meta-row { font-size: 12.5px; color: var(--text); display: flex; justify-content: space-between; padding: 2px 0; }
  .meta-row span:first-child { color: var(--text-dim); }
  .neighbor {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 11.5px; padding: 7px 9px; margin-bottom: 5px;
    border: 1px solid var(--border); border-radius: 6px;
    cursor: pointer; color: var(--text);
    display: flex; justify-content: space-between; gap: 8px;
  }
  .neighbor:hover { border-color: var(--accent); }
  .neighbor .sim { color: var(--text-dim); flex-shrink: 0; }
  .neighbor .txt { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  h1 { font-size: 14px; font-weight: 600; margin: 0 0 2px 0; }
  #panel-sub { font-size: 11.5px; color: var(--text-dim); margin-bottom: 4px; }
</style>
</head>
<body>
<div id="app">
  <div id="canvas-wrap">
    <canvas id="c"></canvas>
    <div id="legend"><h2>Domain (click to isolate)</h2><div id="legend-rows"></div></div>
    <div id="hint">click a point &middot; scroll to zoom &middot; drag to pan</div>
  </div>
  <div id="panel"><div class="empty">Click a point to inspect its formula<br>and nearest neighbors.</div></div>
</div>

<script>
const DATA = __DATA_JSON__;
const DOMAIN_COLOR = __DOMAIN_COLOR_JSON__;
const nodes = DATA.nodes;
const byId = new Map(nodes.map(n => [n.id, n]));

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let W, H, dpr;
function resize() {
  dpr = window.devicePixelRatio || 1;
  W = canvas.clientWidth; H = canvas.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}
window.addEventListener('resize', resize);

// view transform: pan/zoom
let scale = Math.min(W || 800, H || 800) * 0.42;
let offX = 0, offY = 0;
let userScale = 1, panX = 0, panY = 0;

function toScreen(x, y) {
  return [
    W/2 + panX + (x * scale * userScale),
    H/2 + panY - (y * scale * userScale),
  ];
}

let selected = null;
let hovered = null;
let activeDomains = new Set(DATA.domains);

function draw() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0b0d12';
  ctx.fillRect(0, 0, W, H);

  // edges for selected node
  if (selected !== null) {
    const n = byId.get(selected);
    ctx.lineWidth = 1;
    for (const nb of n.neighbors) {
      const other = byId.get(nb.id);
      const [x1, y1] = toScreen(n.x, n.y);
      const [x2, y2] = toScreen(other.x, other.y);
      ctx.strokeStyle = `rgba(94, 200, 248, ${0.25 + nb.sim*0.5})`;
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
  }

  for (const n of nodes) {
    const dimmed = !activeDomains.has(n.domain);
    const [x, y] = toScreen(n.x, n.y);
    if (x < -10 || x > W+10 || y < -10 || y > H+10) continue;
    const isSel = n.id === selected;
    const isNb = selected !== null && byId.get(selected).neighbors.some(nb => nb.id === n.id);
    const r = isSel ? 5.5 : (isNb ? 4 : 2.2);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI*2);
    ctx.fillStyle = dimmed ? 'rgba(255,255,255,0.06)' : DOMAIN_COLOR[n.domain];
    ctx.globalAlpha = dimmed ? 1 : (isSel || isNb ? 1 : 0.75);
    ctx.fill();
    if (isSel) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
}

function nearestNode(mx, my) {
  let best = null, bestD = 14*14; // px hit radius
  for (const n of nodes) {
    if (!activeDomains.has(n.domain)) continue;
    const [x, y] = toScreen(n.x, n.y);
    const d = (x-mx)**2 + (y-my)**2;
    if (d < bestD) { bestD = d; best = n; }
  }
  return best;
}

function renderPanel(n) {
  const panel = document.getElementById('panel');
  if (!n) { panel.innerHTML = '<div class="empty">Click a point to inspect its formula<br>and nearest neighbors.</div>'; return; }
  const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  let html = `<h1>${esc(n.clause_id)}</h1>`;
  html += `<div id="panel-sub">${esc(n.domain)}</div>`;
  html += `<div class="field-label">Formula</div><div id="formula-box">${esc(n.formula)}</div>`;
  html += `<div class="field-label">Details</div>`;
  html += `<div class="meta-row"><span>Problem</span><span>${esc(n.problem_id)}</span></div>`;
  html += `<div class="meta-row"><span>Status</span><span>${esc(n.status)}</span></div>`;
  html += `<div class="meta-row"><span>Role</span><span>${esc(n.role)}</span></div>`;
  html += `<div class="field-label">Nearest neighbors</div>`;
  for (const nb of n.neighbors) {
    const other = byId.get(nb.id);
    html += `<div class="neighbor" data-id="${other.id}">`
          +   `<span class="txt">${esc(other.formula)}</span>`
          +   `<span class="sim">${nb.sim.toFixed(3)}</span>`
          + `</div>`;
  }
  panel.innerHTML = html;
  panel.querySelectorAll('.neighbor').forEach(el => {
    el.addEventListener('click', () => selectNode(parseInt(el.dataset.id)));
  });
}

function selectNode(id) {
  selected = id;
  renderPanel(byId.get(id));
  draw();
}

canvas.addEventListener('click', (e) => {
  const rect = canvas.getBoundingClientRect();
  const n = nearestNode(e.clientX - rect.left, e.clientY - rect.top);
  if (n) selectNode(n.id);
});

canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.08 : 1/1.08;
  userScale = Math.max(0.3, Math.min(20, userScale * factor));
  draw();
}, { passive: false });

let dragging = false, lastX, lastY;
canvas.addEventListener('mousedown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  panX += e.clientX - lastX; panY += e.clientY - lastY;
  lastX = e.clientX; lastY = e.clientY;
  draw();
});

// legend
const legendRows = document.getElementById('legend-rows');
const counts = {};
for (const n of nodes) counts[n.domain] = (counts[n.domain]||0) + 1;
for (const d of DATA.domains) {
  const row = document.createElement('div');
  row.className = 'legend-row';
  row.innerHTML = `<div class="swatch" style="background:${DOMAIN_COLOR[d]}"></div>`
                + `<span>${d}</span><span class="legend-count">${counts[d]||0}</span>`;
  row.addEventListener('click', () => {
    if (activeDomains.has(d) && activeDomains.size === 1) {
      activeDomains = new Set(DATA.domains); // clicking the lone isolated domain resets
    } else {
      activeDomains = new Set([d]); // isolate this one domain
    }
    updateLegendDim();
    draw();
  });
  row.addEventListener('dblclick', (e) => { e.stopPropagation(); activeDomains = new Set(DATA.domains); updateLegendDim(); draw(); });
  legendRows.appendChild(row);
}
function updateLegendDim() {
  [...legendRows.children].forEach((row, i) => {
    row.classList.toggle('dimmed', !activeDomains.has(DATA.domains[i]));
  });
}

resize();
</script>
</body>
</html>
"""

html = HTML.replace("__DATA_JSON__", json.dumps(data).replace("</", "<\\/"))
html = html.replace("__DOMAIN_COLOR_JSON__", json.dumps(domain_color).replace("</", "<\\/"))

out_path = Path(out_file)
if not out_path.is_absolute():
    out_path = REPO_ROOT / out_path
with out_path.open("w", encoding="utf-8") as f:
    f.write(html)

print(f"Wrote {out_path} ({len(html)/1e6:.2f} MB)")