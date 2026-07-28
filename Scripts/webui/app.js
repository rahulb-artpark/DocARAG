"use strict";

/* ============================ state & helpers ============================ */
const state = {
  tree: null,
  graph: null,
  graphNet: null,
  treeNet: null,
  graphBuilt: false,
  treeBuilt: false,
  graphNodesById: new Map(),
  treeNodesById: new Map(),
  entitiesBySourceNode: new Map(), // tree node id -> [entity, ...]
  chatBusy: false,
  currentRun: null,
  libraryTimer: null,
};

const LIBRARY_POLL_MS = 4000;

const PALETTE = [
  "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2",
  "#db2777", "#65a30d", "#9333ea", "#0d9488", "#ea580c", "#4f46e5",
  "#16a34a", "#0ea5e9", "#c026d3", "#ca8a04", "#e11d48", "#475569",
];
const typeColorCache = new Map();
function colorForType(type) {
  const key = String(type || "UNKNOWN");
  if (!typeColorCache.has(key)) {
    let h = 0;
    for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
    typeColorCache.set(key, PALETTE[h % PALETTE.length]);
  }
  return typeColorCache.get(key);
}

const $ = (id) => document.getElementById(id);
const escapeHtml = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clip = (s, n) => {
  s = String(s ?? "").replace(/\s+/g, " ").trim();
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
};

// Render a tree node's content: HTML tables as real tables, markdown as markdown,
// otherwise plain text. Cells that look like formulas are shown in monospace.
const TABLE_TAGS = new Set(["TABLE", "THEAD", "TBODY", "TFOOT", "TR", "TD", "TH", "CAPTION"]);

function renderTableBody(html) {
  // Parser table markup is untrusted document content. Rebuild it from a fixed
  // tag set so nothing else in it can reach the page.
  const parsed = new DOMParser().parseFromString(String(html || ""), "text/html");
  const source = parsed.body.querySelector("table");
  if (!source) return null;
  const build = (node) => {
    if (node.nodeType === Node.TEXT_NODE) return document.createTextNode(node.nodeValue);
    if (node.nodeType !== Node.ELEMENT_NODE || !TABLE_TAGS.has(node.tagName)) return null;
    const el = document.createElement(node.tagName.toLowerCase());
    for (const name of ["colspan", "rowspan"]) {
      const value = node.getAttribute(name);
      if (value && /^[1-9]\d{0,2}$/.test(value)) el.setAttribute(name, value);
    }
    for (const child of node.childNodes) {
      const built = build(child);
      if (built) el.appendChild(built);
    }
    return el;
  };
  return build(source);
}

function nodeLabel(n) {
  // A table often carries no caption of its own; its header row names it far
  // better than its node type does.
  if (n.content && n.content.trim()) return n.content;
  if (n.tableBody) {
    const table = renderTableBody(n.tableBody);
    const row = table && table.querySelector("tr");
    if (row) {
      const cells = [...row.children].map((c) => c.textContent.trim()).filter(Boolean);
      if (cells.length) return cells.join(" · ");
    }
  }
  return n.type;
}

function renderContent(n) {
  const c = (n && n.content) || "";
  if (!c.trim()) return "—";
  if (/^\s*<table/i.test(c)) {
    const table = renderTableBody(c);
    return table
      ? `<div class="tableWrap">${table.outerHTML}</div>`
      : escapeHtml(c);
  }
  const hasMath = /\$\$|\\\(|\\\[|\$[^$]+\$/.test(c);
  if (hasMath) {
    // keep $$...$$ intact on one line so MathJax finds the delimiters (no <br>)
    return `<div class="mathblock">${escapeHtml(c).replace(/\s*\n\s*/g, " ")}</div>`;
  }
  if (window.marked && (/\n\s*\|.*\|/.test(c) || (/\|.*\|.*\n?/.test(c) && c.includes("---"))))
    return marked.parse(c);
  return escapeHtml(c).replace(/\n/g, "<br>");
}

// Render any LaTeX ($...$, $$...$$) inside an element via MathJax.
function typesetMath(el) {
  if (window.MathJax && MathJax.typesetPromise) {
    MathJax.typesetClear && MathJax.typesetClear([el]);
    MathJax.typesetPromise([el]).catch((e) => console.warn("MathJax:", e));
  }
}

// Post-process rendered content: mark formula-like table cells as monospace.
function enhanceContent(root) {
  root.querySelectorAll(".prose td, .prose th").forEach((cell) => {
    const t = cell.textContent.trim();
    const words = t.split(/\s+/).length;
    const looksFormula = (/=|≈|≤|≥|→|∑|Σ|θ|∇/.test(t) || /\b\w+_[a-z0-9{]/i.test(t));
    if (t && words <= 10 && looksFormula) cell.classList.add("cellformula");
  });
}

// Click-to-zoom image lightbox
function ensureLightbox() {
  let lb = document.getElementById("lightbox");
  if (lb) return lb;
  lb = document.createElement("div");
  lb.id = "lightbox";
  lb.className = "lightbox";
  lb.innerHTML = `<span class="lbClose" title="Close (Esc)">×</span><img alt="full image" />`;
  document.body.appendChild(lb);
  lb.addEventListener("click", () => lb.classList.remove("open"));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") lb.classList.remove("open"); });
  return lb;
}
function openLightbox(src) {
  const lb = ensureLightbox();
  lb.querySelector("img").src = src;
  lb.classList.add("open");
}

// vis-network measures its container once at creation; if that happens before
// the flex/grid layout has painted, the canvas is stuck at 0x0 (blank). A
// ResizeObserver re-sizes it whenever the container actually gets dimensions
// (initial paint, tab switch from display:none, or window resize).
function observeSize(getNet, el, onFirstSize) {
  let didFirst = false;
  const ro = new ResizeObserver(() => {
    const net = getNet();
    if (!net) return;
    const w = el.clientWidth, h = el.clientHeight;
    if (w > 0 && h > 0) {
      net.setSize(w + "px", h + "px");
      net.redraw();
      if (!didFirst) { didFirst = true; if (onFirstSize) onFirstSize(); }
    }
  });
  ro.observe(el);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { /* non-json */ }
  if (!res.ok) {
    const msg = (data && (data.detail || data.error)) || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

/* ================================ boot =================================== */
document.addEventListener("DOMContentLoaded", boot);

async function boot() {
  wireTabs();
  wireGraphControls();
  wireTreeControls();
  wireChat();
  wireLibrary();
  wireResizers();
  attachNetControls("#graphView .stage", () => state.graphNet);
  attachNetControls("#treeView .stage", () => state.treeNet);
  enableWheelPan("treeNet", () => state.treeNet);
  enableWheelPan("graphNet", () => state.graphNet);
  try {
    await loadRuns();
    await loadData();
  } catch (err) {
    console.error("BookRAG load failed:", err);
    alert("Failed to load index: " + err.message);
  }
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => activateView(tab.dataset.view));
  });
}

// Make the mouse wheel PAN the view (vertical), Shift+wheel = horizontal,
// Ctrl/Cmd+wheel = zoom. Requires the network's zoomView to be off.
function enableWheelPan(containerId, getNet) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.addEventListener("wheel", (e) => {
    const net = getNet();
    if (!net) return;
    e.preventDefault();
    const scale = net.getScale() || 1;
    if (e.ctrlKey || e.metaKey) {
      net.moveTo({ scale: scale * (e.deltaY < 0 ? 1.12 : 0.89) });
      return;
    }
    const pos = net.getViewPosition();
    let dx = e.deltaX, dy = e.deltaY;
    if (e.shiftKey && dx === 0) { dx = e.deltaY; dy = 0; } // shift+wheel → horizontal
    net.moveTo({ position: { x: pos.x + dx / scale, y: pos.y + dy / scale } });
  }, { passive: false });
}

// Compact, on-theme pan/zoom/fit controls (replaces vis's default buttons).
function attachNetControls(stageSel, getNet) {
  const stage = document.querySelector(stageSel);
  if (!stage || stage.querySelector(".netCtl")) return;
  const ctl = document.createElement("div");
  ctl.className = "netCtl";
  ctl.innerHTML = `
    <button data-act="zoomin" title="Zoom in">+</button>
    <button data-act="zoomout" title="Zoom out">−</button>`;
  stage.appendChild(ctl);
  ctl.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    const net = getNet();
    if (!net) return;
    const act = b.dataset.act;
    if (act === "fit") return net.fit({ animation: true });
    const scale = net.getScale() || 1;
    if (act === "zoomin") return net.moveTo({ scale: scale * 1.25, animation: true });
    if (act === "zoomout") return net.moveTo({ scale: scale * 0.8, animation: true });
  });
}

// Drag the divider to resize the right detail panel (shared by graph & tree).
function wireResizers() {
  document.querySelectorAll(".viewResizer").forEach((rz) => {
    rz.addEventListener("mousedown", (e) => {
      e.preventDefault();
      rz.classList.add("dragging");
      const onMove = (ev) => {
        let w = window.innerWidth - ev.clientX;
        w = Math.max(260, Math.min(760, w));
        document.documentElement.style.setProperty("--detail-w", w + "px");
      };
      const onUp = () => {
        rz.classList.remove("dragging");
        document.body.style.userSelect = "";
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
        if (state.graphNet) state.graphNet.redraw();
        if (state.treeNet) state.treeNet.redraw();
      };
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    });
  });
}

function activateView(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $(name + "View").classList.add("active");

  // vis-network must be (re)fit once its container becomes visible
  if (name === "graph" && state.graph) {
    if (!state.graphBuilt) buildGraphNetwork();
    else { state.graphNet.redraw(); state.graphNet.fit(); }
  }
  if (name === "tree" && state.tree) {
    if (!state.treeBuilt) buildTreeNetwork();
    else { state.treeNet.redraw(); state.treeNet.fit(); }
  }
  if (name === "chat") $("question").focus();
  if (name === "library") loadLibrary();
  else scheduleLibraryPoll(false);
}

/* =============================== library ================================ */
const fmtDuration = (sec) => {
  if (sec === null || sec === undefined) return "—";
  if (sec < 60) return `${sec.toFixed(0)}s`;
  const m = Math.floor(sec / 60);
  return m < 60 ? `${m}m ${Math.round(sec % 60)}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
};

const fmtTokens = (n) => {
  if (!n) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
};

const fmtWhen = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? "—" : d.toLocaleString();
};

function stageHeadline(stage) {
  if (stage.status === "running") {
    const parts = [];
    if (stage.progress) parts.push(`${stage.progress.done}/${stage.progress.total}`);
    if (stage.etaSeconds != null) parts.push(`~${fmtDuration(stage.etaSeconds)} left`);
    else if (stage.elapsed != null) parts.push(`${fmtDuration(stage.elapsed)} elapsed`);
    return parts.join(" · ") || "running…";
  }
  if (stage.seconds != null) return fmtDuration(stage.seconds);
  return stage.status;
}

function stageRows(stage) {
  const rows = [];
  const add = (k, v) => { if (v !== null && v !== undefined && v !== "") rows.push([k, v]); };
  add("Status", stage.status);
  add("Started", stage.startedAt ? fmtWhen(stage.startedAt) : null);
  add("Ended", stage.endedAt ? fmtWhen(stage.endedAt) : null);
  add("Duration", stage.seconds != null ? fmtDuration(stage.seconds) : null);
  if (stage.status === "running") {
    add("Elapsed", stage.elapsed != null ? fmtDuration(stage.elapsed) : null);
    if (stage.progress) {
      const pct = Math.round((stage.progress.done / stage.progress.total) * 100);
      add("Progress", `${stage.progress.done} of ${stage.progress.total} nodes (${pct}%)`);
    }
    if (stage.etaSeconds != null)
      add("Remaining", `~${fmtDuration(stage.etaSeconds)} (${stage.etaBasis})`);
  }
  if (stage.typicalSeconds != null) add("Typical", fmtDuration(stage.typicalSeconds));
  for (const [key, total] of Object.entries(stage.tokens || {}))
    add(key.replace(/_/g, " "), `${fmtTokens(total)} tokens`);
  if (stage.exitCode !== null && stage.exitCode !== undefined && stage.exitCode !== 0)
    add("Exit code", stage.exitCode);
  return rows.length
    ? `<dl class="stageFacts">${rows
        .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd>`)
        .join("")}</dl>`
    : `<p class="prose">Nothing recorded for this stage yet.</p>`;
}

function stageStepHtml(stage) {
  const pct = stage.progress
    ? Math.round((stage.progress.done / stage.progress.total) * 100)
    : null;
  return `<details class="stageStep ${escapeHtml(stage.status)}">
    <summary>
      <span class="nm">${escapeHtml(stage.name)}</span>
      <span class="st">${escapeHtml(stageHeadline(stage))}</span>
      <span class="bar"><i${pct != null ? ` style="width:${pct}%"` : ""}></i></span>
    </summary>
    <div class="stageBody">
      ${stageRows(stage)}
      ${stage.error ? `<div class="runError">${escapeHtml(String(stage.error))}</div>` : ""}
    </div>
  </details>`;
}

function runCardHtml(run) {
  const status = run.overallStatus || (run.viewable ? "succeeded" : "pending");
  const isCurrent = run.runDir === state.currentRun;
  const canOpen = run.viewable && !isCurrent;
  const stages = run.stages.map(stageStepHtml).join("");
  const total = run.stages.reduce((a, s) => a + (s.seconds || 0), 0);
  return `<article class="runCard ${isCurrent ? "current" : ""}" data-run="${escapeHtml(run.runDir)}">
    <div class="runCardTop">
      <span class="runName" title="${escapeHtml(run.name)}">${escapeHtml(run.name)}</span>
      <span class="pill ${escapeHtml(status)}">${escapeHtml(status)}</span>
      ${isCurrent ? '<span class="pill">open</span>' : ""}
      ${canOpen ? '<button class="btn openRun">Open</button>' : ""}
      <button class="btn danger deleteRun" ${isCurrent ? "disabled title='Open another document first'" : ""}>Delete</button>
    </div>
    <div class="runMeta">Started ${escapeHtml(fmtWhen(run.startedAt))}</div>
    <div class="stages">${stages}</div>
    <div class="runStats">
      <div class="runStat"><div class="k">Index time</div><div class="v">${escapeHtml(fmtDuration(total || null))}</div></div>
      <div class="runStat"><div class="k">Tree build</div><div class="v">${escapeHtml(fmtDuration(run.metrics.buildTreeSeconds))}</div></div>
      <div class="runStat"><div class="k">Tokens</div><div class="v">${escapeHtml(fmtTokens(run.metrics.totalTokens))}</div></div>
      <div class="runStat"><div class="k">Viewable</div><div class="v">${run.viewable ? "yes" : "no"}</div></div>
    </div>
    ${run.error ? `<div class="runError">${escapeHtml(String(run.error))}</div>` : ""}
  </article>`;
}

async function loadLibrary() {
  let data;
  try {
    data = await api("/api/library");
  } catch (err) {
    $("runGrid").innerHTML = `<div class="libraryEmpty">Could not load documents: ${escapeHtml(err.message)}</div>`;
    return;
  }
  state.currentRun = data.current;
  const runs = data.runs || [];
  $("libraryCount").textContent = runs.length ? `${runs.length} indexed` : "";
  $("runGrid").innerHTML = runs.length
    ? runs.map(runCardHtml).join("")
    : '<div class="libraryEmpty">No documents yet. Drop a PDF above to index one.</div>';

  $("runGrid").querySelectorAll(".openRun").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const runDir = btn.closest(".runCard").dataset.run;
      btn.disabled = true;
      btn.textContent = "Opening…";
      try {
        await api("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ runDir }),
        });
        resetForNewRun();
        await loadRuns();
        await loadData();
        await loadLibrary();
        activateView("tree");
      } catch (err) {
        alert("Could not open document: " + err.message);
        btn.disabled = false;
        btn.textContent = "Open";
      }
    });
  });

  $("runGrid").querySelectorAll(".deleteRun").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const card = btn.closest(".runCard");
      const runDir = card.dataset.run;
      const name = card.querySelector(".runName").textContent;
      const ok = confirm(
        `Delete "${name}" completely?\n\n` +
          "This removes its vectors from Qdrant, its rows from PostgreSQL, and " +
          "its extracted content from disk. This cannot be undone."
      );
      if (!ok) return;
      btn.disabled = true;
      btn.textContent = "Deleting…";
      try {
        await api(`/api/run?run=${encodeURIComponent(runDir)}`, { method: "DELETE" });
        await loadRuns();
        await loadLibrary();
      } catch (err) {
        alert("Could not delete: " + err.message);
        btn.disabled = false;
        btn.textContent = "Delete";
      }
    });
  });

  const active = runs.some((r) => r.phase === "indexing" || r.overallStatus === "running");
  scheduleLibraryPoll(active);
}

function scheduleLibraryPoll(active) {
  if (state.libraryTimer) { clearTimeout(state.libraryTimer); state.libraryTimer = null; }
  const visible = $("libraryView").classList.contains("active");
  if (active && visible) {
    state.libraryTimer = setTimeout(loadLibrary, LIBRARY_POLL_MS);
  }
}

function setUploadStatus(message, isError) {
  const box = $("uploadStatus");
  box.hidden = false;
  box.className = "uploadStatus" + (isError ? " err" : "");
  box.innerHTML = message;
}

async function uploadPdf(file) {
  if (!file) return;
  if (!/\.pdf$/i.test(file.name)) {
    setUploadStatus("Only PDF files can be indexed.", true);
    return;
  }
  const zone = $("dropZone");
  zone.classList.add("busy");
  setUploadStatus(`<span class="spinner"></span>Uploading and starting <b>${escapeHtml(file.name)}</b>…`);
  try {
    const res = await api("/api/upload", {
      method: "POST",
      headers: {
        "Content-Type": "application/pdf",
        "X-Filename": encodeURIComponent(file.name),
      },
      body: file,
    });
    setUploadStatus(`Indexing <b>${escapeHtml(res.name)}</b> — progress appears below.`);
    await loadLibrary();
  } catch (err) {
    setUploadStatus("Upload failed: " + escapeHtml(err.message), true);
  } finally {
    zone.classList.remove("busy");
    $("fileInput").value = "";
  }
}

function wireLibrary() {
  const zone = $("dropZone");
  const input = $("fileInput");
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => uploadPdf(input.files[0]));
  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.remove("dragging"); }));
  zone.addEventListener("drop", (e) => uploadPdf(e.dataTransfer.files[0]));
}

/* ============================== run picker =============================== */
async function loadRuns() {
  const data = await api("/api/runs");
  const sel = $("runSelect");
  sel.innerHTML = "";
  for (const run of data.runs) {
    const opt = document.createElement("option");
    opt.value = run.path;
    opt.textContent = run.name + (run.hasGraph ? "" : "  (no graph)");
    if (run.path === data.current) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.disabled = data.runs.length <= 1;
  sel.onchange = async () => {
    sel.disabled = true;
    try {
      await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ runDir: sel.value }),
      });
      resetForNewRun();
      await loadData();
    } catch (err) {
      alert("Could not switch document: " + err.message);
    } finally {
      sel.disabled = false;
    }
  };
}

function resetForNewRun() {
  if (state.graphNet) { state.graphNet.destroy(); state.graphNet = null; }
  if (state.treeNet) { state.treeNet.destroy(); state.treeNet = null; }
  state.graphBuilt = state.treeBuilt = false;
  $("messages").innerHTML = "";
}

/* ============================== load data ================================ */
async function loadData() {
  const [tree, graph] = await Promise.all([api("/api/tree"), api("/api/graph")]);
  state.tree = tree;
  state.graph = graph;

  state.treeNodesById = new Map(tree.nodes.map((n) => [n.id, n]));
  state.graphNodesById = new Map((graph.nodes || []).map((n) => [n.id, n]));
  state.treeLevelById = computeTreeLevels(tree.nodes);

  // reverse index: which entities were extracted from each tree node
  state.entitiesBySourceNode = new Map();
  for (const ent of graph.nodes || []) {
    for (const sid of ent.sourceIds || []) {
      if (!state.entitiesBySourceNode.has(sid)) state.entitiesBySourceNode.set(sid, []);
      state.entitiesBySourceNode.get(sid).push(ent);
    }
  }


  buildTypeFilter($("entityTypeFilter"), (graph.nodes || []).map((n) => n.type));
  buildTypeFilter($("treeTypeFilter"), tree.nodes.map((n) => n.type));
  renderEntityList();
  renderTreeList();
  renderGraphLegend();
  renderTreeLegend();

  const active = document.querySelector(".tab.active").dataset.view;
  if (active === "graph") buildGraphNetwork();
  if (active === "tree") buildTreeNetwork();
}

function buildTypeFilter(sel, types) {
  const uniq = [...new Set(types.map((t) => t || "UNKNOWN"))].sort();
  const first = sel.querySelector('option[value=""]');
  sel.innerHTML = "";
  sel.appendChild(first);
  for (const t of uniq) {
    const o = document.createElement("option");
    o.value = t; o.textContent = t;
    sel.appendChild(o);
  }
}

/* =============================== GRAPH =================================== */
function wireGraphControls() {
  $("graphSearch").addEventListener("input", applyGraphFilter);
  $("entityTypeFilter").addEventListener("change", applyGraphFilter);
  $("graphLabels").addEventListener("change", () => { if (state.graphBuilt) applyGraphFilter(); });
  $("graphFit").addEventListener("click", () => state.graphNet && state.graphNet.fit({ animation: true }));
  $("graphPhysics").addEventListener("click", toggleGraphPhysics);
}

let graphPhysicsOn = true;
function toggleGraphPhysics() {
  if (!state.graphNet) return;
  graphPhysicsOn = !graphPhysicsOn;
  state.graphNet.setOptions({ physics: { enabled: graphPhysicsOn } });
  $("graphPhysics").textContent = graphPhysicsOn ? "Freeze" : "Unfreeze";
}

// A force layout over thousands of entities is both unreadable and slow enough
// to hang the tab, so an unfiltered view falls back to the best-connected ones.
// Searching or picking a type narrows it first and keeps everything that matches.
const GRAPH_RENDER_BUDGET = 400;

function visibleEntities() {
  const q = $("graphSearch").value.trim().toLowerCase();
  const type = $("entityTypeFilter").value;
  const matched = (state.graph.nodes || []).filter((n) => {
    const okType = !type || (n.type || "UNKNOWN") === type;
    const hay = `${n.label} ${n.type} ${n.description}`.toLowerCase();
    return okType && (!q || hay.includes(q));
  });
  if (matched.length <= GRAPH_RENDER_BUDGET) return matched;
  return [...matched]
    .sort((a, b) => (b.degree || 0) - (a.degree || 0))
    .slice(0, GRAPH_RENDER_BUDGET);
}

function toVisEntity(n, labelsOn) {
  return {
    id: n.id,
    label: labelsOn ? clip(n.label, 26) : undefined,
    title: `${n.label}\n[${n.type}]  ·  degree ${n.degree}\n${clip(n.description, 160)}`,
    value: (n.degree || 0) + 1,
    color: { background: colorForType(n.type), border: "#ffffff",
      highlight: { background: colorForType(n.type), border: "#111" } },
  };
}

// Physics scaled to the graph: more nodes / higher degree → stronger repulsion,
// longer springs, more overlap avoidance, so it doesn't clump as it grows.
function graphPhysicsFor(nodes) {
  const N = Math.max(1, nodes.length);
  // Only repulsion & central gravity scale with node COUNT (to avoid overlap as
  // the graph grows). Edge length is NOT count-based — it's set per-edge from
  // node degree (see edgeLength) so only hubs get longer edges, not everything.
  const grav = -Math.min(26000, 2200 + N * 22);
  const centralGravity = N > 400 ? 0.3 : N > 120 ? 0.45 : 0.7;
  const springConstant = N > 300 ? 0.02 : 0.045;
  return {
    enabled: true,
    solver: "barnesHut",
    barnesHut: { gravitationalConstant: grav, centralGravity, springLength: 90,
      springConstant, avoidOverlap: 0.55, damping: 0.45 },
    stabilization: { iterations: N > 400 ? 260 : 180, updateInterval: 40 },
    minVelocity: 0.8,
  };
}

// Per-edge rest length, driven by BOTH endpoints' degree (their sum) so lengths
// vary edge-to-edge: hub↔hub longest, hub↔leaf medium, leaf↔leaf shortest.
function edgeLength(l) {
  const ds = (state.graphNodesById.get(l.source) || {}).degree || 0;
  const dt = (state.graphNodesById.get(l.target) || {}).degree || 0;
  return Math.round(Math.min(340, 60 + (ds + dt) * 1.1));
}

function buildGraphNetwork() {
  const container = $("graphNet");
  const nodes = new vis.DataSet();
  const edges = new vis.DataSet();
  state.graphNodes = nodes;
  state.graphEdges = edges;

  state.graphNet = new vis.Network(container, { nodes, edges }, {
    nodes: {
      shape: "dot",
      scaling: {
        min: 8, max: 25,
        label: { enabled: true, min: 12, max: 20 },
        // concave curve → size grows slowly, so high-degree hubs stay modest
        customScalingFunction: (min, max, total, value) =>
          max === min ? 0.5 : Math.pow((value - min) / (max - min), 0.4),
      },
      font: { size: 13, face: "system-ui", color: "#1c2430" },
      borderWidth: 1.5,
    },
    edges: {
      color: { color: "#b3bcc4", highlight: "#0f766e", opacity: 0.65 },
      scaling: {
        min: 1, max: 4.5,
        customScalingFunction: (min, max, total, value) =>
          max === min ? 0.5 : Math.pow((value - min) / (max - min), 0.4),
      },
      selectionWidth: 2,
      smooth: { enabled: true, type: "continuous", roundness: 0.3 },
    },
    interaction: { hover: true, tooltipDelay: 120, hideEdgesOnDrag: true,
      dragView: true, zoomView: false },  // wheel handled as pan (enableWheelPan)
    physics: {
      solver: "barnesHut",
      barnesHut: {
        gravitationalConstant: -4000, // weaker repulsion → less spreading
        centralGravity: 0.85,         // strong pull to center so lone nodes don't fly off
        springLength: 100,
        springConstant: 0.06,
        avoidOverlap: 0.2,
        damping: 0.4,
      },
      stabilization: { iterations: 220, updateInterval: 30 },
      minVelocity: 0.75,
    },
    layout: { improvedLayout: false },
  });

  state.graphNet.on("click", (params) => {
    if (params.nodes.length) selectEntity(params.nodes[0]);
    else if (params.edges.length) selectRelation(params.edges[0]);
  });
  // freeze after every (re)stabilization so it settles then stops jittering
  state.graphNet.on("stabilizationIterationsDone", () => {
    graphPhysicsOn = false;
    state.graphNet.setOptions({ physics: { enabled: false } });
    $("graphPhysics").textContent = "Unfreeze";
    state.graphNet.fit();
  });
  observeSize(() => state.graphNet, container, () => state.graphNet.fit());

  state.graphBuilt = true;
  applyGraphFilter();
}

function applyGraphFilter() {
  if (!state.graphBuilt) return;
  const visN = visibleEntities();
  const ids = new Set(visN.map((n) => n.id));
  const labelsOn = $("graphLabels").checked || visN.length <= 70;
  const links = (state.graph.links || []).filter((l) => ids.has(l.source) && ids.has(l.target));

  // re-tune physics for the current node count / degree, then re-layout
  state.graphNet.setOptions({ physics: graphPhysicsFor(visN) });
  graphPhysicsOn = true;
  $("graphPhysics").textContent = "Freeze";

  state.graphNodes.clear();
  state.graphEdges.clear();
  state.graphNodes.add(visN.map((n) => toVisEntity(n, labelsOn)));
  state.graphEdges.add(links.map((l) => ({
    id: l.id, from: l.source, to: l.target,
    value: Number(l.weight) || 1,
    length: edgeLength(l),   // varies per edge by endpoint degrees (hubs longer)
    title: `${l.sourceName} → ${l.targetName}\n${clip(l.description, 160)}`,
  })));

  const total = (state.graph.nodes || []).length;
  const capped = visN.length >= GRAPH_RENDER_BUDGET && total > GRAPH_RENDER_BUDGET;
  $("graphStats").textContent =
    `${visN.length} of ${total} entities · ${links.length} relationships shown`
    + (capped ? " · most-connected first; search or filter to see others" : "");
  renderEntityList(visN);
  $("graphHint").textContent = labelsOn
    ? "scroll to pan · Ctrl+scroll = zoom"
    : "labels hidden — search/filter or tick Labels · scroll to pan · Ctrl+scroll = zoom";
}

function renderEntityList(list) {
  const nodes = (list || state.graph.nodes || []).slice().sort((a, b) => b.degree - a.degree);
  const root = $("entityList");
  root.innerHTML = "";
  for (const n of nodes.slice(0, 400)) {
    const btn = document.createElement("button");
    btn.className = "railItem";
    btn.innerHTML = `<span class="t">${escapeHtml(n.label)}</span>
      <span class="s" style="color:${colorForType(n.type)}">${escapeHtml(n.type)}</span>
      <span class="s"> · degree ${n.degree}</span>`;
    btn.onclick = () => { activateView("graph"); selectEntity(n.id); };
    root.appendChild(btn);
  }
}

function renderGraphLegend() {
  const types = [...new Set((state.graph.nodes || []).map((n) => n.type || "UNKNOWN"))].sort();
  $("graphLegend").innerHTML = types.map((t) =>
    `<span class="swatch" data-type="${escapeHtml(t)}"><span class="dot" style="background:${colorForType(t)}"></span>${escapeHtml(t)}</span>`
  ).join("");
  $("graphLegend").querySelectorAll(".swatch").forEach((sw) => {
    sw.onclick = () => {
      const cur = $("entityTypeFilter").value;
      $("entityTypeFilter").value = cur === sw.dataset.type ? "" : sw.dataset.type;
      applyGraphFilter();
    };
  });
}

function selectEntity(id) {
  const ent = state.graphNodesById.get(id);
  if (!ent) return;
  if (state.graphBuilt && state.graphNodes.get(id)) {
    state.graphNet.selectNodes([id]);
    state.graphNet.focus(id, { scale: 1.1, animation: true });
  }
  const rels = (state.graph.links || []).filter((l) => l.source === id || l.target === id)
    .sort((a, b) => (Number(b.weight) || 0) - (Number(a.weight) || 0));

  const sources = (ent.sourceIds || []).map((sid, i) => {
    const snip = (ent.sourceSnippets || [])[i] || "";
    return `<div class="card"><div class="h">source node §${escapeHtml(sid)}
      <a class="linkish" data-tree="${escapeHtml(sid)}">view in tree →</a></div>
      <div class="b">${escapeHtml(clip(snip, 260))}</div></div>`;
  }).join("") || `<p class="prose">No source node recorded.</p>`;

  const relHtml = rels.slice(0, 40).map((l) => {
    const other = l.source === id ? l.targetName : l.sourceName;
    const dir = l.source === id ? "→" : "←";
    return `<div class="card"><div class="h">${dir} ${escapeHtml(other)} ${l.weight ? `· w${escapeHtml(l.weight)}` : ""}</div>
      <div class="b">${escapeHtml(clip(l.description, 200))}</div></div>`;
  }).join("") || `<p class="prose">No relationships.</p>`;

  $("graphDetail").innerHTML = `
    <h2>${escapeHtml(ent.label)}</h2>
    <span class="tag" style="background:${colorForType(ent.type)}">${escapeHtml(ent.type)}</span>
    <span class="pill">degree ${ent.degree}</span>
    <div class="section-h">Description (LLM-extracted)</div>
    <div class="prose">${escapeHtml(ent.description || "—")}</div>
    <div class="section-h">Created from ${ent.sourceIds?.length || 0} source node(s)</div>
    ${sources}
    <div class="section-h">Relationships (${rels.length})</div>
    ${relHtml}`;
  wireTreeJumpLinks($("graphDetail"));
}

function selectRelation(edgeId) {
  const l = (state.graph.links || []).find((x) => x.id === edgeId);
  if (!l) return;
  $("graphDetail").innerHTML = `
    <h2>Relationship</h2>
    <div class="kv">
      <dt>source</dt><dd><a class="linkish" data-entity="${escapeHtml(l.source)}">${escapeHtml(l.sourceName)}</a></dd>
      <dt>target</dt><dd><a class="linkish" data-entity="${escapeHtml(l.target)}">${escapeHtml(l.targetName)}</a></dd>
      <dt>weight</dt><dd>${escapeHtml(l.weight ?? "—")}</dd>
    </div>
    <div class="section-h">Description (LLM-extracted)</div>
    <div class="prose">${escapeHtml(l.description || "—")}</div>
    <div class="section-h">Source node(s)</div>
    <div class="prose">${(l.sourceIds || []).map((s) => `<a class="linkish" data-tree="${escapeHtml(s)}">§${escapeHtml(s)}</a>`).join(", ") || "—"}</div>`;
  $("graphDetail").querySelectorAll("[data-entity]").forEach((a) =>
    a.onclick = () => selectEntity(a.dataset.entity));
  wireTreeJumpLinks($("graphDetail"));
}

function wireTreeJumpLinks(root) {
  root.querySelectorAll("[data-tree]").forEach((a) =>
    a.onclick = () => { activateView("tree"); selectTreeNode(Number(a.dataset.tree)); });
}

/* =============================== TREE ==================================== */
function wireTreeControls() {
  $("treeSearch").addEventListener("input", () => renderTreeList());
  $("treeTypeFilter").addEventListener("change", () => renderTreeList());
  $("treeLabels").addEventListener("change", () => { if (state.treeBuilt) rebuildTreeLabels(); });
  $("treeFit").addEventListener("click", () => state.treeNet && state.treeNet.fit({ animation: true }));
}

// Hierarchical layout is superlinear in node count, so a 16k-node document
// hangs the tab before it paints. Draw a bounded slice instead: the shallow
// structure, plus whatever the reader is currently looking at.
const TREE_RENDER_BUDGET = 600;

function treeRenderSubset(all, focusId) {
  if (all.length <= TREE_RENDER_BUDGET) return all;
  const byId = new Map(all.map((n) => [n.id, n]));
  const keep = new Map();

  // The focused node, its ancestors, and its children always survive.
  for (let cur = byId.get(focusId); cur; cur = byId.get(cur.parentId)) keep.set(cur.id, cur);
  if (keep.size) for (const n of all) if (n.parentId === focusId) keep.set(n.id, n);

  // Fill the remainder breadth-first so the document's outline is what shows.
  const level = state.treeLevelById || new Map();
  const rest = all
    .filter((n) => !keep.has(n.id))
    .sort((a, b) => (level.get(a.id) || 0) - (level.get(b.id) || 0) || a.id - b.id);
  for (const n of rest) {
    if (keep.size >= TREE_RENDER_BUDGET) break;
    keep.set(n.id, n);
  }

  // An edge is only drawn when its parent is present, so pull ancestors in.
  for (const n of [...keep.values()]) {
    for (let p = byId.get(n.parentId); p && !keep.has(p.id); p = byId.get(p.parentId)) {
      keep.set(p.id, p);
    }
  }
  return [...keep.values()];
}

// Refill the drawn slice in place. Replacing the network instead loses the
// canvas, and vis re-runs the hierarchical layout on a DataSet change anyway.
function populateTreeNetwork() {
  const labelsOn = $("treeLabels").checked;
  const shown = treeRenderSubset(state.tree.nodes, state.selectedTreeId);
  const shownIds = new Set(shown.map((n) => n.id));
  state.treeShownCount = shown.length;
  state.treeNodes.clear();
  state.treeEdges.clear();
  state.treeNodes.add(shown.map((n) => toVisTreeNode(n, labelsOn)));
  state.treeEdges.add(
    shown.filter((n) => n.parentId !== null && shownIds.has(n.parentId)).map((n) => ({
      from: n.parentId, to: n.id, arrows: "to",
      color: { color: "#cbd3da", highlight: "#0f766e" }, width: 1,
    }))
  );
}

function buildTreeNetwork() {
  const container = $("treeNet");
  const nodes = new vis.DataSet();
  const edges = new vis.DataSet();
  state.treeNodes = nodes;
  state.treeEdges = edges;
  populateTreeNetwork();

  state.treeNet = new vis.Network(container, { nodes, edges }, {
    layout: { hierarchical: { enabled: true, direction: "LR", sortMethod: "directed",
      levelSeparation: 210, nodeSpacing: 95, treeSpacing: 120, blockShifting: true, edgeMinimization: true } },
    nodes: { shape: "box", widthConstraint: { maximum: 165 }, heightConstraint: { minimum: 34 },
      font: { size: 12.5, face: "system-ui" }, margin: 8, borderWidth: 1.5 },
    edges: { smooth: { enabled: true, type: "cubicBezier", forceDirection: "horizontal", roundness: 0.5 } },
    physics: { enabled: false },
    interaction: {
      hover: true, tooltipDelay: 120,
      dragView: true, zoomView: false,    // wheel is handled as pan (see enableWheelPan)
      navigationButtons: false,
      keyboard: { enabled: true, speed: { x: 14, y: 14, zoom: 0.03 }, bindToWindow: false },
    },
  });

  state.treeNet.on("click", (params) => {
    if (params.nodes.length) selectTreeNode(params.nodes[0]);
    else clearTreeHighlight(); // click empty space -> revert to default view
  });
  observeSize(() => state.treeNet, container, () => state.treeNet.fit());
  state.treeBuilt = true;
}

// Staggered levels: each tree depth spans two vis columns, and adjacent
// siblings alternate between them — this ~halves the vertical height.
//   depth 0 (root) -> 0 ; depth d>=1 -> {2d-1, 2d} chosen by sibling parity
function computeTreeLevels(nodes) {
  const childrenOf = new Map();
  for (const n of nodes) {
    if (n.parentId != null) {
      if (!childrenOf.has(n.parentId)) childrenOf.set(n.parentId, []);
      childrenOf.get(n.parentId).push(n.id);
    }
  }
  const depth = new Map();
  const roots = nodes.filter((n) => n.parentId == null).map((n) => n.id);
  const queue = roots.map((r) => [r, 0]);
  while (queue.length) {
    const [id, d] = queue.shift();
    depth.set(id, d);
    for (const k of childrenOf.get(id) || []) queue.push([k, d + 1]);
  }
  const level = new Map();
  for (const n of nodes) {
    const d = depth.get(n.id) ?? 0;
    if (d === 0) { level.set(n.id, 0); continue; }
    const sibs = childrenOf.get(n.parentId) || [];
    const parity = Math.max(0, sibs.indexOf(n.id)) % 2; // alternate adjacent siblings
    level.set(n.id, 2 * d - 1 + parity);
  }
  return level;
}

function toVisTreeNode(n, labelsOn) {
  const c = colorForType(n.type);
  const short = clip(nodeLabel(n), 40);
  return {
    id: n.id,
    level: state.treeLevelById ? state.treeLevelById.get(n.id) : undefined,
    label: labelsOn ? `#${n.id} ${short}` : `#${n.id}`,
    title: `#${n.id} · ${n.type}${n.page ? " · p" + n.page : ""}\n${clip(n.content, 160)}`,
    color: { background: n.type === "TITLE" ? "#eef6ff" : "#ffffff", border: c,
      highlight: { background: "#fff7ed", border: "#111" } },
    font: { color: "#1c2430" },
  };
}

function rebuildTreeLabels() {
  const labelsOn = $("treeLabels").checked;
  // update() inserts ids it does not find, so relabel only what is drawn --
  // otherwise toggling labels would quietly restore the whole tree.
  state.treeNodes.update(state.tree.nodes
    .filter((n) => state.treeNodes.get(n.id))
    .map((n) => ({
      id: n.id, label: labelsOn ? `#${n.id} ${clip(nodeLabel(n), 40)}` : `#${n.id}`,
    })));
}

function renderTreeList() {
  const q = $("treeSearch").value.trim().toLowerCase();
  const type = $("treeTypeFilter").value;
  const list = state.tree.nodes.filter((n) => {
    if (n.type === "root") return false;
    const okType = !type || n.type === type;
    const hay = `${n.content} ${n.type} ${n.summary}`.toLowerCase();
    return okType && (!q || hay.includes(q));
  });
  const drawn = state.treeShownCount || 0;
  $("treeStats").textContent = `${list.length} of ${state.tree.nodes.length} tree nodes`
    + (drawn && drawn < state.tree.nodes.length
        ? ` · ${drawn} drawn; click a node to redraw around it` : "");
  const root = $("treeList");
  root.innerHTML = "";
  for (const n of list.slice(0, 500)) {
    const btn = document.createElement("button");
    btn.className = "railItem";
    const indent = Math.min((n.titleLevel || 0), 4) * 10;
    btn.style.paddingLeft = 10 + indent + "px";
    btn.innerHTML = `<span class="t">#${n.id} ${escapeHtml(clip(n.content || "(no text)", 60))}</span>
      <span class="s" style="color:${colorForType(n.type)}">${escapeHtml(n.type)}</span>
      <span class="s">${n.page ? " · page " + n.page : ""}</span>`;
    btn.onclick = () => { activateView("tree"); selectTreeNode(n.id); };
    root.appendChild(btn);
  }
}

let treeHighlightType = null;
function renderTreeLegend() {
  const types = [...new Set(state.tree.nodes.map((n) => n.type))].filter((t) => t !== "root").sort();
  $("treeLegend").innerHTML = types.map((t) =>
    `<span class="swatch" data-type="${escapeHtml(t)}"><span class="dot" style="background:${colorForType(t)}"></span>${escapeHtml(t)}</span>`).join("");
  $("treeLegend").querySelectorAll(".swatch").forEach((sw) =>
    sw.onclick = () => highlightTreeType(sw.dataset.type));
}

function hexA(hex, a) {
  const m = String(hex).replace("#", "");
  const r = parseInt(m.slice(0, 2), 16), g = parseInt(m.slice(2, 4), 16), b = parseInt(m.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

// Clear any legend highlight and restore the default tree styling.
function clearTreeHighlight() {
  if (!treeHighlightType) return;
  treeHighlightType = null;
  const lg = $("treeLegend");
  if (lg) lg.querySelectorAll(".swatch").forEach((s) => s.classList.remove("active"));
  if (state.treeBuilt) recolorTree(null);
}

// Click a tree legend type -> soft-glow all nodes of that type, gently dim the rest.
function highlightTreeType(type) {
  activateView("tree");
  if (!state.treeBuilt) buildTreeNetwork();
  if (treeHighlightType === type) { clearTreeHighlight(); return; } // toggle off
  treeHighlightType = type;
  $("treeLegend").querySelectorAll(".swatch").forEach((s) =>
    s.classList.toggle("active", s.dataset.type === type));
  const ids = state.tree.nodes.filter((n) => n.type === type).map((n) => n.id);
  recolorTree(new Set(ids));
  if (ids.length) state.treeNet.fit({ nodes: ids, animation: true });
}

function recolorTree(hlSet) {
  state.treeNodes.update(state.tree.nodes.map((n) => {
    const c = colorForType(n.type);
    const base = {
      background: n.type === "TITLE" ? "#eef6ff" : "#ffffff", border: c,
      highlight: { background: "#fff7ed", border: "#111" },
    };
    if (!hlSet) { // default view
      return { id: n.id, opacity: 1, borderWidth: 1.5, color: base,
        font: { color: "#1c2430" }, shadow: { enabled: false } };
    }
    if (hlSet.has(n.id)) { // soft colored glow, normal fill kept
      return { id: n.id, opacity: 1, borderWidth: 2, color: base,
        font: { color: "#1c2430" },
        shadow: { enabled: true, color: hexA(c, 0.5), size: 22, x: 0, y: 0 } };
    }
    return { id: n.id, opacity: 0.5, borderWidth: 1, // gently dimmed
      color: { background: "#f7f9fb", border: "#e6eaee" }, font: { color: "#9aa4ad" },
      shadow: { enabled: false } };
  }));
}

function selectTreeNode(id) {
  const n = state.treeNodesById.get(id);
  if (!n) return;
  state.selectedTreeId = id;
  // Outside the drawn slice: refill it around this node so the reader can walk
  // into parts of a large document the budget left out.
  if (state.treeBuilt && !state.treeNodes.get(id)) {
    populateTreeNetwork();
    renderTreeList();
  }
  if (state.treeBuilt && state.treeNodes.get(id)) {
    state.treeNet.selectNodes([id]);
    state.treeNet.focus(id, { scale: 0.9, animation: true });
  }
  const kids = (n.children || []).map((cid) => {
    const c = state.treeNodesById.get(cid);
    return c ? `<div class="card"><div class="h"><a class="linkish" data-tree="${cid}">#${cid} ${escapeHtml(clip(nodeLabel(c), 46))}</a></div>
      <div class="b">${escapeHtml(c.type)}${c.page ? " · page " + c.page : ""}</div></div>` : "";
  }).join("") || `<p class="prose">No children (leaf node).</p>`;

  const ents = (state.entitiesBySourceNode.get(id) || []);
  const entHtml = ents.length
    ? ents.slice(0, 30).map((e) => `<a class="linkish" data-entity="${escapeHtml(e.id)}">${escapeHtml(e.label)}</a> <span class="s" style="color:${colorForType(e.type)}">(${escapeHtml(e.type)})</span>`).join("<br>")
    : `<p class="prose">No entities were extracted from this node.</p>`;

  const parent = n.parentId != null ? state.treeNodesById.get(n.parentId) : null;
  const img = (n.meta && n.meta.img_path)
    ? `<div class="section-h">Image</div><img class="nodeimg" src="/api/tree-image?id=${id}" alt="node image" />` : "";

  $("treeDetail").innerHTML = `
    <h2>#${id} ${escapeHtml(clip(nodeLabel(n), 90))}</h2>
    <div class="kv">
      <dt>Type</dt><dd><span class="tag" style="background:${colorForType(n.type)}">${escapeHtml(n.type)}</span></dd>
      ${n.page ? `<dt>Page</dt><dd>${n.page}</dd>` : ""}
      <dt>Parent</dt><dd>${parent ? `<a class="linkish" data-tree="${parent.id}">#${parent.id} ${escapeHtml(clip(nodeLabel(parent), 40))}</a>` : "— (root)"}</dd>
      <dt>Children</dt><dd>${(n.children || []).length}</dd>
    </div>
    ${n.tableBody ? `<div class="section-h">Table</div><div class="tableWrap" id="nodeTable"></div>` : ""}
    ${(n.content || "").trim() || !n.tableBody
      ? `<div class="section-h">${n.tableBody ? "Caption &amp; footnotes" : "Content"}</div>
         <div class="prose">${renderContent(n)}</div>`
      : ""}
    ${(n.structures || []).length ? `<div class="section-h">Extracted structure</div>${renderStructures(n.structures)}` : ""}
    ${n.summary ? `<div class="section-h">Summary (LLM-generated) <button class="miniBtn" id="editSummary">edit</button></div><div class="prose" id="summaryText">${escapeHtml(n.summary)}</div>` : ""}
    <div class="section-h">Entities extracted here (${ents.length})</div>
    <div class="prose">${entHtml}</div>
    <div class="section-h">Child nodes</div>
    ${kids}
    ${img}`;
  if (n.tableBody) {
    const table = renderTableBody(n.tableBody);
    const host = $("nodeTable");
    if (table && host) host.appendChild(table);
    else if (host) host.innerHTML = `<p class="prose">The parser produced no readable grid for this table.</p>`;
  }
  wireTreeJumpLinks($("treeDetail"));
  $("treeDetail").querySelectorAll("[data-entity]").forEach((a) =>
    a.onclick = () => { activateView("graph"); selectEntity(a.dataset.entity); });
  if ((n.structures || []).length) drawStructures($("treeDetail"), n.structures);
  enhanceContent($("treeDetail"));
  typesetMath($("treeDetail"));
  const editBtn = $("treeDetail").querySelector("#editSummary");
  if (editBtn) editBtn.onclick = () => beginSummaryEdit(id, n.summary);
  const im = $("treeDetail").querySelector(".nodeimg");
  if (im) im.onclick = () => openLightbox(im.src);
}

const ROLE_COLORS = {
  input:   { background: "#e0f2fe", border: "#0369a1" },
  output:  { background: "#dcfce7", border: "#15803d" },
  process: { background: "#fce7f3", border: "#be185d" },
  data:    { background: "#fef9c3", border: "#a16207" },
  decision:{ background: "#fef3c7", border: "#b45309" },
};

function structureParts(s) {
  const flowConnected = new Set();
  (s.edges || []).forEach((e) => {
    if (e.kind !== "diagram_contains") {
      flowConnected.add(e.source);
      flowConnected.add(e.target);
    }
  });
  // A node with flow arrows through it is part of the flow whatever any
  // containment claim says; only arrow-less parts fold into their parent.
  const contained = new Map();
  (s.edges || []).forEach((e) => {
    if (e.kind !== "diagram_contains") return;
    if (flowConnected.has(e.target)) return;
    if (!contained.has(e.source)) contained.set(e.source, []);
    contained.get(e.source).push(e.target);
  });
  const nested = new Set([].concat(...contained.values()));
  const inFlow = (s.elements || []).filter(
    (el) => flowConnected.has(el.content) && !nested.has(el.content)
  );
  const standalone = (s.elements || []).filter(
    (el) => !flowConnected.has(el.content) && !nested.has(el.content)
  );
  return { inFlow, standalone, contained };
}

function renderStructures(structures) {
  return structures.map((s, i) => {
    const { standalone } = structureParts(s);
    const hasFlow = (s.edges || []).length > 0;
    const chips = standalone.map((el) =>
      `<span class="tag">${escapeHtml(el.content)}${el.role ? ` \u00b7 ${escapeHtml(el.role)}` : ""}</span>`
    ).join(" ");
    const canvas = hasFlow ? `<div class="flowCanvas" data-struct="${i}"></div>` : "";
    const chipBlock = chips
      ? `<div class="structEls">${hasFlow ? `<span class="structNote">also shown, without flow arrows:</span> ` : ""}${chips}</div>`
      : "";
    return `<div class="structBlock"><div class="structKind">${escapeHtml(s.kind)}</div>${canvas}${chipBlock}</div>`;
  }).join("");
}

function drawStructures(host, structures) {
  host.querySelectorAll(".flowCanvas").forEach((canvas) => {
    const s = structures[Number(canvas.dataset.struct)];
    if (!s || typeof vis === "undefined") return;
    const ids = new Map();
    const { inFlow, contained } = structureParts(s);
    const nodes = inFlow.map((el, i) => {
      ids.set(el.content, i);
      const c = ROLE_COLORS[el.role] || { background: "#ffffff", border: "#64748b" };
      const inner = contained.get(el.content) || [];
      const label = inner.length
        ? `${el.content}\n${inner.map((name) => `\u2022 ${name}`).join("\n")}`
        : el.content;
      return {
        id: i,
        label,
        title: el.role || undefined,
        shape: el.role === "decision" ? "diamond" : "box",
        color: { ...c, highlight: c },
        font: { size: 13, face: "system-ui", color: "#1c2430",
          multi: false, align: inner.length ? "left" : "center" },
        margin: inner.length ? 10 : 8,
        borderWidth: inner.length ? 2 : 1,
      };
    });
    // A pair of opposite edges reads as one double-headed arrow, the way the
    // source diagram draws bidirectional flow.
    const seen = new Map();
    const edges = [];
    (s.edges || []).forEach((e) => {
      if (e.kind === "diagram_contains") return;
      const a = ids.get(e.source), b = ids.get(e.target);
      if (a === undefined || b === undefined) return;
      const back = seen.get(`${b}|${a}`);
      if (back !== undefined) {
        edges[back].arrows = { to: { enabled: true }, from: { enabled: true } };
        edges[back].smooth = false;
        edges[back].width = 2;
        return;
      }
      seen.set(`${a}|${b}`, edges.length);
      edges.push({ from: a, to: b, arrows: "to", label: e.label || undefined,
        color: { color: "#64748b", highlight: "#0f766e" }, width: 1.5,
        font: { size: 11, align: "top" } });
    });
    const net = new vis.Network(canvas, {
      nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges),
    }, {
      layout: { hierarchical: { enabled: true, direction: "LR",
        sortMethod: "directed", levelSeparation: 150, nodeSpacing: 70 } },
      physics: false,
      interaction: { dragNodes: true, zoomView: true, dragView: true },
      edges: { smooth: { enabled: true, type: "cubicBezier",
        forceDirection: "horizontal", roundness: 0.4 } },
    });
    net.once("afterDrawing", () => net.fit({ animation: false }));
  });
}

function beginSummaryEdit(nodeId, current) {
  const host = $("summaryText");
  if (!host) return;
  host.innerHTML = `<textarea id="summaryDraft" rows="5" style="width:100%">${escapeHtml(current)}</textarea>
    <div style="margin-top:6px"><button class="miniBtn" id="saveSummary">save</button>
    <span class="structNote">saving reseals the revision and marks vectors stale</span></div>`;
  host.querySelector("#saveSummary").onclick = async () => {
    const text = host.querySelector("#summaryDraft").value.trim();
    if (!text) return;
    try {
      await api("/api/edit-summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodeId, text }),
      });
      await loadData();
      selectTreeNode(nodeId);
    } catch (err) {
      alert("Edit failed: " + err.message);
    }
  };
}

/* =============================== CHAT =================================== */
function wireChat() {
  $("chatForm").addEventListener("submit", submitChat);
  const ta = $("question");
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitChat(e); }
  });
  ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 160) + "px"; });
  addMessage("assistant",
    `<div class="answer">Ask a question about the indexed document. The retrieved tree nodes &amp; graph entities used to answer will appear under each response.</div>`);
}

function addMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  wrap.innerHTML = `<div class="bubble"><div class="who">${role === "user" ? "You" : "BookRAG"}</div>${html}</div>`;
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return wrap;
}

async function submitChat(event) {
  if (event) event.preventDefault();
  if (state.chatBusy) return;
  const ta = $("question");
  const question = ta.value.trim();
  if (!question) return;

  ta.value = ""; ta.style.height = "auto";
  addMessage("user", `<div class="answer">${escapeHtml(question)}</div>`);
  const pending = addMessage("assistant",
    `<div class="loading"><span class="spinner"></span>Thinking… <span class="meta">first question loads the models (~30–60s)</span></div>`);

  state.chatBusy = true;
  $("askBtn").disabled = true;
  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    pending.querySelector(".bubble").innerHTML =
      `<div class="who">BookRAG</div>` + renderAnswer(data);
    linkifyCitations(pending, data.citations);
    wireEvidenceLinks(pending);
    typesetMath(pending);
  } catch (err) {
    pending.querySelector(".bubble").innerHTML =
      `<div class="who">BookRAG</div><p class="error">Error: ${escapeHtml(err.message)}</p>`;
  } finally {
    state.chatBusy = false;
    $("askBtn").disabled = false;
    $("messages").scrollTop = $("messages").scrollHeight;
  }
}

function renderAnswer(data) {
  const md = window.marked ? marked.parse(data.answer || "") : escapeHtml(data.answer || "");
  let html = `<div class="answer">${md}</div>`;
  html += `<div class="meta">${(data.nodeIds || []).length} nodes retrieved · ${data.elapsedSec}s</div>`;
  html += renderReferences(data.citations);
  html += renderRetrievedEntities(data.retrieval);
  return html;
}

// The answer's own [E1] / [Q1-E1] markers are what the model actually cited;
// they are the reference list, not a separate summary of everything fetched.
function renderReferences(citations) {
  if (!citations || !citations.length) return "";
  const cards = citations.map((c) => {
    const img = c.hasImage && c.nodeId != null
      ? `<img class="refThumb" src="/api/tree-image?id=${encodeURIComponent(c.nodeId)}" alt="" loading="lazy" />`
      : "";
    const jump = c.nodeId != null
      ? `<a class="linkish refJump" data-tree="${escapeHtml(c.nodeId)}">view in tree →</a>`
      : "";
    return `<div class="refCard" id="ref-${escapeHtml(c.label)}">
      <div class="refHead">
        <span class="refLabel">${escapeHtml(c.label)}</span>
        ${c.type ? `<span class="refType">${escapeHtml(c.type)}</span>` : ""}
        ${c.page ? `<span class="refPage">p${escapeHtml(c.page)}</span>` : ""}
      </div>
      ${img}
      ${c.snippet ? `<div class="refSnippet">${escapeHtml(clip(c.snippet, 260))}</div>` : ""}
      ${jump}
    </div>`;
  }).join("");
  return `<details class="evidence" open><summary>References (${citations.length})</summary><div class="refList">${cards}</div></details>`;
}

// Entities the retrieval touched but that the answer may not have named
// directly -- kept as a secondary, collapsed fold so it doesn't compete
// with the actual references for attention.
function renderRetrievedEntities(retrieval) {
  if (!retrieval || !retrieval.iterations) return "";
  let inner = "";
  for (const it of retrieval.iterations) {
    const g = it.iteration_graph_nodes || [];
    if (!g.length) continue;
    if (it.sub_query) inner += `<div class="subq">▸ ${escapeHtml(it.sub_query)}</div>`;
    inner += `<div class="ents">entities: ${g.slice(0, 14).map((e) =>
      `<a class="linkish" data-entity="${escapeHtml(e.entity_name)}">${escapeHtml(e.entity_name)}</a>`).join(", ")}</div>`;
  }
  return inner ? `<details class="evidence"><summary>Entities touched</summary>${inner}</details>` : "";
}

// Turn [E1]/[Q1-E1] markers inside the rendered answer into links to their
// reference card. Only labels the backend actually issued are linkified --
// stray bracketed text in the model's prose is left alone. Text nodes only,
// so this cannot corrupt markup or touch code blocks.
function linkifyCitations(scope, citations) {
  const known = new Set((citations || []).map((c) => c.label));
  if (!known.size) return;
  const root = scope.querySelector(".answer");
  if (!root) return;
  const pattern = /\[([A-Za-z0-9][\w-]*)\]/g;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement && node.parentElement.closest("code, pre, a")
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  });
  const targets = [];
  let node;
  while ((node = walker.nextNode())) {
    pattern.lastIndex = 0;
    if (pattern.test(node.textContent)) targets.push(node);
  }
  for (const textNode of targets) {
    const text = textNode.textContent;
    const frag = document.createDocumentFragment();
    let last = 0;
    pattern.lastIndex = 0;
    let m;
    while ((m = pattern.exec(text))) {
      const label = m[1];
      if (!known.has(label)) continue;
      frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      const a = document.createElement("a");
      a.className = "cite";
      a.href = `#ref-${label}`;
      a.dataset.cite = label;
      a.textContent = `[${label}]`;
      frag.appendChild(a);
      last = m.index + m[0].length;
    }
    frag.appendChild(document.createTextNode(text.slice(last)));
    textNode.replaceWith(frag);
  }
}

function wireEvidenceLinks(scope) {
  scope.querySelectorAll("[data-tree]").forEach((a) =>
    a.onclick = () => { activateView("tree"); selectTreeNode(Number(a.dataset.tree)); });
  scope.querySelectorAll("[data-entity]").forEach((a) =>
    a.onclick = () => {
      const ent = (state.graph.nodes || []).find((n) => n.label === a.dataset.entity);
      if (ent) { activateView("graph"); selectEntity(ent.id); }
    });
  scope.querySelectorAll("[data-cite]").forEach((a) =>
    a.onclick = (e) => {
      e.preventDefault();
      const target = scope.querySelector(`#ref-${a.dataset.cite}`);
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.add("flash");
      setTimeout(() => target.classList.remove("flash"), 1200);
    });
}
