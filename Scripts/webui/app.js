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
  conversations: [],
  activeConversationId: null,
  currentRun: null,
  libraryTimer: null,
  selectedTreeId: null,
  treeFilteredIds: [],
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

const LISTING_TYPES = new Set(["CODE", "ALGORITHM"]);

// One numbered line of a listing. The number comes from a CSS counter rather
// than the text so that selecting the listing copies the code and not the
// gutter.
const listingLines = (text) =>
  String(text)
    .replace(/\n+$/, "")
    .split(/\r?\n/)
    .map((line) => `<span class="listingLine">${escapeHtml(line) || " "}</span>`)
    .join("");

// A program is reproduced exactly: its indentation is its syntax, and a "$" in
// it belongs to the program. <pre> is in MathJax's skipHtmlTags, so nothing
// inside here is ever mistaken for mathematics.
function renderProgram(text, language) {
  const tag = language
    ? `<div class="codeLang">${escapeHtml(language)}</div>`
    : "";
  return `${tag}<pre class="contentCode listingNums"><code>${listingLines(text)}</code></pre>`;
}

// Pseudocode is not a program. Its lines carry real mathematics -- "If $N < 1$",
// "Do while $I <= N$", "$I = I + I$" -- and inside a <pre> MathJax skips it, so
// every one of those was displayed as literal dollar-sign source. This keeps the
// line-per-line layout and the leading whitespace in CSS instead of in a <pre>,
// which is what lets the maths be typeset.
function renderPseudocode(text) {
  return `<div class="contentCode listingNums pseudocode">${listingLines(text)}</div>`;
}

// MathJax scans a container's whole text, not one line at a time, so a single
// unpaired "$" could pair with another line's and swallow every line between
// them. A listing only gets the maths treatment when every line closes the
// dollars it opens; anything else is reproduced verbatim, which is the safe
// direction to fail in.
const listingMathIsBalanced = (text) =>
  String(text)
    .split(/\r?\n/)
    .every((line) => ((line.match(/\$/g) || []).length % 2) === 0);

// Pseudocode when it reads as pseudocode, verbatim otherwise.
function renderListing(text, nodeType, language) {
  return nodeType === "ALGORITHM" && listingMathIsBalanced(text)
    ? renderPseudocode(text)
    : renderProgram(text, language);
}

function renderContent(n) {
  const c = (n && n.content) || "";
  if (!c.trim()) return "—";
  // A listing's line breaks and leading whitespace are what make it readable,
  // and for Python they are what make it mean anything. Prose rendering
  // collapses both, so every listing type is laid out line by line -- ALGORITHM
  // was falling through and being shown as a single wrapped line.
  const nodeType = String((n && n.type) || "").toUpperCase();
  if (LISTING_TYPES.has(nodeType)) {
    // ALGORITHM is exactly the pseudocode type, and CODE is exactly the type
    // whose text must survive byte-for-byte. That is the whole distinction, so
    // it decides which of the two renderings applies.
    return renderListing(c, nodeType, n && n.codeLanguage);
  }
  if (n && String(n.type).toUpperCase() === "LIST") {
    const items = c.split(/\r?\n/)
      .map((line) => line.replace(/^\s*(?:[-*+]|\d+[.)])\s+/, "").trim())
      .filter(Boolean);
    if (items.length)
      return `<ul class="contentList">${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }
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
  root.querySelectorAll(".prose td, .prose th, .answer td, .answer th").forEach((cell) => {
    const t = cell.textContent.trim();
    const words = t.split(/\s+/).length;
    const looksFormula = (/=|≈|≤|≥|→|∑|Σ|θ|∇/.test(t) || /\b\w+_[a-z0-9{]/i.test(t));
    if (t && words <= 10 && looksFormula) cell.classList.add("cellformula");
  });
}

// --- Rich text: markdown, mathematics and code in one answer ---------------
// Markdown and LaTeX fight each other. marked reads "\\" as a hard line break,
// "_" and "*" as emphasis and "[...]" as a link, so a formula parsed as markdown
// first comes back mangled and MathJax can no longer read it: an answer
// containing \begin{array}{l} ... \\ ... \end{array} lost its row separators
// before it was ever typeset. Code has the same conflict in reverse -- a "$" in
// a program is a dollar sign, not the start of mathematics.
//
// So mathematics is lifted out of the text before markdown runs and put back
// after. Code spans are lifted first and restored before markdown, purely so
// the maths patterns cannot match inside them; marked itself still renders the
// fences, and because <pre> is in MathJax's skipHtmlTags a formula printed
// inside a code block correctly stays source.
// Written as escapes because the characters themselves are invisible: these
// are private-use code points, which no markdown rule gives meaning to and
// which cannot occur in a real answer.
const CODE_MARK = ["\uE000", "\uE001"];
const MATH_MARK = ["\uE002", "\uE003"];

const CODE_PATTERNS = [/```[\s\S]*?```/g, /~~~[\s\S]*?~~~/g, /`[^`\n]+`/g];
const MATH_PATTERNS = [
  /\$\$[\s\S]+?\$\$/g,
  /\\\[[\s\S]+?\\\]/g,
  /\\\([\s\S]+?\\\)/g,
  // Inline $...$ on a single line. The closing "$" may not be followed by a
  // digit so that "$5 and $6" stays two amounts of money rather than becoming
  // one formula.
  /\$(?!\s)[^$\n]*[^\s$]\$(?!\d)/g,
];

function liftSpans(text, patterns, mark, store) {
  let out = String(text);
  for (const pattern of patterns) {
    out = out.replace(pattern, (match) => {
      store.push(match);
      return `${mark[0]}${store.length - 1}${mark[1]}`;
    });
  }
  return out;
}

function restoreSpans(text, mark, store, transform) {
  const pattern = new RegExp(`${mark[0]}(\\d+)${mark[1]}`, "g");
  return String(text).replace(pattern, (whole, index) => {
    const original = store[Number(index)];
    return original === undefined ? whole : transform(original);
  });
}

// Elements dropped with everything inside them. Anything else unrecognised is
// unwrapped instead, so unexpected markup loses its tag but never its text.
const RICH_TEXT_DROP = new Set([
  "SCRIPT", "STYLE", "IFRAME", "OBJECT", "EMBED", "NOSCRIPT", "TEMPLATE",
  "FORM", "INPUT", "BUTTON", "SELECT", "TEXTAREA", "LINK", "META", "BASE",
]);
const RICH_TEXT_TAGS = new Set([
  "P", "BR", "HR", "SPAN", "DIV", "STRONG", "EM", "B", "I", "U", "S", "DEL",
  "SUP", "SUB", "CODE", "PRE", "BLOCKQUOTE", "UL", "OL", "LI", "A", "IMG",
  "H1", "H2", "H3", "H4", "H5", "H6",
  "TABLE", "THEAD", "TBODY", "TFOOT", "TR", "TD", "TH", "CAPTION",
]);

// A same-origin path or a fragment. An answer that echoes document text could
// carry "javascript:" or an off-site URL, and neither may be followed.
const localUrl = (value) => {
  const v = String(value ?? "").trim();
  return /^[/#][^\s"']*$/.test(v) ? v : null;
};

function sanitizeRichText(html) {
  const parsed = new DOMParser().parseFromString(String(html || ""), "text/html");
  const build = (node, into) => {
    for (const child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        into.appendChild(document.createTextNode(child.nodeValue));
        continue;
      }
      if (child.nodeType !== Node.ELEMENT_NODE) continue;
      const tag = child.tagName.toUpperCase();
      if (RICH_TEXT_DROP.has(tag)) continue;
      if (!RICH_TEXT_TAGS.has(tag)) {
        build(child, into); // unwrap: keep the text, discard the tag
        continue;
      }
      const el = document.createElement(tag.toLowerCase());
      if (tag === "A") {
        const href = localUrl(child.getAttribute("href"));
        if (href) el.setAttribute("href", href);
      } else if (tag === "IMG") {
        const src = localUrl(child.getAttribute("src"));
        if (!src) continue; // an image we cannot vouch for is not shown at all
        el.setAttribute("src", src);
        el.setAttribute("loading", "lazy");
        el.setAttribute("alt", String(child.getAttribute("alt") || ""));
      }
      if (tag === "TD" || tag === "TH") {
        for (const name of ["colspan", "rowspan"]) {
          const value = child.getAttribute(name);
          if (value && /^[1-9]\d{0,2}$/.test(value)) el.setAttribute(name, value);
        }
      }
      // Only the language hint markdown puts on a fence survives, because it is
      // the one class the stylesheet reads.
      const cls = String(child.getAttribute("class") || "");
      const language = cls.split(/\s+/).find((name) => /^language-[\w+#.-]+$/.test(name));
      if (language) el.setAttribute("class", language);
      build(child, el);
      into.appendChild(el);
    }
  };
  const root = document.createElement("div");
  build(parsed.body, root);
  return root.innerHTML;
}

// Markdown + LaTeX + code, rendered safely. Callers must still run typesetMath
// on the container afterwards: this leaves the formulas as source for MathJax.
function renderRichText(text) {
  const source = String(text ?? "");
  if (!source.trim()) return "";
  const code = [];
  const math = [];
  let staged = liftSpans(source, CODE_PATTERNS, CODE_MARK, code);
  staged = liftSpans(staged, MATH_PATTERNS, MATH_MARK, math);
  staged = restoreSpans(staged, CODE_MARK, code, (original) => original);
  const parsed = window.marked ? marked.parse(staged) : escapeHtml(staged).replace(/\n/g, "<br>");
  // Mathematics goes back escaped. MathJax reads text, and the browser decodes
  // "&lt;" to "<" before it does, so "$a &lt; b$" typesets as a < b while the
  // markup itself stays well-formed.
  return restoreSpans(sanitizeRichText(parsed), MATH_MARK, math, escapeHtml);
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

function treeImageUrl(nodeId) {
  // Node ids repeat in every document. Include the selected run in the URL so
  // the browser cannot reuse a crop cached for node #3 from a previous PDF.
  return `/api/tree-image?id=${encodeURIComponent(nodeId)}&run=${encodeURIComponent(state.currentRun || "")}`;
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
    const hasRun = await loadRuns();
    if (hasRun) await loadData();
    else activateView("library");
  } catch (err) {
    console.error("NoeticsRAG load failed:", err);
    alert("Failed to load index: " + err.message);
  }
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => activateView(tab.dataset.view));
  });
}

// A viewport gesture is in progress. Two things watch this: a programmatic
// focus scheduled before the gesture must not yank the view back afterwards
// (compare the sequence you captured), and the tree drops its glow for the
// duration, because canvas shadowBlur is redrawn on every frame of a pan.
const GESTURE_IDLE_MS = 150;
const gestureWatchers = new Set();
let gestureSeq = 0;
let gestureIdle = null;

// Watchers are told the view has settled (true) or is moving (false). The
// settle is timer-driven rather than tied to a gesture-end event, so a wheel
// that never reports an end still restores whatever was suspended.
function markGesture() {
  gestureSeq += 1;
  if (gestureIdle === null) gestureWatchers.forEach((watch) => watch(false));
  else clearTimeout(gestureIdle);
  gestureIdle = setTimeout(() => {
    gestureIdle = null;
    gestureWatchers.forEach((watch) => watch(true));
  }, GESTURE_IDLE_MS);
}

const gestureActive = () => gestureIdle !== null;

const ZOOM_MIN = 0.05;
const ZOOM_MAX = 4;

// Wheel PANS the view, Shift+wheel = horizontal, Ctrl/Cmd+wheel = zoom.
// Requires the network's zoomView to be off.
function enableWheelPan(containerId, getNet) {
  const el = document.getElementById(containerId);
  if (!el) return;
  let wheelFrame = null;
  let pendingX = 0;
  let pendingY = 0;
  let pendingZoom = 0;
  let pointerX = 0;
  let pointerY = 0;
  el.addEventListener("wheel", (e) => {
    const net = getNet();
    if (!net) return;
    e.preventDefault();
    markGesture();
    pointerX = e.clientX;
    pointerY = e.clientY;
    if (e.ctrlKey || e.metaKey) {
      pendingZoom += e.deltaY;
    } else {
      let dx = e.deltaX, dy = e.deltaY;
      if (e.shiftKey && dx === 0) { dx = e.deltaY; dy = 0; }
      pendingX += dx;
      pendingY += dy;
    }
    if (wheelFrame !== null) return;
    wheelFrame = requestAnimationFrame(() => {
      wheelFrame = null;
      const current = getNet();
      if (!current) return;
      const from = current.getScale() || 1;
      const to = pendingZoom
        ? Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, from * Math.exp(-pendingZoom * 0.0015)))
        : from;
      // Anchor the zoom on the cursor: the graph point under it keeps its
      // screen position, which is what makes zooming feel like it tracks the
      // mouse instead of dragging the viewport centre around.
      const box = el.getBoundingClientRect();
      const offX = pointerX - box.left - box.width / 2;
      const offY = pointerY - box.top - box.height / 2;
      const anchor = 1 / from - 1 / to;
      const pos = current.getViewPosition();
      current.moveTo({
        scale: to,
        position: {
          x: pos.x + offX * anchor + pendingX / to,
          y: pos.y + offY * anchor + pendingY / to,
        },
        animation: false,
      });
      pendingX = pendingY = pendingZoom = 0;
    });
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
    else {
      state.treeNet.redraw();
      if (state.selectedTreeId != null) focusTreeNode(state.selectedTreeId, { rebuild: "never" });
      else state.treeNet.fit();
    }
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

function stageRows(stage, waitingOn) {
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
  if (stage.status === "pending")
    add("Waiting", waitingOn || "Pipeline start");
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

function stageStepHtml(stage, waitingOn) {
  const pct = stage.progress
    ? Math.round((stage.progress.done / stage.progress.total) * 100)
    : null;
  const statusLabel = stage.status === "running"
    ? "In progress"
    : stage.status.charAt(0).toUpperCase() + stage.status.slice(1);
  return `<section class="stageStep ${escapeHtml(stage.status)}" aria-label="${escapeHtml(stage.name)} stage: ${escapeHtml(stage.status)}">
    <div class="stageSummary">
      <span class="nm">${escapeHtml(stage.name)}</span>
      <span class="stageStatus">${escapeHtml(statusLabel)}</span>
      <span class="st">${escapeHtml(stageHeadline(stage))}</span>
      <span class="bar"><i${pct != null ? ` style="width:${pct}%"` : ""}></i></span>
    </div>
    <div class="stageBody">
      ${stageRows(stage, waitingOn)}
      ${stage.error ? `<div class="runError">${escapeHtml(String(stage.error))}</div>` : ""}
    </div>
  </section>`;
}

// The reader's own label for a document, shown just before the status pill so
// the eye reaches it on the way to the status rather than hunting for it. An
// untagged document offers a quiet "+ tag" affordance instead of a chip, so the
// row does not fill with placeholders.
const MAX_TAG_LENGTH = 32;

function runTagHtml(tag) {
  const label = String(tag || "").trim();
  return label
    ? `<button class="runTag" type="button" title="Edit tag">${escapeHtml(label)}</button>`
    : `<button class="runTag empty" type="button" title="Add a tag">+ tag</button>`;
}

async function editRunTag(card) {
  const runDir = card.dataset.run;
  const button = card.querySelector(".runTag");
  const current = button.classList.contains("empty") ? "" : button.textContent.trim();
  const entered = prompt(
    `Tag for “${card.querySelector(".runName").textContent}”\n` +
      `Up to ${MAX_TAG_LENGTH} characters. Leave empty to remove the tag.`,
    current,
  );
  if (entered === null) return; // cancelled, not cleared
  try {
    await api("/api/run-tag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ runDir, tag: entered }),
    });
  } catch (err) {
    alert("Could not save the tag: " + err.message);
    return;
  }
  await loadLibrary();
}

// Stop, and start again. Delete used to be the only way to end a run, so the
// only way to stop something was to throw away the hours it had already spent --
// and a run whose driver was killed kept claiming to be running with nothing
// offered but Delete. A stopped or interrupted run keeps its finished stages.
//
// "Rebuild from" discards that stage and everything after it, because a vdb
// built from the old graph cannot stand while the graph is rebuilt. Resuming
// without choosing a stage keeps everything and simply continues.
function runLifecycleHtml(run) {
  if (run.isIndexing) {
    return '<button class="btn stopRun" title="Stop indexing; finished stages are kept">Stop</button>';
  }
  if (run.codeChanged) {
    return `<button class="btn" disabled title="The indexing code changed since this run started, so it cannot be resumed — starting would index from nothing under a new revision. Delete it and add the document again instead.">Resume</button>`;
  }
  if (!run.canResume) return "";
  return `<label class="rebuildFrom" title="Resume continues where it stopped. Rebuilding discards that stage and everything after it.">
      <select class="rebuildSelect">
        <option value="">Resume</option>
        <option value="tree">Rebuild from tree</option>
        <option value="graph">Rebuild from graph</option>
        <option value="vdb">Rebuild from vdb</option>
      </select>
    </label>
    <button class="btn startRun">Start</button>`;
}

function runCardHtml(run) {
  const status = run.overallStatus || (run.viewable ? "succeeded" : "pending");
  const isCurrent = run.runDir === state.currentRun;
  const canOpen = run.viewable && !isCurrent;
  const stages = run.stages
    .map((stage, index) =>
      stageStepHtml(stage, index ? run.stages[index - 1].name : null))
    .join("");
  const total = run.stages.reduce((a, s) => a + (s.seconds || 0), 0);
  return `<article class="runCard ${isCurrent ? "current" : ""}" data-run="${escapeHtml(run.runDir)}">
    <div class="runCardTop">
      <span class="runName" title="${escapeHtml(run.name)}">${escapeHtml(run.name)}</span>
      ${runTagHtml(run.tag)}
      <span class="pill ${escapeHtml(status)}">${escapeHtml(status)}</span>
      ${isCurrent ? '<span class="pill">open</span>' : ""}
      ${runLifecycleHtml(run)}
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

  $("runGrid").querySelectorAll(".stopRun").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const card = btn.closest(".runCard");
      if (!confirm("Stop indexing this document? Finished stages are kept and it can be started again.")) return;
      btn.disabled = true;
      btn.textContent = "Stopping…";
      try {
        await api("/api/run-stop", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ runDir: card.dataset.run }),
        });
      } catch (err) {
        alert("Could not stop this run: " + err.message);
      }
      await loadLibrary();
    });
  });

  $("runGrid").querySelectorAll(".startRun").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const card = btn.closest(".runCard");
      const select = card.querySelector(".rebuildSelect");
      const fromStage = select && select.value ? select.value : null;
      if (fromStage && !confirm(
        `Rebuild from the ${fromStage} stage? That stage and every stage after it ` +
        `will be discarded and built again.`)) return;
      btn.disabled = true;
      btn.textContent = "Starting…";
      try {
        await api("/api/run-start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ runDir: card.dataset.run, fromStage }),
        });
      } catch (err) {
        alert("Could not start this run: " + err.message);
      }
      await loadLibrary();
    });
  });

  $("runGrid").querySelectorAll(".runTag").forEach((btn) => {
    btn.addEventListener("click", () => editRunTag(btn.closest(".runCard")));
  });

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
  state.currentRun = data.current || null;
  const sel = $("runSelect");
  sel.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = data.runs.length
    ? "Select a document…"
    : "No indexed documents";
  placeholder.selected = !state.currentRun;
  sel.appendChild(placeholder);
  for (const run of data.runs) {
    const opt = document.createElement("option");
    opt.value = run.path;
    opt.textContent = run.name + (run.hasGraph ? "" : "  (no graph)");
    if (run.path === data.current) opt.selected = true;
    sel.appendChild(opt);
  }
  setDocumentViewsEnabled(Boolean(state.currentRun));
  sel.disabled = data.runs.length === 0;
  sel.onchange = async () => {
    if (!sel.value) return;
    sel.disabled = true;
    try {
      const result = await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ runDir: sel.value }),
      });
      state.currentRun = result.runDir;
      setDocumentViewsEnabled(true);
      resetForNewRun();
      await loadData();
    } catch (err) {
      alert("Could not switch document: " + err.message);
    } finally {
      sel.disabled = false;
    }
  };
  return Boolean(state.currentRun);
}

function setDocumentViewsEnabled(enabled) {
  document.querySelectorAll('.tab[data-view="graph"], .tab[data-view="tree"], .tab[data-view="chat"]')
    .forEach((tab) => { tab.disabled = !enabled; });
}

function resetForNewRun() {
  if (state.graphNet) { state.graphNet.destroy(); state.graphNet = null; }
  if (state.treeNet) { state.treeNet.destroy(); state.treeNet = null; }
  state.graphBuilt = state.treeBuilt = false;
  state.selectedTreeId = null;
  state.treeFilteredIds = [];
  state.treeMatchCount = 0;
  state.treeHighlit = null;
  treeHighlightType = null;
  // History is stored per run directory, so a different PDF has a different
  // set of conversations. Clearing the id here stops the next question being
  // appended to the previous document's chat.
  state.conversations = [];
  state.activeConversationId = null;
  $("messages").innerHTML = "";
}

/* ============================== load data ================================ */
async function loadData() {
  const [tree, graph] = await Promise.all([api("/api/tree"), api("/api/graph")]);
  state.tree = tree;
  state.graph = graph;

  state.treeNodesById = new Map(tree.nodes.map((n) => [n.id, n]));
  state.treeDepthById = computeTreeDepths(tree.nodes);
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

  // Saved history belongs to this run, so it loads with the run's data. The
  // most recent conversation is reopened rather than leaving an empty pane.
  await loadConversations();
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
  $("treeSearch").addEventListener("input", renderTreeList);
  $("treeSearch").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && state.treeFilteredIds.length) {
      event.preventDefault();
      selectTreeNode(state.treeFilteredIds[0]);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      stepTreeResult(event.key === "ArrowDown" ? 1 : -1);
    } else if (event.key === "Escape") {
      $("treeSearchClear").click();
    }
  });
  $("treeTypeFilter").addEventListener("change", () =>
    applyTreeTypeSelection($("treeTypeFilter").value, false));
  $("treeSearchClear").addEventListener("click", () => {
    $("treeSearch").value = "";
    $("treeTypeFilter").value = "";
    clearTreeHighlight();
    renderTreeList();
    $("treeSearch").focus();
  });
  $("treePrevResult").addEventListener("click", () => stepTreeResult(-1));
  $("treeNextResult").addEventListener("click", () => stepTreeResult(1));
  $("treeLabels").addEventListener("change", () => { if (state.treeBuilt) rebuildTreeLabels(); });
  $("treeFit").addEventListener("click", () => state.treeNet && state.treeNet.fit({ animation: true }));
  $("treeFocus").addEventListener("click", () => {
    if (state.selectedTreeId != null) focusTreeNode(state.selectedTreeId, { rebuild: "never" });
  });
}

// Hierarchical layout is superlinear in node count, so a 16k-node document
// hangs the tab before it paints. Draw a bounded slice instead: the shallow
// structure, plus whatever the reader is currently looking at.
const TREE_RENDER_BUDGET = 600;
const SIBLING_WINDOW = 12;      // siblings kept either side of the focus
const MAX_CHILDREN_DRAWN = 10;  // children drawn per node, so depth survives

function treeRenderSubset(all, focusId) {
  if (all.length <= TREE_RENDER_BUDGET) return all;
  const byId = new Map(all.map((n) => [n.id, n]));
  const childrenOf = new Map();
  for (const n of all) {
    if (n.parentId === null || n.parentId === undefined) continue;
    if (!childrenOf.has(n.parentId)) childrenOf.set(n.parentId, []);
    childrenOf.get(n.parentId).push(n);
  }

  const focus =
    byId.get(focusId) || all.find((n) => n.parentId === null || n.parentId === undefined) || all[0];
  const keep = new Map();

  // The path back to the root gives the node its context.
  for (let cur = focus; cur; cur = byId.get(cur.parentId)) keep.set(cur.id, cur);

  // A window of siblings, not all of them: a title here can have thousands,
  // which would spend the whole budget before anything below is reached.
  if (focus.parentId !== null && focus.parentId !== undefined) {
    const sibs = childrenOf.get(focus.parentId) || [];
    const at = Math.max(0, sibs.findIndex((s) => s.id === focus.id));
    for (const s of sibs.slice(Math.max(0, at - SIBLING_WINDOW), at + SIBLING_WINDOW + 1)) {
      keep.set(s.id, s);
    }
  }

  // Then spend the budget going *down* from the focus. Taking only the first
  // few children of each node keeps breadth from crowding out depth, so the
  // view shows several levels instead of one very wide one.
  const queue = [focus];
  while (queue.length && keep.size < TREE_RENDER_BUDGET) {
    const cur = queue.shift();
    const kids = childrenOf.get(cur.id) || [];
    for (const c of kids.slice(0, MAX_CHILDREN_DRAWN)) {
      if (keep.size >= TREE_RENDER_BUDGET) break;
      keep.set(c.id, c);
      queue.push(c);
    }
  }

  // An edge is only drawn when its parent is present, so pull ancestors in.
  for (const n of [...keep.values()]) {
    for (let p = byId.get(n.parentId); p && !keep.has(p.id); p = byId.get(p.parentId)) {
      keep.set(p.id, p);
    }
  }
  return [...keep.values()];
}

function treeTypeRenderSubset(all, type, focusId) {
  const byId = new Map(all.map((node) => [node.id, node]));
  const matches = all.filter((node) => node.type === type);
  if (!matches.length) return treeRenderSubset(all, focusId);

  const focusIndex = Math.max(0, matches.findIndex((node) => node.id === focusId));
  const ordered = [];
  for (let distance = 0; ordered.length < matches.length; distance += 1) {
    const after = focusIndex + distance;
    const before = focusIndex - distance;
    if (after < matches.length) ordered.push(matches[after]);
    if (distance && before >= 0) ordered.push(matches[before]);
  }

  const keep = new Map();
  for (const match of ordered) {
    const chain = [];
    const seen = new Set();
    for (let current = match; current && !seen.has(current.id); current = byId.get(current.parentId)) {
      seen.add(current.id);
      if (!keep.has(current.id)) chain.push(current);
    }
    if (keep.size && keep.size + chain.length > TREE_RENDER_BUDGET) continue;
    for (const node of chain.reverse()) keep.set(node.id, node);
    if (keep.size >= TREE_RENDER_BUDGET) break;
  }
  return [...keep.values()];
}

// Refill the drawn slice in place. Replacing the network instead loses the
// canvas, and vis re-runs the hierarchical layout on a DataSet change anyway.
// The scoped styling is folded into the add: recolouring afterwards was a
// second DataSet mutation, so a legend click drew the slice unstyled and then
// restyled it a frame later -- the flicker.
function populateTreeNetwork() {
  const labelsOn = $("treeLabels").checked;
  const shown = treeHighlightType
    ? treeTypeRenderSubset(state.tree.nodes, treeHighlightType, state.selectedTreeId)
    : treeRenderSubset(state.tree.nodes, state.selectedTreeId);
  const shownIds = new Set(shown.map((n) => n.id));
  state.treeShownCount = shown.length;
  state.treeHighlit = treeHighlightType
    ? new Set(shown.filter((n) => n.type === treeHighlightType).map((n) => n.id))
    : null;
  state.treeNodes.clear();
  state.treeEdges.clear();
  state.treeNodes.add(shown.map((n) => toVisTreeNode(n, labelsOn, state.treeHighlit)));
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
    if (params.nodes.length) { selectTreeNode(params.nodes[0]); return; }
    // Empty space clears the legend scope and returns to the whole tree. The
    // selection and its detail panel are kept -- only the scope is dropped.
    // Ignored right after a gesture: vis reports a tap for a drag that barely
    // moved, which would throw the scope away mid-pan.
    if (!treeHighlightType || gestureActive()) return;
    clearTreeHighlight(true);
  });
  state.treeNet.on("dragStart", markGesture);
  state.treeNet.on("dragging", markGesture);
  gestureWatchers.add(setTreeGlow);
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

function computeTreeDepths(nodes) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const depths = new Map();
  const depthOf = (node, seen = new Set()) => {
    if (!node || node.parentId == null) return 0;
    if (depths.has(node.id)) return depths.get(node.id);
    if (seen.has(node.id)) return 0;
    seen.add(node.id);
    const depth = 1 + depthOf(byId.get(node.parentId), seen);
    depths.set(node.id, depth);
    return depth;
  };
  for (const node of nodes) depths.set(node.id, depthOf(node));
  return depths;
}

const TREE_GLOW_SIZE = 16;
// shadowBlur is re-rasterised on every frame of a pan, and the cost scales with
// how many nodes carry it. A scope this small is free to keep glowing while the
// view moves; suspending it would only make the glow blink between scroll
// bursts, which is worse than the frames it would save.
const TREE_GLOW_ALWAYS_UNDER = 60;

function glowShadow(type) {
  return { enabled: true, color: hexA(colorForType(type), 0.5), size: TREE_GLOW_SIZE, x: 0, y: 0 };
}

function treeGlowOn(highlit) {
  return !gestureActive() || !highlit || highlit.size < TREE_GLOW_ALWAYS_UNDER;
}

// Every partial update to a drawn node must go through this. vis decides
// between a repaint and a full hierarchical relayout by comparing the incoming
// `level` against the stored one -- an update that omits the field compares a
// number against undefined, so it always reads as a layout change and the whole
// tree is re-laid out. Restating the unchanged level keeps it a repaint.
function treeNodePatch(id, fields) {
  return {
    id,
    level: state.treeLevelById ? state.treeLevelById.get(id) : undefined,
    ...fields,
  };
}

// The one place a tree node's paint is decided, so an add and a recolour cannot
// disagree. `highlit` null means no type is scoped and every node paints
// normally; otherwise members glow and the rest are gently dimmed.
function treeNodeStyle(n, highlit, glow) {
  const fill = {
    background: n.type === "TITLE" ? "#eef6ff" : "#ffffff",
    border: colorForType(n.type),
    highlight: { background: "#fff7ed", border: "#111" },
  };
  if (!highlit) {
    return { opacity: 1, borderWidth: 1.5, color: fill,
      font: { color: "#1c2430" }, shadow: { enabled: false } };
  }
  if (highlit.has(n.id)) {
    return { opacity: 1, borderWidth: 2, color: fill, font: { color: "#1c2430" },
      shadow: glow ? glowShadow(n.type) : { enabled: false } };
  }
  return { opacity: 0.5, borderWidth: 1,
    color: { background: "#f7f9fb", border: "#e6eaee" },
    font: { color: "#9aa4ad" }, shadow: { enabled: false } };
}

function toVisTreeNode(n, labelsOn, highlit) {
  const short = clip(nodeLabel(n), 40);
  return {
    id: n.id,
    level: state.treeLevelById ? state.treeLevelById.get(n.id) : undefined,
    label: labelsOn ? `#${n.id} ${short}` : `#${n.id}`,
    title: `#${n.id} · ${n.type}${n.page ? " · p" + n.page : ""}\n${clip(n.content, 160)}`,
    ...treeNodeStyle(n, highlit, treeGlowOn(highlit)),
  };
}

function rebuildTreeLabels() {
  const labelsOn = $("treeLabels").checked;
  // update() inserts ids it does not find, so relabel only what is drawn --
  // otherwise toggling labels would quietly restore the whole tree.
  state.treeNodes.update(state.tree.nodes
    .filter((n) => state.treeNodes.get(n.id))
    .map((n) => treeNodePatch(n.id, {
      label: labelsOn ? `#${n.id} ${clip(nodeLabel(n), 40)}` : `#${n.id}`,
    })));
}

function renderTreeStats() {
  const total = state.tree.nodes.length;
  const drawn = state.treeShownCount || 0;
  $("treeStats").textContent = `${state.treeMatchCount || 0} matching · ${total} total`
    + (drawn && drawn < total ? ` · ${drawn} on canvas` : "");
}

// Selection moves one class. Re-running renderTreeList would rebuild up to a
// thousand rail buttons in the same frame as a canvas click, which is what made
// clicking a node feel like the canvas had stalled.
function markTreeListSelection(id) {
  const root = $("treeList");
  const previous = root.querySelector(".treeRailItem.active");
  if (previous) previous.classList.remove("active");
  const current = root.querySelector(`[data-node-id="${id}"]`);
  if (!current) return;
  current.classList.add("active");
  current.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function renderTreeList() {
  const q = $("treeSearch").value.trim().toLowerCase();
  const type = $("treeTypeFilter").value;
  const list = state.tree.nodes.filter((n) => {
    if (n.type === "root") return false;
    const okType = !type || n.type === type;
    const hay = `#${n.id} ${n.id} ${n.content} ${n.type} ${n.summary} ${n.caption || ""}`.toLowerCase();
    return okType && (!q || hay.includes(q));
  });
  state.treeFilteredIds = list.map((n) => n.id);
  state.treeMatchCount = list.length;
  renderTreeStats();
  $("treePrevResult").disabled = list.length === 0;
  $("treeNextResult").disabled = list.length === 0;
  const root = $("treeList");
  root.innerHTML = "";
  for (const n of list.slice(0, 1000)) {
    const btn = document.createElement("button");
    btn.className = "railItem treeRailItem";
    btn.dataset.nodeId = n.id;
    btn.classList.toggle("active", n.id === state.selectedTreeId);
    const indent = Math.min(state.treeDepthById?.get(n.id) || 0, 6) * 9;
    btn.style.paddingLeft = 10 + indent + "px";
    btn.innerHTML = `<span class="treeItemTop"><span class="nodeId">#${n.id}</span>
      <span class="typeDot" style="background:${colorForType(n.type)}"></span>
      <span class="nodeType">${escapeHtml(n.type)}</span>
      ${n.page ? `<span class="nodePage">p.${n.page}</span>` : ""}</span>
      <span class="t">${escapeHtml(clip(nodeLabel(n) || "(no text)", 84))}</span>`;
    btn.onclick = () => selectTreeNode(n.id);
    root.appendChild(btn);
  }
  if (!list.length) {
    root.innerHTML = `<div class="treeNoResults">No nodes match this search and type filter.</div>`;
  } else if (list.length > 1000) {
    root.insertAdjacentHTML("beforeend", `<div class="treeNoResults">Showing the first 1,000 matches. Refine the search to narrow the list.</div>`);
  }
}

function stepTreeResult(direction) {
  const ids = state.treeFilteredIds || [];
  if (!ids.length) return;
  let at = ids.indexOf(state.selectedTreeId);
  if (at < 0) at = direction > 0 ? -1 : 0;
  selectTreeNode(ids[(at + direction + ids.length) % ids.length]);
}

let treeHighlightType = null;
function renderTreeLegend() {
  const types = [...new Set(state.tree.nodes.map((n) => n.type))].filter((t) => t !== "root").sort();
  $("treeLegend").innerHTML = types.map((t) =>
    `<span class="swatch" data-type="${escapeHtml(t)}"><span class="dot" style="background:${colorForType(t)}"></span>${escapeHtml(t)}</span>`).join("");
  $("treeLegend").querySelectorAll(".swatch").forEach((sw) =>
    sw.onclick = () => applyTreeTypeSelection(sw.dataset.type, true));
}

function hexA(hex, a) {
  const m = String(hex).replace("#", "");
  const r = parseInt(m.slice(0, 2), 16), g = parseInt(m.slice(2, 4), 16), b = parseInt(m.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

// Clear any legend highlight and restore the default tree styling.
function clearTreeHighlight(clearFilter = false) {
  const hadTypeScope = Boolean(treeHighlightType);
  treeHighlightType = null;
  const lg = $("treeLegend");
  if (lg) lg.querySelectorAll(".swatch").forEach((s) => s.classList.remove("active"));
  if (clearFilter && $("treeTypeFilter")) {
    $("treeTypeFilter").value = "";
    renderTreeList();
  }
  if (!state.treeBuilt) return;
  if (!hadTypeScope) { recolorTree(null); return; }
  // Dropping the scope recomputes the slice, so every drawn node lands at new
  // hierarchical coordinates. Settle the viewport back on the selection instead
  // of leaving the reader looking at wherever the old layout used to be.
  if (state.selectedTreeId != null) {
    focusTreeNode(state.selectedTreeId, { rebuild: "force", animate: false });
  } else {
    populateTreeNetwork();
    state.treeNet.once("afterDrawing", () => state.treeNet.fit());
  }
  renderTreeStats();
}

// Keep legend, type selector, result list, bounded canvas, and selection in one
// state transition. Previously the legend affected only nodes that happened to
// be drawn, so a valid type could appear to do nothing.
function applyTreeTypeSelection(type, toggle) {
  if (!$("treeView").classList.contains("active")) activateView("tree");
  if (!state.treeBuilt) buildTreeNetwork();
  if (toggle && treeHighlightType === type) {
    $("treeTypeFilter").value = "";
    clearTreeHighlight();
    renderTreeList();
    return;
  }
  if (!type) {
    clearTreeHighlight();
    renderTreeList();
    return;
  }

  treeHighlightType = type;
  $("treeTypeFilter").value = type;
  $("treeLegend").querySelectorAll(".swatch").forEach((s) =>
    s.classList.toggle("active", s.dataset.type === type));
  const ids = state.tree.nodes.filter((n) => n.type === type).map((n) => n.id);
  renderTreeList();
  if (!ids.length) {
    recolorTree(new Set());
    return;
  }

  // Select a real match when the current focus is a different type, and force
  // the slice to be rebuilt around the requested type even if that node was
  // already drawn under the previous scope.
  const focusId = ids.includes(state.selectedTreeId) ? state.selectedTreeId : ids[0];
  selectTreeNode(focusId, { rebuild: "force", animate: false });
}

function recolorTree(hlSet) {
  // DataSet.update inserts unknown ids. Updating the full document here used
  // to reinsert thousands of nodes outside the bounded canvas and destroy the
  // hierarchical layout after filtering/highlighting.
  const drawn = state.tree.nodes.filter((n) => state.treeNodes.get(n.id));
  state.treeHighlit = hlSet && new Set([...hlSet].filter((id) => state.treeNodes.get(id)));
  const glow = treeGlowOn(state.treeHighlit);
  state.treeNodes.update(
    drawn.map((n) => treeNodePatch(n.id, treeNodeStyle(n, state.treeHighlit, glow)))
  );
}

// Carry the glow only while the view is still. Nothing else about the styling
// moves, so this is one shadow-only update on the glowing nodes rather than a
// restyle of the slice.
function setTreeGlow(on) {
  const highlit = state.treeHighlit;
  if (!state.treeBuilt || !highlit || highlit.size < TREE_GLOW_ALWAYS_UNDER) return;
  const updates = [];
  for (const id of highlit) {
    const n = state.treeNodesById.get(id);
    if (n && state.treeNodes.get(id)) {
      updates.push(treeNodePatch(id, {
        shadow: on ? glowShadow(n.type) : { enabled: false },
      }));
    }
  }
  if (updates.length) state.treeNodes.update(updates);
}

// `rebuild`: "missing" refills the bounded slice when the node is not drawn,
// "force" refills it even when it is (the scope changed, so the slice must be
// recomputed), "never" leaves the slice alone.
function focusTreeNode(id, { rebuild = "missing", animate = true } = {}) {
  if (!state.treeBuilt) buildTreeNetwork();
  const rebuilding = rebuild === "force"
    || (rebuild === "missing" && !state.treeNodes.get(id));
  const seq = gestureSeq;
  const focus = () => {
    // The user panned or zoomed after this focus was queued, so their gesture
    // wins: vis finalises an in-flight animation on the next moveTo, and
    // proceeding here would teleport the view out from under them. A rebuild is
    // exempt -- the slice it replaced no longer exists at those coordinates, so
    // there is no view of theirs left to protect and skipping would strand them
    // on empty canvas.
    if (!rebuilding && seq !== gestureSeq) return;
    if (!state.treeNet || !state.treeNodes.get(id)) return;
    state.treeNet.selectNodes([id]);
    state.treeNet.focus(id, {
      scale: 0.92,
      // A rebuilt slice has entirely new coordinates, so easing from the old
      // viewport animates between two unrelated views. Cut straight there.
      animation: animate && !rebuilding
        ? { duration: 280, easingFunction: "easeInOutQuad" }
        : false,
    });
  };
  // Hierarchical positions are recomputed asynchronously after a bounded
  // slice is replaced. Focus only after the first completed paint.
  if (rebuilding) {
    state.treeNet.once("afterDrawing", () => requestAnimationFrame(focus));
    populateTreeNetwork();
  } else if (state.treeNodes.get(id)) {
    requestAnimationFrame(focus);
  }
}

function treeBreadcrumbs(node) {
  const chain = [];
  const seen = new Set();
  for (let cur = node; cur && !seen.has(cur.id); cur = state.treeNodesById.get(cur.parentId)) {
    seen.add(cur.id);
    chain.unshift(cur);
  }
  return chain.map((item, index) => {
    const current = index === chain.length - 1;
    const label = item.type === "root" ? "Document" : clip(nodeLabel(item), 28);
    return `${index ? '<span class="crumbSep">›</span>' : ""}${
      current ? `<span class="crumb current">${escapeHtml(label)}</span>`
        : `<button class="crumb" data-tree="${item.id}">${escapeHtml(label)}</button>`
    }`;
  }).join("");
}

function selectTreeNode(id, options = {}) {
  const n = state.treeNodesById.get(id);
  if (!n) return;
  state.selectedTreeId = id;
  focusTreeNode(id, { rebuild: options.rebuild || "missing", animate: options.animate !== false });
  markTreeListSelection(id);
  renderTreeStats();
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
  const siblings = parent
    ? (parent.children || []).map((cid) => state.treeNodesById.get(cid)).filter(Boolean)
    : [];
  const siblingIndex = siblings.findIndex((item) => item.id === id);
  const previousSibling = siblingIndex > 0 ? siblings[siblingIndex - 1] : null;
  const nextSibling = siblingIndex >= 0 && siblingIndex < siblings.length - 1
    ? siblings[siblingIndex + 1] : null;
  const img = (n.hasImage || (n.meta && (n.meta.crop_uri || n.meta.img_path)))
    ? `<div class="section-h">Image</div>
       <img class="nodeimg" src="${treeImageUrl(id)}" alt="Source crop for node ${id}"
            title="Click to view the full-resolution crop"
            onerror="this.nextElementSibling.hidden=false" />
       <p class="prose imageError" hidden>The source crop could not be loaded.</p>` : "";

  $("treeDetail").innerHTML = `
    <nav class="treeBreadcrumbs" aria-label="Node path">${treeBreadcrumbs(n)}</nav>
    <h2>#${id} ${escapeHtml(clip(nodeLabel(n), 90))}</h2>
    <div class="kv">
      <dt>Type</dt><dd><span class="tag" style="background:${colorForType(n.type)}">${escapeHtml(n.type)}</span></dd>
      ${n.page ? `<dt>Page</dt><dd>${n.page}</dd>` : ""}
      <dt>Parent</dt><dd>${parent ? `<a class="linkish" data-tree="${parent.id}">#${parent.id} ${escapeHtml(clip(nodeLabel(parent), 40))}</a>` : "— (root)"}</dd>
      <dt>Children</dt><dd>${(n.children || []).length}</dd>
    </div>
    ${img}
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
    <div class="nodeNav" aria-label="Node navigation">
      <button class="miniBtn" data-tree="${parent?.id ?? ""}" ${parent ? "" : "disabled"}>↑ Parent</button>
      <button class="miniBtn" data-tree="${previousSibling?.id ?? ""}" ${previousSibling ? "" : "disabled"}>← Previous sibling</button>
      <button class="miniBtn" data-tree="${nextSibling?.id ?? ""}" ${nextSibling ? "" : "disabled"}>Next sibling →</button>
    </div>`;
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
const ASSISTANT_NAME = "Noetics-AI";
const SIDEBAR_COLLAPSED_KEY = "noeticsrag.chatSidebarCollapsed";
const CHAT_INTRO =
  `<div class="answer">Ask a question about the indexed document. Your conversation ` +
  `will be saved automatically, including the retrieved tree nodes and graph evidence.</div>`;

function wireChat() {
  $("chatForm").addEventListener("submit", submitChat);
  const ta = $("question");
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitChat(e); }
  });
  ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 160) + "px"; });
  $("newChatBtn").addEventListener("click", startNewChat);
  $("sidebarToggle").addEventListener("click", toggleChatSidebar);
  // The collapsed state is a workspace preference, not document state, so it
  // outlives both the conversation and the selected PDF.
  applySidebarCollapsed(localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1");
  showChatIntro();
}

function applySidebarCollapsed(collapsed) {
  const layout = $("chatLayout");
  const toggle = $("sidebarToggle");
  if (!layout || !toggle) return;
  layout.classList.toggle("sidebarCollapsed", collapsed);
  toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
  toggle.title = collapsed ? "Show conversation list" : "Collapse conversation list";
  const label = toggle.querySelector(".srOnly");
  if (label) label.textContent = toggle.title;
  const chevron = toggle.querySelector("[aria-hidden]");
  if (chevron) chevron.innerHTML = collapsed ? "&#10095;" : "&#10094;";
}

function toggleChatSidebar() {
  const collapsed = !$("chatLayout").classList.contains("sidebarCollapsed");
  applySidebarCollapsed(collapsed);
  try { localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0"); } catch (e) {}
}

function showChatIntro() {
  $("messages").innerHTML = "";
  addMessage("assistant", CHAT_INTRO);
}

function addMessage(role, html) {
  const wrap = document.createElement("div");
  wrap.className = "msg " + role;
  wrap.innerHTML = `<div class="bubble"><div class="who">${role === "user" ? "You" : ASSISTANT_NAME}</div>${html}</div>`;
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return wrap;
}

/* --------------------------- Saved conversations ------------------------- */
// History lives in the run directory server-side, so it follows the document:
// deleting the PDF deletes its chats with it.

function conversationWhen(value) {
  if (!value) return "";
  const when = new Date(value);
  if (isNaN(when)) return "";
  const days = (Date.now() - when.getTime()) / 86400000;
  return days < 1
    ? when.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : when.toLocaleDateString();
}

async function refreshConversations() {
  try {
    const data = await api("/api/conversations");
    state.conversations = data.conversations || [];
  } catch (err) {
    state.conversations = [];
  }
  renderConversationList();
}

function renderConversationList() {
  const list = $("conversationList");
  if (!list) return;
  if (!state.conversations.length) {
    list.innerHTML =
      `<div class="conversationEmpty">No previous chats yet.<br>Start a new conversation below.</div>`;
    return;
  }
  list.innerHTML = state.conversations.map((c) => `
    <div class="conversationItem${c.id === state.activeConversationId ? " active" : ""}" data-conversation="${escapeHtml(c.id)}">
      <button class="conversationOpen" type="button" title="${escapeHtml(c.title || "")}">
        <span class="conversationTitle">${escapeHtml(c.title || "New conversation")}</span>
        <span class="conversationPreview">${escapeHtml(clip(c.preview || "No messages yet", 72))}</span>
        <span class="conversationTime">${escapeHtml(conversationWhen(c.updatedAt))}</span>
      </button>
      <button class="conversationDelete" type="button" title="Delete conversation"
              aria-label="Delete ${escapeHtml(c.title || "conversation")}">×</button>
    </div>`).join("");
  list.querySelectorAll(".conversationOpen").forEach((button) =>
    button.addEventListener("click", () =>
      openConversation(button.closest(".conversationItem").dataset.conversation)));
  list.querySelectorAll(".conversationDelete").forEach((button) =>
    button.addEventListener("click", () =>
      deleteConversation(button.closest(".conversationItem").dataset.conversation)));
}

async function loadConversations() {
  await refreshConversations();
  if (state.conversations.length) await openConversation(state.conversations[0].id);
  else resetChatPane(null);
}

function resetChatPane(conversationId) {
  state.activeConversationId = conversationId;
  showChatIntro();
  renderConversationList();
}

async function startNewChat() {
  if (state.chatBusy) return;
  try {
    const created = await api("/api/conversations", { method: "POST" });
    state.activeConversationId = created.id;
  } catch (err) {
    // Without a saved draft the next question still opens its own conversation
    // server-side, so the chat stays usable.
    state.activeConversationId = null;
  }
  showChatIntro();
  await refreshConversations();
  $("question").focus();
}

async function openConversation(conversationId) {
  if (state.chatBusy) return;
  let conversation;
  try {
    conversation = await api(`/api/conversations/${encodeURIComponent(conversationId)}`);
  } catch (err) {
    await refreshConversations();
    return;
  }
  state.activeConversationId = conversation.id;
  const messages = conversation.messages || [];
  $("messages").innerHTML = "";
  if (!messages.length) addMessage("assistant", CHAT_INTRO);
  for (const message of messages) {
    if (message.role === "user") {
      addMessage("user", `<div class="answer">${escapeHtml(message.content || "")}</div>`);
      continue;
    }
    if (message.error) {
      addMessage("assistant", `<p class="error">Error: ${escapeHtml(message.error)}</p>`);
      continue;
    }
    // A stored answer carries the same shape the live response did, so it is
    // replayed through exactly the same renderer rather than a second one that
    // would drift from it.
    const bubble = addMessage("assistant", renderAnswer(message));
    linkifyCitations(bubble, message.citations);
    inlineCitedFigures(bubble, message.citations);
    wrapAnswerTables(bubble);
    enhanceContent(bubble);
    wireEvidenceLinks(bubble);
    typesetMath(bubble);
  }
  renderConversationList();
  $("messages").scrollTop = $("messages").scrollHeight;
}

async function deleteConversation(conversationId) {
  const conversation = state.conversations.find((c) => c.id === conversationId);
  const label = (conversation && conversation.title) || "this conversation";
  if (!confirm(`Delete “${label}”? This cannot be undone.`)) return;
  try {
    await api(`/api/conversations/${encodeURIComponent(conversationId)}`, { method: "DELETE" });
  } catch (err) {
    alert("Could not delete the conversation: " + err.message);
    return;
  }
  if (state.activeConversationId === conversationId) state.activeConversationId = null;
  await refreshConversations();
  if (state.activeConversationId === null) {
    if (state.conversations.length) await openConversation(state.conversations[0].id);
    else resetChatPane(null);
  }
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
      body: JSON.stringify({ question, conversationId: state.activeConversationId }),
    });
    if (data.conversationId) state.activeConversationId = data.conversationId;
    pending.querySelector(".bubble").innerHTML =
      `<div class="who">${ASSISTANT_NAME}</div>` + renderAnswer(data);
    linkifyCitations(pending, data.citations);
    inlineCitedFigures(pending, data.citations);
    wrapAnswerTables(pending);
    enhanceContent(pending);
    wireEvidenceLinks(pending);
    typesetMath(pending);
  } catch (err) {
    pending.querySelector(".bubble").innerHTML =
      `<div class="who">${ASSISTANT_NAME}</div><p class="error">Error: ${escapeHtml(err.message)}</p>`;
  } finally {
    state.chatBusy = false;
    $("askBtn").disabled = false;
    $("messages").scrollTop = $("messages").scrollHeight;
    refreshConversations();
  }
}

function renderAnswer(data) {
  let html = `<div class="answer">${renderRichText(data.answer)}</div>`;
  html += `<div class="meta">${(data.nodeIds || []).length} nodes retrieved · ${data.elapsedSec}s</div>`;
  html += renderReferences(data.citations);
  html += renderRetrievedEntities(data.retrieval);
  return html;
}

// The answer's own [E1] / [Q1-E1] markers are what the model actually cited;
// they are the reference list, not a separate summary of everything fetched.
// A cited listing or formula has to keep its shape. clip() collapses every run
// of whitespace, which showed a cited algorithm as one unreadable line and left
// a cited formula as source with its line structure gone.
const SNIPPET_MAX_LINES = 12;

function renderSnippet(citation) {
  const text = String((citation && citation.snippet) || "");
  if (!text.trim()) return "";
  const type = String((citation && citation.type) || "").toUpperCase();
  if (LISTING_TYPES.has(type)) {
    const lines = text.replace(/\n+$/, "").split(/\r?\n/);
    const shown = lines.slice(0, SNIPPET_MAX_LINES);
    const hidden = lines.length - shown.length;
    const body = renderListing(shown.join("\n"), type, null);
    const more = hidden > 0 ? `<div class="snipMore">+${hidden} more lines</div>` : "";
    return `<div class="refSnippet">${body}${more}</div>`;
  }
  if (/\$\$|\\\[|\$[^$\n]+\$/.test(text)) {
    // Keep the delimiters on one line so MathJax can find them, as the tree
    // panel's own equation rendering does.
    return `<div class="refSnippet mathblock">${escapeHtml(text.replace(/\s*\n\s*/g, " "))}</div>`;
  }
  return `<div class="refSnippet">${escapeHtml(clip(text, 260))}</div>`;
}

// marked emits a bare <table>, which cannot scroll on a narrow panel and pushed
// the whole conversation sideways. Wrapping it after the fact keeps the table a
// real table and gives the overflow to its own container.
function wrapAnswerTables(scope) {
  scope.querySelectorAll(".answer table").forEach((table) => {
    if (table.parentElement && table.parentElement.classList.contains("tableWrap")) return;
    const wrap = document.createElement("div");
    wrap.className = "tableWrap";
    table.replaceWith(wrap);
    wrap.appendChild(table);
  });
}

// An answer that says "here is the diagram" should show the diagram. The crop
// was only ever reachable in the References fold, so a question answered
// entirely by a figure showed the reader a description of a picture and not the
// picture. Each cited image is placed after the block that first mentions it, so
// it reads as that paragraph's figure rather than as an attachment.
const INLINE_FIGURE_BLOCKS = "P, LI, H1, H2, H3, H4, H5, H6, BLOCKQUOTE, PRE, TABLE";

function inlineFigureHtml(citation) {
  const caption = [citation.label, citation.type, citation.page ? `p${citation.page}` : ""]
    .filter(Boolean)
    .map((part) => escapeHtml(String(part)))
    .join(" · ");
  return `<figure class="answerFigure" data-figure="${escapeHtml(citation.label)}">
    <img src="${treeImageUrl(citation.nodeId)}" alt="${escapeHtml(citation.label)}" loading="lazy" />
    <figcaption>${caption}${
      citation.nodeId != null
        ? ` · <a class="linkish" data-tree="${escapeHtml(citation.nodeId)}">view in tree →</a>`
        : ""
    }</figcaption>
  </figure>`;
}

function inlineCitedFigures(scope, citations) {
  const root = scope.querySelector(".answer");
  if (!root) return;
  const withImages = (citations || []).filter((c) => c.hasImage && c.nodeId != null);
  if (!withImages.length) return;

  for (const citation of withImages) {
    if (root.querySelector(`[data-figure="${CSS.escape(citation.label)}"]`)) continue;
    // linkifyCitations has already turned the marker into an anchor, so the
    // first mention is exactly where the reader met this evidence.
    const marker = root.querySelector(`a.cite[data-cite="${CSS.escape(citation.label)}"]`);
    const figure = document.createElement("div");
    figure.innerHTML = inlineFigureHtml(citation);
    const node = figure.firstElementChild;
    // hasImage says the node owns a crop, not that the crop can still be
    // served. A figure frame around a broken-image icon is worse than no
    // figure, and the reference card still offers the evidence either way.
    const img = node.querySelector("img");
    if (img) img.addEventListener("error", () => node.remove());
    if (!marker) {
      // Cited as an image but never named in the prose: still show it, at the
      // end, rather than silently dropping it.
      root.appendChild(node);
      continue;
    }
    let block = marker.closest(INLINE_FIGURE_BLOCKS);
    // A figure after an <li> would be a direct child of the <ul>, which is not
    // valid markup. Place it after the whole list instead.
    if (block && block.tagName === "LI") block = block.closest("ul, ol") || block;
    if (!block || !root.contains(block)) block = root.lastElementChild;
    if (block && block.parentNode) block.parentNode.insertBefore(node, block.nextSibling);
    else root.appendChild(node);
  }
}

function renderReferences(citations) {
  if (!citations || !citations.length) return "";
  const cards = citations.map((c) => {
    const img = c.hasImage && c.nodeId != null
      ? `<img class="refThumb" src="${treeImageUrl(c.nodeId)}" alt="" loading="lazy" />`
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
      ${renderSnippet(c)}
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
  // A figure, chart or table crop cited in an answer is thumbnail-sized, which
  // is too small to read. The tree panel already zooms its crop on click; an
  // answer's evidence now does the same rather than being a dead image.
  scope.querySelectorAll(".refThumb, .answer img").forEach((img) => {
    img.classList.add("zoomable");
    img.onclick = () => openLightbox(img.src);
  });
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
