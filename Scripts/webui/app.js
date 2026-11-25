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
  // The one shared scope -- drives Library, Graph, Chat and Tree together.
  scopeKind: "document",
  scopeGroupId: null,
  allDocumentCount: null,
  groupDocumentCount: null,
  documentRuns: [],
  groups: [],
  libraryRuns: [],
  // Drive-style library: folder being browsed ("" = My Library root), selection.
  library: {
    folder: "",
    selected: null,
    search: "",
    status: "",
    sort: { key: "name", dir: "asc" },
    detailOpen: true,
    expanded: new Set([""]),
    menuItems: [],
  },
  libraryDragging: false,
  libraryRenderPending: false,
  folderPicker: null,
  addDocsFolder: null,
  pdfPreviewRun: null,
  // Scope to restore when returning to Chat after a citation jump.
  pendingScopeRestore: null,
  crossLink: null,
  graphCrossDocOnly: false,
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

// Same palette, a decorrelated hash -- node fill already encodes entity TYPE
// via colorForType, so a node's border can encode which DOCUMENT it came
// from without the two channels fighting for the same color on the same node.
const docColorCache = new Map();
function colorForDocument(title) {
  const key = String(title || "");
  if (!docColorCache.has(key)) {
    let h = 2166136261;
    for (let i = 0; i < key.length; i++) h = (h ^ key.charCodeAt(i)) * 16777619 >>> 0;
    docColorCache.set(key, PALETTE[h % PALETTE.length]);
  }
  return docColorCache.get(key);
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
  state.library.detailOpen = readPref("noetics.libraryDetail", "open") !== "closed";
  wireTabs();
  wireGraphControls();
  wireGraphLinkBanner();
  wireTreeControls();
  wireChat();
  wireLibrary();
  wireScopeBar();
  wireResizers();
  updateHeaderScopeVisibility("library"); // matches index.html's default active tab
  attachNetControls("#graphView .stage", () => state.graphNet);
  attachNetControls("#treeView .stage", () => state.treeNet);
  enableWheelPan("treeNet", () => state.treeNet);
  enableWheelPan("graphNet", () => state.graphNet);
  try {
    // loadLibrary() bootstraps the shared scope (kind/document/group),
    // the group list, and the header's document picker together.
    await loadLibrary();
    if (state.scopeKind === "document" && state.currentRun) {
      await loadData();
    } else if (state.scopeKind !== "document") {
      await loadGraph();
      await loadConversations();
    } else {
      activateView("library");
    }
  } catch (err) {
    console.error("NoeticsRAG load failed:", err);
    alert("Failed to load index: " + err.message);
  }
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      if (tab.dataset.view === "chat" && state.pendingScopeRestore) returnToChatScope();
      else activateView(tab.dataset.view);
    });
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

// The header's Document/Group/All control only means something where "which
// documents am I looking at" actually changes what's shown: Graph and Chat.
// Library already shows every document (browsed through its own folders) and
// Tree only ever shows one document (via its own picker in the stage bar,
// see renderTreeScopeState) -- the header control there was just noise that
// looked like it was scoping something it wasn't.
function updateHeaderScopeVisibility(view) {
  const bar = document.querySelector(".headerScope");
  if (bar) bar.hidden = view === "library" || view === "tree";
}

function activateView(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
  $(name + "View").classList.add("active");
  updateHeaderScopeVisibility(name);

  // vis-network must be (re)fit once its container becomes visible
  if (name === "graph" && state.graph) {
    if (!state.graphBuilt) buildGraphNetwork();
    else { state.graphNet.redraw(); state.graphNet.fit(); }
  }
  if (name === "tree" && state.scopeKind === "document" && state.tree) {
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

const fmtBytes = (n) => {
  if (n === null || n === undefined) return "—";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
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

// Cross-linking isn't one of the driver's own stages (STAGE_ORDER on the
// backend) -- it's a separate, best-effort process against Postgres that
// most documents never had requested at all. Unlike Tree/Graph/Vdb, which
// always run and so always earn a place in the stage row, this card only
// appears once linking has actually been requested (running/succeeded/
// failed) -- a permanent "pending" tile for something nobody asked for reads
// as a stuck process, not as "nothing to show yet".
function crossLinkCardHtml(run) {
  const cl = run.crossLink;
  if (!cl || cl.status === "pending") return "";
  const status = cl.status;
  const seconds = cl.startedAt && cl.endedAt
    ? (new Date(cl.endedAt) - new Date(cl.startedAt)) / 1000
    : null;
  const elapsed = status === "running" && cl.startedAt
    ? (Date.now() - new Date(cl.startedAt).getTime()) / 1000
    : null;
  const title = status === "running"
    ? '<span class="spinner"></span>Cross-linking…'
    : status === "failed" ? "Cross-linking failed" : "Cross-linked";
  const meta = seconds != null ? fmtDuration(seconds)
    : elapsed != null ? `${fmtDuration(elapsed)} elapsed` : "";

  const statsHtml = cl.stats
    ? `<div class="crossLinkStats">
        <span><b>${cl.stats.nodesConnected}</b> nodes connected</span>
        <span><b>${cl.stats.groupingHubs}</b> grouping hubs</span>
        <span><b>${cl.stats.crossEdges}</b> cross edges</span>
      </div>`
    : "";

  const seenDirs = new Set();
  const links = (cl.linkedWith || []).filter((d) => (seenDirs.has(d.runDir) ? false : seenDirs.add(d.runDir)));
  // Different documents can share a filename (re-uploads, near-duplicate
  // sources) -- a bare name would make two distinct links look like one.
  const nameCounts = {};
  links.forEach((d) => { nameCounts[d.name] = (nameCounts[d.name] || 0) + 1; });
  const chipsHtml = links.length
    ? `<div class="crossLinkChips"><span class="crossLinkLabel">Linked with ${links.length} document${links.length === 1 ? "" : "s"}:</span>${links
        .map((d) => {
          const suffix = nameCounts[d.name] > 1 ? ` (${d.tag || d.runDir.slice(-6)})` : "";
          return `<button type="button" class="crossLinkChip" data-link-a="${escapeHtml(run.runDir)}" data-link-b="${escapeHtml(d.runDir)}"
                    title="View what connects this to ${escapeHtml(d.name)}">${escapeHtml(d.name + suffix)}</button>`;
        })
        .join("")}</div>`
    : "";

  return `<div class="crossLinkCard ${escapeHtml(status)}">
    <div class="crossLinkCardHead">
      <span>${title}</span>
      ${meta ? `<span class="crossLinkCardMeta">${escapeHtml(meta)}</span>` : ""}
    </div>
    ${cl.error ? `<div class="runError">${escapeHtml(String(cl.error))}</div>` : ""}
    ${statsHtml}
    ${chipsHtml}
  </div>`;
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

// The reader's own label for a document. An untagged document offers a quiet
// "+ tag" affordance instead of a chip, so the details panel does not fill with
// placeholders.
const MAX_TAG_LENGTH = 32;

function runTagHtml(tag) {
  const label = String(tag || "").trim();
  return label
    ? `<button class="runTag" type="button" data-act="tag" title="Edit tag">${escapeHtml(label)}</button>`
    : `<button class="runTag empty" type="button" data-act="tag" title="Add a tag">+ tag</button>`;
}

async function editRunTag(runDir) {
  const run = findRun(runDir);
  if (!run) return;
  const entered = prompt(
    `Tag for “${run.name}”\n` +
      `Up to ${MAX_TAG_LENGTH} characters. Leave empty to remove the tag.`,
    run.tag || "",
  );
  if (entered === null) return; // cancelled, not cleared
  try {
    await postJson("/api/run-tag", { runDir, tag: entered });
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

const postJson = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

async function openRunByDir(runDir) {
  try {
    await setScope("document", { runDir });
    activateView("tree");
  } catch (err) {
    alert("Could not open document: " + err.message);
  }
}

/* ------------------------------ folders ----------------------------------- */
// Groups are folders: each has a parentId (null = the library root) and a
// document can be filed in one or more of them. A folder's scope -- chat,
// graph, cross-link -- covers its subfolders too (runs_in_group on the server).

const LIBRARY_ROOT = "";
const LIBRARY_ALL = "__all__";

const ICONS = {
  folder: '<svg class="icoFolder" viewBox="0 0 24 24" aria-hidden="true"><path d="M10 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-8l-2-2z"/></svg>',
  pdf: '<svg class="icoPdf" viewBox="0 0 24 24" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="2.5"/><text x="12" y="15.2" text-anchor="middle">PDF</text></svg>',
  library: '<svg class="icoNav" viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm0 16H5v-4h3.6a3.5 3.5 0 0 0 6.8 0H19v4zm0-6h-5a2 2 0 0 1-4 0H5V5h14v8z"/></svg>',
  all: '<svg class="icoNav" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 5h2v2H3zm4 0h14v2H7zM3 11h2v2H3zm4 0h14v2H7zM3 17h2v2H3zm4 0h14v2H7z"/></svg>',
  caret: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 7l5 5-5 5z"/></svg>',
  chevron: '<svg class="crumbChevron" viewBox="0 0 24 24" aria-hidden="true"><path d="M9.3 6.7l1.4-1.4L17.4 12l-6.7 6.7-1.4-1.4 5.3-5.3z"/></svg>',
};

const byName = (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" });
const folderById = (id) => state.groups.find((g) => g.id === id) || null;
const childFolders = (parentId) =>
  state.groups.filter((g) => (g.parentId || LIBRARY_ROOT) === parentId).sort(byName);
const findRun = (runDir) => state.libraryRuns.find((r) => r.runDir === runDir) || null;
const runFolderIds = (run) => (run.groupIds || []).filter((id) => folderById(id));

function folderSubtreeIds(id) {
  const found = [];
  const stack = [id];
  while (stack.length) {
    const current = stack.pop();
    if (found.includes(current)) continue;
    found.push(current);
    childFolders(current).forEach((c) => stack.push(c.id));
  }
  return found;
}

function folderTrail(id) {
  const trail = [];
  for (let f = folderById(id); f && !trail.includes(f); f = folderById(f.parentId)) trail.unshift(f);
  return trail;
}

const folderPath = (id) => folderTrail(id).map((f) => f.name).join(" / ");

function runsInFolder(id) {
  return state.libraryRuns.filter((r) => {
    const ids = runFolderIds(r);
    return id === LIBRARY_ROOT ? ids.length === 0 : ids.includes(id);
  });
}

function runsInFolderTree(id) {
  const scope = new Set(folderSubtreeIds(id));
  return state.libraryRuns.filter((r) => runFolderIds(r).some((g) => scope.has(g)));
}

const foldersSortedByPath = () =>
  state.groups.map((g) => ({ ...g, path: folderPath(g.id) }))
    .sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true, sensitivity: "base" }));

const runStatus = (run) => run.overallStatus || (run.viewable ? "succeeded" : "pending");

function runStatusBucket(run) {
  const status = runStatus(run);
  if (run.isIndexing || status === "running" || status === "pending") return "indexing";
  if (status === "failed" || status === "interrupted" || run.error) return "attention";
  return "ready";
}

const STATUS_FILTERS = [
  ["", "All"],
  ["ready", "Ready"],
  ["indexing", "Indexing"],
  ["attention", "Needs attention"],
];

// createdAt is epoch seconds from the server; startedAt is ISO.
const runAddedAt = (run) => (run.createdAt ? new Date(run.createdAt * 1000) : run.startedAt || null);

const runIndexSeconds = (run) => run.stages.reduce((a, s) => a + (s.seconds || 0), 0) || null;

function fmtDay(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return "—";
  const now = new Date();
  if (d.toDateString() === now.toDateString())
    return d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  return d.toLocaleDateString(undefined, {
    month: "short", day: "numeric",
    ...(d.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  });
}

function readPref(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}
function writePref(key, value) {
  try { localStorage.setItem(key, value); } catch { /* per-viewer nicety only */ }
}

/* ------------------------------ library ----------------------------------- */

async function loadLibrary() {
  let data;
  try {
    data = await api("/api/library");
  } catch (err) {
    $("driveBody").innerHTML = `<div class="libraryEmpty">Could not load documents: ${escapeHtml(err.message)}</div>`;
    return;
  }
  state.currentRun = data.scopeRunDir;
  state.scopeKind = data.scopeKind || state.scopeKind || "document";
  state.scopeGroupId = data.scopeGroupId || null;
  state.allDocumentCount = data.allDocumentCount;
  state.groupDocumentCount = data.groupDocumentCount;
  state.libraryRuns = data.runs || [];
  await loadGroups();
  loadScopeBar();
  renderLibrary();

  // Cross-linking runs after tree/graph/vdb/persist already report
  // "succeeded", so it needs its own liveness check -- otherwise the poll
  // stops the moment the main pipeline finishes and a still-running link job
  // never gets its status refreshed without a manual reload.
  const active = state.libraryRuns.some((r) =>
    r.phase === "indexing" || r.overallStatus === "running"
    || (r.crossLink && r.crossLink.status === "running"));
  scheduleLibraryPoll(active);
}

async function loadGroups() {
  try {
    const data = await api("/api/groups");
    state.groups = data.groups || [];
  } catch (err) {
    state.groups = [];
  }
}

function scheduleLibraryPoll(active) {
  if (state.libraryTimer) { clearTimeout(state.libraryTimer); state.libraryTimer = null; }
  const visible = $("libraryView").classList.contains("active");
  if (active && visible) {
    state.libraryTimer = setTimeout(loadLibrary, LIBRARY_POLL_MS);
  }
}

function renderLibrary() {
  // A poll must not pull the row out from under a drag in progress.
  if (state.libraryDragging) { state.libraryRenderPending = true; return; }
  const lib = state.library;
  if (lib.folder !== LIBRARY_ROOT && lib.folder !== LIBRARY_ALL && !folderById(lib.folder)) {
    lib.folder = LIBRARY_ROOT;
  }
  if (lib.selected && !(lib.selected.kind === "run" ? findRun(lib.selected.id) : folderById(lib.selected.id))) {
    lib.selected = null;
  }
  if (lib.folder !== LIBRARY_ALL) folderTrail(lib.folder).forEach((f) => lib.expanded.add(f.id).add(f.parentId || LIBRARY_ROOT));
  lib.expanded.add(LIBRARY_ROOT);

  renderDriveTree();
  renderDriveCrumbs();
  renderDriveFilters();
  renderDriveBody();
  renderDriveDetail();

  const where = lib.folder === LIBRARY_ROOT || lib.folder === LIBRARY_ALL ? null : folderById(lib.folder);
  $("dropBarText").innerHTML = where
    ? `Drop a document here or click to upload into <b>${escapeHtml(where.name)}</b>`
    : "Drop a document here or click to upload";
  $("libraryView").classList.toggle("detailOpen", lib.detailOpen);
  $("driveInfoBtn").setAttribute("aria-pressed", String(lib.detailOpen));
  $("driveInfoBtn").classList.toggle("active", lib.detailOpen);
}

function navigateFolder(id) {
  const lib = state.library;
  lib.folder = id;
  lib.selected = null;
  lib.search = "";
  $("driveSearch").value = "";
  renderLibrary();
}

// Selection toggles classes in place: re-rendering the list between the two
// clicks of a double-click would swap the element and swallow the dblclick.
function selectItem(kind, id) {
  state.library.selected = kind ? { kind, id } : null;
  $("driveBody").querySelectorAll("[data-kind]").forEach((el) =>
    el.classList.toggle("selected", el.dataset.kind === kind && el.dataset.id === id));
  renderDriveDetail();
}

/* sidebar */

function navFolderHtml(folder, depth) {
  const lib = state.library;
  const kids = childFolders(folder.id);
  const open = lib.expanded.has(folder.id);
  const active = lib.folder === folder.id;
  return `<div class="navItem${active ? " active" : ""}" style="--depth:${depth}" data-nav="${escapeHtml(folder.id)}"
      data-drop-folder="${escapeHtml(folder.id)}" data-kind="folder" data-id="${escapeHtml(folder.id)}" draggable="true" title="${escapeHtml(folder.name)}">
      <button type="button" class="navCaret${kids.length ? "" : " none"}${open ? " open" : ""}" data-toggle="${escapeHtml(folder.id)}"
        aria-label="${open ? "Collapse" : "Expand"} ${escapeHtml(folder.name)}" aria-expanded="${open}">${ICONS.caret}</button>
      ${ICONS.folder}<span class="navLabel">${escapeHtml(folder.name)}</span>
    </div>
    ${open ? kids.map((k) => navFolderHtml(k, depth + 1)).join("") : ""}`;
}

function renderDriveTree() {
  const lib = state.library;
  const rootKids = childFolders(LIBRARY_ROOT);
  const rootOpen = lib.expanded.has(LIBRARY_ROOT);
  $("driveTree").innerHTML = `
    <div class="navItem${lib.folder === LIBRARY_ROOT ? " active" : ""}" style="--depth:0" data-nav="" data-drop-folder="">
      <button type="button" class="navCaret${rootKids.length ? "" : " none"}${rootOpen ? " open" : ""}" data-toggle=""
        aria-label="${rootOpen ? "Collapse" : "Expand"} My Library" aria-expanded="${rootOpen}">${ICONS.caret}</button>
      ${ICONS.library}<span class="navLabel">My Library</span>
    </div>
    ${rootOpen ? rootKids.map((f) => navFolderHtml(f, 1)).join("") : ""}
    <div class="navItem${lib.folder === LIBRARY_ALL ? " active" : ""}" style="--depth:0" data-nav="${LIBRARY_ALL}">
      <span class="navCaret none"></span>${ICONS.all}<span class="navLabel">All documents</span>
    </div>`;
  const runs = state.libraryRuns;
  const tokens = runs.reduce((a, r) => a + ((r.metrics && r.metrics.totalTokens) || 0), 0);
  $("driveNavFoot").innerHTML = `
    <div><b>${runs.length}</b> document${runs.length === 1 ? "" : "s"} · <b>${state.groups.length}</b> folder${state.groups.length === 1 ? "" : "s"}</div>
    <div>${escapeHtml(fmtTokens(tokens))} tokens indexed</div>`;
}

/* breadcrumbs + filters */

function renderDriveCrumbs() {
  const lib = state.library;
  const box = $("driveCrumbs");
  if (lib.search.trim()) {
    box.innerHTML = `<span class="crumbNow">Search results for “${escapeHtml(lib.search.trim())}”</span>`;
    return;
  }
  if (lib.folder === LIBRARY_ALL) {
    box.innerHTML = '<span class="crumbNow">All documents</span>';
    return;
  }
  const trail = folderTrail(lib.folder);
  const items = [{ id: LIBRARY_ROOT, name: "My Library" }, ...trail];
  box.innerHTML = items.map((f, i) => {
    const last = i === items.length - 1;
    return last
      ? `<button type="button" class="crumbNow crumbMenu" data-crumb-menu="${escapeHtml(f.id)}" aria-haspopup="menu">
           ${escapeHtml(f.name)} <span aria-hidden="true">▾</span></button>`
      : `<button type="button" class="crumbLink" data-nav="${escapeHtml(f.id)}" data-drop-folder="${escapeHtml(f.id)}">${escapeHtml(f.name)}</button>${ICONS.chevron}`;
  }).join("");
}

function renderDriveFilters() {
  const lib = state.library;
  const pool = visibleRuns({ ignoreStatus: true });
  $("driveFilters").hidden = !pool.length && !lib.status;
  $("driveFilters").innerHTML = STATUS_FILTERS.map(([key, label]) => {
    const count = key ? pool.filter((r) => runStatusBucket(r) === key).length : pool.length;
    if (key && !count && lib.status !== key) return "";
    const on = lib.status === key;
    return `<button type="button" class="filterChip${on ? " active" : ""}" data-status="${key}" aria-pressed="${on}">
      ${on && key ? "✓ " : ""}${escapeHtml(label)} <span class="filterCount">${count}</span></button>`;
  }).join("");
}

/* the file list */

function visibleFolders() {
  const lib = state.library;
  const q = lib.search.trim().toLowerCase();
  if (q) return state.groups.filter((g) => g.name.toLowerCase().includes(q)).sort(byName);
  if (lib.folder === LIBRARY_ALL) return [];
  return childFolders(lib.folder);
}

function visibleRuns({ ignoreStatus = false } = {}) {
  const lib = state.library;
  const q = lib.search.trim().toLowerCase();
  let runs = q
    ? state.libraryRuns.filter((r) => `${r.name} ${r.tag || ""}`.toLowerCase().includes(q))
    : lib.folder === LIBRARY_ALL ? state.libraryRuns.slice() : runsInFolder(lib.folder);
  if (!ignoreStatus && lib.status) runs = runs.filter((r) => runStatusBucket(r) === lib.status);
  const dir = lib.sort.dir === "asc" ? 1 : -1;
  const key = lib.sort.key;
  return runs.sort((a, b) => {
    if (key === "created") return dir * (new Date(runAddedAt(a) || 0) - new Date(runAddedAt(b) || 0));
    if (key === "size") return dir * ((a.sizeBytes || 0) - (b.sizeBytes || 0));
    return dir * byName(a, b);
  });
}

const showLocation = () => state.library.folder === LIBRARY_ALL || !!state.library.search.trim();

function runStatusHtml(run) {
  const status = runStatus(run);
  const live = run.stages.find((s) => s.status === "running");
  let sub = "";
  if (live) {
    const pct = live.progress ? ` ${Math.round((live.progress.done / live.progress.total) * 100)}%` : "";
    sub = `<span class="statusSub">${escapeHtml(live.name)}${pct}</span>`;
  } else if (run.crossLink && run.crossLink.status === "running") {
    sub = '<span class="statusSub">cross-linking</span>';
  }
  return `<span class="pill ${escapeHtml(status)}">${escapeHtml(status)}</span>${sub}`;
}

function locationHtml(run) {
  const ids = runFolderIds(run);
  if (!ids.length) return '<span class="locChip root">My Library</span>';
  return ids.map((id) => `<button type="button" class="locChip" data-nav="${escapeHtml(id)}" title="${escapeHtml(folderPath(id))}">
    ${ICONS.folder}${escapeHtml(folderById(id).name)}</button>`).join("");
}

function sortHeadHtml(key, label) {
  const { sort } = state.library;
  const on = sort.key === key;
  const arrow = on ? (sort.dir === "asc" ? "↑" : "↓") : "";
  return `<button type="button" class="colSort${on ? " active" : ""}" data-sort="${key}">${label}<span class="sortArrow">${arrow}</span></button>`;
}

function folderRowHtml(folder) {
  const lib = state.library;
  const selected = lib.selected && lib.selected.kind === "folder" && lib.selected.id === folder.id;
  const items = childFolders(folder.id).length + runsInFolder(folder.id).length;
  return `<div class="dRow folder${selected ? " selected" : ""}" role="row" tabindex="0" draggable="true"
      data-kind="folder" data-id="${escapeHtml(folder.id)}" data-drop-folder="${escapeHtml(folder.id)}">
    <div class="dCell dName">${ICONS.folder}<span class="dLabel">${escapeHtml(folder.name)}</span></div>
    <div class="dCell dStatus muted">${items} item${items === 1 ? "" : "s"}</div>
    ${showLocation() ? `<div class="dCell dLoc">${folder.parentId ? `<button type="button" class="locChip" data-nav="${escapeHtml(folder.parentId)}">${ICONS.folder}${escapeHtml(folderById(folder.parentId).name)}</button>` : '<span class="locChip root">My Library</span>'}</div>` : ""}
    <div class="dCell dWhen">${escapeHtml(fmtDay(folder.createdAt))}</div>
    <div class="dCell dSize muted">—</div>
    <div class="dCell dMore"><button type="button" class="moreBtn" data-more aria-label="More actions for ${escapeHtml(folder.name)}">⋮</button></div>
  </div>`;
}

function runRowHtml(run) {
  const lib = state.library;
  const selected = lib.selected && lib.selected.kind === "run" && lib.selected.id === run.runDir;
  const isCurrent = run.runDir === state.currentRun;
  return `<div class="dRow file${selected ? " selected" : ""}" role="row" tabindex="0" draggable="true"
      data-kind="run" data-id="${escapeHtml(run.runDir)}">
    <div class="dCell dName">${ICONS.pdf}<span class="dLabel" title="${escapeHtml(run.name)}">${escapeHtml(run.name)}</span>
      ${run.tag ? `<span class="rowTag">${escapeHtml(run.tag)}</span>` : ""}
      ${isCurrent ? '<span class="rowOpen" title="Open in Graph, Tree and Chat">open</span>' : ""}</div>
    <div class="dCell dStatus">${runStatusHtml(run)}</div>
    ${showLocation() ? `<div class="dCell dLoc">${locationHtml(run)}</div>` : ""}
    <div class="dCell dWhen">${escapeHtml(fmtDay(runAddedAt(run)))}</div>
    <div class="dCell dSize">${escapeHtml(fmtBytes(run.sizeBytes))}</div>
    <div class="dCell dMore"><button type="button" class="moreBtn" data-more aria-label="More actions for ${escapeHtml(run.name)}">⋮</button></div>
  </div>`;
}

function renderDriveBody() {
  if (state.libraryDragging) { state.libraryRenderPending = true; return; }
  const lib = state.library;
  const body = $("driveBody");
  const focused = document.activeElement && body.contains(document.activeElement)
    ? document.activeElement.closest("[data-kind]") : null;
  const refocus = focused ? { kind: focused.dataset.kind, id: focused.dataset.id } : null;
  body.classList.toggle("withLoc", showLocation());
  body.innerHTML = driveBodyHtml();
  if (refocus) {
    const again = body.querySelector(`[data-kind="${refocus.kind}"][data-id="${CSS.escape(refocus.id)}"]`);
    if (again) again.focus({ preventScroll: true });
  }
}

function driveBodyHtml() {
  const lib = state.library;
  const folders = lib.status ? [] : visibleFolders();
  const runs = visibleRuns();

  if (!folders.length && !runs.length) {
    const q = lib.search.trim();
    return `<div class="driveEmpty">
      ${ICONS.folder}
      <h3>${q ? "No matches" : lib.status ? "Nothing with this status here" : lib.folder === LIBRARY_ROOT || lib.folder === LIBRARY_ALL ? "Your library is empty" : "This folder is empty"}</h3>
      <p>${q ? "Try a different name or tag." : lib.status ? "Clear the filter to see everything." : "Drop a document here, or move documents in from another folder."}</p>
      ${!q && !lib.status && folderById(lib.folder) && state.libraryRuns.length
        ? '<button type="button" class="btn" data-act="add-existing">Add existing documents…</button>' : ""}
    </div>`;
  }

  return `<div class="dTable" role="table" aria-label="Library contents">
    <div class="dRow dHeadRow" role="row">
      <div class="dCell dName">${sortHeadHtml("name", "Name")}</div>
      <div class="dCell dStatus">Status</div>
      ${showLocation() ? '<div class="dCell dLoc">Location</div>' : ""}
      <div class="dCell dWhen">${sortHeadHtml("created", "Added")}</div>
      <div class="dCell dSize">${sortHeadHtml("size", "Size")}</div>
      <div class="dCell dMore"></div>
    </div>
    ${folders.map(folderRowHtml).join("")}
    ${runs.map(runRowHtml).join("")}
  </div>`;
}

/* details panel */

function renderDriveDetail() {
  const lib = state.library;
  const panel = $("driveDetail");
  if (!lib.detailOpen) { panel.innerHTML = ""; return; }
  const sel = lib.selected;
  // Keep a half-made Rebuild choice and the scroll position across polls.
  const shownRun = (panel.querySelector("[data-run]") || {}).dataset?.run;
  const pendingRebuild = panel.querySelector(".rebuildSelect")?.value;
  const scroll = panel.scrollTop;
  $("libraryView").classList.toggle("hasSelection", !!sel);
  if (sel && sel.kind === "run") panel.innerHTML = runDetailHtml(findRun(sel.id));
  else if (sel && sel.kind === "folder") panel.innerHTML = folderDetailHtml(folderById(sel.id));
  else panel.innerHTML = placeDetailHtml();
  if (sel && sel.kind === "run" && shownRun === sel.id) {
    const select = panel.querySelector(".rebuildSelect");
    if (select && pendingRebuild) select.value = pendingRebuild;
    panel.scrollTop = scroll;
  }
}

function detailHeadHtml(icon, name, sub) {
  return `<div class="detailHead">
    ${icon}<div class="detailTitles"><h2 title="${escapeHtml(name)}">${escapeHtml(name)}</h2>${sub ? `<div class="detailSub">${sub}</div>` : ""}</div>
    <button type="button" class="driveIconBtn" data-act="close-detail" aria-label="Close details">×</button>
  </div>`;
}

function runDetailHtml(run) {
  const status = runStatus(run);
  const isCurrent = run.runDir === state.currentRun;
  const stages = run.stages
    .map((stage, index) => stageStepHtml(stage, index ? run.stages[index - 1].name : null))
    .join("");
  const inFolder = folderById(state.library.folder) && runFolderIds(run).includes(state.library.folder);
  return `<div class="detailBody" data-run="${escapeHtml(run.runDir)}">
    ${detailHeadHtml(ICONS.pdf, run.name, `Added ${escapeHtml(fmtWhen(runAddedAt(run)))}`)}
    <div class="detailPills">
      <span class="pill ${escapeHtml(status)}">${escapeHtml(status)}</span>
      ${isCurrent ? '<span class="pill">open</span>' : ""}
      ${runTagHtml(run.tag)}
    </div>
    <div class="detailActions">
      ${run.hasPdf ? '<button class="btn primary" data-act="preview">Preview</button>' : ""}
      ${run.viewable ? `<button class="btn${run.hasPdf ? "" : " primary"}" data-act="open">Open in Tree</button><button class="btn" data-act="chat">Chat</button>` : ""}
      ${runLifecycleHtml(run)}
      ${run.viewable ? '<button class="btn" data-act="link" title="Cross-link this document with others">Link to…</button>' : ""}
    </div>

    <h3 class="detailSection">Location</h3>
    <div class="detailLocation">${locationHtml(run)}</div>
    <div class="detailActions compact">
      <button class="btn" data-act="move">Move to…</button>
      <button class="btn" data-act="add-to">Add to folder…</button>
      ${inFolder ? '<button class="btn" data-act="remove-from">Remove from this folder</button>' : ""}
    </div>

    <h3 class="detailSection">Indexing</h3>
    <div class="stages">${stages}</div>
    <div class="runStats">
      <div class="runStat"><div class="k">Index time</div><div class="v">${escapeHtml(fmtDuration(runIndexSeconds(run)))}</div></div>
      <div class="runStat"><div class="k">Tree build</div><div class="v">${escapeHtml(fmtDuration(run.metrics.buildTreeSeconds))}</div></div>
      <div class="runStat"><div class="k">Tokens</div><div class="v">${escapeHtml(fmtTokens(run.metrics.totalTokens))}</div></div>
      <div class="runStat"><div class="k">Viewable</div><div class="v">${run.viewable ? "yes" : "no"}</div></div>
    </div>
    ${crossLinkCardHtml(run)}
    ${run.error ? `<div class="runError">${escapeHtml(String(run.error))}</div>` : ""}

    <div class="detailDanger">
      <button class="btn danger" data-act="delete" ${isCurrent ? "disabled title='Open another document first'" : ""}>Delete document</button>
    </div>
  </div>`;
}

function folderStatsHtml(direct, nested, subfolders) {
  const ready = nested.filter((r) => runStatusBucket(r) === "ready").length;
  return `<dl class="kv">
    <dt>Here</dt><dd>${direct.length} document${direct.length === 1 ? "" : "s"}</dd>
    <dt>Subfolders</dt><dd>${subfolders}</dd>
    <dt>In total</dt><dd>${nested.length} document${nested.length === 1 ? "" : "s"}, ${ready} ready</dd>
  </dl>`;
}

function folderDetailHtml(folder) {
  const nested = runsInFolderTree(folder.id);
  const canQuery = nested.some((r) => r.viewable);
  const parent = folder.parentId ? folderPath(folder.parentId) : "My Library";
  return `<div class="detailBody" data-folder="${escapeHtml(folder.id)}">
    ${detailHeadHtml(ICONS.folder, folder.name, `in ${escapeHtml(parent)}`)}
    <div class="detailActions">
      <button class="btn primary" data-act="open-folder">Open</button>
      <button class="btn" data-act="chat-folder" ${canQuery ? "" : "disabled title='No indexed documents in this folder yet'"}>Chat</button>
      <button class="btn" data-act="graph-folder" ${canQuery ? "" : "disabled title='No indexed documents in this folder yet'"}>Graph</button>
    </div>
    <h3 class="detailSection">Contents</h3>
    ${folderStatsHtml(runsInFolder(folder.id), nested, folderSubtreeIds(folder.id).length - 1)}
    <p class="detailNote">Chat and Graph cover every document in this folder and its subfolders.</p>
    <h3 class="detailSection">Organize</h3>
    <div class="detailActions compact">
      <button class="btn" data-act="new-subfolder">New subfolder</button>
      <button class="btn" data-act="add-existing">Add documents…</button>
      <button class="btn" data-act="rename-folder">Rename</button>
      <button class="btn" data-act="move-folder">Move to…</button>
    </div>
    <div class="detailDanger">
      <button class="btn danger" data-act="delete-folder">Delete folder</button>
    </div>
  </div>`;
}

function placeDetailHtml() {
  const lib = state.library;
  if (lib.folder === LIBRARY_ALL || lib.search.trim()) {
    const runs = state.libraryRuns;
    return `<div class="detailBody">
      ${detailHeadHtml(ICONS.all, "All documents", "")}
      <dl class="kv">
        ${STATUS_FILTERS.slice(1).map(([key, label]) =>
          `<dt>${escapeHtml(label)}</dt><dd>${runs.filter((r) => runStatusBucket(r) === key).length}</dd>`).join("")}
      </dl>
      <p class="detailNote">Select a file or folder to see its details.</p>
    </div>`;
  }
  const folder = folderById(lib.folder);
  if (folder) return folderDetailHtml(folder);
  const nested = state.libraryRuns;
  return `<div class="detailBody">
    ${detailHeadHtml(ICONS.library, "My Library", "")}
    ${folderStatsHtml(runsInFolder(LIBRARY_ROOT), nested, state.groups.length)}
    <p class="detailNote">Select a file or folder to see its details. Drag documents onto a folder to file them.</p>
  </div>`;
}

/* actions */

async function createFolder(parentId) {
  const name = prompt(parentId ? `New folder in “${folderById(parentId).name}”` : "New folder");
  if (name === null || !name.trim()) return null;
  try {
    const created = await postJson("/api/groups", { name, parentId: parentId || null });
    await loadLibrary();
    return created.group;
  } catch (err) {
    alert("Could not create the folder: " + err.message);
    return null;
  }
}

async function renameFolder(id) {
  const folder = folderById(id);
  if (!folder) return;
  const name = prompt("Rename folder", folder.name);
  if (name === null || !name.trim() || name.trim() === folder.name) return;
  try {
    await postJson("/api/group-rename", { groupId: id, name: name.trim() });
  } catch (err) {
    alert("Could not rename the folder: " + err.message);
    return;
  }
  await loadLibrary();
}

async function deleteFolder(id) {
  const folder = folderById(id);
  if (!folder) return;
  const subfolders = folderSubtreeIds(id).length - 1;
  const ok = confirm(
    `Delete the folder “${folder.name}”${subfolders ? ` and its ${subfolders} subfolder${subfolders === 1 ? "" : "s"}` : ""}?\n\n` +
      "Documents inside are not deleted -- they stay in the library.",
  );
  if (!ok) return;
  try {
    await postJson("/api/group-delete", { groupId: id });
  } catch (err) {
    alert("Could not delete the folder: " + err.message);
    return;
  }
  const lib = state.library;
  if (folderSubtreeIds(id).includes(lib.folder)) lib.folder = folder.parentId || LIBRARY_ROOT;
  lib.selected = null;
  await loadLibrary();
}

async function moveFolder(id, parentId) {
  const folder = folderById(id);
  if (!folder || (folder.parentId || LIBRARY_ROOT) === parentId) return;
  try {
    await postJson("/api/group-move", { groupId: id, parentId: parentId || null });
  } catch (err) {
    alert("Could not move the folder: " + err.message);
    return;
  }
  if (parentId) state.library.expanded.add(parentId);
  await loadLibrary();
}

async function setRunFolders(runDir, groupIds) {
  await postJson("/api/run-groups", { runDir, groupIds: Array.from(new Set(groupIds)) });
}

// Move replaces the folder the document is being viewed in (Drive's
// single-parent move); from anywhere else it files the document in the target
// alone. Moving to My Library unfiles it.
async function moveRun(runDir, toId) {
  const run = findRun(runDir);
  if (!run) return;
  const current = runFolderIds(run);
  const from = state.library.folder;
  let next;
  if (folderById(from) && current.includes(from)) next = current.filter((g) => g !== from);
  else next = [];
  if (toId) next.push(toId);
  if (next.length === current.length && next.every((g) => current.includes(g))) return;
  try {
    await setRunFolders(runDir, next);
  } catch (err) {
    alert("Could not move the document: " + err.message);
    return;
  }
  await loadLibrary();
  if (toId) promptLinkToNewFolderMates(runDir, [toId]);
}

async function addRunToFolder(runDir, toId) {
  const run = findRun(runDir);
  if (!run || runFolderIds(run).includes(toId)) return;
  try {
    await setRunFolders(runDir, [...runFolderIds(run), toId]);
  } catch (err) {
    alert("Could not add the document to the folder: " + err.message);
    return;
  }
  await loadLibrary();
  promptLinkToNewFolderMates(runDir, [toId]);
}

async function removeRunFromFolder(runDir, folderId) {
  const run = findRun(runDir);
  if (!run) return;
  try {
    await setRunFolders(runDir, runFolderIds(run).filter((g) => g !== folderId));
  } catch (err) {
    alert("Could not remove the document from the folder: " + err.message);
    return;
  }
  await loadLibrary();
}

async function deleteRun(runDir) {
  const run = findRun(runDir);
  if (!run) return;
  const ok = confirm(
    `Delete "${run.name}" completely?\n\n` +
      "This removes its vectors from Qdrant, its rows from PostgreSQL, and " +
      "its extracted content from disk. This cannot be undone."
  );
  if (!ok) return;
  try {
    await api(`/api/run?run=${encodeURIComponent(runDir)}`, { method: "DELETE" });
  } catch (err) {
    alert("Could not delete: " + err.message);
    return;
  }
  state.library.selected = null;
  await loadLibrary();
}

async function stopRun(runDir) {
  if (!confirm("Stop indexing this document? Finished stages are kept and it can be started again.")) return;
  try {
    await postJson("/api/run-stop", { runDir });
  } catch (err) {
    alert("Could not stop this run: " + err.message);
  }
  await loadLibrary();
}

async function startRun(runDir, fromStage) {
  if (fromStage && !confirm(
    `Rebuild from the ${fromStage} stage? That stage and every stage after it ` +
    `will be discarded and built again.`)) return;
  try {
    await postJson("/api/run-start", { runDir, fromStage });
  } catch (err) {
    alert("Could not start this run: " + err.message);
  }
  await loadLibrary();
}

async function chatAbout(kind, id) {
  await setScope(kind, kind === "group" ? { groupId: id } : { runDir: id });
  if (state.scopeKind === kind) activateView("chat");
}

async function graphOfFolder(id) {
  await setScope("group", { groupId: id });
  if (state.scopeKind === "group") activateView("graph");
}

// Like Drive, opening a file previews it; Tree/Graph/Chat stay one click away.
function openItem(kind, id) {
  if (kind === "folder") return navigateFolder(id);
  const run = findRun(id);
  if (run && run.hasPdf) openPdfPreview(id);
  else if (run && run.viewable) openRunByDir(id);
  else selectItem("run", id);
}

/* PDF preview */

const sourcePdfUrl = (runDir) => `/api/source-pdf?run=${encodeURIComponent(runDir)}`;

function openPdfPreview(runDir) {
  const run = findRun(runDir);
  if (!run || !run.hasPdf) return;
  closeDriveMenu();
  state.pdfPreviewRun = runDir;
  const url = sourcePdfUrl(runDir);
  $("pdfPreviewName").textContent = run.name;
  $("pdfPreviewTab").href = url;
  $("pdfPreviewDownload").href = url;
  $("pdfPreviewDownload").setAttribute("download", run.name);
  $("pdfPreviewOpen").hidden = !run.viewable;
  $("pdfPreviewFrame").src = url;
  $("pdfPreview").hidden = false;
  $("pdfPreviewClose").focus();
}

function closePdfPreview() {
  if ($("pdfPreview").hidden) return;
  $("pdfPreview").hidden = true;
  // Unload the document so a large PDF does not stay in memory behind the library.
  $("pdfPreviewFrame").src = "about:blank";
  const runDir = state.pdfPreviewRun;
  state.pdfPreviewRun = null;
  const row = runDir && $("driveBody").querySelector(`[data-kind="run"][data-id="${CSS.escape(runDir)}"]`);
  if (row) row.focus({ preventScroll: true });
}

function wirePdfPreview() {
  $("pdfPreviewClose").addEventListener("click", closePdfPreview);
  $("pdfPreviewOpen").addEventListener("click", () => {
    const runDir = state.pdfPreviewRun;
    closePdfPreview();
    if (runDir) openRunByDir(runDir);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("pdfPreview").hidden) closePdfPreview();
  });
}

/* context menu */

function runMenuItems(run) {
  const inFolder = folderById(state.library.folder) && runFolderIds(run).includes(state.library.folder);
  return [
    run.hasPdf && { label: "Preview", act: () => openPdfPreview(run.runDir) },
    run.viewable && { label: "Open in Tree", act: () => openRunByDir(run.runDir) },
    run.viewable && { label: "Chat about this document", act: () => chatAbout("document", run.runDir) },
    { sep: true },
    { label: "Move to…", act: () => openFolderPicker({ mode: "move-run", id: run.runDir }) },
    { label: "Add to folder…", act: () => openFolderPicker({ mode: "add-run", id: run.runDir }) },
    inFolder && { label: "Remove from this folder", act: () => removeRunFromFolder(run.runDir, state.library.folder) },
    { sep: true },
    run.viewable && { label: "Link to…", act: () => openCrossLinkModal({ kind: "retroactive", runDir: run.runDir }) },
    { label: run.tag ? "Edit tag" : "Add tag", act: () => editRunTag(run.runDir) },
    run.isIndexing && { label: "Stop indexing", act: () => stopRun(run.runDir) },
    { label: "Details", act: () => { state.library.detailOpen = true; selectItem("run", run.runDir); renderLibrary(); } },
    { sep: true },
    { label: "Delete", danger: true, disabled: run.runDir === state.currentRun, act: () => deleteRun(run.runDir) },
  ];
}

function folderMenuItems(folder) {
  const canQuery = runsInFolderTree(folder.id).some((r) => r.viewable);
  return [
    { label: "Open", act: () => navigateFolder(folder.id) },
    { label: "Chat with this folder", disabled: !canQuery, act: () => chatAbout("group", folder.id) },
    { label: "View graph", disabled: !canQuery, act: () => graphOfFolder(folder.id) },
    { sep: true },
    { label: "New subfolder", act: () => createFolder(folder.id) },
    { label: "Add documents…", act: () => openAddDocsModal(folder.id) },
    { label: "Rename", act: () => renameFolder(folder.id) },
    { label: "Move to…", act: () => openFolderPicker({ mode: "move-folder", id: folder.id }) },
    { sep: true },
    { label: "Delete folder", danger: true, act: () => deleteFolder(folder.id) },
  ];
}

function placeMenuItems() {
  const here = folderById(state.library.folder) ? state.library.folder : LIBRARY_ROOT;
  return [
    { label: "New folder", act: () => createFolder(here) },
    { label: "Upload document", act: () => $("fileInput").click() },
    here && { sep: true },
    here && { label: "Add existing documents…", act: () => openAddDocsModal(here) },
  ];
}

function openDriveMenu(items, anchor) {
  const menu = $("driveMenu");
  const clean = items.filter(Boolean).filter((it, i, arr) =>
    !it.sep || (i > 0 && i < arr.length - 1 && !arr[i - 1].sep));
  state.library.menuItems = clean;
  menu.innerHTML = clean.map((it, i) => it.sep
    ? '<div class="menuSep" role="separator"></div>'
    : `<button type="button" role="menuitem" class="menuItem${it.danger ? " danger" : ""}" data-menu="${i}" ${it.disabled ? "disabled" : ""}>${escapeHtml(it.label)}</button>`,
  ).join("");
  menu.hidden = false;
  const view = $("libraryView").getBoundingClientRect();
  const box = menu.getBoundingClientRect();
  let x, y;
  if (anchor instanceof Element) {
    const r = anchor.getBoundingClientRect();
    x = r.left; y = r.bottom + 4;
  } else {
    ({ x, y } = anchor);
  }
  x = Math.max(view.left + 8, Math.min(x, view.right - box.width - 8));
  if (y + box.height > view.bottom - 8) y = Math.max(view.top + 8, y - box.height - 8);
  menu.style.left = `${x - view.left}px`;
  menu.style.top = `${y - view.top}px`;
  const first = menu.querySelector(".menuItem:not([disabled])");
  if (first) first.focus();
}

function closeDriveMenu() {
  $("driveMenu").hidden = true;
  state.library.menuItems = [];
}

function itemMenu(kind, id) {
  if (kind === "folder") return folderMenuItems(folderById(id));
  return runMenuItems(findRun(id));
}

/* folder picker (move / add) */

function openFolderPicker(context) {
  const picker = state.folderPicker = { ...context, target: null };
  const run = context.mode !== "move-folder" ? findRun(context.id) : null;
  const folder = context.mode === "move-folder" ? folderById(context.id) : null;
  const name = run ? run.name : folder.name;
  $("folderPickerTitle").textContent = context.mode === "add-run" ? `Add “${name}” to a folder` : `Move “${name}”`;
  $("folderPickerHint").textContent = context.mode === "add-run"
    ? "The document stays in the folders it's already in."
    : run
      ? `Currently in ${runFolderIds(run).map((id) => folderPath(id)).join(", ") || "My Library"}.`
      : `Currently in ${folder.parentId ? folderPath(folder.parentId) : "My Library"}.`;
  $("folderPickerConfirm").textContent = context.mode === "add-run" ? "Add here" : "Move here";
  picker.target = context.mode === "add-run" ? null
    : folderById(state.library.folder) ? state.library.folder : LIBRARY_ROOT;
  renderFolderPicker();
  $("folderPickerModal").hidden = false;
}

function renderFolderPicker() {
  const picker = state.folderPicker;
  const run = picker.mode !== "move-folder" ? findRun(picker.id) : null;
  const blocked = new Set(picker.mode === "move-folder" ? folderSubtreeIds(picker.id) : []);
  const already = new Set(picker.mode === "add-run" && run ? runFolderIds(run) : []);
  const rowHtml = (id, name, depth, icon) => {
    const disabled = blocked.has(id) || already.has(id);
    const note = blocked.has(id) ? "" : already.has(id) ? '<span class="pickNote">already here</span>' : "";
    const selected = picker.target === id;
    return `<button type="button" role="option" class="pickRow${selected ? " selected" : ""}" style="--depth:${depth}"
        data-pick="${escapeHtml(id)}" aria-selected="${selected}" ${disabled ? "disabled" : ""}>
      ${icon}<span>${escapeHtml(name)}</span>${note}</button>`;
  };
  const walk = (parentId, depth) => childFolders(parentId)
    .map((f) => rowHtml(f.id, f.name, depth, ICONS.folder) + (blocked.has(f.id) ? "" : walk(f.id, depth + 1)))
    .join("");
  const rows = (picker.mode === "add-run" ? "" : rowHtml(LIBRARY_ROOT, "My Library", 0, ICONS.library))
    + walk(LIBRARY_ROOT, picker.mode === "add-run" ? 0 : 1);
  $("folderPickerTree").innerHTML = rows || '<div class="modalHint">No folders yet — create one below.</div>';
  $("folderPickerConfirm").disabled = picker.target === null;
}

function closeFolderPicker() {
  $("folderPickerModal").hidden = true;
  state.folderPicker = null;
}

async function confirmFolderPicker() {
  const picker = state.folderPicker;
  if (!picker || picker.target === null) return;
  closeFolderPicker();
  if (picker.mode === "move-folder") await moveFolder(picker.id, picker.target);
  else if (picker.mode === "add-run") await addRunToFolder(picker.id, picker.target);
  else await moveRun(picker.id, picker.target);
}

/* add existing documents to a folder */

function openAddDocsModal(folderId) {
  const folder = folderById(folderId);
  if (!folder) return;
  state.addDocsFolder = folderId;
  $("addDocsTitle").textContent = `Add documents to “${folder.name}”`;
  const runs = state.libraryRuns.filter((r) => !runFolderIds(r).includes(folderId)).sort(byName);
  $("addDocsPicker").innerHTML = runs.length
    ? runs.map((r) => `
        <label class="modalCheck">
          <input type="checkbox" value="${escapeHtml(r.runDir)}" />
          ${ICONS.pdf}<span>${escapeHtml(r.name)}</span>
          <span class="pickNote">${escapeHtml(runFolderIds(r).map((id) => folderById(id).name).join(", ") || "My Library")}</span>
        </label>`).join("")
    : '<div class="modalHint">Every document is already in this folder.</div>';
  $("addDocsModal").hidden = false;
}

function closeAddDocsModal() {
  $("addDocsModal").hidden = true;
  state.addDocsFolder = null;
}

async function saveAddDocs() {
  const folderId = state.addDocsFolder;
  if (!folderId) return;
  const picked = Array.from($("addDocsPicker").querySelectorAll("input:checked")).map((i) => i.value);
  closeAddDocsModal();
  try {
    await Promise.all(picked.map((runDir) =>
      setRunFolders(runDir, [...runFolderIds(findRun(runDir)), folderId])));
  } catch (err) {
    alert("Could not add some documents: " + err.message);
  }
  await loadLibrary();
}

// After filing a document in a folder, offer to catch it up with whatever it
// isn't cross-linked to yet in that folder -- membership itself never links
// anything (see set_run_groups in Scripts/web.py).
function promptLinkToNewFolderMates(runDir, folderIds) {
  const self = findRun(runDir);
  const linkedWith = new Set(
    ((self && self.crossLink && self.crossLink.linkedWith) || []).map((l) => l.runDir),
  );
  const candidates = new Map();
  for (const id of folderIds) {
    for (const r of runsInFolderTree(id)) {
      if (r.runDir === runDir || !r.viewable || linkedWith.has(r.runDir)) continue;
      candidates.set(r.runDir, r);
    }
  }
  if (!self || !self.viewable || !candidates.size) return;
  openCrossLinkModal({
    kind: "group-add",
    runDir,
    candidateRunDirs: Array.from(candidates.keys()),
  });
}

/* upload */

function setUploadStatus(message, isError) {
  const box = $("uploadStatus");
  box.hidden = false;
  box.className = "uploadStatus" + (isError ? " err" : "");
  box.innerHTML = message;
}

const UPLOAD_CONTENT_TYPES = {
  ".pdf": "application/pdf",
  ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
};

async function uploadPdf(file, plan, folderId) {
  if (!file) return;
  const ext = (/\.[^.]+$/.exec(file.name) || [""])[0].toLowerCase();
  if (!UPLOAD_CONTENT_TYPES[ext]) {
    setUploadStatus("Only PDF, Word (.docx), or PowerPoint (.pptx) files can be indexed.", true);
    return;
  }
  const zone = $("dropZone");
  zone.classList.add("busy");
  setUploadStatus(`<span class="spinner"></span>Uploading and starting <b>${escapeHtml(file.name)}</b>…`);
  try {
    const headers = {
      "Content-Type": UPLOAD_CONTENT_TYPES[ext],
      "X-Filename": encodeURIComponent(file.name),
    };
    if (plan) headers["X-Cross-Link-Plan"] = encodeURIComponent(JSON.stringify(plan));
    if (folderId) headers["X-Group-Id"] = encodeURIComponent(folderId);
    const res = await api("/api/upload", { method: "POST", headers, body: file });
    const where = folderById(folderId);
    setUploadStatus(`Indexing <b>${escapeHtml(res.name)}</b>${where ? ` in <b>${escapeHtml(where.name)}</b>` : ""} — progress appears in its row.`);
    await loadLibrary();
  } catch (err) {
    setUploadStatus("Upload failed: " + escapeHtml(err.message), true);
  } finally {
    zone.classList.remove("busy");
    $("fileInput").value = "";
  }
}

function startUpload(file, folderId) {
  if (!file) return;
  const target = folderById(folderId) ? folderId : null;
  openCrossLinkModal({ kind: "ingest", file, folderId: target });
}

/* wiring */

const DRAG_TYPE = "application/x-noetics-item";

function wireLibrary() {
  const lib = state.library;
  const input = $("fileInput");
  const uploadTarget = () => (folderById(lib.folder) ? lib.folder : null);

  $("dropZone").addEventListener("click", () => input.click());
  $("dropZone").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => startUpload(input.files[0], uploadTarget()));

  $("driveNewBtn").addEventListener("click", (e) => {
    e.stopPropagation();
    openDriveMenu(placeMenuItems(), $("driveNewBtn"));
  });

  $("driveSearch").addEventListener("input", (e) => {
    lib.search = e.target.value;
    lib.selected = null;
    renderLibrary();
  });
  $("driveFilters").addEventListener("click", (e) => {
    const chip = e.target.closest("[data-status]");
    if (!chip) return;
    lib.status = lib.status === chip.dataset.status ? "" : chip.dataset.status;
    renderLibrary();
  });
  $("driveInfoBtn").addEventListener("click", () => {
    lib.detailOpen = !lib.detailOpen;
    writePref("noetics.libraryDetail", lib.detailOpen ? "open" : "closed");
    renderLibrary();
  });

  // Navigation shared by the sidebar, breadcrumbs and location chips.
  $("libraryView").addEventListener("click", (e) => {
    const toggle = e.target.closest("[data-toggle]");
    if (toggle) {
      e.stopPropagation();
      const id = toggle.dataset.toggle;
      if (lib.expanded.has(id)) lib.expanded.delete(id); else lib.expanded.add(id);
      renderDriveTree();
      return;
    }
    const menuBtn = e.target.closest("[data-menu]");
    if (menuBtn) {
      const item = lib.menuItems[Number(menuBtn.dataset.menu)];
      closeDriveMenu();
      if (item && item.act) item.act();
      return;
    }
    if (!e.target.closest("#driveMenu")) closeDriveMenu();
    const crumbMenu = e.target.closest("[data-crumb-menu]");
    if (crumbMenu) {
      e.stopPropagation();
      const folder = folderById(crumbMenu.dataset.crumbMenu);
      openDriveMenu(folder ? folderMenuItems(folder) : placeMenuItems(), crumbMenu);
      return;
    }
    const nav = e.target.closest("[data-nav]");
    if (nav && !e.target.closest(".dRow, #driveDetail")) {
      navigateFolder(nav.dataset.nav);
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!$("driveMenu").hidden) closeDriveMenu();
  });

  // Rows and cards: click selects, double-click / Enter opens, ⋮ or right-click menus.
  const body = $("driveBody");
  body.addEventListener("click", (e) => {
    if (e.target.closest("[data-act='add-existing']")) { openAddDocsModal(lib.folder); return; }
    if (e.target.closest("[data-sort]")) {
      const key = e.target.closest("[data-sort]").dataset.sort;
      lib.sort = lib.sort.key === key
        ? { key, dir: lib.sort.dir === "asc" ? "desc" : "asc" }
        : { key, dir: key === "name" ? "asc" : "desc" };
      renderDriveBody();
      return;
    }
    const item = e.target.closest("[data-kind]");
    if (!item) { selectItem(null); return; }
    const locNav = e.target.closest(".locChip[data-nav]");
    if (locNav) { navigateFolder(locNav.dataset.nav); return; }
    if (e.target.closest("[data-more]")) {
      e.stopPropagation();
      selectItem(item.dataset.kind, item.dataset.id);
      const fresh = body.querySelector(`[data-kind="${item.dataset.kind}"][data-id="${CSS.escape(item.dataset.id)}"] [data-more]`);
      openDriveMenu(itemMenu(item.dataset.kind, item.dataset.id), fresh || { x: e.clientX, y: e.clientY });
      return;
    }
    selectItem(item.dataset.kind, item.dataset.id);
  });
  body.addEventListener("dblclick", (e) => {
    const item = e.target.closest(".dRow[data-kind]");
    if (item && !e.target.closest("[data-more], .locChip[data-nav]")) openItem(item.dataset.kind, item.dataset.id);
  });
  body.addEventListener("keydown", (e) => {
    const item = e.target.closest(".dRow[data-kind]");
    if (!item) return;
    if (e.key === "Enter") { e.preventDefault(); openItem(item.dataset.kind, item.dataset.id); }
    if (e.key === " ") { e.preventDefault(); selectItem(item.dataset.kind, item.dataset.id); }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const all = Array.from(body.querySelectorAll(".dRow[data-kind]"));
      const next = all[all.indexOf(item) + (e.key === "ArrowDown" ? 1 : -1)];
      if (next) next.focus();
    }
  });
  $("driveMain").addEventListener("contextmenu", (e) => {
    if (e.target.closest("input, #driveFilters")) return;
    e.preventDefault();
    const item = e.target.closest(".dRow[data-kind]");
    if (item) {
      selectItem(item.dataset.kind, item.dataset.id);
      openDriveMenu(itemMenu(item.dataset.kind, item.dataset.id), { x: e.clientX, y: e.clientY });
    } else {
      openDriveMenu(placeMenuItems(), { x: e.clientX, y: e.clientY });
    }
  });
  $("driveTree").addEventListener("contextmenu", (e) => {
    const item = e.target.closest(".navItem[data-kind='folder']");
    if (!item) return;
    e.preventDefault();
    openDriveMenu(folderMenuItems(folderById(item.dataset.id)), { x: e.clientX, y: e.clientY });
  });

  // Details panel actions.
  $("driveDetail").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-act], .stopRun, .startRun, [data-link-a], .locChip[data-nav]");
    if (!btn) return;
    const runDir = (btn.closest("[data-run]") || {}).dataset?.run;
    const folderId = (btn.closest("[data-folder]") || {}).dataset?.folder;
    if (btn.matches(".locChip[data-nav]")) return navigateFolder(btn.dataset.nav);
    if (btn.dataset.linkA) return viewGraphLink(btn.dataset.linkA, btn.dataset.linkB);
    if (btn.classList.contains("stopRun")) return stopRun(runDir);
    if (btn.classList.contains("startRun")) {
      const select = btn.closest(".detailActions").querySelector(".rebuildSelect");
      return startRun(runDir, select && select.value ? select.value : null);
    }
    const actions = {
      "close-detail": () => { lib.detailOpen = false; writePref("noetics.libraryDetail", "closed"); renderLibrary(); },
      preview: () => openPdfPreview(runDir),
      open: () => openRunByDir(runDir),
      chat: () => chatAbout("document", runDir),
      link: () => openCrossLinkModal({ kind: "retroactive", runDir }),
      tag: () => editRunTag(runDir),
      move: () => openFolderPicker({ mode: "move-run", id: runDir }),
      "add-to": () => openFolderPicker({ mode: "add-run", id: runDir }),
      "remove-from": () => removeRunFromFolder(runDir, lib.folder),
      delete: () => deleteRun(runDir),
      "open-folder": () => navigateFolder(folderId),
      "chat-folder": () => chatAbout("group", folderId),
      "graph-folder": () => graphOfFolder(folderId),
      "new-subfolder": () => createFolder(folderId),
      "add-existing": () => openAddDocsModal(folderId),
      "rename-folder": () => renameFolder(folderId),
      "move-folder": () => openFolderPicker({ mode: "move-folder", id: folderId }),
      "delete-folder": () => deleteFolder(folderId),
    };
    const act = actions[btn.dataset.act];
    if (act) act();
  });

  wireLibraryDragAndDrop();
  wireFolderPicker();
  wirePdfPreview();
  wireCrossLinkModal();
}

// Drag a row onto a folder (row, card, sidebar or breadcrumb) to move it there;
// drop a PDF from the desktop anywhere on the list to upload into this folder.
function wireLibraryDragAndDrop() {
  const lib = state.library;
  const view = $("libraryView");
  const overlay = $("dropOverlay");
  let dragItem = null;
  let fileDragDepth = 0;

  const isFileDrag = (e) => Array.from(e.dataTransfer.types || []).includes("Files");
  const dropFolderOf = (e) => {
    const el = e.target.closest("[data-drop-folder]");
    return el ? { el, id: el.dataset.dropFolder } : null;
  };
  const canDrop = (target) => {
    if (!dragItem || !target) return false;
    if (dragItem.kind === "folder") return !folderSubtreeIds(dragItem.id).includes(target.id)
      && (folderById(dragItem.id).parentId || LIBRARY_ROOT) !== target.id;
    return true;
  };
  const clearHighlights = () => view.querySelectorAll(".dropTarget").forEach((el) => el.classList.remove("dropTarget"));
  const finishDrag = () => {
    dragItem = null;
    state.libraryDragging = false;
    clearHighlights();
    view.querySelectorAll(".dragging").forEach((el) => el.classList.remove("dragging"));
    if (state.libraryRenderPending) { state.libraryRenderPending = false; renderLibrary(); }
  };

  view.addEventListener("dragstart", (e) => {
    const item = e.target.closest("[draggable='true'][data-kind]");
    if (!item) return;
    dragItem = { kind: item.dataset.kind, id: item.dataset.id };
    state.libraryDragging = true;
    item.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData(DRAG_TYPE, JSON.stringify(dragItem));
    e.dataTransfer.setData("text/plain", item.dataset.kind === "run" ? findRun(item.dataset.id).name : folderById(item.dataset.id).name);
  });
  view.addEventListener("dragend", finishDrag);

  view.addEventListener("dragover", (e) => {
    if (isFileDrag(e)) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "copy";
      return;
    }
    const target = dropFolderOf(e);
    clearHighlights();
    if (!canDrop(target)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    target.el.classList.add("dropTarget");
  });

  view.addEventListener("dragenter", (e) => {
    if (!isFileDrag(e)) return;
    fileDragDepth += 1;
    const where = folderById(lib.folder);
    $("dropOverlayText").textContent = `Drop to upload to ${where ? where.name : "My Library"}`;
    overlay.hidden = false;
  });
  view.addEventListener("dragleave", (e) => {
    if (!isFileDrag(e)) return;
    fileDragDepth = Math.max(0, fileDragDepth - 1);
    if (!fileDragDepth) overlay.hidden = true;
  });

  view.addEventListener("drop", async (e) => {
    if (isFileDrag(e)) {
      e.preventDefault();
      fileDragDepth = 0;
      overlay.hidden = true;
      const target = dropFolderOf(e);
      const folderId = target ? target.id : lib.folder;
      startUpload(e.dataTransfer.files[0], folderId);
      return;
    }
    const target = dropFolderOf(e);
    const item = dragItem;
    if (!canDrop(target)) return finishDrag();
    e.preventDefault();
    finishDrag();
    if (item.kind === "folder") await moveFolder(item.id, target.id);
    else await moveRun(item.id, target.id);
  });
}

function wireFolderPicker() {
  $("folderPickerTree").addEventListener("click", (e) => {
    const row = e.target.closest("[data-pick]");
    if (!row || row.disabled) return;
    state.folderPicker.target = row.dataset.pick;
    renderFolderPicker();
  });
  $("folderPickerTree").addEventListener("dblclick", (e) => {
    if (e.target.closest("[data-pick]:not([disabled])")) confirmFolderPicker();
  });
  $("folderPickerNew").addEventListener("click", async () => {
    const picker = state.folderPicker;
    const parent = picker.target && folderById(picker.target) ? picker.target : null;
    const created = await createFolder(parent);
    if (created && state.folderPicker) {
      state.folderPicker.target = created.id;
      renderFolderPicker();
    }
  });
  $("folderPickerCancel").addEventListener("click", closeFolderPicker);
  $("folderPickerConfirm").addEventListener("click", confirmFolderPicker);
  $("folderPickerModal").addEventListener("click", (e) => {
    if (e.target === $("folderPickerModal")) closeFolderPicker();
  });
  $("addDocsCancel").addEventListener("click", closeAddDocsModal);
  $("addDocsSave").addEventListener("click", saveAddDocs);
  $("addDocsModal").addEventListener("click", (e) => {
    if (e.target === $("addDocsModal")) closeAddDocsModal();
  });
}

/* --------------------------- Cross-link modal ------------------------------ */
// One modal, two contexts: "ingest" (shown before /api/upload, the choice is
// sent as a header and applied once the new document's graph stage finishes)
// and "retroactive" (an already-indexed document, applied immediately via
// POST /api/cross-link). Mesh mode needs an already-indexed document to find
// matches from, so it's hidden in ingest context.

function crossLinkOtherRuns() {
  if (state.crossLink && state.crossLink.kind === "group-add") {
    const allowed = new Set(state.crossLink.candidateRunDirs);
    return state.libraryRuns.filter((r) => allowed.has(r.runDir));
  }
  const excluded = state.crossLink && state.crossLink.kind === "retroactive" ? state.crossLink.runDir : null;
  return state.libraryRuns.filter((r) => r.viewable && r.runDir !== excluded);
}

function renderCrossLinkDocsPicker() {
  const box = $("crossLinkDocsPicker");
  const others = crossLinkOtherRuns();
  // group-add: every listed document is missing from this run's existing
  // links, so all of them start checked -- the point of the prompt is
  // catching up the gap, not picking a subset.
  const preChecked = state.crossLink && state.crossLink.kind === "group-add";
  box.innerHTML = others.length
    ? others.map((r) => `
        <label class="modalCheck">
          <input type="checkbox" value="${escapeHtml(r.runDir)}"${preChecked ? " checked" : ""} />
          <span>${escapeHtml(r.name)}</span>
        </label>`).join("")
    : '<div class="modalHint">No other indexed documents yet.</div>';
}

function renderCrossLinkGroupPicker() {
  const sel = $("crossLinkGroupPicker");
  sel.innerHTML = state.groups.length
    ? foldersSortedByPath().map((g) =>
        `<option value="${escapeHtml(g.id)}">${escapeHtml(g.path)} (${runsInFolderTree(g.id).length})</option>`).join("")
    : '<option value="">No folders yet</option>';
}

function updateCrossLinkModeUI() {
  const mode = document.querySelector('input[name="crossLinkMode"]:checked').value;
  $("crossLinkDocsPicker").hidden = mode !== "documents";
  $("crossLinkGroupPicker").hidden = mode !== "group";
}

function openCrossLinkModal(context) {
  state.crossLink = context;
  const isIngest = context.kind === "ingest";
  const isGroupAdd = context.kind === "group-add";
  $("crossLinkTitle").textContent = isGroupAdd
    ? "Link to the rest of this folder?"
    : isIngest
    ? "Cross-link this document?"
    : "Cross-link this document with…";
  $("crossLinkHint").textContent = isGroupAdd
    ? `Not yet cross-linked with ${context.candidateRunDirs.length} document(s) in the folder you just filed it in.`
    : isIngest
    ? "Connect matching entities in this document to others already indexed, so chat and the graph can answer across documents."
    : "Extend this already-indexed document's cross-document connections.";
  // group-add always means "pick documents" against a fixed, already-known
  // candidate list -- the mode choice itself is moot, so the radios are
  // hidden rather than shown with only one real option.
  $("crossLinkOptions").hidden = isGroupAdd;
  $("crossLinkMeshOption").hidden = isIngest;
  $("crossLinkAutoOption").hidden = !isIngest;
  $("crossLinkSkipOption").hidden = !isIngest;
  // Uploading into a folder that already holds indexed documents defaults to
  // linking against that folder.
  const intoFolder = isIngest && context.folderId
    && runsInFolderTree(context.folderId).some((r) => r.viewable);
  const defaultMode = isGroupAdd || !isIngest ? "documents" : intoFolder ? "group" : "auto";
  document.querySelector(`input[name="crossLinkMode"][value="${defaultMode}"]`).checked = true;
  renderCrossLinkDocsPicker();
  renderCrossLinkGroupPicker();
  if (intoFolder) $("crossLinkGroupPicker").value = context.folderId;
  updateCrossLinkModeUI();
  $("crossLinkModal").hidden = false;
}

function closeCrossLinkModal() {
  $("crossLinkModal").hidden = true;
  state.crossLink = null;
  $("fileInput").value = "";
}

async function confirmCrossLink() {
  const mode = document.querySelector('input[name="crossLinkMode"]:checked').value;
  const context = state.crossLink;
  if (!context) return;

  let targetRunDirs, groupId;
  if (mode === "documents") {
    targetRunDirs = Array.from($("crossLinkDocsPicker").querySelectorAll("input:checked")).map((i) => i.value);
    if (!targetRunDirs.length) { alert("Pick at least one document."); return; }
  } else if (mode === "group") {
    groupId = $("crossLinkGroupPicker").value;
    if (!groupId) { alert("Pick a folder."); return; }
  }

  if (context.kind === "ingest") {
    const plan = { mode, targetRunDirs, groupId };
    $("crossLinkModal").hidden = true;
    const { file, folderId } = context;
    state.crossLink = null;
    await uploadPdf(file, plan, folderId);
    return;
  }

  try {
    await api("/api/cross-link", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ runDir: context.runDir, mode, targetRunDirs, groupId }),
    });
  } catch (err) {
    alert("Could not start cross-linking: " + err.message);
    return;
  }
  $("crossLinkModal").hidden = true;
  state.crossLink = null;
  pollCrossLinkStatus(context.runDir);
}

// Cross-linking runs detached (it can involve several LLM calls), so this
// polls the status Scripts/pipeline.py records in Postgres and surfaces it
// through the same status box the upload flow uses.
const CROSS_LINK_POLL_MS = 2000;
const CROSS_LINK_POLL_MAX = 45;

async function pollCrossLinkStatus(runDir, attempt) {
  attempt = attempt || 0;
  let status;
  try {
    status = await api(`/api/cross-link-status?run=${encodeURIComponent(runDir)}`);
  } catch (err) {
    return; // best-effort; the library grid still reflects the outcome next poll
  }
  if (status.status === "running" || status.status === "pending") {
    if (attempt >= CROSS_LINK_POLL_MAX) return;
    setUploadStatus(`<span class="spinner"></span>Cross-linking…`);
    setTimeout(() => pollCrossLinkStatus(runDir, attempt + 1), CROSS_LINK_POLL_MS);
    return;
  }
  if (status.status === "failed") {
    setUploadStatus("Cross-linking failed: " + escapeHtml(status.error || "unknown error"), true);
  } else if (status.status === "succeeded") {
    setUploadStatus("Cross-linking complete.");
  }
  await loadLibrary();
}

function wireCrossLinkModal() {
  document.querySelectorAll('input[name="crossLinkMode"]').forEach((radio) =>
    radio.addEventListener("change", updateCrossLinkModeUI));
  $("crossLinkCancel").addEventListener("click", closeCrossLinkModal);
  $("crossLinkConfirm").addEventListener("click", confirmCrossLink);
  $("crossLinkModal").addEventListener("click", (e) => {
    if (e.target === $("crossLinkModal")) closeCrossLinkModal();
  });
}

/* ============================== scope bar ================================= */
// One shared control -- Document / Folder / All -- drives Library, Graph,
// Chat and Tree together. Tree is the one view that only makes sense in
// Document scope; the others read whatever this is set to.

function loadScopeBar() {
  // Only completed documents belong in the picker -- chat/graph can't do
  // anything with one still indexing. Derived from state.libraryRuns
  // (kept fresh by loadLibrary()'s own polling) rather than a second fetch.
  state.documentRuns = state.libraryRuns.filter((r) => r.viewable);
  renderScopeBar();
}

function renderScopeBar() {
  [["headerScopeDocumentBtn", "document"], ["headerScopeGroupBtn", "group"], ["headerScopeAllBtn", "all"]]
    .forEach(([id, value]) => {
      const btn = $(id);
      const active = state.scopeKind === value;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-pressed", String(active));
    });

  const docPicker = $("headerDocumentPicker");
  docPicker.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = state.documentRuns.length ? "Select a document…" : "No indexed documents";
  placeholder.selected = !state.currentRun;
  docPicker.appendChild(placeholder);
  for (const run of state.documentRuns) {
    const opt = document.createElement("option");
    opt.value = run.runDir;
    opt.textContent = run.name;
    if (run.runDir === state.currentRun) opt.selected = true;
    docPicker.appendChild(opt);
  }
  docPicker.disabled = state.documentRuns.length === 0;
  docPicker.hidden = state.scopeKind !== "document";

  const groupPicker = $("headerGroupPicker");
  groupPicker.innerHTML = state.groups.length
    ? foldersSortedByPath().map((g) => `<option value="${escapeHtml(g.id)}">${escapeHtml(g.path)}</option>`).join("")
    : '<option value="">No folders yet</option>';
  if (state.scopeGroupId) groupPicker.value = state.scopeGroupId;
  groupPicker.hidden = state.scopeKind !== "group";

  const hint = $("headerScopeHint");
  if (state.scopeKind === "all" && state.allDocumentCount != null) {
    hint.textContent = `${state.allDocumentCount} document${state.allDocumentCount === 1 ? "" : "s"}`;
  } else if (state.scopeKind === "group" && state.groupDocumentCount != null) {
    hint.textContent = `${state.groupDocumentCount} document${state.groupDocumentCount === 1 ? "" : "s"}`;
  } else {
    hint.textContent = "";
  }

  const hasAnyDocs = state.documentRuns.length > 0;
  document.querySelectorAll('.tab[data-view="graph"], .tab[data-view="tree"], .tab[data-view="chat"]')
    .forEach((tab) => { tab.disabled = !hasAnyDocs; });

  const crossDocRow = $("graphCrossDocOnlyRow");
  if (crossDocRow) crossDocRow.hidden = state.scopeKind === "document";

  renderTreeScopeState();

  const foot = $("chatSidebarFoot");
  if (foot) {
    foot.textContent = state.scopeKind === "all"
      ? "Conversations are saved for all documents."
      : state.scopeKind === "group"
        ? "Conversations are saved for this folder."
        : "Conversations are saved for this document.";
  }
}

function wireScopeBar() {
  $("headerScopeDocumentBtn").addEventListener("click", () => setScope("document", {
    runDir: state.currentRun || (state.documentRuns[0] && state.documentRuns[0].runDir),
  }));
  $("headerScopeGroupBtn").addEventListener("click", () =>
    setScope("group", { groupId: state.scopeGroupId || (foldersSortedByPath()[0] || {}).id }));
  $("headerScopeAllBtn").addEventListener("click", () => setScope("all"));
  $("headerDocumentPicker").addEventListener("change", (e) => {
    if (e.target.value) setScope("document", { runDir: e.target.value });
  });
  $("headerGroupPicker").addEventListener("change", (e) => setScope("group", { groupId: e.target.value }));
}

async function setScope(kind, { runDir, groupId, fromJump = false } = {}) {
  if (kind === "group" && !groupId) {
    alert("Create a folder first (Library → New → New folder).");
    return;
  }
  const unchanged = kind === state.scopeKind
    && (runDir === undefined || runDir === state.currentRun)
    && (kind !== "group" || groupId === state.scopeGroupId);
  if (unchanged) return;
  if (!fromJump) state.pendingScopeRestore = null;

  let data;
  try {
    data = await api("/api/scope", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind, runDir, groupId }),
    });
  } catch (err) {
    alert("Could not switch scope: " + err.message);
    return;
  }
  state.scopeKind = data.kind;
  if (data.runDir) state.currentRun = data.runDir;
  state.scopeGroupId = data.groupId || null;
  if (data.kind === "all") state.allDocumentCount = data.documentCount;
  if (data.kind === "group") state.groupDocumentCount = data.documentCount;
  // Group/All defaults to showing only what actually connects the
  // documents -- the full merged entity soup isn't the point of a
  // cross-document view. Still togglable per view session.
  state.graphCrossDocOnly = data.kind !== "document";
  const crossDocCheckbox = $("graphCrossDocOnly");
  if (crossDocCheckbox) crossDocCheckbox.checked = state.graphCrossDocOnly;

  renderScopeBar();
  await loadLibrary();

  if (data.kind === "document") {
    resetForNewRun();
    await loadData();
  } else {
    await loadGraph();
    await loadConversations();
  }
}

function renderTreeScopeState() {
  const needsDocument = state.scopeKind !== "document";
  const empty = $("treeScopeEmpty");
  if (empty) empty.hidden = !needsDocument;
  const net = $("treeNet");
  const legend = $("treeLegend");
  if (net) net.style.display = needsDocument ? "none" : "";
  if (legend) legend.style.display = needsDocument ? "none" : "";

  // Tree only ever needs one document -- it carries its own picker in the
  // stage bar rather than relying on the header's Document/Group/All
  // control (which is hidden on this tab; see updateHeaderScopeVisibility),
  // since Group/All are never valid choices for a tree.
  const picker = $("treeDocumentPicker");
  if (picker) {
    picker.innerHTML = state.documentRuns.length
      ? state.documentRuns.map((r) => `<option value="${escapeHtml(r.runDir)}">${escapeHtml(r.name)}</option>`).join("")
      : '<option value="">No indexed documents</option>';
    picker.disabled = state.documentRuns.length === 0;
    if (!needsDocument && state.currentRun) picker.value = state.currentRun;
  }
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
  $("graphLinkBanner").hidden = true;
  const [tree, graph] = await Promise.all([api("/api/tree"), api("/api/graph")]);
  state.tree = tree;

  state.treeNodesById = new Map(tree.nodes.map((n) => [n.id, n]));
  state.treeDepthById = computeTreeDepths(tree.nodes);
  state.treeLevelById = computeTreeLevels(tree.nodes);

  buildTypeFilter($("treeTypeFilter"), tree.nodes.map((n) => n.type));
  renderTreeList();
  renderTreeLegend();

  applyGraphPayload(graph);

  const active = document.querySelector(".tab.active").dataset.view;
  if (active === "tree") buildTreeNetwork();

  // Saved history belongs to this run, so it loads with the run's data. The
  // most recent conversation is reopened rather than leaving an empty pane.
  await loadConversations();
}

// Applies a fetched /api/graph payload (any scope) to graph-related state and
// re-renders whatever's already visible, so a scope switch never needs a
// full loadData() re-run of the tree side.
function applyGraphPayload(graph) {
  state.graph = graph;
  state.graphNodesById = new Map((graph.nodes || []).map((n) => [n.id, n]));

  // reverse index: which entities were extracted from each tree node
  // (only meaningful in Document scope -- corpus scope nodes carry
  // documentVersionIds instead of tree-node sourceIds).
  state.entitiesBySourceNode = new Map();
  for (const ent of graph.nodes || []) {
    for (const sid of ent.sourceIds || []) {
      if (!state.entitiesBySourceNode.has(sid)) state.entitiesBySourceNode.set(sid, []);
      state.entitiesBySourceNode.get(sid).push(ent);
    }
  }

  buildTypeFilter($("entityTypeFilter"), (graph.nodes || []).map((n) => n.type));
  renderEntityList();
  renderGraphLegend();

  if ($("graphView").classList.contains("active")) buildGraphNetwork();
  else state.graphBuilt = false;
}

// Cross-link chips open a one-off, read-only view of exactly two documents'
// merged graph -- what actually connects them -- without touching the
// shared scope. "Back to scope" restores the normal scoped view.
async function viewGraphLink(a, b) {
  activateView("graph");
  const nameOf = (dir) => {
    const run = state.libraryRuns.find((r) => r.runDir === dir);
    return (run && run.name) || dir;
  };
  $("graphHint").textContent = "Loading…";
  let graph;
  try {
    graph = await api(`/api/graph-link?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`);
  } catch (err) {
    $("graphHint").textContent = "Could not load link: " + err.message;
    return;
  }
  state.graphCrossDocOnly = true;
  const crossDocCheckbox = $("graphCrossDocOnly");
  if (crossDocCheckbox) crossDocCheckbox.checked = true;
  applyGraphPayload(graph);
  $("graphLinkBannerText").textContent = `Showing what connects ${nameOf(a)} and ${nameOf(b)}`;
  $("graphLinkBanner").hidden = false;
}

function wireGraphLinkBanner() {
  $("graphLinkBannerBack").addEventListener("click", () => {
    $("graphLinkBanner").hidden = true;
    loadGraph();
  });
}

async function loadGraph() {
  // No params: the server always builds the graph for the one shared scope
  // (state.scopeKind/scopeGroupId), set via setScope().
  $("graphLinkBanner").hidden = true;
  $("graphHint").textContent = "Loading…";
  let graph;
  try {
    graph = await api("/api/graph");
  } catch (err) {
    $("graphHint").textContent = "Could not load graph: " + err.message;
    return;
  }
  applyGraphPayload(graph);
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
  $("graphCrossDocOnly").addEventListener("change", (e) => {
    state.graphCrossDocOnly = e.target.checked;
    if (state.graphBuilt) applyGraphFilter();
  });
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
  const crossOnly = state.graphCrossDocOnly;
  const matched = (state.graph.nodes || []).filter((n) => {
    const okType = !type || (n.type || "UNKNOWN") === type;
    const okCross = !crossOnly || n.crossDocument;
    const hay = `${n.label} ${n.type} ${n.description}`.toLowerCase();
    return okType && okCross && (!q || hay.includes(q));
  });
  if (matched.length <= GRAPH_RENDER_BUDGET) return matched;
  return [...matched]
    .sort((a, b) => (b.degree || 0) - (a.degree || 0))
    .slice(0, GRAPH_RENDER_BUDGET);
}

// Node fill = entity type (colorForType). Node border = provenance: which
// document(s) it came from -- only meaningful once more than one document is
// in play (Group/All scope with 2+ documents). A single-document border color
// identifies that document; a thick accent ring flags an entity whose own
// name/type recurred across documents (merged into one node), which reads
// differently from a merely cross-linked entity.
function toVisEntity(n, labelsOn) {
  const isHub = n.type === "GROUPING";
  const docTitles = [...new Set(n.documentTitles || [])];
  const docLine = docTitles.length ? `\nDocuments: ${docTitles.join(", ")}` : "";
  const multiDocScope = (state.graph.documentCount || 0) > 1;

  let borderColor = isHub ? "#111" : "#ffffff";
  let borderWidth = isHub ? 2 : 1.5;
  if (multiDocScope && !isHub) {
    if (docTitles.length > 1) {
      borderColor = "#0f766e"; // this entity itself spans multiple documents
      borderWidth = 3;
    } else if (docTitles.length === 1) {
      borderColor = colorForDocument(docTitles[0]);
      borderWidth = 2.5;
    }
  }

  return {
    id: n.id,
    label: labelsOn ? clip(n.label, 26) : undefined,
    title: `${n.label}\n[${n.type}]  ·  degree ${n.degree}\n${clip(n.description, 160)}${docLine}`,
    value: (n.degree || 0) + 1,
    shape: isHub ? "box" : "dot",
    borderWidth,
    color: { background: colorForType(n.type), border: borderColor,
      highlight: { background: colorForType(n.type), border: borderColor } },
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

// Same-document relation: default grey, solid (set at the network level).
// Cross-document relation (two different documents' entities, LLM-judged):
// amber, dashed, arrowed -- these are the ones a document-scoped graph could
// never show. Grouping membership (an entity attached to a topic hub): violet,
// dotted -- a looser, structural link, not a factual claim, so it reads
// differently from an actual relation even though both cross documents.
const CROSS_DOC_RELATION_COLOR = "#d97706";
const GROUPING_EDGE_COLOR = "#7c3aed";

function toVisEdge(l) {
  const isGrouping = l.name === "grouping_member";
  const cross = !!l.crossDocument;
  const kindLabel = isGrouping ? "Grouping" : cross ? "Cross-document" : "";
  return {
    id: l.id, from: l.source, to: l.target,
    value: Number(l.weight) || 1,
    length: edgeLength(l),   // varies per edge by endpoint degrees (hubs longer)
    title: `${kindLabel ? kindLabel + " · " : ""}${l.sourceName} → ${l.targetName}\n${clip(l.description, 160)}`,
    dashes: isGrouping ? [2, 4] : cross ? [8, 4] : false,
    color: isGrouping
      ? { color: GROUPING_EDGE_COLOR, highlight: GROUPING_EDGE_COLOR, opacity: 0.55 }
      : cross
      ? { color: CROSS_DOC_RELATION_COLOR, highlight: CROSS_DOC_RELATION_COLOR, opacity: 0.85 }
      : undefined,
    width: cross && !isGrouping ? 2.25 : undefined,
    arrows: cross && !isGrouping ? "to" : undefined,
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
  if (state.graphNet) { state.graphNet.destroy(); state.graphNet = null; }
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
  const links = (state.graph.links || []).filter(
    (l) => ids.has(l.source) && ids.has(l.target) && (!state.graphCrossDocOnly || l.crossDocument),
  );

  // re-tune physics for the current node count / degree, then re-layout
  state.graphNet.setOptions({ physics: graphPhysicsFor(visN) });
  graphPhysicsOn = true;
  $("graphPhysics").textContent = "Freeze";

  state.graphNodes.clear();
  state.graphEdges.clear();
  state.graphNodes.add(visN.map((n) => toVisEntity(n, labelsOn)));
  state.graphEdges.add(links.map((l) => toVisEdge(l)));

  const total = (state.graph.nodes || []).length;
  const capped = visN.length >= GRAPH_RENDER_BUDGET && total > GRAPH_RENDER_BUDGET;
  if (visN.length === 0 && state.graphCrossDocOnly && total > 0) {
    $("graphStats").textContent =
      `0 of ${total} entities shown — none are cross-document yet in this scope. Uncheck "Cross-document only" to see everything.`;
  } else {
    $("graphStats").textContent =
      `${visN.length} of ${total} entities · ${links.length} relationships shown`
      + (capped ? " · most-connected first; search or filter to see others" : "");
  }
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
  const typeSwatches = types.map((t) =>
    `<span class="swatch" data-type="${escapeHtml(t)}"><span class="dot" style="background:${colorForType(t)}"></span>${escapeHtml(t)}</span>`
  ).join("");

  // Node fill already keys off entity type (above); border and edge style key
  // off document/cross-document status, which only means anything once more
  // than one document is actually in the graph (Group/All scope, 2+ docs).
  const multiDoc = (state.graph.documentCount || 0) > 1;
  let edgeKey = "";
  let docSwatches = "";
  if (multiDoc) {
    edgeKey = `<span class="legendDivider"></span>
      <span class="legendLine"><span class="ln"></span>Same-document relation</span>
      <span class="legendLine"><span class="ln cross"></span>Cross-document relation</span>
      <span class="legendLine"><span class="ln grouping"></span>Grouping membership</span>
      <span class="legendLine"><span class="dot ring" style="border-color:#0f766e"></span>Entity spans documents</span>`;
    const docTitles = [...new Set((state.graph.nodes || []).flatMap((n) => n.documentTitles || []))].sort();
    if (docTitles.length) {
      docSwatches = `<span class="legendDivider"></span>` + docTitles.map((title) =>
        `<span class="swatch static"><span class="dot ring" style="border-color:${colorForDocument(title)}"></span>${escapeHtml(title)}</span>`
      ).join("");
    }
  }

  $("graphLegend").innerHTML = typeSwatches + edgeKey + docSwatches;
  $("graphLegend").querySelectorAll(".swatch:not(.static)").forEach((sw) => {
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

  const isCorpusNode = !ent.sourceIds && ent.documentTitles;
  const sources = isCorpusNode
    ? (
        [...new Set(ent.documentTitles)].map((title) =>
          `<div class="card"><div class="h">${escapeHtml(title)}</div></div>`).join("")
        || `<p class="prose">No document recorded.</p>`
      )
    : (ent.sourceIds || []).map((sid, i) => {
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
    <div class="section-h">${isCorpusNode
      ? `Appears in ${new Set(ent.documentTitles).size} document(s)`
      : `Created from ${ent.sourceIds?.length || 0} source node(s)`}</div>
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
  $("treeDocumentPicker").addEventListener("change", (e) => {
    if (e.target.value) setScope("document", { runDir: e.target.value });
  });
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
      ? `<div class="section-h">${n.tableBody ? "Caption &amp; footnotes" : "Content"}
           <button class="miniBtn" id="editContent">edit</button></div>
         <div class="prose" id="contentText">${renderContent(n)}</div>`
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
  const editContentBtn = $("treeDetail").querySelector("#editContent");
  if (editContentBtn) editContentBtn.onclick = () => beginContentEdit(id, n.content);
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
    <span class="structNote" id="saveNote">saving re-embeds this node and re-attests the index</span></div>`;
  const button = host.querySelector("#saveSummary");
  const note = host.querySelector("#saveNote");
  button.onclick = async () => {
    const text = host.querySelector("#summaryDraft").value.trim();
    if (!text) return;
    // Saving re-embeds and re-attests before it answers, which takes seconds
    // rather than milliseconds. Without a busy state the button looked broken
    // and invited a second click, which would queue a second reconcile.
    button.disabled = true;
    button.textContent = "saving…";
    note.textContent = "re-embedding and re-attesting; this takes a few seconds";
    try {
      await api("/api/edit-summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodeId, text }),
      });
      await loadData();
      selectTreeNode(nodeId);
    } catch (err) {
      button.disabled = false;
      button.textContent = "save";
      note.textContent = "";
      alert("Edit failed: " + err.message);
    }
  };
}

function beginContentEdit(nodeId, current) {
  const host = $("contentText");
  if (!host) return;
  host.innerHTML = `<textarea id="contentDraft" rows="7" style="width:100%">${escapeHtml(current || "")}</textarea>
    <div style="margin-top:6px"><button class="miniBtn" id="saveContent">save</button>
    <span class="structNote" id="contentNote">a correction is recorded as a human edit of the
    original source, which stays cited alongside it</span></div>`;
  const button = host.querySelector("#saveContent");
  const note = host.querySelector("#contentNote");
  button.onclick = async () => {
    const text = host.querySelector("#contentDraft").value.trim();
    if (!text) return;
    button.disabled = true;
    button.textContent = "saving…";
    note.textContent = "re-deriving evidence, re-embedding and re-attesting…";
    try {
      await api("/api/edit-content", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodeId, text }),
      });
      await loadData();
      selectTreeNode(nodeId);
    } catch (err) {
      button.disabled = false;
      button.textContent = "save";
      note.textContent = "";
      alert("Edit failed: " + err.message);
    }
  };
}

/* =============================== CHAT =================================== */
const ASSISTANT_NAME = "Noetics-AI";
const SIDEBAR_COLLAPSED_KEY = "noeticsrag.chatSidebarCollapsed";
const CITATIONS_HIDDEN_KEY = "noeticsrag.chatCitationsHidden";
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
  const cites = $("citeToggle");
  if (cites) {
    cites.addEventListener("change", () => applyCitationsShown(cites.checked, true));
    applyCitationsShown(localStorage.getItem(CITATIONS_HIDDEN_KEY) !== "1", false);
  }
  showChatIntro();
}

// Whether an answer shows its own [E1] / [Q1-E1] markers is a reading
// preference, so it is applied to the whole transcript at once rather than per
// message: the same conversation reads either way without being asked again,
// and switching costs nothing because every marker is already its own element.
// Only the inline markers go -- the References list stays, since it is the
// evidence itself rather than an interruption in the prose.
function applyCitationsShown(shown, persist) {
  const shell = $("chatShell");
  const toggle = $("citeToggle");
  if (shell) shell.classList.toggle("citationsHidden", !shown);
  if (toggle) {
    toggle.checked = shown;
    toggle.title = shown
      ? "Hide the [E1] markers inside answers"
      : "Show the [E1] markers inside answers";
  }
  if (persist) {
    try { localStorage.setItem(CITATIONS_HIDDEN_KEY, shown ? "0" : "1"); } catch (e) {}
  }
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
  const run = escapeHtml(citation.runDir || "");
  // Keyed by node, not by label: one crop belongs to one node however many
  // evidence scopes cite it.
  return `<figure class="answerFigure" data-figure="${escapeHtml(String(citation.nodeId))}">
    <img src="${treeImageUrl(citation.nodeId)}" alt="${escapeHtml(citation.label)}" loading="lazy" />
    <figcaption>${caption}${
      citation.nodeId != null
        ? ` · <a class="linkish" data-tree="${escapeHtml(citation.nodeId)}" data-run="${run}">view in tree →</a>`
        : ""
    }</figcaption>
  </figure>`;
}

// Which crops an answer should show, and under which label each is anchored.
//
// One figure per *node*, not per label. A decomposed question gives every
// sub-question its own evidence scope, so one table reaches the answer as
// Q1-E5, Q2-E8 and Q3-E7 -- three labels, one crop -- and keying the
// "already drawn" check on the label drew the same table three times.
//
// The anchor is a label the prose actually uses, which is not always the
// lowest-numbered scope: in that answer Q1-E5 is never named and Q2-E8 is, so
// anchoring on the first label would have parked the figure at the end of the
// answer instead of beside the paragraph it illustrates.
//
// Kept free of the DOM so it can be tested on plain data; `isMentioned` is the
// only thing the caller has to look up.
function figuresToInline(citations, isMentioned) {
  const byNode = new Map();
  for (const citation of citations || []) {
    if (!citation || !citation.hasImage || citation.nodeId == null) continue;
    const key = String(citation.nodeId);
    if (!byNode.has(key)) byNode.set(key, []);
    byNode.get(key).push(citation);
  }
  const chosen = [];
  for (const group of byNode.values()) {
    chosen.push(group.find((c) => isMentioned(c.label)) || group[0]);
  }
  return chosen;
}

function inlineCitedFigures(scope, citations) {
  const root = scope.querySelector(".answer");
  if (!root) return;
  const withImages = figuresToInline(citations, (label) =>
    Boolean(root.querySelector(`a.cite[data-cite="${CSS.escape(label)}"]`)),
  );
  if (!withImages.length) return;

  for (const citation of withImages) {
    if (root.querySelector(`[data-figure="${CSS.escape(String(citation.nodeId))}"]`)) continue;
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

function renderReferenceCard(c) {
  const img = c.hasImage && c.nodeId != null
    ? `<img class="refThumb" src="${treeImageUrl(c.nodeId)}" alt="" loading="lazy" />`
    : "";
  const run = escapeHtml(c.runDir || "");
  const jumps = c.nodeId != null
    ? `<a class="linkish refJump" data-tree="${escapeHtml(c.nodeId)}" data-run="${run}">view in tree →</a>
       <a class="linkish refJump" data-graph-from-tree="${escapeHtml(c.nodeId)}" data-run="${run}">view in graph →</a>`
    : "";
  return `<div class="refCard" id="ref-${escapeHtml(c.label)}">
    <div class="refHead">
      <span class="refLabel">${escapeHtml(c.label)}</span>
      ${c.type ? `<span class="refType">${escapeHtml(c.type)}</span>` : ""}
      ${c.page ? `<span class="refPage">p${escapeHtml(c.page)}</span>` : ""}
    </div>
    ${img}
    ${renderSnippet(c)}
    ${jumps}
  </div>`;
}

function renderReferences(citations) {
  if (!citations || !citations.length) return "";
  // A single-document answer's references all share one source -- grouping
  // by it would just repeat the same heading over every card. A group/all-
  // scope answer draws on more than one document, and the reader has to be
  // able to tell which source backs which citation at a glance.
  const titles = new Set(citations.map((c) => c.documentTitle).filter(Boolean));
  let body;
  if (titles.size > 1) {
    const order = [];
    const byTitle = new Map();
    for (const c of citations) {
      const title = c.documentTitle || "Unknown document";
      if (!byTitle.has(title)) { byTitle.set(title, []); order.push(title); }
      byTitle.get(title).push(c);
    }
    body = order.map((title) => `
      <div class="refGroup">
        <div class="refGroupTitle">${escapeHtml(title)}</div>
        <div class="refList">${byTitle.get(title).map(renderReferenceCard).join("")}</div>
      </div>`).join("");
  } else {
    body = `<div class="refList">${citations.map(renderReferenceCard).join("")}</div>`;
  }
  // Closed by default, like every other evidence panel. Reopening it on every
  // answer buried the next question under a full reference list each time the
  // reader switched conversations; clicking a citation still opens it.
  return `<details class="evidence"><summary>References (${citations.length})</summary>${body}</details>`;
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
      // The whitespace around a marker belongs to the marker, and goes inside
      // its wrapper so that hiding inline references closes the prose up
      // instead of leaving the gaps behind. The space in front always counts.
      // The space behind counts when the model wrote "[E1] ." and the stop
      // would otherwise be left stranded, or when the marker opens the text
      // and so has no space of its own to give back.
      const before = text.slice(last, m.index);
      const lead = before.match(/[ \t]+$/);
      const afterIndex = m.index + m[0].length;
      const rest = text.slice(afterIndex);
      const spacing = rest.match(/^[ \t]+/);
      const tail =
        spacing && (!lead || /^[.,;:!?)]/.test(rest.slice(spacing[0].length)))
          ? spacing[0]
          : "";
      frag.appendChild(
        document.createTextNode(lead ? before.slice(0, -lead[0].length) : before)
      );
      const a = document.createElement("a");
      a.className = "cite";
      a.href = `#ref-${label}`;
      a.dataset.cite = label;
      a.textContent = `[${label}]`;
      // The wrapper is unstyled and is not a block, so the chip looks exactly
      // as it did and inlineCitedFigures still finds the marker's paragraph.
      const wrap = document.createElement("span");
      wrap.className = "citeWrap";
      if (lead) wrap.appendChild(document.createTextNode(lead[0]));
      wrap.appendChild(a);
      if (tail) wrap.appendChild(document.createTextNode(tail));
      frag.appendChild(wrap);
      last = afterIndex + tail.length;
    }
    frag.appendChild(document.createTextNode(text.slice(last)));
    textNode.replaceWith(frag);
  }
}

async function jumpToDocumentScope(runDir) {
  if (!runDir) return state.scopeKind === "document";
  if (state.scopeKind !== "document" || state.currentRun !== runDir) {
    if (!state.pendingScopeRestore) {
      state.pendingScopeRestore = {
        kind: state.scopeKind, runDir: state.currentRun, groupId: state.scopeGroupId,
      };
    }
    await setScope("document", { runDir, fromJump: true });
  }
  return true;
}

async function returnToChatScope() {
  const restore = state.pendingScopeRestore;
  state.pendingScopeRestore = null;
  if (restore) {
    await setScope(restore.kind, { runDir: restore.runDir, groupId: restore.groupId, fromJump: true });
  }
  activateView("chat");
}

async function jumpToTreeNode(runDir, nodeId) {
  activateView("tree");
  if (!(await jumpToDocumentScope(runDir))) return;
  selectTreeNode(Number(nodeId));
}

async function jumpToGraphFromTreeNode(runDir, nodeId) {
  activateView("graph");
  if (!(await jumpToDocumentScope(runDir))) return;
  const ents = state.entitiesBySourceNode.get(Number(nodeId)) || [];
  if (!ents.length) {
    alert("No graph entity was extracted from this specific passage.");
    return;
  }
  selectEntity(ents[0].id);
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
    a.onclick = () => jumpToTreeNode(a.dataset.run, a.dataset.tree));
  scope.querySelectorAll("[data-graph-from-tree]").forEach((a) =>
    a.onclick = () => jumpToGraphFromTreeNode(a.dataset.run, a.dataset.graphFromTree));
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
      // The references panel is collapsed until asked for, and a card inside a
      // closed <details> has no box to scroll to -- following a citation would
      // do nothing at all. Open it first, then scroll.
      const panel = target.closest("details");
      if (panel && !panel.open) panel.open = true;
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      target.classList.add("flash");
      setTimeout(() => target.classList.remove("flash"), 1200);
    });
}
