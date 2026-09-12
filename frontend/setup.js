/* Scene setup.
 *
 * Everything is drawn on one still frame. Geometry is stored in the frame's
 * own pixel coordinates together with frame_size, so the backend can rescale
 * it if the stream resolution ever changes rather than silently putting the
 * zones in the wrong place.
 */

const $ = (id) => document.getElementById(id);
const cv = $("setup-canvas");
const ctx = cv.getContext("2d");

let scene = null;
let mode = null;            // null | "zone" | "corridor" | "calibration"
let pending = [];           // points collected so far, in frame pixels
let frameSize = [0, 0];

const HINTS = {
  zone: "Click the corners of the zone. Enter or double-click to close it.",
  corridor: "Click the corners of the corridor. Enter or double-click to close it.",
  calibration: "Click 4 ground points, clockwise from the top-left.",
};

/* ------------------------------------------------------------ frame setup */

function loadFrame() {
  const img = $("frame");
  img.onload = () => {
    frameSize = [img.naturalWidth, img.naturalHeight];
    cv.width = img.naturalWidth;
    cv.height = img.naturalHeight;
    $("waiting").hidden = true;
    fit();
    draw();
  };
  img.onerror = () => { $("waiting").textContent = "no frame yet - is the worker running?"; };
  img.src = `/frames/latest.jpg?t=${Date.now()}`;
}

function fit() {
  const img = $("frame");
  if (!img.clientWidth) return;
  cv.style.width = `${img.clientWidth}px`;
  cv.style.height = `${img.clientHeight}px`;
}

// Canvas is displayed scaled; convert a click back into frame pixels.
function toFrame(ev) {
  const r = cv.getBoundingClientRect();
  return [
    ((ev.clientX - r.left) / r.width) * cv.width,
    ((ev.clientY - r.top) / r.height) * cv.height,
  ];
}

/* ---------------------------------------------------------------- drawing */

function poly(points, close = true) {
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  if (close) ctx.closePath();
}

function label(text, x, y, colour) {
  ctx.font = "600 14px ui-monospace, Consolas, monospace";
  const w = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(9,13,18,.85)";
  ctx.fillRect(x - w / 2 - 6, y - 20, w + 12, 21);
  ctx.fillStyle = colour;
  ctx.textAlign = "center";
  ctx.fillText(text, x, y - 5);
  ctx.textAlign = "left";
}

function centroid(p) {
  return [p.reduce((a, q) => a + q[0], 0) / p.length,
          p.reduce((a, q) => a + q[1], 0) / p.length];
}

function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!scene) return;

  scene.zones.forEach((z) => {
    poly(z.polygon);
    ctx.strokeStyle = "#4b9fe1"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "rgba(75,159,225,.10)"; ctx.fill();
    const [cx, cy] = centroid(z.polygon);
    label(z.name, cx, cy, "#9fd0f5");
  });

  scene.corridors.forEach((c) => {
    poly(c.polygon);
    ctx.setLineDash([9, 6]);
    ctx.strokeStyle = "#3fb950"; ctx.lineWidth = 2; ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(63,185,80,.10)"; ctx.fill();
    const [cx, cy] = centroid(c.polygon);
    label(c.name, cx, cy, "#8fe0a0");
  });

  const cal = scene.calibration.image_points || [];
  if (cal.length === 4) drawCal(cal, "#d29922");

  if (pending.length) {
    const colour = mode === "calibration" ? "#f0b429" : "#e8eef6";
    poly(pending, false);
    ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.stroke(); ctx.setLineDash([]);
    pending.forEach(([x, y], i) => {
      ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fillStyle = colour; ctx.fill();
      ctx.fillStyle = "#0d1117"; ctx.font = "700 10px monospace";
      ctx.textAlign = "center"; ctx.fillText(i + 1, x, y + 3.5); ctx.textAlign = "left";
    });
  }
}

function drawCal(pts, colour) {
  poly(pts);
  ctx.strokeStyle = colour; ctx.lineWidth = 2.5; ctx.setLineDash([7, 5]);
  ctx.stroke(); ctx.setLineDash([]);
  const w = scene.calibration.world_points;
  pts.forEach(([x, y], i) => {
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = colour; ctx.fill();
    if (w && w[i]) label(`${w[i][0]}, ${w[i][1]} m`, x, y - 6, colour);
  });
}

/* ------------------------------------------------------------ interaction */

cv.addEventListener("click", (ev) => {
  if (!mode) return;
  pending.push(toFrame(ev));

  if (mode === "calibration" && pending.length === 4) {
    commitCalibration();
  }
  draw();
});

cv.addEventListener("dblclick", () => { if (mode !== "calibration") closeShape(); });

window.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") closeShape();
  if (ev.key === "Escape") cancel();
  if (ev.key === "z" && (ev.ctrlKey || ev.metaKey)) undo();
});

function setMode(m) {
  mode = m;
  pending = [];
  $("mode-hint").textContent = m ? HINTS[m] : "Pick a step on the right";
  $("click-hint").textContent = m ? HINTS[m] : "";
  draw();
}

function cancel() { setMode(null); }
function undo() { pending.pop(); draw(); }

function closeShape() {
  if (!mode || mode === "calibration") return;
  if (pending.length < 3) {
    flash("A shape needs at least 3 points.", false);
    return;
  }
  const isZone = mode === "zone";
  const name = prompt(isZone ? "Zone name" : "Corridor name",
                      isZone ? `Zone ${scene.zones.length + 1}`
                             : `Corridor ${scene.corridors.length + 1}`);
  if (name) {
    const id = slug(name, isZone ? scene.zones : scene.corridors);
    if (isZone) scene.zones.push({ id, name, polygon: pending, neighbors: [], note: "" });
    else scene.corridors.push({ id, name, polygon: pending, width_m: null, note: "" });
    renderLists();
  }
  setMode(null);
}

function slug(name, existing) {
  let base = name.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || "item";
  let id = base, n = 2;
  while (existing.some((e) => e.id === id)) id = `${base}_${n++}`;
  return id;
}

function commitCalibration() {
  const w = parseFloat($("cal-w").value);
  const h = parseFloat($("cal-h").value);
  if (!(w > 0 && h > 0)) {
    flash("Enter the real width and depth first.", false);
    setMode(null);
    return;
  }
  // Clockwise from top-left, matching the instruction on screen.
  scene.calibration = {
    image_points: pending,
    world_points: [[0, 0], [w, 0], [w, h], [0, h]],
    note: $("cal-note").value || "",
  };
  setMode(null);
  renderCalStatus();
  flash(`Calibrated against ${w} x ${h} m = ${(w * h).toFixed(0)} m2.`, true);
}

/* ----------------------------------------------------------------- lists */

function renderLists() {
  $("zone-list").innerHTML = scene.zones.map((z, i) => `
    <div class="item">
      <span class="nm">${z.name}<div class="sub">${z.polygon.length} points${
        z.neighbors.length ? ` &middot; next to ${z.neighbors.length}` : ""}</div></span>
      <button data-nb="${i}">Neighbours</button>
      <button class="danger" data-dz="${i}">Delete</button>
    </div>`).join("") || `<p>No zones yet.</p>`;

  $("corridor-list").innerHTML = scene.corridors.map((c, i) => `
    <div class="item">
      <span class="nm">${c.name}<div class="sub">${c.polygon.length} points</div></span>
      <button class="danger" data-dc="${i}">Delete</button>
    </div>`).join("") || `<p>No corridors yet.</p>`;

  $("zone-list").querySelectorAll("[data-dz]").forEach((b) => {
    b.onclick = () => { scene.zones.splice(+b.dataset.dz, 1); renderLists(); draw(); };
  });
  $("zone-list").querySelectorAll("[data-nb]").forEach((b) => {
    b.onclick = () => editNeighbours(+b.dataset.nb);
  });
  $("corridor-list").querySelectorAll("[data-dc]").forEach((b) => {
    b.onclick = () => { scene.corridors.splice(+b.dataset.dc, 1); renderLists(); draw(); };
  });
  draw();
}

function editNeighbours(i) {
  const z = scene.zones[i];
  const others = scene.zones.filter((o) => o.id !== z.id);
  if (!others.length) { flash("Draw another zone first.", false); return; }
  // Neighbours are what convergence is measured across: two adjacent zones
  // whose crowds walk into each other is the thing worth catching.
  const answer = prompt(
    `Which zones does "${z.name}" border?\n\n` +
    others.map((o, n) => `${n + 1}. ${o.name}`).join("\n") +
    `\n\nEnter numbers separated by commas:`,
    z.neighbors.map((id) => others.findIndex((o) => o.id === id) + 1)
      .filter((n) => n > 0).join(",")
  );
  if (answer === null) return;
  z.neighbors = answer.split(",")
    .map((s) => others[parseInt(s.trim(), 10) - 1])
    .filter(Boolean).map((o) => o.id);
  renderLists();
}

function renderCalStatus() {
  const c = scene.calibration;
  const el = $("cal-status");
  const hd = $("step-cal").querySelector(".hd");
  if (c.image_points && c.image_points.length === 4) {
    const [w] = c.world_points[1], [, h] = c.world_points[2];
    el.textContent = `calibrated: ${w} x ${h} m${c.note ? ` (${c.note})` : ""}`;
    el.style.color = "#3fb950";
    hd.classList.add("done");
  } else {
    el.textContent = "not calibrated - densities will be indicative only";
    el.style.color = "#d29922";
    hd.classList.remove("done");
  }
}

/* ------------------------------------------------------------ load / save */

async function load() {
  scene = await (await fetch("/api/scene")).json();
  $("scene-name").value = scene.name;
  $("src-kind").value = scene.source.kind;
  $("src-uri").value = scene.source.uri;
  $("count-scale").value = scene.count_scale ?? 1.0;
  if (scene.calibration.world_points?.length === 4) {
    $("cal-w").value = scene.calibration.world_points[1][0];
    $("cal-h").value = scene.calibration.world_points[2][1];
    $("cal-note").value = scene.calibration.note || "";
  }
  renderCalStatus();
  renderLists();
  loadFrame();
}

async function save() {
  scene.name = $("scene-name").value || "Unnamed scene";
  scene.source = { kind: $("src-kind").value, uri: $("src-uri").value };
  scene.count_scale = parseFloat($("count-scale").value) || 1.0;
  if (frameSize[0]) scene.frame_size = frameSize;

  const res = await fetch("/api/scene", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(scene),
  });
  if (res.ok) {
    scene = await res.json();
    flash(`Saved. ${scene.zones.length} zones, ${scene.corridors.length} corridors applied.`, true);
    renderCalStatus();
  } else {
    const err = await res.json().catch(() => ({}));
    flash(`Save failed: ${err.detail?.[0]?.msg || res.statusText}`, false);
  }
}

function flash(msg, ok) {
  const el = $("save-status");
  el.textContent = msg;
  el.className = `status-line ${ok ? "ok" : "err"}`;
}

async function loadPresets() {
  try {
    const list = await (await fetch("/api/presets")).json();
    $("presets").innerHTML = `<option value="">-</option>` +
      list.map((p) => `<option value="${p.id}">${p.name} (${p.zones} zones)</option>`).join("");
  } catch { /* presets are optional */ }
}

/* --------------------------------------------------------------- wiring */

$("btn-zone").onclick = () => setMode("zone");
$("btn-corridor").onclick = () => setMode("corridor");
$("btn-cal").onclick = () => setMode("calibration");
$("btn-cal-clear").onclick = () => {
  scene.calibration = { image_points: [], world_points: [], note: "" };
  renderCalStatus(); draw();
};
$("btn-undo").onclick = undo;
$("btn-cancel").onclick = cancel;
$("btn-refresh").onclick = loadFrame;
$("btn-save").onclick = save;
$("btn-reload").onclick = load;
$("btn-preset").onclick = async () => {
  const id = $("presets").value;
  if (!id) return;
  const res = await fetch(`/api/presets/${id}/apply`, { method: "POST" });
  if (res.ok) { await load(); flash(`Loaded preset "${id}".`, true); }
  else flash(`Could not load preset "${id}".`, false);
};

// Keep the reported count visible so count_scale can be set against it.
setInterval(async () => {
  try {
    const s = await (await fetch("/api/state")).json();
    if (s.totals) $("reported-now").value = `${s.totals.count.toFixed(0)} people`;
  } catch { /* worker may not be publishing yet */ }
}, 2000);

window.addEventListener("resize", () => { fit(); draw(); });
loadPresets();
load();
