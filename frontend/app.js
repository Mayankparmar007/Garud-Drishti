/* Control room view.
 *
 * One websocket carries the state. Imagery is fetched separately and keyed to
 * the sequence number in that state, so the overlay can never be drawn against
 * a frame it does not belong to.
 *
 * All drawing happens in frame pixel coordinates; the canvas is scaled by CSS
 * to sit exactly on the displayed image, so nothing here has to know how big
 * the window is.
 */

const $ = (id) => document.getElementById(id);

const state = {
  latest: null,
  layers: { heat: true, zones: true, flow: true, corridors: true },
  heatImg: new Image(),
  heatSeq: -1,
  frameSeq: -1,
  focus: null,
  lastRx: 0,
};

/* ---------------------------------------------------------------- sockets */

let ws = null;
let retry = 1000;

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws/state`);

  ws.onopen = () => { retry = 1000; setConn("live", "live"); };

  ws.onmessage = (ev) => {
    state.latest = JSON.parse(ev.data);
    state.lastRx = performance.now();
    render(state.latest);
  };

  ws.onclose = () => {
    setConn("down", "reconnecting");
    // Back off, but never so far that a demo sits dead for a minute.
    setTimeout(connect, retry);
    retry = Math.min(retry * 1.6, 8000);
  };

  ws.onerror = () => ws.close();
}

function setConn(cls, text) {
  $("conn").className = `dot ${cls}`;
  $("conn-text").textContent = text;
}

/* ------------------------------------------------------------------ media */

function refreshMedia(s) {
  if (s.seq !== state.frameSeq) {
    state.frameSeq = s.seq;
    $("frame").src = `/frames/latest.jpg?s=${s.seq}`;
  }
  if (state.layers.heat && s.heat_url && s.seq !== state.heatSeq) {
    state.heatSeq = s.seq;
    const img = new Image();
    img.onload = () => { state.heatImg = img; draw(); };
    img.src = `${s.heat_url}?s=${s.seq}`;
  }
}

$("frame").onload = () => {
  $("waiting").hidden = true;
  draw();
};

/* ---------------------------------------------------------------- drawing */

function syncCanvas(s) {
  const cv = $("overlay");
  const img = $("frame");
  if (s.scene.frame_w && cv.width !== s.scene.frame_w) {
    cv.width = s.scene.frame_w;
    cv.height = s.scene.frame_h;
  }
  // Match the canvas to however the browser chose to letterbox the image.
  if (img.clientWidth) {
    cv.style.width = `${img.clientWidth}px`;
    cv.style.height = `${img.clientHeight}px`;
  }
}

const RISK_STROKE = { green: "#3fb950", amber: "#d29922", red: "#f0523f" };
const CORRIDOR_STROKE = { clear: "#3fb950", restricted: "#d29922", blocked: "#f0523f" };

function draw() {
  const s = state.latest;
  if (!s) return;
  syncCanvas(s);

  const cv = $("overlay");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);

  if (state.layers.heat && state.heatImg.width) {
    ctx.drawImage(state.heatImg, 0, 0, cv.width, cv.height);
  }
  if (state.layers.corridors) s.corridors.forEach((c) => drawCorridor(ctx, c));
  if (state.layers.zones) s.zones.forEach((z) => drawZone(ctx, z));
  if (state.layers.flow) s.zones.forEach((z) => drawArrow(ctx, z));
}

function path(ctx, poly) {
  ctx.beginPath();
  poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
}

function centroid(poly) {
  let x = 0, y = 0;
  poly.forEach((p) => { x += p[0]; y += p[1]; });
  return [x / poly.length, y / poly.length];
}

function drawZone(ctx, z) {
  const focused = state.focus === z.id;
  path(ctx, z.polygon);
  ctx.lineWidth = z.risk === "red" ? 3.5 : focused ? 3 : 2;
  ctx.strokeStyle = RISK_STROKE[z.risk];
  ctx.setLineDash([]);
  if (z.risk !== "green") {
    ctx.shadowColor = RISK_STROKE[z.risk];
    ctx.shadowBlur = z.risk === "red" ? 14 : 7;
  }
  ctx.stroke();
  ctx.shadowBlur = 0;

  if (focused) {
    ctx.fillStyle = "rgba(75,159,225,.12)";
    ctx.fill();
  }

  // Label: name and the number the operator actually acts on.
  const [cx, cy] = centroid(z.polygon);
  const label = `${z.name}  ${z.density.toFixed(2)} p/m2`;
  ctx.font = "600 15px ui-monospace, Consolas, monospace";
  const w = ctx.measureText(label).width;
  ctx.fillStyle = "rgba(9,13,18,.82)";
  ctx.fillRect(cx - w / 2 - 7, cy - 30, w + 14, 23);
  ctx.fillStyle = z.risk === "green" ? "#dce3ec" : RISK_STROKE[z.risk];
  ctx.textAlign = "center";
  ctx.fillText(label, cx, cy - 13);
  ctx.textAlign = "left";
}

function drawCorridor(ctx, c) {
  path(ctx, c.polygon);
  ctx.setLineDash([9, 6]);
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = CORRIDOR_STROKE[c.status];
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle =
    c.status === "blocked" ? "rgba(240,82,63,.14)"
    : c.status === "restricted" ? "rgba(210,153,34,.12)"
    : "rgba(63,185,80,.09)";
  ctx.fill();
}

function drawArrow(ctx, z) {
  if (z.flow_dir_deg === null || z.mean_speed_ms < 0.06) return;
  const [cx, cy] = centroid(z.polygon);

  // Bearing is clockwise from up-screen; image y grows downward.
  const th = (z.flow_dir_deg * Math.PI) / 180;
  const ux = Math.sin(th), uy = -Math.cos(th);
  const len = Math.min(20 + z.mean_speed_ms * 55, 95);
  const [x0, y0] = [cx - ux * len / 2, cy - uy * len / 2];
  const [x1, y1] = [cx + ux * len / 2, cy + uy * len / 2];

  ctx.strokeStyle = "rgba(255,255,255,.9)";
  ctx.fillStyle = "rgba(255,255,255,.9)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();

  const h = 11;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - ux * h - uy * h * 0.55, y1 - uy * h + ux * h * 0.55);
  ctx.lineTo(x1 - ux * h + uy * h * 0.55, y1 - uy * h - ux * h * 0.55);
  ctx.closePath();
  ctx.fill();
}

/* ---------------------------------------------------------------- panels */

const fmtTime = (iso) =>
  new Date(iso).toLocaleTimeString("en-GB", { hour12: false });

function trendArrow(z) {
  if (z.trend === "rising") return `<span class="arrow-up">&#9650; ${z.trend_rate_per_min.toFixed(2)}</span>`;
  if (z.trend === "falling") return `<span class="arrow-down">&#9660; ${z.trend_rate_per_min.toFixed(2)}</span>`;
  return `<span>&#9644; steady</span>`;
}

function ttcText(z) {
  if (z.time_to_critical_s === null) return "";
  const t = z.time_to_critical_s;
  const s = t >= 60 ? `${Math.round(t / 60)} min` : `${Math.round(t)} s`;
  return `<span class="k">to critical</span> ${s}`;
}

const BAND_TEXT = {
  free: "Free flow", busy: "Busy", crowded: "Crowded",
  very_crowded: "Very crowded", critical: "Crush risk",
};

function renderZones(s) {
  $("zone-count").textContent = s.zones.length;
  if (!s.zones.length) {
    $("zones").innerHTML = `<div class="empty">No zones configured.<br>
      <a href="setup.html" style="color:#4b9fe1">Draw them in scene setup</a>.</div>`;
    return;
  }

  const order = { red: 0, amber: 1, green: 2 };
  const sorted = [...s.zones].sort(
    (a, b) => order[a.risk] - order[b.risk] || b.density - a.density
  );

  $("zones").innerHTML = sorted.map((z) => {
    const why = z.headline
      ? (z.signals.find((g) => g.name === z.headline) || {}).detail || ""
      : "";
    return `
    <div class="zone ${z.risk} ${state.focus === z.id ? "focus" : ""}" data-zone="${z.id}">
      <div class="zone-top">
        <span class="zone-name">${z.name}</span>
        <span class="zone-density">${z.density.toFixed(2)}<span class="u">p/m&sup2;</span></span>
      </div>
      <div class="zone-band">${BAND_TEXT[z.band]} &middot; ${z.count.toFixed(0)} people over ${z.area_m2.toFixed(0)} m&sup2;</div>
      <div class="zone-meta">
        <span><span class="k">speed</span> ${z.mean_speed_ms.toFixed(2)} m/s</span>
        <span><span class="k">trend</span> ${trendArrow(z)}</span>
        ${ttcText(z) ? `<span>${ttcText(z)}</span>` : ""}
      </div>
      ${why ? `<div class="zone-why">${why}</div>` : ""}
    </div>`;
  }).join("");

  $("zones").querySelectorAll(".zone").forEach((el) => {
    el.onmouseenter = () => { state.focus = el.dataset.zone; draw(); };
    el.onmouseleave = () => { state.focus = null; draw(); };
  });
}

function renderCorridors(s) {
  if (!s.corridors.length) {
    $("corridors").innerHTML = `<div class="empty">No corridors drawn.</div>`;
    return;
  }
  $("corridors").innerHTML = s.corridors.map((c) => `
    <div class="corridor">
      <span class="pill ${c.status}">${c.status}</span>
      <span class="nm">${c.name}</span>
      <span class="val">${c.density.toFixed(1)} p/m&sup2; &middot; ${c.mean_speed_ms.toFixed(2)} m/s</span>
    </div>`).join("");
}

function renderAlerts(s) {
  $("alert-count").textContent = s.alerts.length;
  if (!s.alerts.length) {
    $("alerts").innerHTML = `<div class="empty">No active alerts.</div>`;
    return;
  }
  $("alerts").innerHTML = s.alerts.map((a) => `
    <div class="alert ${a.severity}">
      ${a.thumb ? `<img class="thumb" src="${a.thumb}" alt="evidence">` : ""}
      <div class="body">
        <div class="hdr">
          <span class="sev">${a.severity.toUpperCase()}</span>
          <span class="zn">${a.zone_name}</span>
          <span class="tm">${fmtTime(a.ts)}</span>
        </div>
        <div class="msg">${a.message}</div>
        <div class="act">${a.action}</div>
      </div>
    </div>`).join("");
}

function render(s) {
  $("scene-name").textContent = s.scene.name;
  $("uncal").hidden = s.scene.calibrated;
  $("total-count").textContent = s.totals.count.toFixed(0);
  $("total-density").textContent = s.totals.density.toFixed(2);
  $("latency").textContent = s.latency_ms;
  $("fps").textContent = s.fps.toFixed(1);
  $("clock").textContent = fmtTime(s.ts);
  $("model-note").textContent = s.model ? ` Model: ${s.model}.` : "";

  refreshMedia(s);
  renderZones(s);
  renderCorridors(s);
  renderAlerts(s);
  draw();
}

/* --------------------------------------------------------------- controls */

document.querySelectorAll(".toggle").forEach((el) => {
  el.onclick = () => {
    const key = el.dataset.layer;
    state.layers[key] = !state.layers[key];
    el.classList.toggle("on", state.layers[key]);
    if (key === "heat" && state.layers.heat) state.heatSeq = -1; // force a refetch
    draw();
  };
});

async function loadLegend() {
  try {
    const meta = await (await fetch("/api/meta")).json();
    $("legend").insertAdjacentHTML("beforeend", meta.legend.map((b) =>
      `<span class="sw" style="background:${b.css}">${b.from}+ ${b.label}</span>`
    ).join(""));
  } catch { /* legend is decoration; never block the view on it */ }
}

// A socket that stops delivering is worse than one that closes: the numbers
// freeze but still look live. Flag it if nothing arrives for three seconds.
setInterval(() => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const age = (performance.now() - state.lastRx) / 1000;
  if (age > 3) setConn("stale", `stale ${age.toFixed(0)}s`);
  else setConn("live", "live");
}, 1000);

window.addEventListener("resize", draw);
loadLegend();
connect();
