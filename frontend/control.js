/* UKSI P-007 — control room.
 *
 * One canvas, three layers, one state feed.
 *
 * The canvas is sized internally to the frame the worker is actually
 * processing, so zone polygons -- which arrive already rescaled to that size
 * by compile_scene -- can be drawn at 1:1 with no client-side transform to get
 * wrong. CSS scales the whole thing to fit the pane.
 *
 * DOM nodes for zones and alerts are cached by id and updated in place rather
 * than re-rendered. That is partly to stop the panel flickering once a second,
 * and partly because re-creating an alert thumbnail would re-request it, and
 * every evidence request is an audit log entry. A dashboard left open should
 * not write a thousand rows saying it is still open.
 */

(() => {
  'use strict';

  const {
    bandLabel, num, duration, clock, relTime, el,
    TREND_GLYPH, RISK_COLOUR, CORRIDOR_COLOUR,
    getJSON, postJSON, connectState, centroid, polygonPath,
  } = UKSI;

  const $ = (id) => document.getElementById(id);

  const canvas = $('view');
  const ctx = canvas.getContext('2d');

  const show = {
    heat: true,
    vectors: true,
    labels: true,
  };

  /* Latest state, and the two image layers. Images are held as decoded
   * Image objects and swapped on load, so a paint never waits on the network
   * and a half-arrived JPEG never reaches the screen. */
  let state = null;
  let thresholds = null;
  const layer = {
    frame: { img: new Image(), ready: false, loading: false, tick: 0 },
    heat: { img: new Image(), ready: false, loading: false, seq: -1 },
  };

  const seenAlerts = new Set();
  let firstStateHandled = false;

  // -- image layers -------------------------------------------------------

  function refresh(name, url) {
    const L = layer[name];
    if (L.loading) return;  // one request in flight at a time; never queue
    L.loading = true;
    const img = new Image();
    img.onload = () => {
      L.img = img;
      L.ready = true;
      L.loading = false;
    };
    img.onerror = () => { L.loading = false; };
    img.src = url;
  }

  // The frame is sampled at 3fps by the worker, so asking more often than
  // that only spends bandwidth re-fetching a picture we already have.
  setInterval(() => {
    layer.frame.tick += 1;
    refresh('frame', `/frames/latest.jpg?s=${layer.frame.tick}`);
  }, 330);

  // -- canvas -------------------------------------------------------------

  function sizeCanvas() {
    if (!state) return;
    const w = state.scene.frame_w || 960;
    const h = state.scene.frame_h || 544;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function paint() {
    requestAnimationFrame(paint);
    const w = canvas.width, h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    if (layer.frame.ready) {
      ctx.drawImage(layer.frame.img, 0, 0, w, h);
    } else {
      ctx.fillStyle = '#0b0e13';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#4a5361';
      ctx.font = '16px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('waiting for the feed…', w / 2, h / 2);
      ctx.textAlign = 'left';
    }

    // The heat PNG carries its own per-pixel alpha straight from the density
    // band LUT, so it composites as-is. It arrives at half resolution because
    // the field is smoothed over about a metre of ground -- scaling it up
    // loses nothing that was ever measured.
    if (show.heat && layer.heat.ready) {
      ctx.drawImage(layer.heat.img, 0, 0, w, h);
    }

    if (show.vectors && state) {
      drawCorridors(state.corridors || []);
      drawZones(state.zones || []);
    }
  }

  function drawCorridors(corridors) {
    for (const c of corridors) {
      if (!c.polygon || c.polygon.length < 3) continue;
      const colour = CORRIDOR_COLOUR[c.status] || '#8b96a5';
      polygonPath(ctx, c.polygon);
      ctx.setLineDash([9, 7]);
      ctx.lineWidth = 2;
      ctx.strokeStyle = colour;
      ctx.stroke();
      ctx.setLineDash([]);

      if (c.status !== 'clear') {
        ctx.fillStyle = colour + '22';
        ctx.fill();
      }

      if (show.labels) {
        const mid = centroid(c.polygon);
        if (mid) pill(`${c.name} · ${c.status}`, mid[0], mid[1], colour, 11);
      }
    }
  }

  function drawZones(zones) {
    for (const z of zones) {
      if (!z.polygon || z.polygon.length < 3) continue;
      const colour = RISK_COLOUR[z.risk] || '#8b96a5';

      polygonPath(ctx, z.polygon);
      ctx.lineWidth = z.risk === 'green' ? 1.6 : 2.6;
      ctx.strokeStyle = colour;
      ctx.stroke();
      if (z.risk !== 'green') {
        ctx.fillStyle = colour + (z.risk === 'red' ? '2e' : '1c');
        ctx.fill();
      }

      const mid = centroid(z.polygon);
      if (!mid) continue;
      drawFlow(z, mid[0], mid[1], colour);

      if (show.labels) {
        const trend = TREND_GLYPH[z.trend] || '';
        pill(`${z.name} · ${num(z.density)} p/m² ${trend}`, mid[0], mid[1] - 26, colour, 13);
      }
    }
  }

  /* Flow arrow.
   *
   * The heading is real -- degrees clockwise from up-screen, straight from the
   * optical flow stage. The length is deliberately only indicative: turning
   * m/s into pixels needs the ground scale at that point in the image, which
   * lives in the homography on the server. The number in the zone card is the
   * measurement; the arrow says which way and roughly how fast.
   *
   * A stalled zone gets a ring, not a short arrow. A one-pixel arrow would
   * still point somewhere, and at a standstill there is no direction to claim.
   */
  function drawFlow(z, cx, cy, colour) {
    if (z.flow_dir_deg === null || z.flow_dir_deg === undefined) return;

    const free = (thresholds && thresholds.free_speed_ms) || 1.0;
    const slow = (thresholds && thresholds.slow_speed_ms) || 0.35;

    if (z.mean_speed_ms < slow * 0.5) {
      ctx.beginPath();
      ctx.arc(cx, cy, 9, 0, Math.PI * 2);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = colour;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      return;
    }

    const theta = z.flow_dir_deg * Math.PI / 180;
    const dx = Math.sin(theta), dy = -Math.cos(theta);
    const len = 22 + 58 * Math.min(z.mean_speed_ms / free, 1.6);
    const tipX = cx + dx * len, tipY = cy + dy * len;

    ctx.lineWidth = 3;
    ctx.strokeStyle = colour;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx - dx * 6, cy - dy * 6);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    const head = 9;
    const a = theta + Math.PI * 0.82, b = theta - Math.PI * 0.82;
    ctx.beginPath();
    ctx.moveTo(tipX, tipY);
    ctx.lineTo(tipX + Math.sin(a) * head, tipY - Math.cos(a) * head);
    ctx.lineTo(tipX + Math.sin(b) * head, tipY - Math.cos(b) * head);
    ctx.closePath();
    ctx.fillStyle = colour;
    ctx.fill();
  }

  function pill(text, cx, cy, colour, size) {
    ctx.font = `600 ${size}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    const w = ctx.measureText(text).width + 14;
    const h = size + 10;
    const x = cx - w / 2, y = cy - h / 2;

    ctx.fillStyle = 'rgba(8, 11, 16, .80)';
    ctx.beginPath();
    if (ctx.roundRect) { ctx.roundRect(x, y, w, h, 4); } else { ctx.rect(x, y, w, h); }
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = colour;
    ctx.stroke();

    ctx.fillStyle = '#e6edf3';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, cx, cy + 0.5);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  // -- top bar ------------------------------------------------------------

  function renderTop(s) {
    $('scene-name').textContent = s.scene.name;
    document.title = `${s.scene.name} — UKSI P-007`;

    const src = s.source || {};
    $('src-dot').className = 'dot ' + (src.connected ? 'live' : 'down');
    $('src-text').textContent = src.detail || src.kind || '—';
    $('src-text').title = `${src.kind}${src.uri ? ' · ' + src.uri : ''}`;

    $('m-latency').textContent = `${s.latency_ms} ms`;
    $('m-proc').textContent = `${s.proc_ms} ms · ${num(s.fps, 1)} fps`;
    $('m-count').textContent = num(s.totals.count, 0);
    $('m-zones').textContent = `${s.totals.zones_amber}A / ${s.totals.zones_red}R`;
    $('m-model').textContent = s.model || '—';
    $('m-model').title = s.model || '';
    $('m-clock').textContent = clock(new Date(s.ts));

    // Counting error, but only where there is a known answer to compare with.
    // On real footage there is none, and the field stays hidden rather than
    // showing a zero that would read as "no error".
    const errBox = $('m-err');
    if (s.debug && s.debug.count_error_pct !== null && s.debug.count_error_pct !== undefined) {
      errBox.style.display = '';
      const pct = s.debug.count_error_pct;
      $('m-err-v').textContent = `${pct > 0 ? '+' : ''}${num(pct, 1)}%`;
      $('m-err-v').title = `estimated ${s.debug.estimated_count} vs ground truth ${s.debug.truth_count}`;
    } else {
      errBox.style.display = 'none';
    }
  }

  function renderBanner(s) {
    const box = $('banner');
    let msg = '';
    if (!s.scene.calibrated) {
      msg = 'Scene is not calibrated — densities below are indicative only, not persons per square metre. '
          + 'Set four ground points in scene setup.';
    } else if (!s.source.connected) {
      msg = `Feed is not delivering frames — ${s.source.detail || 'no detail'}. Measurements are the last known values.`;
    } else if (!s.zones.length) {
      msg = 'No zones are defined, so nothing is being measured. Draw zones in scene setup.';
    }
    box.textContent = msg;
    box.classList.toggle('hidden', !msg);
  }

  // -- zones --------------------------------------------------------------

  const zoneNodes = new Map();
  let zoneOrder = '';

  function zoneCard(z) {
    const card = el('div', 'zone');
    card.dataset.id = z.id;

    const head = el('div', 'zone-head');
    head.appendChild(el('div', 'zone-name'));
    head.appendChild(el('span', 'chip'));
    card.appendChild(head);

    const figs = el('div', 'zone-figures');
    const d = el('div', 'density');
    d.appendChild(el('span', 'dv'));
    const unit = el('small', null, 'p/m²');
    d.appendChild(unit);
    figs.appendChild(d);
    figs.appendChild(el('span', 'band-text'));
    card.appendChild(figs);

    card.appendChild(el('div', 'zone-meta'));
    card.appendChild(el('div', 'headline'));
    card.appendChild(el('div', 'signals'));

    // Signals are the "why", and they are collapsed by default because an
    // operator reads the colour first and the reasoning only when it matters.
    card.addEventListener('click', () => card.classList.toggle('open'));
    return card;
  }

  function updateZone(card, z) {
    card.className = `zone ${z.risk}${card.classList.contains('open') ? ' open' : ''}`;
    card.querySelector('.zone-name').textContent = z.name;

    const chip = card.querySelector('.chip');
    chip.className = `chip ${z.risk}`;
    chip.textContent = z.risk;

    card.querySelector('.dv').textContent = num(z.density);
    card.querySelector('.band-text').textContent = bandLabel(z.band);

    const meta = card.querySelector('.zone-meta');
    meta.textContent = '';
    meta.appendChild(el('span', null, `${num(z.count, 0)} people`));
    meta.appendChild(el('span', null, `${num(z.area_m2, 0)} m²`));
    meta.appendChild(el('span', null,
      `${TREND_GLYPH[z.trend] || ''} ${z.trend_rate_per_min >= 0 ? '+' : ''}${num(z.trend_rate_per_min)}/min`));
    meta.appendChild(el('span', null, `${num(z.mean_speed_ms)} m/s`));
    if (z.time_to_critical_s !== null && z.time_to_critical_s !== undefined) {
      const t = el('span', 'ttc' + (z.risk === 'red' ? ' red' : ''),
        `critical in ${duration(z.time_to_critical_s)}`);
      meta.appendChild(t);
    }

    const hl = card.querySelector('.headline');
    hl.textContent = z.headline || '';
    hl.style.display = z.headline ? '' : 'none';

    const sigs = card.querySelector('.signals');
    sigs.textContent = '';
    for (const s of z.signals || []) {
      const row = el('div', 'sig');
      row.appendChild(el('span', `pip ${s.level}`));
      const body = el('div');
      body.appendChild(el('div', 'lbl', s.name));
      body.appendChild(el('div', 'dtl', s.detail));
      row.appendChild(body);
      sigs.appendChild(row);
    }
  }

  const RISK_RANK = { red: 2, amber: 1, green: 0 };

  function renderZones(zones) {
    const host = $('zones');
    $('zones-count').textContent = zones.length ? `${zones.length}` : '0';

    if (!zones.length) {
      if (!host.querySelector('.empty')) {
        host.textContent = '';
        host.appendChild(el('div', 'empty', 'No zones defined.'));
      }
      zoneNodes.clear();
      return;
    }
    const emptyNode = host.querySelector('.empty');
    if (emptyNode) emptyNode.remove();

    const live = new Set();
    for (const z of zones) {
      live.add(z.id);
      let card = zoneNodes.get(z.id);
      if (!card) {
        card = zoneCard(z);
        zoneNodes.set(z.id, card);
        host.appendChild(card);
      }
      updateZone(card, z);
    }
    for (const [id, card] of zoneNodes) {
      if (!live.has(id)) { card.remove(); zoneNodes.delete(id); }
    }

    // Worst first, so the zone that needs attention is never below the fold.
    // Nodes are moved, not rebuilt, so an expanded card stays expanded.
    const sorted = [...zones].sort((a, b) =>
      (RISK_RANK[b.risk] - RISK_RANK[a.risk]) || (b.density - a.density));
    const key = sorted.map((z) => z.id).join('|');
    if (key !== zoneOrder) {
      zoneOrder = key;
      for (const z of sorted) host.appendChild(zoneNodes.get(z.id));
    }
  }

  // -- corridors ----------------------------------------------------------

  function renderCorridors(corridors) {
    const host = $('corridors');
    const blocked = corridors.filter((c) => c.status !== 'clear').length;
    $('corr-count').textContent = corridors.length
      ? (blocked ? `${blocked} of ${corridors.length} not clear` : 'all clear')
      : '0';

    host.textContent = '';
    if (!corridors.length) {
      host.appendChild(el('div', 'empty', 'No evacuation corridors marked.'));
      return;
    }
    for (const c of corridors) {
      const row = el('div', 'corridor');
      row.appendChild(el('span', `chip ${c.status}`, c.status));
      row.appendChild(el('span', 'cname', c.name));
      row.appendChild(el('span', 'cdens', `${num(c.density)} p/m² · ${num(c.mean_speed_ms)} m/s`));
      host.appendChild(row);
    }
  }

  // -- alerts -------------------------------------------------------------

  const alertNodes = new Map();
  const alertData = new Map();
  let alertOrder = '';

  function alertCard(a) {
    const card = el('div', 'alert');
    const head = el('div', 'alert-head');
    head.appendChild(el('span', 'chip'));
    head.appendChild(el('span', 'alert-zone'));
    head.appendChild(el('span', 'alert-time'));
    card.appendChild(head);
    card.appendChild(el('div', 'alert-msg'));
    card.appendChild(el('div', 'alert-action'));
    card.appendChild(el('div', 'escalations'));

    // The thumbnail is created once and never re-created. Fetching it is an
    // audited event -- "someone looked at the evidence" -- and that record is
    // only meaningful if it means a person opened it, not that a page
    // repainted.
    if (a.thumb) {
      const img = el('img', 'alert-thumb');
      img.loading = 'lazy';
      img.alt = `Evidence thumbnail for ${a.zone_name}, downscaled`;
      img.src = a.thumb;
      img.addEventListener('click', () => window.open(a.thumb, '_blank', 'noopener'));
      card.appendChild(img);
    }
    return card;
  }

  function updateAlert(card, a) {
    card.className = `alert ${a.severity}${a.state === 'resolved' ? ' resolved' : ''}`;
    const chip = card.querySelector('.chip');
    chip.className = `chip ${a.state === 'resolved' ? 'neutral' : a.severity}`;
    chip.textContent = a.state === 'resolved' ? 'cleared' : a.severity;
    card.querySelector('.alert-zone').textContent = a.zone_name;
    const t = card.querySelector('.alert-time');
    t.textContent = relTime(a.ts);
    t.title = a.ts;
    card.querySelector('.alert-msg').textContent = a.message;

    const act = card.querySelector('.alert-action');
    act.textContent = a.action || '';
    act.style.display = a.action ? '' : 'none';

    const esc = card.querySelector('.escalations');
    const bits = [`trigger: ${a.metric}`];
    if (a.escalations) bits.push(`escalated ${a.escalations}×`);
    esc.textContent = bits.join(' · ');
  }

  function mergeAlerts(list) {
    for (const a of list) alertData.set(a.id, a);
  }

  function renderAlerts() {
    const host = $('alerts');
    const all = [...alertData.values()].sort((x, y) => {
      const ax = x.state === 'active' ? 1 : 0, ay = y.state === 'active' ? 1 : 0;
      return (ay - ax) || (String(y.ts).localeCompare(String(x.ts)));
    }).slice(0, 20);

    $('alerts-count').textContent = alertData.size
      ? `${[...alertData.values()].filter((a) => a.state === 'active').length} active`
      : 'none';

    if (!all.length) {
      if (!host.querySelector('.empty')) {
        host.textContent = '';
        host.appendChild(el('div', 'empty', 'No alerts. Nothing has held a risk condition long enough to fire.'));
      }
      return;
    }
    const emptyNode = host.querySelector('.empty');
    if (emptyNode) emptyNode.remove();

    const live = new Set(all.map((a) => a.id));
    for (const a of all) {
      let card = alertNodes.get(a.id);
      if (!card) {
        card = alertCard(a);
        alertNodes.set(a.id, card);
        host.appendChild(card);
      }
      updateAlert(card, a);
    }
    for (const [id, card] of alertNodes) {
      if (!live.has(id)) { card.remove(); alertNodes.delete(id); }
    }
    const key = all.map((a) => a.id).join('|');
    if (key !== alertOrder) {
      alertOrder = key;
      for (const a of all) host.appendChild(alertNodes.get(a.id));
    }
  }

  /* Announce a genuinely new alert.
   *
   * The whole point of the system is the moment a zone is called before it
   * looks bad on screen, so that moment gets a flash and a toast rather than
   * quietly appearing in a list.
   */
  function announce(list) {
    for (const a of list) {
      if (seenAlerts.has(a.id)) continue;
      seenAlerts.add(a.id);
      if (!firstStateHandled) continue;  // history loaded at startup is not news
      if (a.state !== 'active') continue;

      toast(`${a.severity.toUpperCase()} — ${a.zone_name}: ${a.message}`, a.severity === 'red');
      const stage = $('stage');
      stage.classList.remove('flash-amber', 'flash-red');
      void stage.offsetWidth;  // restart the animation
      stage.classList.add(a.severity === 'red' ? 'flash-red' : 'flash-amber');
      if (!$('audit-box').open) loadAudit();
    }
  }

  // -- audit --------------------------------------------------------------

  async function loadAudit() {
    let rows;
    try { rows = await getJSON('/api/audit?limit=60'); } catch (e) { return; }
    const host = $('audit');
    host.textContent = '';
    if (!rows.length) {
      host.appendChild(el('div', 'empty', 'No entries.'));
      return;
    }
    for (const r of rows) {
      const row = el('div', 'audit-row');
      row.appendChild(el('span', 'at', clock(new Date(/[Zz]$/.test(r.ts) ? r.ts : r.ts + 'Z'))));
      const what = el('span', 'aa');
      what.appendChild(el('b', null, r.action));
      what.appendChild(document.createTextNode(` ${r.actor}${r.detail ? ' · ' + r.detail : ''}`));
      row.appendChild(what);
      host.appendChild(row);
    }
  }

  $('audit-box').addEventListener('toggle', () => {
    if ($('audit-box').open) loadAudit();
  });

  // -- legend -------------------------------------------------------------

  function renderLegend(meta) {
    const host = $('legend-rows');
    host.textContent = '';
    const rows = meta.legend || [];
    rows.forEach((r, i) => {
      const next = rows[i + 1];
      const row = el('div', 'row');
      const sw = el('span', 'swatch');
      sw.style.background = r.css;
      row.appendChild(sw);
      const range = next ? `${r.from}–${next.from}` : `${r.from}+`;
      row.appendChild(el('span', null, `${range}  ${r.label}`));
      host.appendChild(row);
    });
    $('legend-scale').textContent =
      `sampled at ${meta.sample_fps} fps · evidence kept ${meta.retention.evidence_hours} h`;
  }

  // -- toast --------------------------------------------------------------

  let toastTimer = null;
  function toast(msg, bad) {
    const t = $('toast');
    t.textContent = msg;
    t.className = 'toast show' + (bad ? ' bad' : '');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.className = 'toast'; }, bad ? 8000 : 4500);
  }

  // -- demo control -------------------------------------------------------

  function renderDemo(s) {
    const synthetic = s.source && s.source.kind === 'synthetic';
    const hint = $('demo-hint');
    for (const b of ['demo-jump', 'demo-reset', 'demo-t']) $(b).disabled = !synthetic;
    hint.textContent = synthetic
      ? 'Demo control: moves the simulator along its scripted arc. It steps the '
        + 'simulation, not the clock — speeds stay in real m/s and trends in real p/m²/min.'
      : `Demo control is only available on the synthetic source (this scene is "${s.source.kind}").`;
  }

  async function jump(t) {
    try {
      const r = await postJSON('/api/demo/scenario', { t_s: t });
      toast(`Scenario at t=${r.t_s}s. Trend history cleared; give it ~${
        Math.round((thresholds && thresholds.trend_window_s) || 60)}s to refill.`);
    } catch (e) {
      toast(String(e.message || e), true);
    }
  }

  $('demo-jump').addEventListener('click', () => jump(Number($('demo-t').value) || 0));
  $('demo-reset').addEventListener('click', () => jump(0));

  // -- toggles ------------------------------------------------------------

  for (const [id, key] of [['t-heat', 'heat'], ['t-vectors', 'vectors'], ['t-labels', 'labels']]) {
    $(id).addEventListener('change', (e) => { show[key] = e.target.checked; });
  }

  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    const map = { h: 't-heat', z: 't-vectors', l: 't-labels' };
    const id = map[e.key.toLowerCase()];
    if (id) { const box = $(id); box.checked = !box.checked; box.dispatchEvent(new Event('change')); }
  });

  // -- state feed ---------------------------------------------------------

  function onState(s) {
    state = s;
    sizeCanvas();
    renderTop(s);
    renderBanner(s);
    renderZones(s.zones || []);
    renderCorridors(s.corridors || []);
    renderDemo(s);

    mergeAlerts(s.alerts || []);
    renderAlerts();
    announce(s.alerts || []);
    firstStateHandled = true;

    if (s.heat_url && layer.heat.seq !== s.seq) {
      layer.heat.seq = s.seq;
      refresh('heat', `${s.heat_url}?s=${s.seq}`);
    }
  }

  connectState({
    onState,
    onStatus(status, detail) {
      $('ws-dot').className = 'dot ' + status;
      $('ws-text').textContent = status === 'live' ? 'live' : detail || status;
    },
  });

  // -- boot ---------------------------------------------------------------

  (async () => {
    try {
      const meta = await getJSON('/api/meta');
      UKSI.meta = meta;
      thresholds = meta.thresholds;
      renderLegend(meta);
    } catch (e) {
      console.warn('meta unavailable', e);
    }
    try {
      // Alert history first, so a control room opened mid-event does not look
      // like nothing has ever happened.
      mergeAlerts(await getJSON('/api/alerts?limit=20'));
      renderAlerts();
      for (const a of alertData.keys()) seenAlerts.add(a);
    } catch (e) {
      console.warn('alert history unavailable', e);
    }
  })();

  requestAnimationFrame(paint);
})();
