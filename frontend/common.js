/* UKSI P-007 — shared client helpers.
 *
 * Small on purpose. The dashboard is a plain page: no framework, no build
 * step, nothing to install before a demo. Everything here is shared between
 * the control room and the scene setup screen so the two cannot drift.
 */

const UKSI = (() => {
  'use strict';

  // -- formatting ---------------------------------------------------------

  const BAND_FALLBACK = {
    free: 'Free flow',
    busy: 'Busy',
    crowded: 'Crowded',
    very_crowded: 'Very crowded',
    critical: 'Crush risk',
  };

  function bandLabel(band) {
    return (UKSI.meta && UKSI.meta.band_labels && UKSI.meta.band_labels[band])
      || BAND_FALLBACK[band] || band;
  }

  function num(v, digits = 2) {
    return (v === null || v === undefined || Number.isNaN(v)) ? '—' : v.toFixed(digits);
  }

  // Time-to-critical is the one field that must never round a live risk to
  // "0 min": under a minute it is reported in seconds.
  function duration(seconds) {
    if (seconds === null || seconds === undefined) return '—';
    if (seconds <= 0) return 'now';
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
    return '> 1 hr';
  }

  function clock(d) {
    return d.toLocaleTimeString('en-GB', { hour12: false });
  }

  function relTime(iso) {
    if (!iso) return '';
    // The API emits "...Z" without a timezone suffix on some fields; make it
    // explicit so the browser does not read it as local time.
    const s = /[Zz]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + 'Z';
    const then = new Date(s).getTime();
    if (Number.isNaN(then)) return '';
    const secs = Math.max(0, (Date.now() - then) / 1000);
    if (secs < 5) return 'now';
    if (secs < 60) return `${Math.floor(secs)}s ago`;
    if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
    if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
    return `${Math.floor(secs / 86400)}d ago`;
  }

  const TREND_GLYPH = { rising: '↑', falling: '↓', steady: '→' };
  const TREND_WORD = { rising: 'rising', falling: 'falling', steady: 'steady' };

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  // -- HTTP ---------------------------------------------------------------

  async function getJSON(url) {
    const r = await fetch(url, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    let payload = null;
    try { payload = await r.json(); } catch (e) { /* empty body is fine */ }
    if (!r.ok) {
      const detail = payload && payload.detail ? payload.detail : `HTTP ${r.status}`;
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return payload;
  }

  // -- websocket ----------------------------------------------------------

  /* Reconnecting state feed.
   *
   * Backs off to a ceiling rather than retrying at a fixed rate: a control
   * room left open overnight should not hammer a server that is down, and the
   * connection dot has to tell the operator the truth the whole time.
   */
  function connectState({ onState, onStatus, path = '/ws/state' }) {
    let ws = null;
    let attempt = 0;
    let timer = null;
    let closed = false;

    function setStatus(state, detail) {
      if (onStatus) onStatus(state, detail);
    }

    function open() {
      if (closed) return;
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      try {
        ws = new WebSocket(`${proto}//${location.host}${path}`);
      } catch (e) {
        schedule('error', String(e));
        return;
      }

      ws.onopen = () => {
        attempt = 0;
        setStatus('live', 'connected');
      };
      ws.onmessage = (ev) => {
        let payload;
        try { payload = JSON.parse(ev.data); } catch (e) { return; }
        onState(payload);
      };
      ws.onerror = () => { /* onclose always follows; handled there */ };
      ws.onclose = () => {
        ws = null;
        if (!closed) schedule('down', 'reconnecting');
      };
    }

    function schedule(state, detail) {
      attempt += 1;
      setStatus(state, detail);
      const wait = Math.min(1000 * Math.pow(1.6, attempt), 15000);
      clearTimeout(timer);
      timer = setTimeout(open, wait);
    }

    open();
    return {
      close() { closed = true; clearTimeout(timer); if (ws) ws.close(); },
      get socket() { return ws; },
    };
  }

  // -- canvas geometry ----------------------------------------------------

  /* Bearing from an image-space flow vector, in degrees clockwise from
   * up-screen. Mirrors _bearing_from_image_vector in uksi/worker/flow.py so
   * the arrow on screen points where the engine says the crowd is going. */
  function bearingDeg(vx, vy) {
    return ((Math.atan2(vx, -vy) * 180 / Math.PI) + 360) % 360;
  }

  function centroid(polygon) {
    if (!polygon || !polygon.length) return null;
    let x = 0, y = 0;
    for (const [px, py] of polygon) { x += px; y += py; }
    return [x / polygon.length, y / polygon.length];
  }

  function polygonPath(ctx, polygon) {
    ctx.beginPath();
    polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
  }

  // -- risk palette -------------------------------------------------------

  const RISK_COLOUR = { green: '#3fb950', amber: '#e3a008', red: '#f04a4a' };
  const CORRIDOR_COLOUR = { clear: '#3fb950', restricted: '#e3a008', blocked: '#f04a4a' };

  return {
    bandLabel, num, duration, clock, relTime, el,
    TREND_GLYPH, TREND_WORD, RISK_COLOUR, CORRIDOR_COLOUR,
    getJSON, postJSON, connectState,
    bearingDeg, centroid, polygonPath,
    meta: null,
  };
})();
