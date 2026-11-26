// Timeline canvas: ruler, segment lane, vocal envelope, overview strip, playhead.
// Pointer: drag = pan, pinch = zoom, tap = select segment / seek. Wheel: pan, ctrl+wheel zoom.

const TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
const FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

export function fmtTime(t, dec = 1) {
  if (!Number.isFinite(t)) t = 0;
  const neg = t < 0; t = Math.abs(t);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  let s = (t % 60).toFixed(dec);
  if (parseFloat(s) >= 60) s = (0).toFixed(dec); // rounding edge
  const ss = s.padStart(dec ? 3 + dec : 2, '0');
  const body = h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
  return (neg ? '-' : '') + body;
}

export class Timeline {
  constructor(canvas, opts = {}) {
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.opts = opts;
    this.segs = [];
    this.dur = 1;
    this.peaks = null;
    this.rate = 1;
    this.viewStart = 0;
    this.viewDur = 1;
    this.time = 0;
    this.selectedId = null;
    this.effective = (s) => !!s.keep;
    this.lastUser = 0;
    this.pointers = new Map();
    this.g = null;
    this.w = 0; this.h = 0;
    this._raf = 0;
    this.readColors();
    const mq = matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener?.('change', () => { this.readColors(); this.draw(); });
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(canvas);
    canvas.addEventListener('pointerdown', (e) => this.onDown(e));
    canvas.addEventListener('pointermove', (e) => this.onMove(e));
    canvas.addEventListener('pointerup', (e) => this.onUp(e));
    canvas.addEventListener('pointercancel', (e) => this.onCancel(e));
    canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  readColors() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    this.col = {
      bg: v('--tl-bg'), grid: v('--tl-grid'), text: v('--tl-text'), wave: v('--tl-wave'),
      stripe: v('--tl-stripe'), playhead: v('--tl-playhead'), sel: v('--tl-sel'), view: v('--tl-view'),
      keep: v('--keep'), keepInk: v('--keep-ink'), remove: v('--remove'), removeSoft: v('--remove-soft'),
      warn: v('--warn'), warnEdge: v('--warn-edge'), ink: v('--ink'), surface: v('--surface'), surface3: v('--surface-3'),
    };
    // diagonal stripe pattern for low-confidence segments
    const p = document.createElement('canvas');
    const dpr = Math.max(1, Math.round(window.devicePixelRatio || 1));
    const n = 7 * dpr;
    p.width = p.height = n;
    const pc = p.getContext('2d');
    pc.strokeStyle = this.col.stripe;
    pc.lineWidth = 1.6 * dpr;
    pc.beginPath();
    for (let k = -1; k <= 1; k++) { pc.moveTo(k * n, n); pc.lineTo(k * n + n, 0); }
    pc.stroke();
    this.stripe = this.ctx.createPattern(p, 'repeat');
    if (this.stripe && this.stripe.setTransform) this.stripe.setTransform(new DOMMatrix().scale(1 / dpr));
  }

  setData({ duration, segments, peaks, rate }) {
    const first = this.segs.length === 0;
    this.dur = Math.max(0.1, duration || 1);
    this.segs = segments;
    if (peaks !== undefined) { this.peaks = peaks; this.rate = rate || 1; }
    if (first) { this.viewStart = 0; this.viewDur = this.dur; }
    this.clampView();
    this.draw();
  }

  setSegments(segments) { this.segs = segments; this.draw(); }
  setSelected(id) { this.selectedId = id; this.draw(); }
  setTime(t) { this.time = t; this.draw(); }

  get minDur() { return Math.min(this.dur, 1); }
  clampView() {
    this.viewDur = Math.min(this.dur, Math.max(this.minDur, this.viewDur));
    this.viewStart = Math.min(this.dur - this.viewDur, Math.max(0, this.viewStart));
  }
  tx(t) { return ((t - this.viewStart) / this.viewDur) * this.w; }
  timeAt(x) { return this.viewStart + (x / this.w) * this.viewDur; }

  zoomBy(factor, anchor) {
    if (anchor === undefined) {
      anchor = (this.time >= this.viewStart && this.time <= this.viewStart + this.viewDur)
        ? this.time : this.viewStart + this.viewDur / 2;
    }
    const frac = (anchor - this.viewStart) / this.viewDur;
    this.viewDur = Math.min(this.dur, Math.max(this.minDur, this.viewDur * factor));
    this.viewStart = anchor - frac * this.viewDur;
    this.clampView();
    this.draw();
    this.opts.onView?.();
  }
  fit() { this.viewStart = 0; this.viewDur = this.dur; this.draw(); this.opts.onView?.(); }

  ensureVisible(start, end) {
    const vs = this.viewStart, ve = vs + this.viewDur;
    const margin = this.viewDur * 0.08;
    if (start >= vs + margin * 0.25 && end <= ve - margin * 0.25) return;
    if (end - start > this.viewDur * 0.8) this.viewStart = start - margin;
    else this.viewStart = (start + end) / 2 - this.viewDur / 2;
    this.clampView();
    this.draw();
  }

  follow(t) {
    if (performance.now() - this.lastUser < 2500 || this.g) return;
    const vs = this.viewStart, ve = vs + this.viewDur;
    if (t < vs || t > ve - this.viewDur * 0.04) {
      this.viewStart = t - this.viewDur * 0.05;
      this.clampView();
    }
  }

  resize() {
    const r = this.c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = Math.max(1, r.width); this.h = Math.max(1, r.height);
    this.c.width = Math.round(this.w * dpr);
    this.c.height = Math.round(this.h * dpr);
    this.dpr = dpr;
    this.drawNow();
  }

  layout() {
    const H = this.h;
    const ruler = 20;
    const markTop = ruler + 3;
    const blockTop = markTop + 15;
    const blockBottom = blockTop + 46;
    const ovH = 20;
    const ovTop = H - ovH - 5;
    const waveTop = blockBottom + 6;
    const waveBottom = ovTop - 8;
    return { ruler, markTop, blockTop, blockBottom, waveTop, waveBottom, ovTop, ovH };
  }

  draw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.drawNow(); });
  }

  drawNow() {
    if (!this.w) return;
    const { ctx, col } = this;
    const L = this.layout();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = col.bg;
    ctx.fillRect(0, 0, this.w, this.h);
    this.drawRuler(L);
    this.drawWave(L);
    this.drawSegments(L);
    this.drawPlayhead(L);
    this.drawOverview(L);
  }

  drawRuler(L) {
    const { ctx, col } = this;
    const pps = this.w / this.viewDur;
    let step = TICKS[TICKS.length - 1];
    for (const s of TICKS) { if (s * pps >= 70) { step = s; break; } }
    const minor = step / (step >= 60 && step % 300 === 0 ? 5 : (String(step)[0] === '2' ? 4 : 5));
    const t0 = Math.floor(this.viewStart / minor) * minor;
    const t1 = this.viewStart + this.viewDur;
    ctx.fillStyle = col.grid;
    for (let t = t0; t <= t1; t += minor) {
      const x = Math.round(this.tx(t)) + 0.5;
      const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
      ctx.fillRect(x - 0.5, major ? 12 : 16, 1, major ? 8 : 4);
      if (major) ctx.fillRect(x - 0.5, L.waveTop, 1, L.waveBottom - L.waveTop);
    }
    ctx.fillStyle = col.text;
    ctx.font = `500 11px ${FONT}`;
    ctx.textBaseline = 'top';
    const dec = step < 1 ? 1 : 0;
    for (let t = Math.floor(this.viewStart / step) * step; t <= t1; t += step) {
      const x = this.tx(t);
      if (x < -40) continue;
      ctx.fillText(fmtTime(t, dec), x + 3, 1);
    }
    ctx.fillStyle = col.grid;
    ctx.fillRect(0, L.ruler, this.w, 1);
    // centre line of the envelope
    const mid = (L.waveTop + L.waveBottom) / 2;
    ctx.fillRect(0, Math.round(mid), this.w, 1);
  }

  visibleRange() {
    const vs = this.viewStart, ve = vs + this.viewDur;
    const segs = this.segs;
    // binary search first segment ending after vs
    let lo = 0, hi = segs.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (segs[m].end < vs) lo = m + 1; else hi = m; }
    // segments are sorted by start; ends are nearly monotonic, step back a little for overlaps
    lo = Math.max(0, lo - 8);
    let end = lo;
    while (end < segs.length && segs[end].start <= ve) end++;
    return [lo, end];
  }

  drawWave(L) {
    if (!this.peaks || !this.peaks.length) return;
    const { ctx, col } = this;
    const mid = (L.waveTop + L.waveBottom) / 2;
    const half = (L.waveBottom - L.waveTop) / 2 - 1;
    const paths = { keep: new Path2D(), remove: new Path2D(), none: new Path2D() };
    const pk = this.peaks, rate = this.rate, n = pk.length;
    const segs = this.segs;
    let j = this.visibleRange()[0];
    const spp = this.viewDur / this.w;
    for (let px = 0; px < this.w; px++) {
      const ta = this.viewStart + px * spp;
      const tb = ta + spp;
      let i0 = Math.floor(ta * rate), i1 = Math.max(i0 + 1, Math.ceil(tb * rate));
      if (i0 >= n) break;
      i1 = Math.min(i1, n);
      let a = 0;
      for (let i = Math.max(0, i0); i < i1; i++) if (pk[i] > a) a = pk[i];
      if (a <= 0.004) continue;
      const tc = ta + spp / 2;
      while (j < segs.length && segs[j].end < tc) j++;
      let cls = 'none';
      if (j < segs.length && segs[j].start <= tc) cls = this.effective(segs[j]) ? 'keep' : 'remove';
      const hh = Math.max(0.5, a * half);
      paths[cls].rect(px, mid - hh, 1, hh * 2);
    }
    ctx.fillStyle = col.wave; ctx.fill(paths.none);
    ctx.fillStyle = col.keep; ctx.fill(paths.keep);
    ctx.globalAlpha = 0.55; ctx.fillStyle = col.remove; ctx.fill(paths.remove); ctx.globalAlpha = 1;
  }

  drawSegments(L) {
    const { ctx, col } = this;
    const [a, b] = this.visibleRange();
    const laneH = L.blockBottom - L.blockTop;
    let sel = null;
    for (let i = a; i < b; i++) {
      const s = this.segs[i];
      let x0 = this.tx(s.start), x1 = this.tx(s.end);
      if (x1 < -2 || x0 > this.w + 2) continue;
      const bw = Math.max(2, x1 - x0);
      const keep = this.effective(s);
      const conf = Math.max(0, Math.min(1, +s.confidence || 0));
      const h = Math.round(laneH * (0.42 + 0.58 * conf));
      const y = L.blockBottom - h;
      const r = Math.min(4, bw / 2);
      ctx.beginPath();
      roundRect(ctx, x0, y, bw, h, r);
      if (keep) {
        ctx.fillStyle = col.keep; ctx.fill();
      } else {
        ctx.fillStyle = col.removeSoft; ctx.fill();
        ctx.lineWidth = 1.5; ctx.strokeStyle = col.remove; ctx.stroke();
      }
      if (s.low_confidence && this.stripe) {
        ctx.save(); ctx.clip(); ctx.fillStyle = this.stripe; ctx.fillRect(x0, y, bw, h); ctx.restore();
      }
      // label icon
      const isz = 13;
      if (bw >= 18 && h >= 18) {
        const iconCol = keep ? col.keepInk : col.remove;
        let cx = x0 + bw / 2;
        if (!keep && bw >= 44) cx = x0 + bw / 2 - 8;
        if (s.low_confidence) { // plate behind icon so it reads on top of stripes
          ctx.fillStyle = keep ? col.keep : col.surface;
          ctx.beginPath(); ctx.arc(cx, y + h / 2, 9, 0, Math.PI * 2); ctx.fill();
        }
        drawLabel(ctx, s.label, cx, y + h / 2, isz, iconCol);
        if (!keep && bw >= 44) drawCross(ctx, x0 + bw / 2 + 9, y + h / 2, 9, col.remove);
      }
      // marker row: warning for low confidence, star for user override
      const my = L.markTop + 6;
      const cx = x0 + bw / 2;
      if (s.low_confidence && s.override) {
        drawWarn(ctx, cx + 6, my, 12, col);
        drawStar(ctx, cx - 7, my, 11, col);
      } else if (s.low_confidence) {
        drawWarn(ctx, cx, my, 12, col);
      } else if (s.override) {
        drawStar(ctx, cx, my, 11, col);
      }
      if (this.selectedId !== null && String(s.id) === this.selectedId) sel = { x0, bw };
    }
    if (sel) {
      ctx.fillStyle = col.view;
      ctx.fillRect(sel.x0, L.waveTop, sel.bw, L.waveBottom - L.waveTop);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = col.sel;
      ctx.beginPath();
      roundRect(ctx, sel.x0 - 3, L.markTop - 2, sel.bw + 6, L.blockBottom - L.markTop + 5, 6);
      ctx.stroke();
      ctx.lineWidth = 1;
      ctx.strokeStyle = col.bg;
      ctx.beginPath();
      roundRect(ctx, sel.x0 - 1.2, L.markTop - 0.2, sel.bw + 2.4, L.blockBottom - L.markTop + 1.4, 5);
      ctx.stroke();
    }
  }

  drawPlayhead(L) {
    const { ctx, col } = this;
    const x = this.tx(this.time);
    if (x < -1 || x > this.w + 1) return;
    ctx.fillStyle = col.playhead;
    ctx.fillRect(Math.round(x) - 1, L.ruler, 2, L.waveBottom - L.ruler);
    ctx.beginPath();
    ctx.moveTo(x - 6, L.ruler - 8); ctx.lineTo(x + 6, L.ruler - 8); ctx.lineTo(x, L.ruler + 1); ctx.closePath();
    ctx.fill();
  }

  drawOverview(L) {
    const { ctx, col } = this;
    const y = L.ovTop, h = L.ovH, W = this.w;
    ctx.fillStyle = col.surface3;
    ctx.beginPath(); roundRect(ctx, 0.5, y, W - 1, h, 4); ctx.fill();
    const k = W / this.dur;
    for (const s of this.segs) {
      const x = s.start * k, w = Math.max(1, (s.end - s.start) * k);
      if (this.effective(s)) { ctx.fillStyle = col.keep; ctx.fillRect(x, y + 3, w, h - 6); }
      else { ctx.fillStyle = col.remove; ctx.fillRect(x, y + h / 2, w, h / 2 - 3); }
    }
    ctx.fillStyle = col.warn;
    for (const s of this.segs) if (s.low_confidence) ctx.fillRect(s.start * k, y, Math.max(2, (s.end - s.start) * k), 3);
    // viewport
    const vx = this.viewStart * k, vw = Math.max(4, this.viewDur * k);
    ctx.fillStyle = col.view;
    ctx.fillRect(vx, y, vw, h);
    ctx.strokeStyle = col.sel; ctx.lineWidth = 1.5;
    ctx.beginPath(); roundRect(ctx, Math.max(0.75, vx), y + 0.75, Math.min(vw, W - 1.5), h - 1.5, 3); ctx.stroke();
    ctx.fillStyle = col.playhead;
    ctx.fillRect(Math.round(this.time * k) - 1, y - 2, 2, h + 4);
  }

  // ---------------- hit testing & input ----------------
  zoneAt(y) {
    const L = this.layout();
    if (y >= L.ovTop - 5) return 'overview';
    if (y < L.ruler) return 'ruler';
    return 'main';
  }

  hitSegment(x) {
    const t = this.timeAt(x);
    const slop = 10 * this.viewDur / this.w;
    let best = null, bestD = Infinity;
    const [a, b] = this.visibleRange();
    for (let i = a; i < b; i++) {
      const s = this.segs[i];
      let d = 0;
      if (t < s.start) d = s.start - t; else if (t > s.end) d = t - s.end;
      if (d > slop) continue;
      // prefer containing segments, then the shortest one (overlaps), then the closest
      const score = d * 1000 + (d === 0 ? (s.end - s.start) : 0);
      if (score < bestD) { bestD = score; best = s; }
    }
    return best;
  }

  pos(e) { const r = this.c.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }

  onDown(e) {
    if (e.button !== undefined && e.button > 0) return;
    const p = this.pos(e);
    try { this.c.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    this.pointers.set(e.pointerId, p);
    this.lastUser = performance.now();
    if (this.pointers.size === 1) {
      const zone = this.zoneAt(p.y);
      this.g = { type: zone === 'overview' ? 'overview' : 'pending', x0: p.x, y0: p.y, vs0: this.viewStart, zone };
      if (zone === 'overview') this.centerFromOverview(p.x);
    } else if (this.pointers.size === 2) {
      this.startPinch();
    }
  }

  startPinch() {
    const [a, b] = [...this.pointers.values()];
    const cx = (a.x + b.x) / 2;
    this.g = { type: 'pinch', d0: Math.max(20, Math.abs(a.x - b.x)), dur0: this.viewDur, tc: this.timeAt(cx) };
  }

  onMove(e) {
    if (!this.pointers.has(e.pointerId)) return;
    const p = this.pos(e);
    this.pointers.set(e.pointerId, p);
    const g = this.g;
    if (!g) return;
    this.lastUser = performance.now();
    if (g.type === 'pending' && Math.hypot(p.x - g.x0, p.y - g.y0) > 7) {
      g.type = 'pan'; this.c.classList.add('panning');
    }
    if (g.type === 'pan') {
      this.viewStart = g.vs0 - (p.x - g.x0) * (this.viewDur / this.w);
      this.clampView(); this.draw();
    } else if (g.type === 'pinch' && this.pointers.size >= 2) {
      const [a, b] = [...this.pointers.values()];
      const d = Math.max(20, Math.abs(a.x - b.x));
      const cx = (a.x + b.x) / 2;
      this.viewDur = Math.min(this.dur, Math.max(this.minDur, g.dur0 * g.d0 / d));
      this.viewStart = g.tc - (cx / this.w) * this.viewDur;
      this.clampView(); this.draw();
    } else if (g.type === 'overview') {
      this.centerFromOverview(p.x);
    }
  }

  onUp(e) {
    if (!this.pointers.has(e.pointerId)) return;
    const p = this.pos(e);
    this.pointers.delete(e.pointerId);
    const g = this.g;
    this.c.classList.remove('panning');
    if (g && g.type === 'pending') {
      this.g = null;
      const t = Math.max(0, Math.min(this.dur, this.timeAt(p.x)));
      const seg = g.zone === 'main' ? this.hitSegment(p.x) : null;
      if (seg) this.opts.onSegmentTap?.(seg, t);
      else this.opts.onSeek?.(t);
      return;
    }
    if (g && g.type === 'pinch' && this.pointers.size === 1) {
      const [q] = [...this.pointers.values()];
      this.g = { type: 'pan', x0: q.x, y0: q.y, vs0: this.viewStart };
      return;
    }
    if (this.pointers.size === 0) { this.g = null; this.opts.onView?.(); }
  }

  onCancel(e) {
    this.pointers.delete(e.pointerId);
    if (this.pointers.size === 0) { this.g = null; this.c.classList.remove('panning'); }
  }

  centerFromOverview(x) {
    const t = (x / this.w) * this.dur;
    this.viewStart = t - this.viewDur / 2;
    this.clampView(); this.draw();
  }

  onWheel(e) {
    const scale = e.deltaMode === 1 ? 16 : (e.deltaMode === 2 ? this.w : 1);
    const dx = e.deltaX * scale, dy = e.deltaY * scale;
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const p = this.pos(e);
      this.lastUser = performance.now();
      this.zoomBy(Math.exp(dy * 0.006), this.timeAt(p.x));
      return;
    }
    const horizontal = Math.abs(dx) > Math.abs(dy);
    const fullyOut = this.viewDur >= this.dur - 1e-6;
    if (!horizontal && (fullyOut || e.shiftKey === false && fullyOut)) return; // let the page scroll
    e.preventDefault();
    this.lastUser = performance.now();
    const d = horizontal ? dx : dy;
    this.viewStart += d * (this.viewDur / this.w);
    this.clampView(); this.draw();
  }
}

// ---------------- drawing helpers ----------------
function roundRect(ctx, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

export function drawLabel(ctx, label, cx, cy, s, color) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = Math.max(1.6, s * 0.14); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.beginPath();
  if (label === 'female') {
    ctx.arc(cx, cy - s * 0.17, s * 0.27, 0, Math.PI * 2);
    ctx.moveTo(cx, cy + s * 0.1); ctx.lineTo(cx, cy + s * 0.5);
    ctx.moveTo(cx - s * 0.18, cy + s * 0.32); ctx.lineTo(cx + s * 0.18, cy + s * 0.32);
  } else if (label === 'male') {
    const ox = cx - s * 0.1, oy = cy + s * 0.12, r = s * 0.27;
    ctx.arc(ox, oy, r, 0, Math.PI * 2);
    const ex = cx + s * 0.42, ey = cy - s * 0.4;
    ctx.moveTo(ox + r * 0.7, oy - r * 0.7); ctx.lineTo(ex, ey);
    ctx.moveTo(ex - s * 0.26, ey); ctx.lineTo(ex, ey); ctx.lineTo(ex, ey + s * 0.26);
  } else {
    ctx.arc(cx - s * 0.16, cy, s * 0.27, 0, Math.PI * 2);
    ctx.moveTo(cx + s * 0.16 + s * 0.27, cy);
    ctx.arc(cx + s * 0.16, cy, s * 0.27, 0, Math.PI * 2);
  }
  ctx.stroke();
  ctx.restore();
}

function drawCross(ctx, cx, cy, s, color) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(cx - s / 2, cy - s / 2); ctx.lineTo(cx + s / 2, cy + s / 2);
  ctx.moveTo(cx + s / 2, cy - s / 2); ctx.lineTo(cx - s / 2, cy + s / 2);
  ctx.stroke();
  ctx.restore();
}

function drawWarn(ctx, cx, cy, s, col) {
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(cx, cy - s * 0.55); ctx.lineTo(cx + s * 0.6, cy + s * 0.5); ctx.lineTo(cx - s * 0.6, cy + s * 0.5); ctx.closePath();
  ctx.fillStyle = col.warn; ctx.fill();
  ctx.lineWidth = 1; ctx.lineJoin = 'round'; ctx.strokeStyle = col.warnEdge; ctx.stroke();
  ctx.fillStyle = '#1b1300';
  ctx.fillRect(cx - 0.9, cy - s * 0.2, 1.8, s * 0.38);
  ctx.fillRect(cx - 0.9, cy + s * 0.27, 1.8, 1.8);
  ctx.restore();
}

function drawStar(ctx, cx, cy, s, col) {
  ctx.save();
  ctx.beginPath();
  const R = s / 2, r = R * 0.45;
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + i * Math.PI / 5;
    const rr = i % 2 ? r : R;
    ctx.lineTo(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr);
  }
  ctx.closePath();
  ctx.fillStyle = col.ink; ctx.fill();
  ctx.lineWidth = 1; ctx.strokeStyle = col.bg; ctx.stroke();
  ctx.restore();
}
