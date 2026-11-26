// Voice Filter web UI. Plain ES module, no dependencies. API contract: PLAN.md section 5.
import { Timeline, fmtTime } from './timeline.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const ACTIVE = new Set(['queued', 'analyzing', 'rendering']);
const REVIEWABLE = new Set(['review', 'done']);
const VIDEO_EXT = ['mp4', 'mkv', 'mov', 'webm', 'avi', 'm4v'];
const FALLBACK_PRESETS = [
  { id: 'fast', name: 'Fast', description: 'Quickest; good for a first look.' },
  { id: 'balanced', name: 'Balanced', description: 'Recommended quality and speed.' },
  { id: 'best', name: 'Best', description: 'Slowest, most accurate, handles overlapping voices.' },
];
const STATE_LABEL = {
  queued: 'Queued', analyzing: 'Analyzing', review: 'Ready for review',
  rendering: 'Exporting', done: 'Done', error: 'Failed',
};
const LABEL_TEXT = { female: 'Female', male: 'Male', overlap: 'Overlap' };
const KIND_TEXT = { speech: 'Speech', nonverbal: 'Non-verbal', mixed: 'Mixed' };

const S = {
  info: null,
  jobs: [],
  jobsLoaded: false,
  file: null,
  upload: null,
  review: null,
};

// ------------------------------------------------------------------ utils
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (name, cls = '') => `<svg class="icon ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const labelIcon = (label) => icon(label === 'female' || label === 'male' ? label : 'overlap');
const pct = (x) => `${Math.round((+x || 0) * 100)}%`;

function fmtDur(sec) {
  if (!Number.isFinite(sec)) return '–';
  if (sec < 60) return `${sec.toFixed(1)} s`;
  return fmtTime(sec, 0);
}
function fmtBytes(n) {
  if (!Number.isFinite(n)) return '';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1000 && i < u.length - 1) { n /= 1000; i++; }
  return `${n.toFixed(n >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}
function fmtEta(sec) {
  if (!Number.isFinite(sec) || sec < 0) return '';
  if (sec < 60) return `${Math.ceil(sec)} s left`;
  if (sec < 3600) return `${Math.round(sec / 60)} min left`;
  return `${Math.floor(sec / 3600)} h ${Math.round((sec % 3600) / 60)} min left`;
}
function fmtCreated(v) {
  if (v === null || v === undefined || v === '') return '';
  let d;
  if (typeof v === 'number') d = new Date(v < 1e12 ? v * 1000 : v);
  else d = new Date(v);
  if (Number.isNaN(d.getTime())) return '';
  const time = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const today = new Date().toDateString() === d.toDateString();
  return today ? time : `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`;
}
const store = {
  get(k, d) { try { const v = localStorage.getItem(`vf.${k}`); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`vf.${k}`, JSON.stringify(v)); } catch { /* private mode */ } },
};

class ApiError extends Error {
  constructor(msg, status) { super(msg); this.status = status; }
}
async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.body = JSON.stringify(opts.json);
    init.headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
    delete init.json;
  }
  let r;
  try {
    r = await fetch(`api/${path}`, init);
  } catch {
    throw new ApiError('Cannot reach the Voice Filter server. Is it still running?', 0);
  }
  if (!r.ok) throw new ApiError(await errorText(r), r.status);
  const ct = r.headers.get('content-type') || '';
  return ct.includes('json') ? r.json() : r.text();
}
async function errorText(r) {
  let msg = `${r.status} ${r.statusText}`.trim();
  try {
    const j = await r.json();
    if (j && j.detail) msg = typeof j.detail === 'string' ? j.detail : j.detail.map?.((d) => d.msg).join('; ') || JSON.stringify(j.detail);
    else if (j && j.error) msg = j.error;
  } catch { /* not json */ }
  return msg;
}

function toast(text, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = text;
  const box = $('#toasts');
  box.append(el);
  while (box.children.length > 2) box.firstElementChild.remove();
  setTimeout(() => el.remove(), kind === 'error' ? 7000 : 3500);
}

function confirmDialog({ title, text, ok = 'Delete' }) {
  const d = $('#confirm');
  $('#confirm-title').textContent = title;
  $('#confirm-text').textContent = text;
  $('#confirm-ok').textContent = ok;
  d.returnValue = '';
  if (typeof d.showModal !== 'function') return Promise.resolve(window.confirm(`${title}\n\n${text}`));
  d.showModal();
  return new Promise((res) => d.addEventListener('close', () => res(d.returnValue === 'ok'), { once: true }));
}

async function copyText(t) {
  try {
    if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(t); return true; }
  } catch { /* fall through */ }
  const ta = document.createElement('textarea');
  ta.value = t; ta.setAttribute('readonly', '');
  ta.style.cssText = 'position:fixed;top:0;left:0;opacity:0;';
  document.body.append(ta);
  ta.select(); ta.setSelectionRange(0, t.length);
  let ok = false;
  try { ok = document.execCommand('copy'); } catch { ok = false; }
  ta.remove();
  return ok;
}

function setProgress(el, frac, indeterminate = false) {
  const v = Math.max(0, Math.min(1, +frac || 0));
  el.classList.toggle('indeterminate', indeterminate);
  el.firstElementChild.style.width = `${(v * 100).toFixed(1)}%`;
  if (indeterminate) el.removeAttribute('aria-valuenow'); else el.setAttribute('aria-valuenow', String(Math.round(v * 100)));
}

// ------------------------------------------------------------------ info / header
async function loadInfo() {
  try {
    S.info = await api('info');
  } catch (e) {
    S.info = null;
    $('#device-name').textContent = 'Offline';
    toast(e.message, 'error');
  }
  renderInfo();
  renderPresets();
}

function renderInfo() {
  const info = S.info;
  if (!info) return;
  const gpu = info.gpu_name && String(info.device || '').toLowerCase() !== 'cpu';
  const chip = $('#device-chip');
  const short = gpu ? String(info.gpu_name).replace(/^NVIDIA\s+/i, '').replace(/^GeForce\s+/i, '') : 'CPU';
  $('#device-name').textContent = matchMedia('(min-width: 700px)').matches ? (gpu ? info.gpu_name : 'CPU only') : short;
  chip.classList.toggle('cpu', !gpu);
  chip.title = gpu ? `Processing on ${info.gpu_name}` : 'No GPU found: processing runs on the CPU and is slow';
  chip.setAttribute('aria-label', gpu ? `Device: ${info.gpu_name}` : 'Device: CPU, no GPU found');
  const urls = (info.lan_urls || []).filter(Boolean);
  const onPhone = matchMedia('(pointer: coarse)').matches && urls.some((u) => u.startsWith(location.origin));
  $('#connect').hidden = !urls.length || onPhone;
  $('#lan-urls').innerHTML = urls.map((u, i) => `
    <li><code id="lan-${i}">${esc(u)}</code>
      <button type="button" class="btn small" data-copy="${esc(u)}" aria-label="Copy ${esc(u)}">${icon('copy')}<span>Copy</span></button></li>`).join('');
}

$('#lan-urls').addEventListener('click', async (e) => {
  const b = e.target.closest('[data-copy]');
  if (!b) return;
  const ok = await copyText(b.dataset.copy);
  const span = b.querySelector('span');
  if (ok) {
    span.textContent = 'Copied';
    b.querySelector('use').setAttribute('href', '#i-check');
    setTimeout(() => { span.textContent = 'Copy'; b.querySelector('use').setAttribute('href', '#i-copy'); }, 1600);
  } else {
    const code = b.parentElement.querySelector('code');
    const sel = getSelection(); const rg = document.createRange();
    rg.selectNodeContents(code); sel.removeAllRanges(); sel.addRange(rg);
    toast('Copy is blocked here; the address is selected, copy it manually.');
  }
});

// ------------------------------------------------------------------ new job form
function renderPresets() {
  const presets = S.info?.presets?.length ? S.info.presets : FALLBACK_PRESETS;
  const saved = store.get('preset', 'balanced');
  const cur = presets.some((p) => p.id === saved) ? saved : (presets.find((p) => p.id === 'balanced') || presets[0]).id;
  $('#preset-group').innerHTML = presets.map((p) => `
    <label><input type="radio" name="preset" value="${esc(p.id)}" ${p.id === cur ? 'checked' : ''}
      aria-describedby="preset-desc"><span>${esc(p.name || p.id)}</span></label>`).join('');
  updatePresetDesc();
}
function currentPreset() { return $('#preset-group input:checked')?.value || 'balanced'; }
function updatePresetDesc() {
  const presets = S.info?.presets?.length ? S.info.presets : FALLBACK_PRESETS;
  const p = presets.find((x) => x.id === currentPreset());
  $('#preset-desc').textContent = p?.description || '';
}
$('#preset-group').addEventListener('change', () => { store.set('preset', currentPreset()); updatePresetDesc(); });
$('#opt-bg').checked = store.get('keep_bg', true);
$('#opt-auto').checked = store.get('auto_export', false);
$('#opt-bg').addEventListener('change', (e) => store.set('keep_bg', e.target.checked));
$('#opt-auto').addEventListener('change', (e) => store.set('auto_export', e.target.checked));

function isVideo(file) {
  if (!file) return false;
  if ((file.type || '').startsWith('video/')) return true;
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  return VIDEO_EXT.includes(ext);
}
function setFile(file) {
  if (S.upload) return;
  if (file && !isVideo(file)) {
    toast(`“${file.name}” does not look like a video. Choose an MP4, MKV, MOV, WebM, AVI or M4V file.`, 'error');
    return;
  }
  S.file = file || null;
  const dz = $('#dropzone');
  dz.classList.toggle('has-file', !!S.file);
  $('.dz-empty', dz).hidden = !!S.file;
  $('.dz-file', dz).hidden = !S.file;
  if (S.file) {
    $('#dz-name').textContent = S.file.name;
    $('#dz-size').textContent = fmtBytes(S.file.size);
  }
  $('#start-btn').disabled = !S.file;
}
$('#file-input').addEventListener('change', (e) => setFile(e.target.files[0]));

(function setupDrop() {
  const dz = $('#dropzone');
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  window.addEventListener('dragenter', (e) => {
    if (!hasFiles(e) || $('#view-home').hidden) return;
    depth++; dz.classList.add('dragover');
  });
  window.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) dz.classList.remove('dragover'); });
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0; dz.classList.remove('dragover');
    if ($('#view-home').hidden) return;
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });
})();

$('#job-form').addEventListener('submit', (e) => {
  e.preventDefault();
  if (S.file && !S.upload) startUpload();
});
$('#upload-cancel').addEventListener('click', () => S.upload?.xhr.abort());
window.addEventListener('beforeunload', (e) => {
  if (S.upload) { e.preventDefault(); e.returnValue = ''; }
});

function startUpload() {
  const file = S.file;
  const fd = new FormData();
  fd.append('preset', currentPreset());
  fd.append('keep_background', $('#opt-bg').checked ? 'true' : 'false');
  fd.append('auto_export', $('#opt-auto').checked ? 'true' : 'false');
  fd.append('file', file, file.name);
  const xhr = new XMLHttpRequest();
  const up = { xhr, file, t0: performance.now(), lastT: performance.now(), lastB: 0, rate: 0 };
  S.upload = up;
  setFormBusy(true);
  const bar = $('#upload-bar');
  const text = $('#upload-text');
  setProgress(bar, 0);
  text.textContent = `Starting upload of ${fmtBytes(file.size)}…`;

  xhr.upload.onprogress = (ev) => {
    if (!ev.lengthComputable) { setProgress(bar, 0, true); text.textContent = `${fmtBytes(ev.loaded)} sent`; return; }
    const now = performance.now();
    const dt = (now - up.lastT) / 1000;
    if (dt > 0.25) {
      const inst = (ev.loaded - up.lastB) / dt;
      up.rate = up.rate ? up.rate * 0.7 + inst * 0.3 : inst;
      up.lastT = now; up.lastB = ev.loaded;
    }
    const frac = ev.loaded / ev.total;
    setProgress(bar, frac);
    const parts = [`${Math.floor(frac * 100)}%`, `${fmtBytes(ev.loaded)} of ${fmtBytes(ev.total)}`];
    if (up.rate > 0 && frac < 1) parts.push(`${fmtBytes(up.rate)}/s`, fmtEta((ev.total - ev.loaded) / up.rate));
    text.textContent = parts.join(' · ');
  };
  xhr.upload.onload = () => { setProgress(bar, 1, true); text.textContent = 'Upload complete, starting the job…'; };
  xhr.onload = () => {
    const ok = xhr.status >= 200 && xhr.status < 300;
    let body = null;
    try { body = JSON.parse(xhr.responseText); } catch { /* ignore */ }
    finishUpload();
    if (ok) {
      toast(`Uploaded “${file.name}”. Analysis started.`);
      setFile(null);
      $('#file-input').value = '';
      poll();
    } else {
      let msg = `${xhr.status} ${xhr.statusText}`;
      if (body?.detail) msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
      toast(`Upload failed: ${msg}`, 'error');
    }
  };
  xhr.onerror = () => { finishUpload(); toast('Upload failed: the connection to the server was lost.', 'error'); };
  xhr.onabort = () => { finishUpload(); toast('Upload canceled.'); };
  xhr.open('POST', 'api/jobs');
  xhr.send(fd);
}
function finishUpload() { S.upload = null; setFormBusy(false); }
function setFormBusy(busy) {
  $('#upload-status').hidden = !busy;
  $('#start-btn').hidden = busy;
  $('#start-btn').disabled = busy || !S.file;
  $('#file-input').disabled = busy;
  $$('#preset-group input, #opt-bg, #opt-auto').forEach((i) => { i.disabled = busy; });
}

// ------------------------------------------------------------------ jobs list + polling
let pollTimer = null;
let pollBusy = false;
let pollAgain = false;
let pollFailures = 0;

async function poll() {
  if (pollBusy) { pollAgain = true; return; }
  clearTimeout(pollTimer); pollTimer = null;
  pollBusy = true;
  try {
    await refreshJobs();
    pollFailures = 0;
  } catch (e) {
    pollFailures++;
    if (pollFailures === 2) toast(`${e.message} Retrying…`, 'error');
  } finally {
    pollBusy = false;
  }
  if (pollAgain) { pollAgain = false; poll(); return; }
  const active = S.jobs.some((j) => ACTIVE.has(j.state));
  if ((active || pollFailures) && !document.hidden) {
    pollTimer = setTimeout(poll, pollFailures ? Math.min(10000, 1500 * 2 ** pollFailures) : 1500);
  }
}
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });

async function refreshJobs() {
  const jobs = await api('jobs');
  const prev = new Map(S.jobs.map((j) => [j.id, j]));
  S.jobs = Array.isArray(jobs) ? jobs : [];
  S.jobsLoaded = true;
  if (prev.size) {
    for (const j of S.jobs) {
      const p = prev.get(j.id);
      if (!p || p.state === j.state) continue;
      if (j.state === 'review') toast(`Ready for review: ${j.filename}`);
      else if (j.state === 'done' && p.state === 'rendering') toast(`Export finished: ${j.filename}`);
      else if (j.state === 'error') toast(`Failed: ${j.filename}`, 'error');
    }
  }
  renderJobs();
  if (S.review) {
    const j = S.jobs.find((x) => x.id === S.review.id);
    if (j) onReviewJobUpdate(j);
  }
}

function jobStructureKey(j) {
  return [j.state, j.has_output, j.error, j.filename, j.preset, j.duration, j.created].join('|');
}
function renderJobs() {
  const list = $('#jobs-list');
  if (!S.jobs.length) {
    list.innerHTML = `<p class="empty">${S.jobsLoaded ? 'No jobs yet. Choose a video to start.' : 'Loading…'}</p>`;
    return;
  }
  $('.empty', list)?.remove();
  const seen = new Set();
  let prevEl = null;
  for (const j of S.jobs) {
    seen.add(j.id);
    let el = list.querySelector(`[data-job="${CSS.escape(j.id)}"]`);
    const key = jobStructureKey(j);
    if (!el) {
      el = document.createElement('article');
      el.className = 'card job';
      el.dataset.job = j.id;
    }
    if (el.dataset.key !== key) {
      const focusedAction = el.contains(document.activeElement) ? document.activeElement.dataset.action : null;
      el.innerHTML = jobCardHtml(j);
      el.dataset.key = key;
      if (focusedAction) el.querySelector(`[data-action="${focusedAction}"]`)?.focus();
    }
    updateJobProgress(el, j);
    const want = prevEl ? prevEl.nextSibling : list.firstChild;
    if (want !== el) list.insertBefore(el, want);
    prevEl = el;
  }
  $$('[data-job]', list).forEach((el) => { if (!seen.has(el.dataset.job)) el.remove(); });
}

function presetName(id) {
  const p = (S.info?.presets || FALLBACK_PRESETS).find((x) => x.id === id);
  return p?.name || id || '';
}
function stateChip(state) {
  const cls = ACTIVE.has(state) ? 'active' : state;
  const ic = { review: 'review', done: 'check', error: 'warn', queued: 'list' }[state] || 'gpu';
  return `<span class="state-chip ${esc(cls)}">${icon(ic)}${esc(STATE_LABEL[state] || state)}</span>`;
}
function jobCardHtml(j) {
  const meta = [presetName(j.preset), j.duration ? fmtDur(j.duration) : '', j.keep_background === false ? 'voices only' : '', fmtCreated(j.created)]
    .filter(Boolean).join(' · ');
  const id = encodeURIComponent(j.id);
  const actions = [];
  if (REVIEWABLE.has(j.state)) {
    actions.push(`<a class="btn primary" data-action="review" href="#/review/${id}">${icon('review')}Review</a>`);
    actions.push(`<button type="button" class="btn" data-action="export" data-id="${esc(j.id)}">${icon('export')}${j.state === 'done' ? 'Export again' : 'Export'}</button>`);
  }
  if (j.has_output && j.state !== 'rendering') {
    actions.push(`<a class="btn" data-action="download" href="api/jobs/${id}/download" download>${icon('download')}Download</a>`);
  }
  actions.push(`<button type="button" class="btn ghost del" data-action="delete" data-id="${esc(j.id)}" aria-label="Delete ${esc(j.filename)}">${icon('trash')}Delete</button>`);
  return `
    <div class="job-head">
      <div class="job-title">
        <div class="job-name" title="${esc(j.filename)}">${esc(j.filename)}</div>
        <div class="job-meta">${esc(meta)}</div>
      </div>
      ${stateChip(j.state)}
    </div>
    ${ACTIVE.has(j.state) ? `
      <div class="job-msg"><span class="m"></span><span class="p"></span></div>
      <div class="progress" role="progressbar" aria-label="Progress of ${esc(j.filename)}" aria-valuemin="0" aria-valuemax="100"><span></span></div>` : ''}
    ${j.state === 'error' ? `<div class="err" role="alert">${icon('warn')}<div><strong>Processing failed.</strong> ${esc(j.error || j.message || 'Unknown error')}</div></div>` : ''}
    ${j.state === 'review' ? `<p class="job-meta">${esc(j.message && j.message !== 'Ready for review' ? j.message : 'Check the flagged segments, then export.')}</p>` : ''}
    <div class="job-actions">${actions.join('')}</div>`;
}
function updateJobProgress(el, j) {
  if (!ACTIVE.has(j.state)) return;
  const msg = j.message || STATE_LABEL[j.state];
  $('.job-msg .m', el).textContent = j.state === 'queued' ? (j.message || 'Waiting for the GPU') : msg;
  $('.job-msg .p', el).textContent = j.state === 'queued' ? '' : pct(j.progress);
  setProgress($('.progress', el), j.progress, j.state === 'queued');
}

$('#jobs-list').addEventListener('click', async (e) => {
  const b = e.target.closest('[data-action]');
  if (!b) return;
  const job = S.jobs.find((j) => j.id === (b.dataset.id || b.closest('[data-job]')?.dataset.job));
  if (b.dataset.action === 'export' && job) startExport(job.id);
  if (b.dataset.action === 'delete' && job) deleteJob(job);
});

async function deleteJob(job) {
  const active = ACTIVE.has(job.state);
  const ok = await confirmDialog({
    title: 'Delete this job?',
    text: `“${job.filename}” and its analysis${job.has_output ? ' and exported video' : ''} will be deleted from the PC.${active ? ' Processing will be stopped.' : ''}`,
  });
  if (!ok) return;
  try {
    await api(`jobs/${encodeURIComponent(job.id)}`, { method: 'DELETE' });
    S.jobs = S.jobs.filter((j) => j.id !== job.id);
    renderJobs();
    if (S.review?.id === job.id) location.hash = '#/';
    toast(`Deleted “${job.filename}”.`);
  } catch (err) {
    toast(`Delete failed: ${err.message}`, 'error');
  }
}

async function startExport(id) {
  const R = S.review;
  try {
    if (R && R.id === id) await waitForOverrides(R);
    await api(`jobs/${encodeURIComponent(id)}/export`, { method: 'POST' });
    if (R && R.id === id) R.changedSinceExport = false;
    toast('Export started.');
  } catch (err) {
    toast(`Export failed to start: ${err.message}`, 'error');
  }
  poll();
}

// ------------------------------------------------------------------ routing
function route() {
  const m = location.hash.match(/^#\/review\/(.+)$/);
  if (m) openReview(decodeURIComponent(m[1]));
  else showHome();
}
window.addEventListener('hashchange', route);

function showHome() {
  closeReview();
  $('#view-review').hidden = true;
  $('#view-home').hidden = false;
  document.title = 'Voice Filter';
}

// ------------------------------------------------------------------ review
const audio = $('#audio');
const canvas = $('#timeline');
const timeline = new Timeline(canvas, {
  onSegmentTap: (seg, t) => { selectSegment(seg.id, { from: 'timeline' }); seekTo(t); },
  onSeek: (t) => { seekTo(t); if (S.review?.selectedId && !wideLayout()) selectSegment(null); },
});

function wideLayout() { return matchMedia('(min-width: 1000px)').matches; }

function closeReview() {
  const R = S.review;
  if (!R) return;
  audio.pause();
  audio.removeAttribute('src');
  audio.load();
  cancelAnimationFrame(R.raf);
  S.review = null;
  document.body.classList.remove('sheet-open');
  $('#seg-panel').classList.remove('open');
}

async function openReview(id) {
  if (S.review?.id === id) return;
  closeReview();
  $('#view-home').hidden = true;
  $('#view-review').hidden = false;
  window.scrollTo(0, 0);
  const R = {
    id, job: S.jobs.find((j) => j.id === id) || null, analysis: null, segs: [], segById: new Map(), spk: new Map(),
    flagged: [], selectedId: null, filter: store.get('filter', 'all'), sort: store.get('sort', 'time'),
    source: store.get('source', 'preview'), tab: 'segs', pending: { segments: {}, speakers: {} }, inflight: null,
    previewVersion: 0, curSrc: null, pendingSeek: null, stopAt: null, time: 0, raf: 0, changedSinceExport: false,
    loading: false,
  };
  S.review = R;
  $('#rv-layout').hidden = true;
  $('#rv-loading').hidden = false;
  $('#rv-loading').textContent = 'Loading analysis…';
  renderReviewHeader();
  try {
    R.job = await api(`jobs/${encodeURIComponent(id)}`);
  } catch (e) {
    if (S.review !== R) return;
    toast(e.status === 404 ? 'That job no longer exists.' : e.message, 'error');
    location.hash = '#/';
    return;
  }
  if (S.review !== R) return;
  renderReviewHeader();
  if (REVIEWABLE.has(R.job.state) || R.job.state === 'rendering') loadAnalysis(R);
  else showNotReady(R);
  poll();
}

function showNotReady(R) {
  const j = R.job;
  $('#rv-loading').textContent = j.state === 'error'
    ? `This job failed: ${j.error || j.message || 'unknown error'}`
    : `This video is still being analyzed (${pct(j.progress)}). The review opens automatically when it is ready.`;
}

async function loadAnalysis(R) {
  if (R.loading || R.analysis) return;
  R.loading = true;
  const base = `jobs/${encodeURIComponent(R.id)}`;
  try {
    const [analysis, peaks] = await Promise.all([
      api(`${base}/analysis`),
      api(`${base}/peaks`).catch(() => null),
    ]);
    if (S.review !== R) return;
    applyAnalysis(R, analysis);
    R.peaks = peaks;
    $('#rv-loading').hidden = true;
    $('#rv-layout').hidden = false;
    timeline.resize();
    timeline.setData({ duration: R.analysis.duration, segments: R.segs, peaks: peaks?.peaks || null, rate: peaks?.rate || 1 });
    timeline.fit();
    timeline.effective = (s) => effective(R, s);
    setSource(R.source, false);
    renderTime();
    renderReview();
    renderTabs();
  } catch (e) {
    if (S.review !== R) return;
    $('#rv-loading').textContent = `Could not load the analysis: ${e.message}`;
  } finally {
    R.loading = false;
  }
}

function onReviewJobUpdate(j) {
  const R = S.review;
  R.job = j;
  renderReviewHeader();
  if (!R.analysis && !R.loading) {
    if (REVIEWABLE.has(j.state) || j.state === 'rendering') loadAnalysis(R);
    else showNotReady(R);
  }
}

function applyAnalysis(R, a) {
  R.analysis = a;
  const thr = Number.isFinite(+a.low_conf_threshold) ? +a.low_conf_threshold : 0.5;
  R.segs = [...(a.segments || [])].sort((x, y) => x.start - y.start || x.end - y.end);
  R.segs.forEach((s, i) => {
    s._i = i;
    if (s.low_confidence === undefined || s.low_confidence === null) s.low_confidence = (+s.confidence || 0) < thr;
  });
  R.segById = new Map(R.segs.map((s) => [String(s.id), s]));
  R.spk = new Map((a.speakers || []).map((s) => [String(s.id), s]));
  R.flagged = R.segs.filter((s) => s.low_confidence);
  if (R.selectedId !== null && !R.segById.has(R.selectedId)) R.selectedId = null;
  timeline.setSegments(R.segs);
}

function effective(R, seg) {
  if (seg.override) return seg.override === 'keep';
  const sp = R.spk.get(String(seg.speaker));
  if (sp && sp.override) return sp.override === 'keep';
  return !!seg.keep;
}
function speakerName(id) { return typeof id === 'number' ? `Speaker ${id + 1}` : String(id); }

function renderReviewHeader() {
  const R = S.review;
  if (!R) return;
  const j = R.job;
  $('#rv-title').textContent = j?.filename || 'Loading…';
  document.title = `${j?.filename || 'Review'} · Voice Filter`;
  $('#rv-sub').textContent = j ? [presetName(j.preset), STATE_LABEL[j.state] || j.state, j.duration ? fmtDur(j.duration) : ''].filter(Boolean).join(' · ') : '';
  const exportBtn = $('#rv-export');
  const can = j && REVIEWABLE.has(j.state) && !!R.analysis;
  exportBtn.disabled = !can;
  $('span', exportBtn).textContent = j?.state === 'rendering' ? 'Exporting…' : 'Export';
  $('#rv-export-again').hidden = j?.state !== 'done';
  exportBtn.setAttribute('aria-label', j?.state === 'done' ? 'Export again' : (j?.state === 'rendering' ? 'Exporting' : 'Export video'));
  const dl = $('#rv-download');
  const dlHref = `api/jobs/${encodeURIComponent(R.id)}/download`;
  dl.hidden = !(j?.has_output && j.state !== 'rendering');
  dl.href = dlHref;
  const st = $('#export-status');
  if (j?.state === 'rendering') {
    st.hidden = false;
    $('#export-text').textContent = `Exporting video… ${j.message ? `${j.message}` : ''}`;
    setProgress($('#export-bar'), j.progress, j.stage === 'queued');
    $('#export-bar').hidden = false;
    $('#export-dl').hidden = true;
  } else if (j?.state === 'done' && j.has_output && R.exportSeen) {
    st.hidden = false;
    $('#export-text').textContent = R.changedSinceExport
      ? 'You changed decisions after this export. Export again to apply them.'
      : 'Export finished.';
    $('#export-bar').hidden = true;
    $('#export-dl').hidden = false;
    $('#export-dl').href = dlHref;
  } else if (j?.state === 'error' && R.exportSeen) {
    st.hidden = false;
    $('#export-text').textContent = `Export failed: ${j.error || j.message || 'unknown error'}`;
    $('#export-bar').hidden = true;
    $('#export-dl').hidden = true;
  } else {
    st.hidden = true;
  }
  if (j?.state === 'rendering') R.exportSeen = true;
}

$('#rv-export').addEventListener('click', () => {
  const R = S.review;
  if (!R) return;
  R.exportSeen = true;
  startExport(R.id);
});

// ---------- summary / flagged / lists
function renderReview() {
  const R = S.review;
  if (!R || !R.analysis) return;
  renderSummary(R);
  renderFlagNav(R);
  renderSegList(R);
  renderSpeakers(R);
  renderDetail(R);
  timeline.setSelected(R.selectedId);
  timeline.draw();
  renderReviewHeader();
}

function renderSummary(R) {
  let keepT = 0, remT = 0, nOver = 0;
  for (const s of R.segs) {
    const d = Math.max(0, s.end - s.start);
    if (effective(R, s)) keepT += d; else remT += d;
    if (s.override) nOver++;
  }
  for (const sp of R.spk.values()) if (sp.override) nOver++;
  const total = keepT + remT || 1;
  $('#summary').innerHTML = `
    <span class="pill"><span class="dot keep" aria-hidden="true"></span>Keeping <b>${fmtDur(keepT)}</b> <span class="muted">(${pct(keepT / total)})</span></span>
    <span class="pill"><span class="dot remove" aria-hidden="true"></span>Removing <b>${fmtDur(remT)}</b></span>
    <span class="pill">${icon('warn')}<b>${R.flagged.length}</b> unsure</span>
    ${nOver ? `<span class="pill">${icon('pin')}<b>${nOver}</b> change${nOver === 1 ? '' : 's'}</span>` : ''}`;
}

function renderFlagNav(R) {
  const n = R.flagged.length;
  const sel = R.segById.get(R.selectedId);
  const idx = sel && sel.low_confidence ? R.flagged.indexOf(sel) : -1;
  const span = $('#flag-count span');
  span.textContent = idx >= 0 ? `${idx + 1}/${n}` : String(n);
  $('#flag-count').setAttribute('aria-label', idx >= 0 ? `Unsure segment ${idx + 1} of ${n}` : `${n} unsure segments`);
  $('#flag-count').title = `${n} segments the model is unsure about`;
  $('#flag-prev').disabled = !n;
  $('#flag-next').disabled = !n;
}

const FILTERS = [
  { id: 'all', name: 'All', test: () => true },
  { id: 'flagged', name: 'Unsure', icon: 'warn', test: (R, s) => s.low_confidence },
  { id: 'removed', name: 'Removed', test: (R, s) => !effective(R, s) },
  { id: 'kept', name: 'Kept', test: (R, s) => effective(R, s) },
  { id: 'changed', name: 'Changed', icon: 'pin', test: (R, s) => !!s.override },
];

function renderSegList(R) {
  const chips = $('#seg-filters');
  chips.innerHTML = FILTERS.map((f) => {
    const n = R.segs.filter((s) => f.test(R, s)).length;
    return `<button type="button" class="chip" data-filter="${f.id}" aria-pressed="${R.filter === f.id}">${f.icon ? icon(f.icon) : ''}${f.name}<span class="n">${n}</span></button>`;
  }).join('');
  $('#seg-sort').value = R.sort;
  const f = FILTERS.find((x) => x.id === R.filter) || FILTERS[0];
  let rows = R.segs.filter((s) => f.test(R, s));
  if (R.sort === 'conf') rows = [...rows].sort((a, b) => a.confidence - b.confidence || a.start - b.start);
  const list = $('#seg-list');
  const focusedId = list.contains(document.activeElement) ? document.activeElement.dataset.id : null;
  const scroll = list.scrollTop;
  list.innerHTML = rows.length ? rows.map((s) => segRowHtml(R, s)).join('') : '<li class="list-empty">No segments match this filter.</li>';
  list.scrollTop = scroll;
  if (focusedId) list.querySelector(`[data-id="${CSS.escape(focusedId)}"]`)?.focus({ preventScroll: true });
}

function segRowHtml(R, s) {
  const keep = effective(R, s);
  const id = String(s.id);
  const lbl = LABEL_TEXT[s.label] || s.label;
  const aria = [
    `${fmtTime(s.start)}`, `${(s.end - s.start).toFixed(1)} seconds`, lbl, speakerName(s.speaker),
    keep ? 'kept' : 'removed', `${pct(s.confidence)} sure`, s.low_confidence ? 'unsure' : '', s.override ? 'changed by you' : '',
  ].filter(Boolean).join(', ');
  return `<li><button type="button" class="seg-row" data-id="${esc(id)}" aria-current="${R.selectedId === id}" aria-label="${esc(aria)}">
    <span class="badge ${keep ? 'keep' : 'remove'}">${icon(keep ? 'check' : 'cross')}</span>
    <span class="sr-main">
      <span class="sr-l1"><span class="t">${fmtTime(s.start)}</span>${labelIcon(s.label)}<span>${esc(lbl)}</span><span class="who">· ${esc(speakerName(s.speaker))} · ${esc(KIND_TEXT[s.kind] || s.kind || '')}</span></span>
      <span class="sr-l2"><span class="mini ${s.low_confidence ? 'low' : ''}"><span style="width:${pct(s.confidence)}"></span></span>${pct(s.confidence)} sure · ${(s.end - s.start).toFixed(1)} s</span>
    </span>
    <span class="sr-flags">${s.low_confidence ? icon('warn') : ''}${s.override ? icon('pin') : ''}</span>
  </button></li>`;
}

$('#seg-filters').addEventListener('click', (e) => {
  const b = e.target.closest('[data-filter]');
  const R = S.review;
  if (!b || !R) return;
  R.filter = b.dataset.filter;
  store.set('filter', R.filter);
  renderSegList(R);
  $(`[data-filter="${R.filter}"]`)?.focus();
});
$('#seg-sort').addEventListener('change', (e) => {
  const R = S.review;
  if (!R) return;
  R.sort = e.target.value;
  store.set('sort', R.sort);
  $('#seg-list').scrollTop = 0;
  renderSegList(R);
});
$('#seg-list').addEventListener('click', (e) => {
  const b = e.target.closest('.seg-row');
  if (!b) return;
  selectSegment(b.dataset.id, { from: 'list', seek: true });
});

function renderSpeakers(R) {
  const list = $('#spk-list');
  const focused = list.contains(document.activeElement) ? document.activeElement.dataset.key : null;
  list.innerHTML = [...R.spk.values()].map((sp) => {
    const id = String(sp.id);
    const segs = R.segs.filter((s) => String(s.speaker) === id);
    const tot = Number.isFinite(+sp.total_dur) ? +sp.total_dur : segs.reduce((a, s) => a + s.end - s.start, 0);
    const modelKeepT = segs.reduce((a, s) => a + (s.keep ? s.end - s.start : 0), 0);
    const segTot = segs.reduce((a, s) => a + s.end - s.start, 0) || 1;
    const nSegOver = segs.filter((s) => s.override).length;
    const ov = sp.override || null;
    const effKeep = ov ? ov === 'keep' : modelKeepT / segTot >= 0.5;
    const name = speakerName(sp.id);
    const lbl = LABEL_TEXT[sp.label] || sp.label || '';
    return `<li class="spk" data-spk="${esc(id)}">
      <div class="spk-head">${labelIcon(sp.label)}<span class="spk-name">${esc(name)}</span>
        <span class="spk-label">${esc(lbl)} · p(female) ${(+sp.p_female).toFixed(2)}</span>
        <span class="badge ${effKeep ? 'keep' : 'remove'}" title="${effKeep ? 'Mostly kept' : 'Mostly removed'}">${icon(effKeep ? 'check' : 'cross')}</span></div>
      <div class="spk-meta">${fmtDur(tot)} total · ${sp.n_segments ?? segs.length} segments${nSegOver ? ` · ${nSegOver} changed one by one` : ''}</div>
      <div class="spk-state">${ov
        ? `<strong>You: ${ov === 'keep' ? 'keep' : 'remove'} all</strong> <span class="muted">(model keeps ${pct(modelKeepT / segTot)} of this speaker)</span>`
        : `Model decides per segment <span class="muted">(keeps ${pct(modelKeepT / segTot)})</span>`}</div>
      <div class="spk-actions" role="group" aria-label="${esc(name)}">
        <button type="button" class="btn keep-btn" data-key="${esc(id)}:keep" data-val="keep" aria-pressed="${ov === 'keep'}">${icon('check')}Keep all</button>
        <button type="button" class="btn remove-btn" data-key="${esc(id)}:remove" data-val="remove" aria-pressed="${ov === 'remove'}">${icon('cross')}Remove all</button>
        <button type="button" class="btn" data-key="${esc(id)}:reset" data-val="reset" ${ov ? '' : 'disabled'}>${icon('reset')}Reset</button>
      </div></li>`;
  }).join('') || '<li class="list-empty">No speakers found.</li>';
  if (focused) list.querySelector(`[data-key="${CSS.escape(focused)}"]`)?.focus();
}
$('#spk-list').addEventListener('click', (e) => {
  const b = e.target.closest('[data-val]');
  const R = S.review;
  if (!b || !R) return;
  const sp = R.spk.get(b.closest('[data-spk]').dataset.spk);
  if (!sp) return;
  const v = b.dataset.val === 'reset' ? null : b.dataset.val;
  setSpeakerOverride(R, sp, v);
});

// tabs
function renderTabs() {
  const R = S.review;
  const segs = !R || R.tab === 'segs';
  $('#tab-segs').setAttribute('aria-selected', String(segs));
  $('#tab-spk').setAttribute('aria-selected', String(!segs));
  $('#tab-segs').tabIndex = segs ? 0 : -1;
  $('#tab-spk').tabIndex = segs ? -1 : 0;
  $('#panel-segs').hidden = !segs;
  $('#panel-spk').hidden = segs;
}
function setTab(tab, focus) {
  if (!S.review) return;
  S.review.tab = tab;
  renderTabs();
  if (focus) $(tab === 'segs' ? '#tab-segs' : '#tab-spk').focus();
}
$('#tab-segs').addEventListener('click', () => setTab('segs'));
$('#tab-spk').addEventListener('click', () => setTab('spk'));
$('.tabs').addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
    e.preventDefault();
    setTab(S.review?.tab === 'segs' ? 'spk' : 'segs', true);
  }
});

// ---------- selection & detail
function selectSegment(id, { from = '', seek = false } = {}) {
  const R = S.review;
  if (!R) return;
  R.selectedId = id === null || id === undefined ? null : String(id);
  const seg = R.segById.get(R.selectedId);
  const panel = $('#seg-panel');
  panel.classList.toggle('open', !!seg);
  document.body.classList.toggle('sheet-open', !!seg);
  if (seg) {
    // coming from the list, keys or flag navigation: zoom in if the segment would be too small to see
    if (from !== 'timeline') timeline.zoomToFit(seg.start, seg.end);
    timeline.ensureVisible(seg.start, seg.end);
    if (seek) seekTo(seg.start);
    const keep = effective(R, seg);
    $('#sel-announce').textContent = `Segment ${seg._i + 1} of ${R.segs.length}, ${fmtTime(seg.start)} to ${fmtTime(seg.end)}, ${LABEL_TEXT[seg.label] || seg.label}, ${speakerName(seg.speaker)}, ${keep ? 'kept' : 'removed'}, ${pct(seg.confidence)} sure${seg.low_confidence ? ', unsure' : ''}.`;
  }
  renderReview();
  if (seg && from !== 'list') scrollRowIntoView(R.selectedId);
  if (seg && !wideLayout()) keepTimelineAboveSheet();
}

function keepTimelineAboveSheet() {
  // the bottom sheet covers the lower part of the screen: keep toolbar, canvas and transport visible above it
  requestAnimationFrame(() => {
    const sheetTop = window.innerHeight - $('#seg-panel').offsetHeight; // not getBoundingClientRect: sheet may be mid-transition
    const top = $('.tl-toolbar').getBoundingClientRect().top;
    const bottom = $('.transport').getBoundingClientRect().bottom;
    const canvasBottom = $('.tl-wrap').getBoundingClientRect().bottom;
    if (top >= 8 && bottom <= sheetTop - 4) return;
    let delta = top - 8;
    if (bottom - delta > sheetTop - 4) delta = Math.max(delta, canvasBottom - (sheetTop - 4));
    const smooth = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    window.scrollBy({ top: delta, behavior: smooth ? 'smooth' : 'auto' });
  });
}

function scrollRowIntoView(id) {
  const list = $('#seg-list');
  const row = list.querySelector(`[data-id="${CSS.escape(id)}"]`);
  if (!row) return;
  const top = row.offsetTop - list.offsetTop;
  const bottom = top + row.offsetHeight;
  if (top < list.scrollTop) list.scrollTop = top - 8;
  else if (bottom > list.scrollTop + list.clientHeight) list.scrollTop = bottom - list.clientHeight + 8;
}

function renderDetail(R) {
  const seg = R.segById.get(R.selectedId);
  $('#seg-empty').hidden = !!seg;
  $('#seg-detail').hidden = !seg;
  if (!seg) return;
  const keep = effective(R, seg);
  const sp = R.spk.get(String(seg.speaker));
  const thr = +R.analysis.low_conf_threshold;
  const lbl = LABEL_TEXT[seg.label] || seg.label;
  $('#sd-title').innerHTML = `${labelIcon(seg.label)}Segment ${seg._i + 1} <span class="muted">of ${R.segs.length}</span>`;
  $('#sd-time').textContent = `${fmtTime(seg.start, 2)} – ${fmtTime(seg.end, 2)}`;
  let why;
  const modelTxt = seg.keep ? 'keep' : 'remove';
  if (seg.override) why = `Changed by you (model said ${modelTxt}).`;
  else if (sp?.override) why = `Set for all of ${esc(speakerName(sp.id))} by you (model said ${modelTxt}).`;
  else why = `Model decision: ${esc(lbl.toLowerCase())} voice.`;
  if (seg.low_confidence) why += ` <strong>The model is unsure</strong> (${pct(seg.confidence)} sure).`;
  const dec = $('#sd-decision');
  dec.classList.toggle('flagged', !!seg.low_confidence);
  dec.innerHTML = `<span class="badge ${keep ? 'keep' : 'remove'}">${icon(keep ? 'check' : 'cross')}</span>
    <div><strong>${keep ? 'Kept' : 'Removed'}</strong> ${seg.low_confidence ? icon('warn') : ''}<br><span>${why}</span></div>`;
  $('#sd-keep').setAttribute('aria-pressed', String(keep));
  $('#sd-remove').setAttribute('aria-pressed', String(!keep));
  $('#sd-reset').disabled = !seg.override;
  $('#sd-prev').disabled = seg._i === 0;
  $('#sd-next').disabled = seg._i === R.segs.length - 1;
  const pf = Math.max(0, Math.min(1, +seg.p_female || 0));
  const conf = Math.max(0, Math.min(1, +seg.confidence || 0));
  $('#sd-grid').innerHTML = `
    <dt>Voice</dt><dd>${labelIcon(seg.label)}<span>${esc(lbl)} · ${esc(KIND_TEXT[seg.kind] || seg.kind || '–')} · ${(seg.end - seg.start).toFixed(2)} s</span></dd>
    <dt>Speaker</dt><dd><span>${esc(speakerName(seg.speaker))}${sp ? ` <span class="muted">· ${esc(LABEL_TEXT[sp.label] || sp.label || '')}${sp.override ? `, you: ${sp.override} all` : ''}</span>` : ''}</span></dd>
    <dt>p(female)</dt><dd><span class="num">${pf.toFixed(2)}</span>
      <span class="meter-wrap"><span class="meter scale" role="img" aria-label="Probability female ${pf.toFixed(2)}"><i style="left:calc(50% - 1px)"></i><b style="left:${(pf * 100).toFixed(1)}%"></b></span>
      <span class="meter-ends" aria-hidden="true"><span>male</span><span>female</span></span></span></dd>
    <dt>Confidence</dt><dd><span class="num">${conf.toFixed(2)}</span>
      <span class="meter-wrap"><span class="meter ${seg.low_confidence ? 'low' : ''}" role="img" aria-label="Confidence ${pct(conf)}${Number.isFinite(thr) ? `, unsure below ${pct(thr)}` : ''}"><span style="width:${(conf * 100).toFixed(1)}%"></span>${Number.isFinite(thr) ? `<i style="left:calc(${(thr * 100).toFixed(1)}% - 1px)" title="Unsure below ${pct(thr)}"></i>` : ''}</span></span></dd>`;
  $('#sd-title').title = `Segment id ${seg.id}`;
}

function selected() { const R = S.review; return R ? R.segById.get(R.selectedId) : null; }
function refocusTimeline(e) {
  // after a mouse click, return focus to the timeline so keyboard shortcuts keep working
  if (e && e.detail > 0 && wideLayout()) canvas.focus({ preventScroll: true });
}
$('#sd-keep').addEventListener('click', (e) => { const s = selected(); if (s) setSegOverride(S.review, s, 'keep'); refocusTimeline(e); });
$('#sd-remove').addEventListener('click', (e) => { const s = selected(); if (s) setSegOverride(S.review, s, 'remove'); refocusTimeline(e); });
$('#sd-reset').addEventListener('click', (e) => { const s = selected(); if (s) setSegOverride(S.review, s, null); refocusTimeline(e); });
$('#sd-play').addEventListener('click', () => { const s = selected(); if (s) playRange(s.start, s.end); });
$('#sd-prev').addEventListener('click', () => stepSegment(-1));
$('#sd-next').addEventListener('click', () => stepSegment(1));
$('#sd-close').addEventListener('click', () => selectSegment(null));

function stepSegment(dir) {
  const R = S.review;
  if (!R || !R.segs.length) return;
  const cur = R.segById.get(R.selectedId);
  let target;
  if (cur) target = R.segs[Math.max(0, Math.min(R.segs.length - 1, cur._i + dir))];
  else if (dir > 0) target = R.segs.find((s) => s.start >= R.time - 1e-3) || R.segs[R.segs.length - 1];
  else target = [...R.segs].reverse().find((s) => s.start < R.time) || R.segs[0];
  selectSegment(target.id, { seek: true });
}
function flagStep(dir) {
  const R = S.review;
  if (!R || !R.flagged.length) return;
  const cur = R.segById.get(R.selectedId);
  const ref = cur ? cur.start : R.time;
  const list = R.flagged;
  let target;
  if (dir > 0) target = list.find((s) => s.start > ref + 1e-6) || list[0];
  else target = [...list].reverse().find((s) => s.start < ref - 1e-6) || list[list.length - 1];
  selectSegment(target.id, { seek: true });
}
$('#flag-next').addEventListener('click', (e) => { flagStep(1); refocusTimeline(e); });
$('#flag-prev').addEventListener('click', (e) => { flagStep(-1); refocusTimeline(e); });

// ---------- overrides (optimistic, coalesced, reconciled with the server response)
function setSegOverride(R, seg, val) {
  if ((seg.override ?? null) === val) return;
  seg.override = val;
  queueOverride(R, 'segments', seg.id, val);
}
function setSpeakerOverride(R, sp, val) {
  if ((sp.override ?? null) === val) return;
  sp.override = val;
  queueOverride(R, 'speakers', sp.id, val);
}
function queueOverride(R, kind, id, val) {
  R.pending[kind][String(id)] = val;
  if (R.job?.state === 'done') R.changedSinceExport = true;
  renderReview();
  flushOverrides(R);
}
const hasPending = (p) => Object.keys(p.segments).length || Object.keys(p.speakers).length;
function reapplyPending(R) {
  for (const [id, v] of Object.entries(R.pending.segments)) { const s = R.segById.get(id); if (s) s.override = v; }
  for (const [id, v] of Object.entries(R.pending.speakers)) { const s = R.spk.get(id); if (s) s.override = v; }
}
function flushOverrides(R) {
  if (R.inflight || !hasPending(R.pending)) return R.inflight;
  const body = R.pending;
  R.pending = { segments: {}, speakers: {} };
  R.inflight = (async () => {
    try {
      const a = await api(`jobs/${encodeURIComponent(R.id)}/overrides`, { method: 'POST', json: body });
      if (S.review !== R) return;
      R.previewVersion++;
      if (a && Array.isArray(a.segments)) { applyAnalysis(R, a); reapplyPending(R); }
    } catch (e) {
      if (S.review !== R) return;
      toast(`Could not save your change: ${e.message}`, 'error');
      try { applyAnalysis(R, await api(`jobs/${encodeURIComponent(R.id)}/analysis`)); reapplyPending(R); } catch { /* keep local */ }
    } finally {
      R.inflight = null;
    }
    if (S.review !== R) return;
    renderReview();
    if (hasPending(R.pending)) await flushOverrides(R);
  })();
  return R.inflight;
}
async function waitForOverrides(R) {
  while (R.inflight || hasPending(R.pending)) {
    await (R.inflight || flushOverrides(R));
  }
}

// ---------- audio
function audioUrl(R, kind) {
  return `api/jobs/${encodeURIComponent(R.id)}/audio/${kind}${kind === 'preview' ? `?v=${R.previewVersion}` : ''}`;
}
function ensureSrc(R) {
  const want = audioUrl(R, R.source);
  if (R.curSrc === want) return;
  R.pendingSeek = R.time;
  R.curSrc = want;
  audio.src = want;
}
function setSource(kind, user = true) {
  const R = S.review;
  if (!R) return;
  R.source = ['original', 'vocals', 'preview'].includes(kind) ? kind : 'preview';
  store.set('source', R.source);
  $$('#source-group [role="radio"]').forEach((b) => {
    const on = b.dataset.src === R.source;
    b.setAttribute('aria-checked', String(on));
    b.tabIndex = on ? 0 : -1;
  });
  if (user) {
    const playing = !audio.paused;
    ensureSrc(R);
    if (playing) audio.play().catch(() => {});
  }
}
$('#source-group').addEventListener('click', (e) => {
  const b = e.target.closest('[data-src]');
  if (b) setSource(b.dataset.src);
});
$('#source-group').addEventListener('keydown', (e) => {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return;
  e.preventDefault();
  const order = ['original', 'vocals', 'preview'];
  const i = order.indexOf(S.review?.source);
  const n = order[(i + (e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : 2)) % 3];
  setSource(n);
  $(`#source-group [data-src="${n}"]`).focus();
});

function seekTo(t) {
  const R = S.review;
  if (!R) return;
  const dur = R.analysis?.duration || 0;
  t = Math.max(0, Math.min(dur || t, t));
  R.time = t;
  R.stopAt = null;
  if (audio.readyState >= 1 && R.curSrc) {
    try { audio.currentTime = Math.min(t, Number.isFinite(audio.duration) ? audio.duration : t); } catch { R.pendingSeek = t; }
  } else {
    R.pendingSeek = t;
  }
  timeline.setTime(t);
  renderTime();
}
function renderTime() {
  const R = S.review;
  if (!R) return;
  $('#time-display').textContent = `${fmtTime(R.time)} / ${fmtTime(R.analysis?.duration || 0, 0)}`;
}
async function play() {
  const R = S.review;
  if (!R) return;
  ensureSrc(R);
  const dur = R.analysis?.duration || 0;
  if (R.time >= dur - 0.05 && R.stopAt === null) seekTo(0);
  try {
    await audio.play();
  } catch (e) {
    if (e.name !== 'AbortError') toast(`Cannot play audio: ${e.message}`, 'error');
  }
}
function togglePlay() { if (audio.paused) play(); else audio.pause(); }
function playRange(a, b) {
  const R = S.review;
  if (!R) return;
  ensureSrc(R);
  seekTo(a);
  R.stopAt = b;
  play();
}
$('#play-btn').addEventListener('click', togglePlay);
audio.addEventListener('loadedmetadata', () => {
  const R = S.review;
  if (!R || R.pendingSeek === null) return;
  try { audio.currentTime = Math.min(R.pendingSeek, audio.duration || R.pendingSeek); } catch { /* ignore */ }
  R.pendingSeek = null;
});
audio.addEventListener('play', () => { setPlayIcon(true); tick(); });
audio.addEventListener('pause', () => { setPlayIcon(false); });
audio.addEventListener('ended', () => { setPlayIcon(false); if (S.review) S.review.stopAt = null; });
audio.addEventListener('error', () => {
  const R = S.review;
  if (!R || !audio.getAttribute('src')) return;
  toast(`Could not load the ${R.source} audio.`, 'error');
  R.curSrc = null;
});
function setPlayIcon(playing) {
  const b = $('#play-btn');
  b.querySelector('use').setAttribute('href', playing ? '#i-pause' : '#i-play');
  b.setAttribute('aria-label', playing ? 'Pause' : 'Play');
}
function tick() {
  const R = S.review;
  if (!R) return;
  cancelAnimationFrame(R.raf);
  const loop = () => {
    if (S.review !== R) return;
    if (R.pendingSeek === null) R.time = audio.currentTime;
    if (R.stopAt !== null && R.time >= R.stopAt) { audio.pause(); R.stopAt = null; }
    timeline.time = R.time;
    if (!audio.paused) timeline.follow(R.time);
    timeline.draw();
    renderTime();
    if (!audio.paused) R.raf = requestAnimationFrame(loop);
  };
  R.raf = requestAnimationFrame(loop);
}

$('#zoom-in').addEventListener('click', (e) => { timeline.zoomBy(0.5); refocusTimeline(e); });
$('#zoom-out').addEventListener('click', (e) => { timeline.zoomBy(2); refocusTimeline(e); });
$('#zoom-fit').addEventListener('click', (e) => { timeline.fit(); refocusTimeline(e); });

// ---------- keyboard
document.addEventListener('keydown', (e) => {
  const R = S.review;
  if (!R || !R.analysis || $('#view-review').hidden || $('#confirm').open) return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target;
  const tag = t.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || t.isContentEditable) return;
  const inGroup = t.closest('[role="radiogroup"], [role="tablist"]');
  const isControl = tag === 'BUTTON' || tag === 'A';
  const seg = selected();
  let handled = true;
  switch (e.key) {
    case 'ArrowRight': if (inGroup) return; stepSegment(1); break;
    case 'ArrowLeft': if (inGroup) return; stepSegment(-1); break;
    case 'k': case 'K': if (seg) setSegOverride(R, seg, 'keep'); break;
    case 'r': case 'R': if (seg) setSegOverride(R, seg, 'remove'); break;
    case 'u': case 'U': if (seg) setSegOverride(R, seg, null); break;
    case ' ': if (isControl) return; togglePlay(); break;
    case 'Enter': if (isControl) return; if (seg) playRange(seg.start, seg.end); break;
    case 'f': case 'F': flagStep(e.shiftKey ? -1 : 1); break;
    case '+': case '=': timeline.zoomBy(0.5); break;
    case '-': case '_': timeline.zoomBy(2); break;
    case '0': timeline.fit(); break;
    case 'Escape': if (seg) selectSegment(null); else handled = false; break;
    default: handled = false;
  }
  if (handled) e.preventDefault();
});

window.addEventListener('resize', () => { if (S.review?.analysis) timeline.draw(); });

// ------------------------------------------------------------------ boot
// read-only handle for automated UI tests and debugging in the browser console
window.__voicefilter = { state: S, timeline };
renderJobs();
loadInfo();
route();
poll();
