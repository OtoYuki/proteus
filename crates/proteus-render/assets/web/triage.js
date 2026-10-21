// Proteus triage table: sort, filter and read a campaign's rows. The data is in the page
// (`PROTEUS_TRIAGE`); nothing is fetched. Every number is the one `analyze --export` writes.
'use strict';
(() => {
  const D = window.PROTEUS_TRIAGE;
  const $ = (id) => document.getElementById(id);
  const rows = D.rows;
  // Which columns to show first. The rest are there and appear with "every column".
  const shown = new Set(D.columns);
  let sortKey = D.sort || 'model';
  let sortAsc = false;
  let all = false;

  const isNumber = (k) => rows.some((r) => typeof r[k] === 'number');
  const label = (k) => k.replace(/_pct$/, ' %').replace(/_/g, ' ');
  const columns = () => (all ? D.allColumns : D.columns);

  /** A cell's text: a number to the precision the column deserves, or an empty mark. */
  function text(v, k) {
    if (v === null || v === undefined || v === '') return '—';
    if (typeof v !== 'number') return String(v);
    if (!isFinite(v)) return '—';
    if (Number.isInteger(v)) return String(v);
    const abs = Math.abs(v);
    return v.toFixed(abs >= 1000 ? 0 : abs >= 100 ? 1 : abs >= 1 ? 2 : 3);
  }

  function matching() {
    const q = $('filter').value.trim().toLowerCase();
    const onlyOver = $('over') && $('over').checked;
    return rows.filter((r) => {
      if (onlyOver && !(typeof r[D.cutoffColumn] === 'number' && r[D.cutoffColumn] >= D.cutoff)) return false;
      if (!q) return true;
      return String(r.model).toLowerCase().includes(q) || String(r.file).toLowerCase().includes(q);
    });
  }

  function sorted(list) {
    const k = sortKey;
    const dir = sortAsc ? 1 : -1;
    // Missing values rank last whichever way the column is sorted: they are not small, they
    // are absent, and a run of blanks at the top reads as a result.
    return list.slice().sort((a, b) => {
      const x = a[k], y = b[k];
      const xm = x === null || x === undefined || x === '' || (typeof x === 'number' && !isFinite(x));
      const ym = y === null || y === undefined || y === '' || (typeof y === 'number' && !isFinite(y));
      if (xm || ym) return xm && ym ? 0 : xm ? 1 : -1;
      if (typeof x === 'number' && typeof y === 'number') return (x - y) * dir;
      return String(x).localeCompare(String(y), undefined, { numeric: true }) * dir;
    });
  }

  function toast(text) {
    const t = $('toast');
    t.textContent = text;
    t.hidden = false;
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => { t.hidden = true; }, 2200);
  }

  function draw() {
    const cols = columns();
    const list = sorted(matching());
    const thead = $('head'), tbody = $('body');
    thead.textContent = '';
    const hr = document.createElement('tr');
    for (const k of cols) {
      const th = document.createElement('th');
      th.textContent = label(k);
      th.scope = 'col';
      if (k === sortKey) th.setAttribute('aria-sort', sortAsc ? 'ascending' : 'descending');
      th.title = k + ' — click to sort';
      th.addEventListener('click', () => {
        // A new column starts the way that column is usually read: names up, numbers down.
        if (sortKey === k) sortAsc = !sortAsc;
        else { sortKey = k; sortAsc = !isNumber(k); }
        draw();
      });
      hr.append(th);
    }
    thead.append(hr);

    tbody.textContent = '';
    const frag = document.createDocumentFragment();
    for (const r of list) {
      const tr = document.createElement('tr');
      for (const k of cols) {
        const td = document.createElement('td');
        const v = r[k];
        td.textContent = text(v, k);
        if (td.textContent === '—') td.className = 'none';
        if (k === 'model') {
          td.className = 'model';
          td.title = r.file + ' — click to copy a view command';
          td.addEventListener('click', () => {
            const cmd = 'proteus view ' + quote(r.file);
            if (navigator.clipboard && navigator.clipboard.writeText) {
              navigator.clipboard.writeText(cmd).then(() => toast('copied: ' + cmd), () => toast(cmd));
            } else toast(cmd);
          });
        } else if (k === D.cutoffColumn && typeof v === 'number' && v >= D.cutoff) {
          td.className = 'over';
          td.title = 'at or above the ' + D.cutoff + ' line';
        }
        tr.append(td);
      }
      frag.append(tr);
    }
    tbody.append(frag);
    $('empty').hidden = list.length > 0;
    $('count').textContent = list.length === rows.length
      ? rows.length + (rows.length === 1 ? ' model' : ' models')
      : list.length + ' of ' + rows.length + ' models';
  }

  /** A path as a shell would take it back. */
  function quote(p) {
    return /^[\w@%+=:,./-]+$/.test(p) ? p : "'" + p.replace(/'/g, "'\\''") + "'";
  }

  $('filter').addEventListener('input', draw);
  if ($('over')) $('over').addEventListener('change', draw);
  $('all').addEventListener('change', (e) => { all = e.target.checked; draw(); });
  window.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) {
      if (e.key === 'Escape') { e.target.value = ''; e.target.blur(); draw(); }
      return;
    }
    if (e.key === '/') { $('filter').focus(); e.preventDefault(); }
  });
  draw();
})();
