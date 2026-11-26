// ─────────────────────────────────────────────────────────────────────────────
// EPD BACKTEST HARNESS - v2, 2026-08-24
//
// Built 2026-08-24, then RE-VERIFIED and corrected on the same day by running it
// over [n] symbols. Everything here has been executed against EPD v8.0 on the live
// chart. Paste the whole file into `ui_evaluate` in ONE call; it installs
// `window.__epd`.
//
// Gate before trusting anything:  priceAll() on CBRS 2026-08-12 must return
// pct = [n] and pctLvl = [n]. Both were reproduced exactly.
//
// ★ THE FOUR TRAPS, each of which silently returns a convincing wrong number ★
//
//   1. The `+` add marker is a plotshape, NOT a label - invisible to labels().
//      Reading labels alone prices 100 units where the truth is 150. See addFills().
//   2. An in-page symbol switch can SILENTLY FAIL and leave the previous ticker
//      on screen. The first [n]-symbol pass had clusters of symbols all
//      measured against the wrong data - identical bar counts gave it away. Always
//      assert the ticker EXACTLY (substring matching makes `U` match `BATS:UBER`),
//      and re-assert AFTER the history load. See setSymHard().
//   3. A symbol switch loads only ~5 days of 1-minute history. An older earnings
//      date then returns an empty label set that is indistinguishable from a real
//      "no trade". loadRange() forces the deep fetch; `covered` records whether it
//      actually reached. NEVER report "no trade" when covered is false.
//   4. useHold (in_1) must be OFF. It is a viewing feature; with it on the stop
//      cannot close a trade and every number is wrong.
//
// ─────────────────────────────────────────────────────────────────────────────
(function () {
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  window.__epd = {

    // ── locating the study and the price series ──────────────────────────────
    epd() {
      const cwc = window.TradingViewApi._chartWidgetCollection;
      const m = cwc.activeChartWidget.value().model();
      return m.panes()[0].sourcesByGroup().all()
        .find(s => { try { return /EPD|Earnings Print/i.test(s.title()); } catch (e) { return false; } });
    },
    // NEVER match a TradingView source by its MINIFIED CLASS NAME. Those names are
    // regenerated on every bundle ship: this used to be `s.constructor.name === 'bs'`
    // and on 2026-08-31 the build renamed it to `'Cs'`, so main() returned undefined
    // and every single bar read threw. Use the stable model API, then fall back to the
    // semantic `isMainSeries()` predicate - never to a class name.
    main() {
      const cwc = window.TradingViewApi._chartWidgetCollection;
      const m = cwc.activeChartWidget.value().model();
      try { const ms = m.mainSeries(); if (ms && ms.data) return ms; } catch (e) {}
      return m.panes()[0].sourcesByGroup().all()
        .find(s => { try { return typeof s.isMainSeries === 'function' && s.isMainSeries(); } catch (e) { return false; } });
    },
    ready() { try { return this.epd().status().type === 2; } catch (e) { return false; } },

    // Ticker only, upper case, no exchange prefix. Compare with === , never indexOf.
    tick() {
      const c = (window.TradingViewApi.activeChart().symbol() || '').toUpperCase();
      return c.indexOf(':') >= 0 ? c.split(':')[1] : c;
    },

    // ── inputs ───────────────────────────────────────────────────────────────
    set(o) {
      const c = this.epd().properties().inputs.childs();
      Object.keys(o).forEach(k => c[k].setValue(o[k]));
    },
    read(ks) {
      const c = this.epd().properties().inputs.childs();
      const o = {};
      ks.forEach(k => { try { o[k] = c[k].value(); } catch (e) { o[k] = 'ERR'; } });
      return o;
    },
    // Every input with its human title and current value - use this after ANY
    // input-list change instead of trusting a remembered index.
    inputMeta() {
      const md = this.epd().metaInfo(), c = this.epd().properties().inputs.childs(), out = [];
      (md.inputs || []).forEach(inp => {
        let v = '?'; try { v = c[inp.id].value(); } catch (e) {}
        out.push({ id: inp.id, name: inp.name, val: v });
      });
      return out;
    },

    // ── WAIT FOR A REAL RECOMPUTE ────────────────────────────────────────────
    // status().type goes 2 -> 1 (busy) -> 2 (ready). A blind sleep returns the
    // PREVIOUS configuration's numbers. Check the returned `timeout` flag.
    async recompute(obj, opt) {
      opt = opt || {};
      const minMs = opt.minMs || 2200, maxMs = opt.maxMs || 60000, settleN = opt.settleN || 8;
      this.set(obj);
      const t0 = performance.now();
      let sawBusy = false, stable = 0;
      while (performance.now() - t0 < maxMs) {
        await sleep(50);
        let st = -1; try { st = this.epd().status().type; } catch (e) {}
        if (st === 1) { sawBusy = true; stable = 0; }
        else if (st === 2) { stable++; }
        else { stable = 0; }
        const el = performance.now() - t0;
        if (sawBusy && stable >= settleN && el > minMs) return { ms: Math.round(el), busy: true };
        if (!sawBusy && el > 5000 && stable >= settleN) return { ms: Math.round(el), busy: false };
      }
      return { ms: Math.round(performance.now() - t0), busy: sawBusy, timeout: true };
    },

    // ── SYMBOL, ASSERTED ─────────────────────────────────────────────────────
    // Trap 2. The plain callback fires (or times out) without the chart having
    // changed. Poll the actual symbol instead and retry.
    async setSymHard(sym, maxTries) {
      const want = sym.toUpperCase();
      for (let k = 0; k < (maxTries || 4); k++) {
        try { window.TradingViewApi.activeChart().setSymbol(sym); } catch (e) {}
        const t0 = performance.now();
        while (performance.now() - t0 < 9000) {
          await sleep(200);
          if (this.tick() === want) return { ok: true, tries: k + 1, sym: this.tick() };
        }
      }
      return { ok: false, sym: this.tick() };
    },

    // ── HISTORY, DEEP ────────────────────────────────────────────────────────
    // Trap 3. THIS is the call that forces the fetch, and it works in-page - the
    // MCP chart_set_visible_range route is not needed and snaps to a narrower
    // window anyway. On CSCO this took 2,834 bars -> 11,226.
    loadRange(from, to) {
      try {
        const m = window.TradingViewApi._chartWidgetCollection.activeChartWidget.value().model();
        m.mainSeries().loadDataTo({ type: 'time-range', from: from, to: to });
        return true;
      } catch (e) { return String(e); }
    },

    // Settle on THIS symbol: bars stable, labels stable, study ready.
    async waitStable(wantSym, maxMs) {
      const want = (wantSym || '').toUpperCase();
      const t0 = performance.now();
      let stable = 0, lastN = -1, lastL = -1, n = -1, L = -1, first = null;
      while (performance.now() - t0 < (maxMs || 25000)) {
        await sleep(150);
        let ok = false;
        try {
          const symOk = !want || this.tick() === want;
          const b = this.main().data().bars();
          n = b.size(); first = n ? b.valueAt(b.firstIndex())[0] : null;
          L = this.labels().length;
          ok = symOk && this.epd().status().type === 2 && n > 0;
        } catch (e) {}
        if (ok && n === lastN && L === lastL) stable++; else stable = 0;
        lastN = n; lastL = L;
        if (stable >= 10) return { ms: Math.round(performance.now() - t0), n, firstTs: first,
          first: first ? new Date(first * 1000).toISOString().slice(0, 16) : null, labels: L };
      }
      return { ms: Math.round(performance.now() - t0), settled: false, n, firstTs: first,
        first: first ? new Date(first * 1000).toISOString().slice(0, 16) : null, labels: L };
    },

    // ── SETTLE THAT PROVES THE DATA BELONGS TO THE NEW SYMBOL ────────────────
    // waitStable() above only waits for bar/label COUNTS to stop changing. Straight
    // after a symbol switch those counts are still the PREVIOUS symbol's, and they
    // are perfectly stable, so waitStable can settle on stale data and every number
    // that follows describes the wrong ticker. Observed 2026-08-31: the chart read
    // CBRS while serving NBIS's 13,638 bars, zero labels and an empty earnings feed.
    //
    // This settle instead requires the EARNINGS FEED for this symbol to be non-empty
    // and to have stopped changing - a condition stale data cannot satisfy, because
    // the previous symbol's feed is a different event list.
    // `settled === false` means NO DATA. It must never be read as "no trade".
    async waitFeed(wantSym, wideFrom, wideTo, maxMs) {
      const want = (wantSym || '').toUpperCase();
      const t0 = performance.now();
      let stable = 0, lastFp = null, n = -1, first = null, nEv = 0;
      while (performance.now() - t0 < (maxMs || 40000)) {
        await sleep(250);
        let ok = false, fp = null;
        try {
          if (this.tick() !== want) { stable = 0; continue; }
          const b = this.main().data().bars();
          n = b.size(); first = n ? b.valueAt(b.firstIndex())[0] : null;
          const ev = this.earnings(wideFrom, wideTo);
          nEv = (ev.events || []).length;
          fp = (ev.events || []).map(x => x.ts).join(',') + '|' + n + '|' + this.labels().length;
          ok = this.epd().status().type === 2 && n > 0 && nEv > 0;
        } catch (e) {}
        if (ok && fp === lastFp) stable++; else stable = 0;
        lastFp = fp;
        if (stable >= 8) return { ms: Math.round(performance.now() - t0), n, nEv, firstTs: first,
          first: first ? new Date(first * 1000).toISOString().slice(0, 16) : null,
          labels: this.labels().length, settled: true };
      }
      return { ms: Math.round(performance.now() - t0), n, nEv, firstTs: first,
        first: first ? new Date(first * 1000).toISOString().slice(0, 16) : null, settled: false };
    },

    // ── LABELS ───────────────────────────────────────────────────────────────
    // MUST read _primitivesDataById. The materialised `_primitiveById` only holds
    // labels currently ON SCREEN - on CRWV it gave 8 where the truth was 9.
    // t=text, tt=tooltip, y=price, x=index into _graphics._indexes -> bar index.
    // Vocabulary: ▲/▼ print, ⟲ re-entry ARMED (not an entry), ✓ entry,
    // T1/T2/T3 (combined as "T1·T2"), ✕ stop, ⊘ conviction/session-end, HOLD, !
    labels() {
      const g = this.epd()._graphics;
      const idx = g._indexes || [];
      const out = [];
      g.dwglabels().forEach(inner => {
        inner.forEach(coll => {
          const pd = coll._primitivesDataById;
          if (!pd) return;
          pd.forEach(v => out.push({ id: v.id, xi: v.x,
            bar: (idx[v.x] !== undefined ? idx[v.x] : null),
            text: v.t, price: v.y, tooltip: v.tt || '' }));
        });
      });
      out.sort((a, b) => (a.bar - b.bar) || (a.id - b.id));
      return out;
    },

    // ── BARS ─────────────────────────────────────────────────────────────────
    bars() {
      const b = this.main().data().bars();
      return { n: b.size(),
        first: new Date(b.valueAt(b.firstIndex())[0] * 1000).toISOString(),
        last: new Date(b.valueAt(b.lastIndex())[0] * 1000).toISOString() };
    },
    barAt(barIdx) {
      try {
        const v = this.main().data().bars().valueAt(barIdx);
        return v ? { ts: v[0], open: v[1], high: v[2], low: v[3], close: v[4] } : null;
      } catch (e) { return null; }
    },
    barByTime(ts) {
      const b = this.main().data().bars();
      for (let i = b.firstIndex(); i <= b.lastIndex(); i++) {
        const v = b.valueAt(i);
        if (v && v[0] === ts) return { i, open: v[1], high: v[2], low: v[3], close: v[4] };
      }
      return null;
    },

    // ── PLOT SERIES - this is how you find the + ADD markers ─────────────────
    // Trap 1. Resolve by TITLE; the index moves if plots change.
    // _data.valueAt(i) = [time, plot_0, plot_1, ...]  so plot_N is at index N+1.
    plotIndexByTitle(rx) {
      const md = this.epd().metaInfo();
      for (let i = 0; i < md.plots.length; i++) {
        const p = md.plots[i], stl = md.styles && md.styles[p.id];
        const title = stl && stl.title ? stl.title : (p.id || '');
        if (rx.test(title)) return { plot: i, valueIdx: i + 1, title };
      }
      return null;
    },
    addFills(fromTs, toTs) {
      const info = this.plotIndexByTitle(/ADD: count so far/i);
      if (!info) return { err: 'add-count plot not found' };
      const d = this.epd()._data, out = [];
      let prev = null;
      for (let i = d.firstIndex(); i <= d.lastIndex(); i++) {
        const v = d.valueAt(i);
        if (!v) continue;
        const ts = v[0], cnt = v[info.valueIdx];
        if (ts < fromTs || ts > toTs) { prev = cnt; continue; }
        if (prev !== null && cnt > prev) {
          const bar = this.barByTime(ts);
          out.push({ ts, time: new Date(ts * 1000).toISOString().slice(0, 16), close: bar ? bar.close : null });
        }
        prev = cnt;
      }
      return { plot: info.title, fills: out };
    },

    // ── EARNINGS DATES ───────────────────────────────────────────────────────
    // On a DAILY chart this returns the full history (including future events).
    // On 1-minute the daily timestamp is a SCHEDULED time (often 13:30Z) - the
    // exact release minute comes from the 1-minute chart's own source. Note the
    // confirm often PRECEDES the stamped release by a few minutes: the wire hits
    // the tape before TradingView stamps it. That is normal, not an error.
    earnings(fromTs, toTs) {
      const cwc = window.TradingViewApi._chartWidgetCollection;
      const m = cwc.activeChartWidget.value().model();
      // Same brittleness class as main(): do not identify a source by a loose string
      // another source can also satisfy. The study is called "Earnings Print Detector",
      // so a bare /Earnings/i can select the INDICATOR instead of the events feed and
      // silently return zero release dates. Exclude the study, then match the feed.
      const study = this.epd();
      const cands = m.panes()[0].sourcesByGroup().all()
        .filter(s => s !== study)
        .filter(s => { try { return /Earnings/i.test(s.title()); } catch (e) { return false; } });
      const es = cands.find(s => { try { return /^earnings$/i.test(String(s.title()).trim()); } catch (e) { return false; } })
        || cands[0];
      if (!es) return { err: 'no earnings source' };
      try {
        const d = es._data, out = [];
        for (let i = d.firstIndex(); i <= d.lastIndex(); i++) {
          const v = d.valueAt(i);
          if (!v || !v[0]) continue;
          if (fromTs && (v[0] < fromTs || v[0] > toTs)) continue;
          out.push({ ts: v[0], iso: new Date(v[0] * 1000).toISOString().slice(0, 16) });
        }
        return { n: out.length, events: out };
      } catch (e) { return { err: String(e) }; }
    },

    // Batch earnings verification - run on a DAILY chart. [n] symbols in 90s.
    earnScan(syms, fromTs, toTs) {
      window.__es = { status: 'running', done: 0, total: syms.length, hit: [], miss: [], err: [] };
      (async () => {
        for (const s of syms) {
          try {
            await new Promise(res => {
              let d = false;
              window.TradingViewApi.activeChart().setSymbol(s, () => { if (!d) { d = true; res(); } });
              setTimeout(() => { if (!d) { d = true; res(); } }, 6000);
            });
            await sleep(1300);
            const er = this.earnings(fromTs, toTs);
            if (er.err) window.__es.err.push(s + ':' + er.err);
            else if (er.n) window.__es.hit.push({ sym: s, dates: er.events.map(e => e.iso) });
            else window.__es.miss.push(s);
          } catch (e) { window.__es.err.push(s + ':' + String(e)); }
          window.__es.done++;
        }
        window.__es.status = 'done';
      })();
      return 'scanning ' + syms.length;
    },

    // ── PRICE EVERY TRADE IN A WINDOW ────────────────────────────────────────
    // One row per ✓ confirmation, because each deploys capital independently.
    // A ⟲ only ARMS a re-entry; the trade begins at the ✓ that follows it.
    //
    // Model (the owner's, confirmed 2026-08-24):
    //   size    100 units at the ✓ close, +50 at EACH + add close
    //   exits   30% at EACH take-profit reached, remainder at the exit marker
    //   prices  pct    = every fill at the candle CLOSE   (what the owner asked for)
    //           pctLvl = fills at the actual T/stop LEVELS (the conservative read)
    //   result  % on TOTAL capital deployed = dir * (avgExit - blended) / blended
    //
    // Same-bar ordering is safe: the TP label is created before the exit label in
    // the Pine, so a lower id sorts first and the TP is banked before the remainder.
    priceAll(fromTs, toTs, opt) {
      opt = opt || {};
      const TP = opt.tpFrac === undefined ? 0.30 : opt.tpFrac;
      const L = this.labels().map(l => {
        const b = this.barAt(l.bar);
        return Object.assign({}, l, { ts: b ? b.ts : null, close: b ? b.close : null });
      }).filter(l => l.ts !== null && l.ts >= fromTs && l.ts <= toTs);

      const trades = [];
      let i = 0;
      while (i < L.length) {
        if (L[i].text !== '✓') { i++; continue; }
        const conf = L[i];
        const mDir = /CONTINUATION CONFIRMED\s+(LONG|SHORT)/i.exec(conf.tooltip);
        const mEnt = /entry \(this close\)\s+([\d.]+)/i.exec(conf.tooltip);
        const mTgt = /1R target\s+([\d.]+)/i.exec(conf.tooltip);
        if (!mDir || !mEnt || !mTgt) { trades.push({ bad: 'parse', ts: conf.ts }); i++; continue; }

        const dir = /LONG/i.test(mDir[1]) ? 1 : -1;
        const entry = parseFloat(mEnt[1]);
        const R = Math.abs(entry - parseFloat(mTgt[1]));

        let j = i + 1, exit = null;
        const tps = [];
        for (; j < L.length; j++) {
          const t = L[j].text || '';
          if (t === '✕' || t === '⊘') { exit = L[j]; break; }
          if (t === '✓') break;
          const m = t.match(/T[1234]/g);          // "T1·T2" means BOTH on one bar
          if (m) m.forEach(k => tps.push({ k, lab: L[j] }));
        }

        const endTs = exit ? exit.ts : toTs;
        const fills = (this.addFills(conf.ts, endTs).fills || []);
        let shares = 100 / entry, capital = 100;
        fills.forEach(f => { if (f.close) { shares += 50 / f.close; capital += 50; } });
        const blended = capital / shares;

        let outFrac = 0, avgC = 0, avgL = 0;
        const tpRows = [];
        tps.slice(0, 4).forEach(t => {
          // v10.0 - the ladder is [n], so the R figure in the
          // tooltip is fractional now. The old whole-number pattern would silently stop
          // matching and fall back to the bar close for EVERY target.
          // v10.0 - the rungs are [n], so the R figure in
          // each tooltip is fractional now. The old whole-number pattern would stop
          // matching and every target price would fall back to the bar close.
          const mm = new RegExp('TP' + t.k[1] + '\\s+\\+[\\d.]+R\\s+([\\d.]+)').exec(t.lab.tooltip || '');
          const lvl = mm ? parseFloat(mm[1]) : t.lab.close;
          tpRows.push({ t: t.k, close: t.lab.close, level: lvl, ts: t.lab.ts });
          avgC += TP * t.lab.close; avgL += TP * lvl; outFrac += TP;
        });

        const rem = 1 - outFrac;
        let openTrade = false, exLvl = null;
        if (rem > 1e-9) {
          if (!exit) openTrade = true;
          else {
            const ms = /entry\s+[\d.]+\s+stop\s+([\d.]+)/.exec(exit.tooltip || '');
            exLvl = (exit.text === '✕' && ms) ? parseFloat(ms[1]) : exit.close;
            avgC += rem * exit.close; avgL += rem * exLvl;
          }
        }

        const rec = {
          tConf: new Date(conf.ts * 1000).toISOString().slice(0, 16), tsConf: conf.ts,
          dir: dir > 0 ? 'LONG' : 'SHORT', entry, R: +R.toFixed(4),
          nAdds: fills.length, addTimes: fills.map(f => f.time + '@' + f.close),
          capital, blended: +blended.toFixed(4),
          nTP: tpRows.length, tps: tpRows,
          exitT: exit ? new Date(exit.ts * 1000).toISOString().slice(0, 16) : null,
          exitReason: exit ? ((exit.tooltip || '').split('\n')[0] || '').trim() : null,
          exitClose: exit ? exit.close : null, exitLevel: exLvl,
          open: openTrade
        };
        if (!openTrade) {
          rec.avgExit = +avgC.toFixed(4);
          rec.pct    = +(100 * dir * (avgC - blended) / blended).toFixed(4);
          rec.pctLvl = +(100 * dir * (avgL - blended) / blended).toFixed(4);
          rec.rMult  = +((dir * (avgC - blended)) / R).toFixed(4);
        }
        trades.push(rec);
        i = exit ? (j + 1) : (i + 1);
      }
      return trades;
    },

    // ── BATCH ────────────────────────────────────────────────────────────────
    // jobs: [{sym, earnIso, from, to, expectTs, minFirstTs}]
    // Poll window.__bt. Read `covered` on EVERY row before believing a zero.
    runBatch(jobs, opt) {
      opt = opt || {};
      const E = this;
      window.__bt = { status: 'running', done: 0, total: jobs.length, out: [] };
      (async () => {
        for (const jb of jobs) {
          const rec = { sym: jb.sym, expect: jb.earnIso };
          try {
            const sw = await E.setSymHard(jb.sym);
            rec.symOK = sw.ok; rec.symActual = sw.sym; rec.symTries = sw.tries;
            if (!sw.ok) { rec.fail = 'symbol-switch-failed'; window.__bt.out.push(rec); window.__bt.done++; continue; }

            await sleep(300);
            E.loadRange(jb.from, jb.to);
            let w = await E.waitStable(jb.sym, 25000);
            if (!w.firstTs || w.firstTs > jb.minFirstTs) {   // second pull if history fell short
              E.loadRange(jb.from, jb.to);
              const w2 = await E.waitStable(jb.sym, 25000);
              if (w2.firstTs && (!w.firstTs || w2.firstTs < w.firstTs)) w = w2;
              rec.retried = true;
            }
            // re-assert AFTER the load - this is where the stale-symbol bug hid
            if (E.tick() !== jb.sym.toUpperCase()) {
              rec.fail = 'symbol-drifted:' + E.tick(); window.__bt.out.push(rec); window.__bt.done++; continue;
            }
            rec.wait = w; rec.firstBar = w.first; rec.nBars = w.n; rec.settled = w.settled !== false;

            const er = E.earnings(jb.from, jb.to);
            rec.earn = er.events ? er.events.map(e => e.iso) : er;
            let f = jb.from, t = jb.to;
            if (er.events && er.events.length) {
              let best = er.events[0];
              if (jb.expectTs) best = er.events.reduce((a, b) =>
                Math.abs(b.ts - jb.expectTs) < Math.abs(a.ts - jb.expectTs) ? b : a);
              rec.release = best.iso; rec.releaseTs = best.ts;
              f = best.ts - 3600; t = best.ts + 2 * 86400;
            }
            // ★ COVERAGE AUDIT - an empty trade list is a real "no trade" ONLY if this is true
            rec.covered = rec.releaseTs ? (w.firstTs !== null && w.firstTs <= rec.releaseTs - 1800) : false;
            rec.leadHrs = rec.releaseTs && w.firstTs ? +(((rec.releaseTs - w.firstTs) / 3600).toFixed(1)) : null;
            rec.win = [new Date(f * 1000).toISOString().slice(0, 16), new Date(t * 1000).toISOString().slice(0, 16)];
            rec.trades = rec.covered ? E.priceAll(f, t) : [];
            rec.symFinal = E.tick();
            rec.nLabels = E.labels().length;
          } catch (e) { rec.err = String(e); }
          window.__bt.out.push(rec); window.__bt.done++;
          await sleep(400);
        }
        window.__bt.status = 'done';
      })();
      return 'batch started: ' + jobs.length;
    }
  };

  return 'EPD backtest harness v2 installed';
})()

// ─── HOW A RUN GOES ──────────────────────────────────────────────────────────
//
//  1. verify the study:      window.__epd.epd().title()   -> "EPD v8.0"
//  2. set the measurement:   await window.__epd.recompute({in_0:true, in_1:false})
//     (ui_evaluate does NOT await a promise - assign to a global and poll it)
//  3. dates, on a DAILY chart:  window.__epd.earnScan([...], from, to)
//  4. back to 1-minute, then:   window.__epd.runBatch(jobs)
//  5. AUDIT BEFORE ANALYSING:
//       - every row has fail == undefined and symFinal === sym
//       - no two rows share (nBars, firstBar) unless you have checked why
//       - covered === true, or the row is "no data", NOT "no trade"
//
// ─── AFTER ANY CHANGE, RE-GATE ON CBRS ───────────────────────────────────────
//   chart_set_symbol CBRS, then priceAll(1786492800, 1786665600)
//   -> pct [n], pctLvl [n], nAdds 1, blended [n], capital 150
