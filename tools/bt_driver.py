"""bt_driver - run the backtest harness over a roster, from outside the page.

WHY THIS EXISTS. The v8-v10 runs drove tools/backtest-harness.js by hand through
single ui_evaluate calls and transcribed the output. That works, and it also means
every run re-invents the glue. This is the glue, written once:

  * pastes the harness WHOLE (its own rule: never rebuild it) and uses its own
    pricing, add-fill, symbol-switch, history-load and settle routines untouched;
  * adds only what the harness does not record but the v10 report used - the
    label sequence (so each trade's PRINT time can be gated against the release)
    and the liquidity screen that excluded HRL;
  * talks to TradingView over the desktop app's debug port and writes the raw
    result to disk, so nothing large passes through a chat tool.

THE CHART IS THE OWNER'S LIVE TRADING TERMINAL. The earnings watcher must be stopped
first (it writes to the same study's inputs); the arrival symbol, timeframe and
inputs are recorded and restored, and the run refuses to start if HOLD is on.

    python tools/bt_driver.py --roster roster.json --out raw.json [--atr 1.80] [--only A,B]
"""
import argparse, asyncio, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tv_push                                                      # noqa: E402

HARNESS = os.path.join(HERE, "backtest-harness.js")

# Orchestration only. Every measurement below is a call into the pasted harness.
WRAPPER = r"""
(function(){
  const E = window.__epd, sleep = ms => new Promise(r => setTimeout(r, ms));
  window.__epd2 = {
    // The label sequence in a window: text + bar time. The v10 report gated every
    // trade on its PRINT marker (the triangle before the confirm) being within a
    // few hours of the release; the harness prices trades but keeps no labels.
    labSeq(fromTs, toTs) {
      return E.labels().map(l => { const b = E.barAt(l.bar);
        return { t: l.text, ts: b ? b.ts : null }; })
        .filter(x => x.ts !== null && x.ts >= fromTs && x.ts <= toTs);
    },
    // HRL was dropped because 22 of 31 post-print bars had zero range. Same screen:
    // the first 90 minutes from the release.
    liq(fromTs) {
      // E.bars() is only a summary {n, first, last}; read the raw series the way
      // the harness's own barAt() does. valueAt = [time, open, high, low, close, vol].
      const s = E.main().data().bars(), w = [];
      for (let i = s.firstIndex(); i <= s.lastIndex(); i++) {
        const v = s.valueAt(i);
        if (v && v[0] >= fromTs && v[0] < fromTs + 90 * 60)
          w.push({ ts: v[0], high: v[2], low: v[3], close: v[4], volume: v[5] });
      }
      if (!w.length) return { n: 0 };
      const med = a => { const s = a.slice().sort((p, q) => p - q); return s.length ? s[s.length >> 1] : null; };
      return { n: w.length,
               zero: w.filter(x => x.high === x.low).length,
               medVol: med(w.map(x => x.volume || 0)),
               medRangePct: med(w.map(x => x.close ? 100 * (x.high - x.low) / x.close : 0)) };
    },
    // The v10 report's ceiling section: the best price each trade reached BEFORE its
    // own initial stop was touched, and whether that peak came after the system had
    // already exited. The stop bar's own favourable extreme is excluded - inside a
    // minute bar the order of high and low is unknowable, so it cannot be credited.
    paths(trades, toTs) {
      const s = E.main().data().bars(), bars = [];
      for (let i = s.firstIndex(); i <= s.lastIndex(); i++) {
        const v = s.valueAt(i);
        if (v && v[0] <= toTs) bars.push([v[0], v[2], v[3], v[4]]);
      }
      return (trades || []).map(tr => {
        if (!tr || tr.bad || !tr.R) return null;
        const d = tr.dir === 'LONG' ? 1 : -1, stop = tr.entry - d * tr.R;
        let best = null, bestT = null, stopT = null, lastC = null;
        for (const b of bars) {
          if (b[0] <= tr.tsConf) continue;               // the confirm bar cannot count
          const fav = d > 0 ? b[1] : b[2], adv = d > 0 ? b[2] : b[1];
          if ((d > 0 && adv <= stop) || (d < 0 && adv >= stop)) { stopT = b[0]; break; }
          if (best === null || d * (fav - best) > 0) { best = fav; bestT = b[0]; }
          lastC = b[3];
        }
        const exitTs = tr.exitT ? Date.parse(tr.exitT + ':00Z') / 1000 : null;
        return { best, bestT, stopT, lastC,
                 mfeR: best === null ? null : +((d * (best - tr.entry)) / tr.R).toFixed(4),
                 peakAfterExit: (bestT && exitTs) ? bestT > exitTs : null };
      });
    },
    // WHAT IS ACTUALLY LOADED, not what the chart says it is showing. The harness
    // asserts the chart's symbol STRING, which updates even when the data does not.
    // On 2026-09-29 a bare "SAIL" resolved to Steel Authority of India (NSE), the
    // data connection broke, and 32 following symbols each showed that stock's 300
    // stale bars under their own names - every name check passed. This reads the
    // loaded series' own identity and freshness instead.
    loaded() {
      const ms = E.main(), si = (ms && ms.symbolInfo && ms.symbolInfo()) || {};
      const b = ms.data().bars(), l = b.valueAt(b.lastIndex()), f = b.valueAt(b.firstIndex());
      return { name: String(si.name || '').toUpperCase(), exch: String(si.exchange || ''),
               listed: String(si.listed_exchange || si.exchange || ''), desc: si.description || '',
               n: b.size(), firstTs: f ? f[0] : null, lastTs: l ? l[0] : null };
    },
    US: /^(NASDAQ|NYSE|AMEX|NYSE ARCA|NYSE AMERICAN|ARCA|CBOE|CBOE ONE|BATS|OTC)/i,
    async setSymVerified(tv, bare) {
      // The harness's own switch, driven with the QUALIFIED symbol, then the tick
      // compared bare - the harness strips the prefix, so this is its assertion,
      // unchanged, plus the data check it lacks.
      for (let k = 0; k < 4; k++) {
        try { window.TradingViewApi.activeChart().setSymbol(tv); } catch (e) {}
        const t0 = performance.now();
        while (performance.now() - t0 < 9000) {
          await sleep(250);
          if (E.tick() !== bare) continue;
          const L = this.loaded();
          if (L.name === bare && this.US.test(L.exch) && L.lastTs &&
              (Date.now() / 1000 - L.lastTs) < 4 * 86400)
            return { ok: true, tries: k + 1, loaded: L };
        }
      }
      return { ok: false, loaded: this.loaded() };
    },
    run(jobs) {
      window.__bt2 = { status: 'running', done: 0, total: jobs.length, out: [] };
      (async () => {
        let prevSig = null;
        for (const jb of jobs) {
          const rec = { sym: jb.sym, tv: jb.tv, grp: jb.grp, expect: jb.earnIso };
          try {
            const sw = await this.setSymVerified(jb.tv || jb.sym, jb.sym.toUpperCase());
            rec.symOK = sw.ok; rec.symActual = E.tick(); rec.loadedAs = sw.loaded;
            if (!sw.ok) { rec.fail = 'symbol-not-loaded:' + JSON.stringify(sw.loaded); window.__bt2.out.push(rec); window.__bt2.done++; continue; }
            await sleep(300);
            E.loadRange(jb.from, jb.to);
            let w = await E.waitStable(jb.sym, 25000);
            if (!w.firstTs || w.firstTs > jb.minFirstTs) {
              E.loadRange(jb.from, jb.to);
              const w2 = await E.waitStable(jb.sym, 25000);
              if (w2.firstTs && (!w.firstTs || w2.firstTs < w.firstTs)) w = w2;
              rec.retried = true;
            }
            if (E.tick() !== jb.sym.toUpperCase()) {
              rec.fail = 'symbol-drifted:' + E.tick(); window.__bt2.out.push(rec); window.__bt2.done++; continue;
            }
            // Re-assert on the DATA after the load, and refuse a series identical to
            // the previous symbol's - the harness's own audit rule ("no two rows share
            // nBars and firstBar"), enforced here instead of left to a later read.
            const L2 = this.loaded();
            rec.loadedAfter = L2;
            if (L2.name !== jb.sym.toUpperCase() || !this.US.test(L2.exch)) {
              rec.fail = 'data-not-this-symbol:' + L2.exch + ':' + L2.name; window.__bt2.out.push(rec); window.__bt2.done++; continue;
            }
            const sig = L2.n + '|' + L2.firstTs + '|' + L2.lastTs;
            if (sig === prevSig) {
              rec.fail = 'identical-series-to-previous-symbol'; window.__bt2.out.push(rec); window.__bt2.done++; continue;
            }
            prevSig = sig;
            rec.firstBar = w.first; rec.firstTs = w.firstTs; rec.nBars = w.n; rec.settled = w.settled !== false;
            const er = E.earnings(jb.from, jb.to);
            rec.earn = er.events ? er.events.map(e => e.iso) : er;
            let f = jb.from, t = jb.to;
            if (er.events && er.events.length) {
              let best = er.events.reduce((a, b) =>
                Math.abs(b.ts - jb.expectTs) < Math.abs(a.ts - jb.expectTs) ? b : a);
              rec.release = best.iso; rec.releaseTs = best.ts;
              f = best.ts - 3600; t = best.ts + 2 * 86400;
            }
            rec.covered = rec.releaseTs ? (w.firstTs !== null && w.firstTs <= rec.releaseTs - 1800) : false;
            rec.win = [f, t];
            rec.trades = rec.covered ? E.priceAll(f, t) : [];
            rec.labSeq = rec.covered ? this.labSeq(f, t) : [];
            rec.paths = rec.covered ? this.paths(rec.trades, t) : [];
            rec.liq = rec.releaseTs ? this.liq(rec.releaseTs) : null;
            rec.symFinal = E.tick();
          } catch (e) { rec.err = String(e); }
          window.__bt2.out.push(rec); window.__bt2.done++;
          await sleep(400);
        }
        window.__bt2.status = 'done';
      })();
      return 'started ' + jobs.length;
    }
  };
  return 'wrapper installed';
})()
"""

STATE_JS = r"""
(function(){
  try {
    const E = window.__epd;
    const st = E.epd();
    return JSON.stringify({ ok: true, study: String(st.title()), sym: E.tick(), full: window.TradingViewApi.activeChart().symbol(),
      res: window.TradingViewApi.activeChart().resolution(),
      inputs: E.read(['in_0', 'in_1', 'in_4', 'in_70']) });
  } catch (e) { return JSON.stringify({ ok: false, err: String(e) }); }
})()
"""


async def ev(ws, js, ident, timeout=60):
    await ws.send(json.dumps({"id": ident, "method": "Runtime.evaluate",
                              "params": {"expression": js, "returnByValue": True,
                                         "awaitPromise": False}}))
    t0 = time.time()
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(1, timeout - (time.time() - t0))))
        if msg.get("id") == ident:
            res = msg.get("result", {}).get("result", {})
            v = res.get("value")
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except Exception:
                    return v
            return v if v is not None else msg


async def run(a):
    import websockets
    roster = json.load(open(a.roster, encoding="utf-8-sig"))   # tolerate a BOM
    jobs = [j for j in roster if not a.only or j["sym"] in a.only]
    log = open(a.out + ".log", "a", encoding="utf-8")

    def say(m):
        line = "%s  %s" % (time.strftime("%H:%M:%S"), m)
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    ws = await websockets.connect(tv_push.find_chart_target(), max_size=256 * 1024 * 1024)
    await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
    await asyncio.wait_for(ws.recv(), 10)
    i = 10
    i += 1; r = await ev(ws, open(HARNESS, encoding="utf-8").read(), i)
    say("harness: %s" % r)
    i += 1; say("wrapper: %s" % await ev(ws, WRAPPER, i))
    i += 1; arrival = await ev(ws, STATE_JS, i)
    say("arrival: %s" % json.dumps(arrival))
    if not arrival.get("ok") or not str(arrival.get("study", "")).startswith("EPD v10"):
        say("REFUSED - study not found or not v10"); return 2
    if arrival["inputs"].get("in_1") is not False:
        say("REFUSED - HOLD (in_1) is ON; every number would be wrong"); return 2

    want_atr = a.atr if a.atr is not None else arrival["inputs"]["in_70"]
    if arrival["inputs"]["in_70"] != want_atr:
        i += 1
        await ev(ws, "window.__epd.set({in_70: %s}); 'set'" % want_atr, i)
        await asyncio.sleep(6)                     # let the study recompute
    i += 1; st = await ev(ws, STATE_JS, i)
    say("measuring with: %s" % json.dumps(st["inputs"]))

    i += 1; say(await ev(ws, "window.__epd2.run(%s)" % json.dumps(jobs), i))
    last = -1
    while True:
        await asyncio.sleep(8)
        i += 1
        p = await ev(ws, "JSON.stringify({s: window.__bt2.status, d: window.__bt2.done, "
                         "n: window.__bt2.total, last: (window.__bt2.out.slice(-1)[0]||{}).sym})", i)
        if p.get("d") != last:
            say("progress %s/%s  last=%s" % (p.get("d"), p.get("n"), p.get("last")))
            last = p.get("d")
        if p.get("s") == "done":
            break
    i += 1
    out = await ev(ws, "JSON.stringify(window.__bt2.out)", i, timeout=180)
    payload = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "inputs": st["inputs"],
               "study": st["study"], "rows": out}
    json.dump(payload, open(a.out, "w", encoding="utf-8"), indent=1)
    say("wrote %s  (%d rows, %d trades)" % (a.out, len(out),
                                           sum(len(r.get("trades") or []) for r in out)))

    # Restore exactly what the chart arrived with.
    if not a.no_restore:
        i += 1
        await ev(ws, "window.__epd.set({in_70: %s}); 'ok'" % arrival["inputs"]["in_70"], i)
        i += 1
        await ev(ws, "window.TradingViewApi.activeChart().setSymbol(%s); 'ok'"
                 % json.dumps(a.restore_symbol or arrival.get("full") or arrival["sym"]), i)
        await asyncio.sleep(4)
        i += 1
        say("restored: %s" % json.dumps(await ev(ws, STATE_JS, i)))
    await ws.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atr", type=float, default=None, help="in_70 for this pass")
    ap.add_argument("--only", type=lambda s: [x.strip().upper() for x in s.split(",") if x.strip()],
                    default=None)
    ap.add_argument("--restore-symbol", default=None)
    ap.add_argument("--no-restore", action="store_true")
    a = ap.parse_args()
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
