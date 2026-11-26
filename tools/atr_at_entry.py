"""atr_at_entry - recover which measure actually SET the stop on every past trade.

WHY THIS EXISTS. The question:

    Some stops (CoreWeave) look right on the chart and others (PANW, BABA) look
    too wide. Does a different measure set each group, so that the wide pathway
    can be tightened while the other stays as it is?

There are two pathways. The Pine takes the LARGEST of three
measures, trims with a cap, then re-asserts the volatility floor last:

    riskPx = max(structure, earnings move, volatility)
    riskPx = min(riskPx, 0.65 x the whole move)          # trim an absurd stop
    riskPx = max(riskPx, volatility)                     # but never below noise

Because the volatility floor is re-asserted LAST, the test is exact rather than
approximate: a trade is VOLATILITY-DETERMINED if and only if its final R equals
riskAtr. Anything larger was set by structure, by the earnings move, or by the
cap landing above the floor.

    riskAtr = riskAtrMult x atrLive = 1.80 x ta.sma(ta.tr(true), 10)

`atrLive` is a plain 10-bar simple average of true range, so it reproduces
exactly from 1-minute bars - no need to re-implement the rest of the stop logic,
and no need to touch the shipped indicator to instrument it.

This reads bars straight off the TradingView desktop app over its debug port,
which is the only sanctioned price source for this project. It changes the chart
SYMBOL as it walks the list and puts the original symbol back when it finishes.

    python tools/atr_at_entry.py --out backtest/stop-pathway-v10.0.json
"""
import argparse, asyncio, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tv_push                                                       # noqa: E402

RISK_ATR_MULT = 1.80          # pine input riskAtrMult
ATR_TRAIL_LEN = 10            # pine input atrTrailLen
STOP_RETRACE = 0.60           # pine input stopRetrace

# Walks the loaded 1-minute series and returns, for each requested timestamp, the
# 10-bar SMA of true range ENDING AT that bar - which is what ta.sma(ta.tr(true),
# 10) evaluates to on the confirm bar - plus the bar itself and the few bars
# before it, so the print candle's extreme can be recovered too.
JS = r"""
(function(){
  try{
    var cwc = window.TradingViewApi._chartWidgetCollection;
    var m = cwc.activeChartWidget.value().model();
    var main = m.panes().reduce(function(a,p){
      return a.concat(p.sourcesByGroup().all());
    }, []).find(function(s){
      try { return typeof s.isMainSeries === 'function' && s.isMainSeries(); }
      catch(e){ return false; }
    });
    if(!main) return JSON.stringify({ok:false, err:'no main series'});
    var sym = String(main.symbolInfo().name || '');
    var b = main.data().bars();
    var lo = b.firstIndex(), hi = b.lastIndex();
    var ts = [], px = [];
    for (var i = lo; i <= hi; i++){
      var v = b.valueAt(i);
      if (v) { ts.push(v[0]); px.push([v[1], v[2], v[3], v[4]]); }
    }
    var want = __WANT__;              // array of unix seconds
    var out = [];
    for (var k = 0; k < want.length; k++){
      var j = ts.indexOf(want[k]);
      if (j < 0) { out.push({t: want[k], found: false}); continue; }
      // ta.tr(true) uses the previous close, and treats the first bar as h-l.
      var sum = 0, n = 0;
      for (var q = j - (__LEN__ - 1); q <= j; q++){
        if (q < 0) continue;
        var h = px[q][1], l = px[q][2];
        var tr = h - l;
        if (q > 0){
          var pc = px[q-1][3];
          tr = Math.max(tr, Math.abs(h - pc), Math.abs(l - pc));
        }
        sum += tr; n++;
      }
      var pre = [];
      for (var q2 = Math.max(0, j - 4); q2 <= j; q2++){
        pre.push({t: ts[q2], o: px[q2][0], h: px[q2][1], l: px[q2][2], c: px[q2][3]});
      }
      out.push({t: want[k], found: true, atr: n ? sum / n : null, n: n,
                bar: {o: px[j][0], h: px[j][1], l: px[j][2], c: px[j][3]}, pre: pre});
    }
    return JSON.stringify({ok:true, symbol:sym, nBars: ts.length,
                           first: ts[0], last: ts[ts.length-1], rows: out});
  } catch(e){ return JSON.stringify({ok:false, err:String(e)}); }
})()
"""

SET_SYM = r"""
(function(){
  try{
    window.TradingViewApi.activeChart().setSymbol("__SYM__");
    return JSON.stringify({ok:true});
  } catch(e){ return JSON.stringify({ok:false, err:String(e)}); }
})()
"""

# Switching symbol loads only the visible window - about 470 one-minute bars, or
# one session. Every entry here is weeks back, so without this the reader
# returned "0 entries matched" for all of them, which looks exactly like a
# correct answer of "no data" and is not one.
LOAD_RANGE = r"""
(function(){
  try{
    var m = window.TradingViewApi._chartWidgetCollection.activeChartWidget.value().model();
    m.mainSeries().loadDataTo({type:'time-range', from: __FROM__, to: __TO__});
    return JSON.stringify({ok:true});
  } catch(e){ return JSON.stringify({ok:false, err:String(e)}); }
})()
"""

# loadDataTo only ever EXTENDS backwards from what is already on screen, and the
# series caps out around 11,000 one-minute bars. Names that trade heavily in
# extended hours (CoreWeave, Coherent, Lumentum) burn that budget before reaching
# an early-August print, and every one of them reported "0 entries matched" -
# indistinguishable from a correct "no data". Moving the VISIBLE RANGE onto the
# date first makes TradingView fetch that window directly instead.
GOTO = r"""
(function(){
  try{
    var c = window.TradingViewApi.activeChart();
    c.setVisibleRange({from: __FROM__, to: __TO__});
    return JSON.stringify({ok:true});
  } catch(e){
    try{
      var m = window.TradingViewApi._chartWidgetCollection.activeChartWidget.value().model();
      m.gotoTime(__FROM__);
      return JSON.stringify({ok:true, via:'gotoTime'});
    } catch(e2){ return JSON.stringify({ok:false, err:String(e2)}); }
  }
})()
"""


async def _eval(ws, js, ident):
    await ws.send(json.dumps({"id": ident, "method": "Runtime.evaluate",
                              "params": {"expression": js, "returnByValue": True,
                                         "awaitPromise": False}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == ident:
            res = msg.get("result", {}).get("result", {})
            if res.get("type") == "string":
                return json.loads(res["value"])
            return {"ok": False, "err": json.dumps(msg)[:300]}


async def run(jobs, settle=3.0, tries=14):
    """jobs: {symbol: [unix seconds, ...]}. Returns {symbol: result}."""
    import websockets
    url = tv_push.find_chart_target()
    out, ident = {}, 100
    async with websockets.connect(url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
        await ws.recv()
        original = None
        for sym, want in jobs.items():
            ident += 1
            r0 = await _eval(ws, JS.replace("__WANT__", "[]").replace("__LEN__", "1"), ident)
            if original is None and r0.get("ok"):
                original = r0.get("symbol")
            ident += 1
            await _eval(ws, SET_SYM.replace("__SYM__", sym), ident)
            # A fixed sleep lies here - the previous symbol's series stays loaded
            # for a moment and the read comes back describing the WRONG ticker.
            # Poll until the series actually reports the symbol we asked for.
            got = None
            for _ in range(tries):
                await asyncio.sleep(settle / 3.0)
                ident += 1
                probe = await _eval(ws, JS.replace("__WANT__", "[]").replace("__LEN__", "1"),
                                    ident)
                if probe.get("ok") and (probe.get("symbol") or "").upper().endswith(sym.upper()):
                    got = probe
                    if probe.get("nBars", 0) > 400:
                        break
            if not got:
                out[sym] = {"ok": False, "err": "symbol never settled"}
                print("  %-6s did not load" % sym, flush=True)
                continue
            # Pull history back past the OLDEST entry we need, with margin for the
            # ten bars before it that the ATR average is taken over.
            need = min(want) - 4 * 3600
            ident += 1
            await _eval(ws, GOTO.replace("__FROM__", str(need - 6 * 3600))
                        .replace("__TO__", str(max(want) + 2 * 3600)), ident)
            await asyncio.sleep(2.5)
            r, best_first = {}, None
            # TradingView returns history a chunk at a time, so one loadDataTo is
            # not enough to reach three weeks back. Keep pulling while the oldest
            # bar keeps moving earlier; stop when it covers the entry or stalls.
            for _attempt in range(8):
                ident += 1
                await _eval(ws, LOAD_RANGE.replace("__FROM__", str(need))
                            .replace("__TO__", str(max(want) + 3600)), ident)
                for _ in range(15):
                    await asyncio.sleep(1.0)
                    ident += 1
                    r = await _eval(ws, JS.replace("__WANT__", json.dumps(want))
                                    .replace("__LEN__", str(ATR_TRAIL_LEN)), ident)
                    if r.get("ok") and all(x.get("found") for x in r.get("rows", [])):
                        break
                    if r.get("ok") and r.get("first") and best_first and r["first"] < best_first:
                        break
                if r.get("ok") and all(x.get("found") for x in r.get("rows", [])):
                    break
                f = r.get("first")
                if f is not None and best_first is not None and f >= best_first:
                    break                                  # history stopped growing
                best_first = f if f is not None else best_first
            out[sym] = r
            hit = sum(1 for x in r.get("rows", []) if x.get("found"))
            cov = r.get("first")
            print("  %-6s %6d bars back to %s   %d/%d entries matched"
                  % (sym, r.get("nBars", 0),
                     time.strftime("%m-%d %H:%M", time.gmtime(cov)) if cov else "?",
                     hit, len(want)), flush=True)
        if original:
            ident += 1
            await _eval(ws, SET_SYM.replace("__SYM__", original), ident)
            print("chart put back on %s" % original, flush=True)
    return out


def main():
    import csv
    from datetime import datetime, timezone
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=os.path.join(
        HERE, "..", "backtest", "trades-v10.0-2026-09-01.csv"))
    ap.add_argument("--out", default=os.path.join(
        HERE, "..", "backtest", "stop-pathway-v10.0.json"))
    ap.add_argument("--only", default="", help="comma-separated symbols, for a spot check")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.trades)))
    only = {s.strip().upper() for s in a.only.split(",") if s.strip()}
    jobs, meta = {}, []
    for r in rows:
        if only and r["sym"].upper() not in only:
            continue
        try:
            t = int(datetime.strptime(r["confirm"], "%Y-%m-%dT%H:%M")
                    .replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            continue
        jobs.setdefault(r["sym"], [])
        if t not in jobs[r["sym"]]:
            jobs[r["sym"]].append(t)
        meta.append({"sym": r["sym"], "seq": r["seq"], "confirmTs": t,
                     "confirm": r["confirm"], "dir": r["dir"],
                     "entry": float(r["entry"]), "R": float(r["R"]),
                     "pct": r["pct"], "rMult": r["rMult"],
                     "exitReason": r["exitReason"], "printIso": r["printIso"]})
    print("reading %d symbols, %d trades" % (len(jobs), len(meta)), flush=True)
    got = asyncio.run(run(jobs))
    json.dump({"meta": meta, "bars": got,
               "riskAtrMult": RISK_ATR_MULT, "atrTrailLen": ATR_TRAIL_LEN,
               "stopRetrace": STOP_RETRACE,
               "read_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(a.out, "w"), indent=1)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
