"""Try to recover the trades the pathway scan could not read, and prove why.

VERDICT, measured 2026-09-05 and reproducible: they cannot be read. TradingView
will not serve one-minute bars before a per-symbol horizon, and for these names
that horizon now sits AFTER the trades.

    symbol   bars    one-minute history reaches back to
    CRWV     11,395  2026-08-17 08:00
    COHR     10,368  2026-08-17 08:00
    LITE     10,767  2026-08-17 08:00
    NBIS     12,407  2026-08-17 08:00
    TGT      10,301  2026-08-03 08:00
    HD       11,229  2026-08-03 08:00

Those trades confirm on 2026-08-11 and 08-12, five days the wrong side of it.

IT IS NOT A BAR-COUNT CAP, which is what an earlier note in this project claimed.
NBIS holds 12,407 bars and reaches only 08-17, while HD holds 11,229 and reaches
08-03. Two symbols, more bars, less depth. The horizons cluster on two dates,
both at 08:00 UTC and exactly fourteen days apart, which points at a server-side
per-symbol limit rather than anything the client controls.

NOR IS IT IMPATIENCE. Forty consecutive loadDataTo calls on CRWV, anchored as far
back as April, left the series frozen at exactly 11,395 bars every single time.

AND IT DECAYS. The first scan of these symbols reached 2026-08-03 in late August.
The same call reaches only 08-17 today. The window moves forward with the
calendar, so backtest bars must be captured while the prints are recent - once
the horizon passes a trade, that trade is gone for good.

This script is kept because it is the check, not the fix: run it after any change
to the loader, or if TradingView's depth ever improves, and it will recover the
trades automatically if they become reachable.

    python backtest/recover_missing.py

Run with the earnings watcher stopped; it drives the same chart.
"""
import asyncio, json, os, sys, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "tools")))
import atr_at_entry as A
import tv_push

SCAN = os.path.join(HERE, "stop-pathway-v10.0.json")
OUT = os.path.join(HERE, "recovered-v10.0.json")
WANT = [("COHR", "1"), ("CRWV", "1"), ("LITE", "1"), ("LITE", "2"),
        ("NBIS", "1"), ("NBIS", "2")]
TRIES, STALL_LIMIT = 30, 12
ATR_LEN = 10

READ = r"""
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
    var want = __WANT__, to = __TO__, L = __LEN__;
    var j = ts.indexOf(want);
    var res = {ok:true, sym:sym, n:ts.length, first:ts[0], found:(j>=0)};
    if (j >= 0){
      var sum = 0, k = 0;
      for (var q = j - (L - 1); q <= j; q++){
        if (q < 0) continue;
        var h = px[q][1], l = px[q][2], tr = h - l;
        if (q > 0){
          var pc = px[q-1][3];
          tr = Math.max(tr, Math.abs(h - pc), Math.abs(l - pc));
        }
        sum += tr; k++;
      }
      res.atr = k ? sum / k : null;
      res.bar = {o: px[j][0], h: px[j][1], l: px[j][2], c: px[j][3]};
      var bars = [];
      for (var z = j; z < ts.length && ts[z] <= to; z++){
        bars.push({t: ts[z], o: px[z][0], h: px[z][1], l: px[z][2], c: px[z][3]});
      }
      res.bars = bars;
    }
    return JSON.stringify(res);
  } catch(e){ return JSON.stringify({ok:false, err:String(e)}); }
})()
"""


async def run():
    import websockets
    scan = json.load(open(SCAN))
    meta = {(m["sym"], m["seq"]): m for m in scan["meta"]}
    ws = await websockets.connect(tv_push.find_chart_target(), max_size=128 * 1024 * 1024)
    await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
    await ws.recv()
    ident, out = 700, {}
    for sym, seq in WANT:
        m = meta.get((sym, seq))
        if not m:
            print("  %s#%s not in the scan" % (sym, seq))
            continue
        ct = m["confirmTs"]
        ident += 1
        await A._eval(ws, A.SET_SYM.replace("__SYM__", sym), ident)
        await asyncio.sleep(4.0)
        best, stall, r = None, 0, {}
        for k in range(TRIES):
            ident += 1
            # `to` MUST run through to the present. Asking for a window that ENDS
            # in the past - confirm + 4 days - is what made every one of these
            # look unreachable: the series would not extend backwards for it, and
            # stalled at exactly 2026-08-17 every time. The same call with `to` set
            # to now reached 2026-08-03 on the same symbols minutes earlier. That
            # was the whole "11,000 bar cap", and it was never a cap.
            await A._eval(ws, A.LOAD_RANGE.replace("__FROM__", str(ct - 6 * 86400))
                          .replace("__TO__", str(ct + 40 * 86400)), ident)
            await asyncio.sleep(2.5)
            ident += 1
            r = await A._eval(ws, READ.replace("__WANT__", str(ct))
                              .replace("__TO__", str(ct + 3 * 86400))
                              .replace("__LEN__", str(ATR_LEN)), ident)
            if not r.get("ok"):
                break
            if (r.get("sym") or "").upper().split(":")[-1] != sym.upper():
                continue                     # a silent symbol-switch failure
            if r.get("found"):
                break
            f = r.get("first")
            if best is not None and f is not None and f >= best:
                stall += 1
            else:
                stall = 0
            best = f if best is None else min(best, f or best)
            if stall >= STALL_LIMIT:
                break
        key = "%s#%s" % (sym, seq)
        if r.get("found"):
            ok = abs(r["bar"]["c"] - m["entry"]) < 0.011
            out[key] = {"sym": sym, "seq": seq, "atr": r["atr"], "bar": r["bar"],
                        "barCloseMatches": ok, "bars": r.get("bars") or [],
                        "entry": m["entry"], "R": m["R"], "dir": m["dir"],
                        "pct": m.get("pct"), "exitReason": m.get("exitReason"),
                        "confirmTs": ct}
            print("  %-8s RECOVERED  atr=%.4f  close %.2f vs entry %.2f %s  %d post bars"
                  % (key, r["atr"], r["bar"]["c"], m["entry"],
                     "OK" if ok else "MISMATCH - excluded", len(r.get("bars") or [])))
        else:
            e = r.get("first")
            print("  %-8s still unreachable, earliest %s"
                  % (key, dt.datetime.fromtimestamp(e, dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
                     if e else "?"))
    await ws.close()
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nrecovered %d of %d - wrote %s" % (len(out), len(WANT), OUT))


if __name__ == "__main__":
    asyncio.run(run())
