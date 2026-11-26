"""atr_sweep - what a TIGHTER volatility floor would actually have done.

Job 2 split the trades by pathway: some of the v10.0 trades had their stop set by
the ATR floor (the VOLATILITY pathway) rather than by structure. The obvious next
question is whether lowering `riskAtrMult` from 1.80 changes that group's exits.


IT CANNOT BE ANSWERED BY ARITHMETIC. R sets the targets and the trail as well as
the stop, so a tighter floor moves every exit, not just the losing one. And it
cannot be answered from the stored path facts either, because their touch times
were computed for the ORIGINAL R. It needs the bars.

ONLY THOSE TRADES CHANGE. Lowering the multiplier can only shrink riskAtr, and a
structural trade already had riskStr above it, so it stays structural and its R is
untouched. That makes this a small, bounded read of a few symbols.

THE EXIT MODEL, taken from the Pine's own rules:
  * entry is the confirm bar's CLOSE, and that bar cannot stop you out or fill a
    target - the same exclusion scenarios.py uses.
  * initial stop  = entry - dir * R
  * T1/T2/T3 at 1R/2R/3R, 30% of the position each; 10% rides.
  * past +3R the stop trails 2R behind the peak, which is exactly the slack the
    ladder was already keeping, so the trail is continuous with it.
  * anything still open at the end of the measured window closes at the last close.

Run with the earnings watcher STOPPED - it drives the same chart.

    python backtest/atr_sweep.py
"""
import asyncio, json, os, sys, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "tools")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "tools")))
import atr_at_entry as A
import tv_push

PATHWAY = os.path.join(HERE, "pathway-v10.0.json")
PATHS = os.path.join(HERE, "paths-v10.0-2026-09-01.json")
OUT = os.path.join(HERE, "atr-sweep-v10.0.json")

MULTS = (1.80, 1.60, 1.50, 1.35, 1.20)
TP, TARGETS, TRAIL_R, TRAIL_FROM = 0.30, (1, 2, 3), 2.0, 3.0

BARS_JS = r"""
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
    var from = __FROM__, to = __TO__;
    var out = [];
    for (var i = lo; i <= hi; i++){
      var v = b.valueAt(i);
      if (!v) continue;
      if (v[0] >= from && v[0] <= to) out.push({t:v[0], o:v[1], h:v[2], l:v[3], c:v[4]});
    }
    return JSON.stringify({ok:true, symbol:sym, nBars:(hi-lo+1), bars:out});
  } catch(e){ return JSON.stringify({ok:false, err:String(e)}); }
})()
"""


def simulate(bars, entry, direction, R, last_close):
    """Return percent P/L under the ladder-plus-trail model described above."""
    if R <= 0 or not bars:
        return None
    stop = entry - direction * R
    peak_r, banked, legs, hit = 0.0, 0.0, [], [False] * len(TARGETS)
    for b in bars:
        hi, lo = b["h"], b["l"]
        fav = (hi - entry) * direction if direction > 0 else (entry - lo) * direction * -1
        fav = ((hi - entry) if direction > 0 else (entry - lo))
        peak_r = max(peak_r, fav / R)
        # Targets first: within a minute bar the order is unknowable, and taking
        # the target first is the assumption the rest of this project already
        # makes for the ladder.
        for i, n in enumerate(TARGETS):
            if hit[i]:
                continue
            px = entry + direction * n * R
            if (direction > 0 and hi >= px) or (direction < 0 and lo <= px):
                hit[i] = True
                legs.append((TP, px))
                banked += TP
        if peak_r >= TRAIL_FROM:
            peak_px = entry + direction * peak_r * R
            trailed = peak_px - direction * TRAIL_R * R
            stop = max(stop, trailed) if direction > 0 else min(stop, trailed)
        if (direction > 0 and lo <= stop) or (direction < 0 and hi >= stop):
            rem = 1.0 - banked
            if rem > 1e-9:
                legs.append((rem, stop))
            banked = 1.0
            break
    rem = 1.0 - banked
    if rem > 1e-9:
        legs.append((rem, last_close))
    avg = sum(w * p for w, p in legs)
    return 100.0 * direction * (avg - entry) / entry


async def capture(trades):
    import websockets
    ws_url = tv_push.find_chart_target()
    got = {}
    async with websockets.connect(ws_url, max_size=96 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
        await ws.recv()
        ident = 100
        for t in trades:
            sym = t["sym"]
            ident += 1
            await A._eval(ws, A.SET_SYM.replace("__SYM__", sym), ident)
            await asyncio.sleep(3.5)
            frm = t["confirmTs"] - 3600
            to = t["confirmTs"] + 3 * 86400
            for attempt in range(5):
                ident += 1
                await A._eval(ws, A.LOAD_RANGE.replace("__FROM__", str(frm - 6 * 3600))
                              .replace("__TO__", str(to)), ident)
                await asyncio.sleep(2.2)
                ident += 1
                r = await A._eval(ws, BARS_JS.replace("__FROM__", str(t["confirmTs"]))
                                  .replace("__TO__", str(to)), ident)
                if not r.get("ok"):
                    continue
                # A silent symbol-switch failure reads exactly like "no data".
                if (r.get("symbol") or "").upper().split(":")[-1] != sym.upper():
                    continue
                if len(r.get("bars") or []) > 30:
                    break
            key = "%s#%s" % (sym, t["seq"])
            got[key] = r.get("bars") or []
            print("  %-6s#%-2s %5d bars from the confirm" % (sym, t["seq"], len(got[key])))
    return got


def main():
    rows = json.load(open(PATHWAY))
    paths = {(p["sym"], str(p["seq"])): p for p in json.load(open(PATHS))}
    scan = json.load(open(os.path.join(HERE, "stop-pathway-v10.0.json")))
    meta = {(m["sym"], m["seq"]): m for m in scan["meta"]}

    vol = [r for r in rows if r.get("path") == "VOLATILITY" and r.get("barCloseMatches")]
    print("volatility-set trades: %d" % len(vol))
    for r in vol:
        m = meta.get((r["sym"], r["seq"]))
        r["confirmTs"] = m["confirmTs"]
        r["atr"] = r["riskAtr"] / scan.get("riskAtrMult", 1.80)

    bars = asyncio.run(capture(vol))

    out = {"model": "ladder 30%% at 1R/2R/3R, 2R trail past +3R, confirm bar excluded",
           "mults": list(MULTS), "trades": []}
    print("\n%-9s %-6s %8s" % ("trade", "actual", "") + "".join("  x%.2f" % k for k in MULTS))
    totals = {k: [] for k in MULTS}
    for r in vol:
        key = "%s#%s" % (r["sym"], r["seq"])
        bs = bars.get(key) or []
        if len(bs) < 10:
            print("  %-9s no bars - excluded" % key)
            continue
        lastc = bs[-1]["c"]
        d = 1 if r["dir"].upper().startswith("LONG") else -1
        rec = {"trade": key, "actual": r["pct"], "bars": len(bs), "byMult": {}}
        line = "%-9s %+6.2f%%  " % (key, r["pct"])
        for k in MULTS:
            R = k * r["atr"]
            v = simulate(bs[1:], r["entry"], d, R, lastc)
            rec["byMult"]["%.2f" % k] = v
            if v is not None:
                totals[k].append(v)
            line += "  %+6.2f%%" % (v if v is not None else float("nan"))
        out["trades"].append(rec)
        print(line)

    print("\n%-18s" % "MEAN" + "".join("  %+6.2f%%" % (sum(v) / len(v) if v else 0)
                                       for v in (totals[k] for k in MULTS)))
    print("%-18s" % "TOTAL" + "".join("  %+6.2f%%" % sum(totals[k]) for k in MULTS))
    print("%-18s" % "LOSERS" + "".join("  %6d" % len([x for x in totals[k] if x < 0])
                                       for k in MULTS))
    out["summary"] = {"%.2f" % k: {"n": len(totals[k]),
                                   "mean": (sum(totals[k]) / len(totals[k])) if totals[k] else None,
                                   "total": sum(totals[k]),
                                   "losers": len([x for x in totals[k] if x < 0])}
                      for k in MULTS}
    json.dump(out, open(OUT, "w"), indent=1)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
