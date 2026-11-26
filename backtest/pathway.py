"""pathway - split the v10.0 trades by WHICH MEASURE SET THE STOP, and score each.

The question this exists to test:

    Some stops (CoreWeave) look right on the chart and others (PANW, BABA) look
    too wide. Does a different measure set each group, so that the wide pathway
    can be tightened while the other stays as it is?

THE TEST IS EXACT, NOT APPROXIMATE. The Pine re-asserts the volatility floor as
the LAST operation on riskPx:

    riskPx = max(structure, earnings move, volatility)
    riskPx = min(riskPx, 0.65 x the whole move)
    riskPx = max(riskPx, volatility)          <- last word

so R equals riskAtr exactly when the floor is what set it, and exceeds it
otherwise. riskAtr = 1.80 x ta.sma(ta.tr(true), 10), read off the same
1-minute bars TradingView showed at the time.

    python backtest/pathway.py
"""
import csv, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
TRADES = os.path.join(HERE, "trades-v10.0-2026-09-01.csv")
PATHS = os.path.join(HERE, "paths-v10.0-2026-09-01.json")
SCAN = os.path.join(HERE, "stop-pathway-v10.0.json")

TOL = 0.02          # R within 2% of riskAtr is the floor binding, not a coincidence


def f(x):
    try:
        return float(x)
    except Exception:
        return None


def classify():
    d = json.load(open(SCAN))
    mult = d.get("riskAtrMult", 1.80)
    trades = {(r["sym"], r["seq"]): r for r in csv.DictReader(open(TRADES))}
    paths = {(p["sym"], str(p["seq"])): p for p in json.load(open(PATHS))}
    out = []
    for m in d["meta"]:
        b = d["bars"].get(m["sym"], {})
        row = next((r for r in b.get("rows", [])
                    if r.get("t") == m["confirmTs"] and r.get("found")), None)
        t = trades.get((m["sym"], m["seq"]))
        rec = {"sym": m["sym"], "seq": m["seq"], "dir": m["dir"], "entry": m["entry"],
               "R": m["R"], "pct": f(m["pct"]), "rMult": f(m["rMult"]),
               "exitReason": m["exitReason"], "confirm": m["confirm"],
               "grp": (t or {}).get("grp", "")}
        rec["rPctOfPrice"] = 100.0 * m["R"] / m["entry"] if m["entry"] else None
        if not row or not row.get("atr"):
            rec["path"] = "no data"
            out.append(rec)
            continue
        ra = mult * row["atr"]
        rec["riskAtr"] = ra
        rec["ratio"] = m["R"] / ra if ra else None
        # The bar this ATR was read from must be the bar the trade entered on.
        # If its close does not match the recorded entry, the read is of the
        # wrong bar and the classification would be fiction.
        rec["barCloseMatches"] = abs(row["bar"]["c"] - m["entry"]) < 0.011
        rec["path"] = "VOLATILITY" if abs(m["R"] - ra) / m["R"] < TOL else "STRUCTURAL"
        p = paths.get((m["sym"], str(m["seq"])))
        if p:
            # MFE MUST BE BOUNDED BY THE ACTUAL EXIT. Reporting a peak the trade
            # was no longer open for is how "NBIS reached [n]R" got said about a
            # trade that closed at [n]%.
            rec["maeR"] = p.get("maeR")
            rec["mfeR"] = p.get("mfeR")
        out.append(rec)
    return out


def scoreboard(rows, label):
    if not rows:
        return "  %-12s  no trades" % label
    pct = [r["pct"] for r in rows if r["pct"] is not None]
    rm = [r["rMult"] for r in rows if r["rMult"] is not None]
    wid = [r["rPctOfPrice"] for r in rows if r["rPctOfPrice"] is not None]
    wins = [p for p in pct if p > 0]
    return ("  %-12s n=%-3d  mean %+6.2f%%  median %+6.2f%%  win %3.0f%%  "
            "meanR %+5.2f  stop width %4.2f%% of price"
            % (label, len(rows), st.mean(pct) if pct else 0,
               st.median(pct) if pct else 0,
               100.0 * len(wins) / len(pct) if pct else 0,
               st.mean(rm) if rm else 0, st.mean(wid) if wid else 0))


def main():
    rows = classify()
    bad = [r for r in rows if r.get("path") != "no data" and not r.get("barCloseMatches")]
    nod = [r for r in rows if r.get("path") == "no data"]
    print("=" * 78)
    print("WHICH MEASURE SET THE STOP — v10.0, %d trades" % len(rows))
    print("=" * 78)
    if nod:
        print("no bar data (NOT counted as anything): " +
              ", ".join("%s#%s" % (r["sym"], r["seq"]) for r in nod))
    if bad:
        print("bar close did not match the recorded entry, EXCLUDED: " +
              ", ".join("%s#%s" % (r["sym"], r["seq"]) for r in bad))
    good = [r for r in rows if r.get("path") in ("VOLATILITY", "STRUCTURAL")
            and r.get("barCloseMatches")]

    print("\n%-6s %-4s %-6s %8s %9s %6s %8s %8s  %s"
          % ("sym", "seq", "dir", "R", "riskAtr", "R/atr", "R %px", "P/L %", "set by"))
    for r in sorted(good, key=lambda x: -(x["ratio"] or 0)):
        print("%-6s %-4s %-6s %8.3f %9.3f %6.2f %7.2f%% %+7.2f%%  %s"
              % (r["sym"], r["seq"], r["dir"][:5], r["R"], r["riskAtr"], r["ratio"],
                 r["rPctOfPrice"], r["pct"] or 0, r["path"]))

    vol = [r for r in good if r["path"] == "VOLATILITY"]
    stru = [r for r in good if r["path"] == "STRUCTURAL"]
    print("\n" + "-" * 78)
    print("SPLIT BY PATHWAY")
    print(scoreboard(good, "all"))
    print(scoreboard(vol, "VOLATILITY"))
    print(scoreboard(stru, "STRUCTURAL"))

    print("\n" + "-" * 78)
    print("WHERE THE LOSSES SIT")
    for name, grp in (("VOLATILITY", vol), ("STRUCTURAL", stru)):
        lose = [r for r in grp if (r["pct"] or 0) < 0]
        if not grp:
            continue
        tot = sum(r["pct"] for r in grp if r["pct"] is not None)
        lost = sum(r["pct"] for r in lose)
        print("  %-11s %2d of %2d losing   they cost %+.2f%%   pathway total %+.2f%%"
              % (name, len(lose), len(grp), lost, tot))
        for r in sorted(lose, key=lambda x: x["pct"])[:5]:
            print("      %-6s#%-2s %+6.2f%%  stop %4.2f%% of price  %s"
                  % (r["sym"], r["seq"], r["pct"], r["rPctOfPrice"], r["exitReason"]))

    print("\n" + "-" * 78)
    print("HOW EACH PATHWAY EXITS")
    # "STOP" alone lumps the two opposite outcomes together: a TRAILED stop is a
    # winner being banked, an INITIAL stop is the trade being wrong. Averaging
    # them said "[n] of [n] stopped out, mean [n]%", which reads like a healthy
    # pathway and describes nothing.
    for name, grp in (("VOLATILITY", vol), ("STRUCTURAL", stru)):
        if not grp:
            continue
        init = [r for r in grp if (r["exitReason"] or "").upper().startswith("INITIAL")]
        trail = [r for r in grp if (r["exitReason"] or "").upper().startswith("TRAILED")]
        other = [r for r in grp if r not in init and r not in trail]
        for lab, g in (("initial stop", init), ("trailed stop", trail), ("other", other)):
            if not g:
                continue
            print("  %-11s %-13s n=%-2d  mean %+6.2f%%  stop width %4.2f%% of price"
                  % (name, lab, len(g), st.mean([r["pct"] for r in g]),
                     st.mean([r["rPctOfPrice"] for r in g])))

    print("\n" + "-" * 78)
    print("MAE CONSISTENCY CHECK  —  run this BEFORE trusting any excursion figure")
    # maeR in the paths file is measured across the whole WINDOW, not bounded by
    # the trade's exit. Some trades contradict their own exit reason because of
    # it: a few show excursions past [n]R while exiting on a TRAILED stop (the
    # move happened after they closed), and one shows [n]R while the trade CSV
    # records an INITIAL STOP hit with no stop touch found at
    # all. Any "would a tighter stop have helped" claim built on this number is
    # unsound, which is why that section was removed rather than caveated.
    bad = []
    for r in rows:
        mae = r.get("maeR")
        init = "INITIAL" in (r["exitReason"] or "").upper()
        if (init and (mae is None or mae > -0.999)) or \
           ((not init) and mae is not None and mae < -1.05):
            bad.append("%s#%s %s maeR=%s" % (r["sym"], r["seq"],
                                             (r["exitReason"] or "")[:18], mae))
    print("  %d of %d trades have an excursion inconsistent with their exit."
          % (len(bad), len(rows)))
    for b in bad:
        print("      " + b)
    print("  So MAE here is NOT exit-bounded. Sizing the stop from it would need a"
          "\n  fresh bar read per trade, not this file.")
    # maeR is the worst excursion in units of the R the trade actually used, so a
    # trade at [n]R survived with part of its stop unused; scaling R down by more
    # than that factor would have stopped it out instead.


if __name__ == "__main__":
    main()
