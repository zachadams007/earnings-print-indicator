"""v10.x - the September 2026 out-of-sample set, measured and analysed.

Everything the report says is recomputed here from the raw harness output, so a
reader can re-run it and get the same numbers. Nothing is typed in by hand.

INPUTS (backtest/)
  roster-v10x.json                 the roster, with groups
  roster-v10x-2x-etfs.json         the verified 2x-leveraged-ETF overlay
  raw-v10x-atr150-2026-09-29.json  full roster at the live setting (ATR floor 1.50)
  raw-v10x-atr150-paths-*.json     traded names re-run at 1.50 with price paths
  raw-v10x-atr180-*.json           traded names at the previous setting (1.80)

THE RULES, each inherited from the v8-v10 runs and each there because breaking
it once produced a wrong headline:
  * candle-close fills are the headline; level fills are the secondary column
  * 25% off at each of the four take-profits (0.60R/1.40R/2.40R/4.00R), the
    remainder at the exit marker; 20%-at-each is carried as the comparison
  * 100 units at the confirm close, +50 at every add; % of total capital deployed
  * a trade counts only if its PRINT (the triangle) is within 6 hours of the
    verified release - the error that once credited LUNR with a next-day trade
  * covered === false is NO DATA, never "no trade"
  * HRL-class illiquidity is screened on flat-bar fraction, not price or size
  * an open trade (no exit marker yet) is excluded, never priced at a guess

    python backtest/analyse_v10x.py
"""
import csv, glob, json, math, os, random, statistics as st, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
PRINT_GATE_H = 6.0
FLAT_SCREEN = 0.60           # more than 60% zero-range bars in the first 90 minutes
W_HEAD, W_ALT = 0.25, 0.20


def J(name):
    return json.load(open(os.path.join(HERE, name), encoding="utf-8"))


def latest(pattern):
    fs = sorted(glob.glob(os.path.join(HERE, pattern)))
    return os.path.basename(fs[-1]) if fs else None


def iso(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M") if ts else ""


def bucket(reason):
    r = (reason or "").upper()
    for k, lab in (("TRAILED", "trail"), ("INITIAL", "init"), ("FLAT TAPE", "flat"),
                   ("REVERSAL", "rev"), ("SESSION END", "sess")):
        if k in r:
            return lab
    return "other"


def price(tr, w, level):
    """P/L at weighting w, rebuilt from the component prices - the same formula as
    analyse.py's rebuild_v10(), written from the position model."""
    d = 1 if tr["dir"] == "LONG" else -1
    tps = tr.get("tps") or []
    px = [t["level"] if level else t["close"] for t in tps[:4]]
    rem = 1.0 - w * len(px)
    ex = tr.get("exitLevel") if (level and tr.get("exitLevel") is not None) else tr.get("exitClose")
    if rem > 1e-9 and ex is None:
        return None
    avg = sum(w * p for p in px) + (rem * ex if rem > 1e-9 else 0.0)
    return 100.0 * d * (avg - tr["blended"]) / tr["blended"]


def stats(a):
    a = [x for x in a if x is not None]
    if not a:
        return dict(n=0)
    return dict(n=len(a), mean=st.mean(a), median=st.median(a), sum=sum(a),
                win=100.0 * sum(1 for x in a if x > 0) / len(a),
                sd=st.stdev(a) if len(a) > 1 else 0.0, best=max(a), worst=min(a))


def boot(a, iters=10000, seed=12345):
    a = [x for x in a if x is not None]
    if len(a) < 2:
        return (None, None)
    rnd = random.Random(seed)
    ms = sorted(st.mean(rnd.choice(a) for _ in a) for _ in range(iters))
    return ms[int(0.025 * iters)], ms[int(0.975 * iters) - 1]


def load_pass(name):
    """Rows -> (per-symbol records, trade list) with every gate applied.

    `name` may be a list: rows from later files replace the same symbol in earlier
    ones, so a one-name correction run can be layered over a full pass.
    """
    names = name if isinstance(name, list) else [name]
    raw = J(names[0])
    if len(names) > 1:
        rows = {r["sym"]: r for r in raw["rows"]}
        for extra in names[1:]:
            for r in J(extra)["rows"]:
                rows[r["sym"]] = r
        raw = dict(raw, rows=list(rows.values()))
    roster = {j["sym"]: j for j in J("roster-v10x.json")}
    x2 = {k: v for k, v in J("roster-v10x-2x-etfs.json").items() if not k.startswith("_")}
    syms, trades = [], []
    for r in raw["rows"]:
        s = r["sym"]
        rec = dict(sym=s, grp=r.get("grp") or roster.get(s, {}).get("grp", ""),
                   x2=s in x2, expect=r.get("expect"), release=r.get("release") or "",
                   covered=r.get("covered"), firstBar=r.get("firstBar") or "",
                   nBars=r.get("nBars"), fail=r.get("fail") or r.get("err") or "",
                   liq=r.get("liq") or {}, nTrades=0, status="")
        lq = rec["liq"]
        rec["flatFrac"] = (lq.get("zero", 0) / lq["n"]) if lq.get("n") else None
        rec["leadHrs"] = (round((r["releaseTs"] - r["firstTs"]) / 3600.0, 1)
                          if r.get("releaseTs") and r.get("firstTs") else None)
        exp_ts = roster.get(s, {}).get("expectTs")
        if rec["fail"]:
            rec["status"] = "error"
        elif not r.get("releaseTs") and r.get("firstTs") and exp_ts and r["firstTs"] > exp_ts - 1800:
            # History starts AFTER the expected release: nothing could be seen, so
            # this is NO DATA - never "no release" and never "no trade".
            rec["status"] = "no-data"
        elif not r.get("releaseTs"):
            rec["status"] = "no-release-found"
        elif not r.get("covered"):
            rec["status"] = "no-data"
        elif rec["flatFrac"] is not None and rec["flatFrac"] > FLAT_SCREEN:
            rec["status"] = "illiquid"
        labs = [l for l in (r.get("labSeq") or []) if l["t"] in ("▲", "▼")]
        kept = 0
        for i, tr in enumerate(r.get("trades") or []):
            if not tr or tr.get("bad"):
                continue
            pr = [l for l in labs if l["ts"] <= tr["tsConf"]]
            pts = pr[-1]["ts"] if pr else None
            t = dict(sym=s, grp=rec["grp"], x2=rec["x2"], release=rec["release"],
                     releaseTs=r.get("releaseTs"), seq=i + 1, confirm=tr["tConf"],
                     tsConf=tr["tsConf"], dir=tr["dir"], entry=tr["entry"], R=tr["R"],
                     adds=tr["nAdds"], capital=tr["capital"], blended=tr["blended"],
                     nTP=tr["nTP"], tps=tr.get("tps") or [], exitTime=tr.get("exitT") or "",
                     exitReason=(tr.get("exitReason") or "").strip(),
                     exitClose=tr.get("exitClose"), exitLevel=tr.get("exitLevel"),
                     open=bool(tr.get("open")), printIso=iso(pts),
                     printHrs=round((pts - r["releaseTs"]) / 3600.0, 2) if pts and r.get("releaseTs") else None,
                     confHrs=round((tr["tsConf"] - pts) / 3600.0, 2) if pts else None,
                     path=((r.get("paths") or [None] * (i + 1))[i] if r.get("paths") else None))
            t["excluded"] = ""
            if rec["status"] == "illiquid":
                t["excluded"] = "illiquid tape"
            elif t["open"]:
                t["excluded"] = "still open at data end"
            elif t["printHrs"] is None or abs(t["printHrs"]) > PRINT_GATE_H:
                t["excluded"] = "print not at the release (%sh)" % t["printHrs"]
            t["pct"] = price(tr, W_HEAD, False)
            t["pctLvl"] = price(tr, W_HEAD, True)
            t["pct20"] = price(tr, W_ALT, False)
            t["pctLvl20"] = price(tr, W_ALT, True)
            d = 1 if t["dir"] == "LONG" else -1
            if t["pct"] is not None:
                t["rMult"] = t["pct"] / 100.0 * t["blended"] / t["R"]
            t["bucket"] = bucket(t["exitReason"])
            trades.append(t)
            if not t["excluded"]:
                kept += 1
        rec["nTrades"] = kept
        if not rec["status"]:
            rec["status"] = "traded" if kept else "no-trade"
        syms.append(rec)
    return raw, syms, trades


def ceiling(t):
    p = t.get("path") or {}
    if p.get("best") is None:
        return 0.0
    d = 1 if t["dir"] == "LONG" else -1
    return 100.0 * d * (p["best"] - t["blended"]) / t["blended"]


def main():
    f150 = "raw-v10x-atr150-2026-09-29.json"
    fpth = latest("raw-v10x-atr150-paths-*.json")
    f180 = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "raw-v10x-atr180-*.json"))) or None
    raw, syms, trades = load_pass(f150)
    keep = [t for t in trades if not t["excluded"] and t["pct"] is not None]

    # DETERMINISM. The first attempt broke at SAIL (see the method notes), but the
    # names measured BEFORE it are genuine reads of the same bars. Re-measured
    # now, their trades must come out identical to the cent - if they do not, the
    # measurement is not repeatable and nothing else on the page can be trusted.
    det = None
    broken = os.environ.get("EPD_FIRST_ATTEMPT")
    if broken and os.path.exists(broken):
        b = json.load(open(broken, encoding="utf-8"))
        prior = {}
        for r in b["rows"]:
            if (r.get("nBars") or 0) < 2000 or not r.get("covered"):
                continue                              # the post-SAIL stale rows
            for tr in (r.get("trades") or []):
                if tr and not tr.get("bad"):
                    prior[(r["sym"], tr["tConf"])] = tr
        same = diff = 0
        mism = []
        for t in trades:
            u = prior.get((t["sym"], t["confirm"]))
            if u is None:
                continue
            if u["entry"] == t["entry"] and u["R"] == t["R"] and u["nAdds"] == t["adds"] \
                    and (u.get("exitT") or "") == t["exitTime"]:
                same += 1
            else:
                diff += 1
                mism.append("%s %s" % (t["sym"], t["confirm"]))
        missing = [k for k in prior if not any((t["sym"], t["confirm"]) == k for t in trades)]
        det = dict(same=same, diff=diff, mismatches=mism, missing=["%s %s" % k for k in missing])

    out = {"inputs": raw["inputs"], "study": raw["study"], "measured_at": raw["measured_at"]}
    out["symbols"] = syms
    out["counts"] = {k: sum(1 for s in syms if s["status"] == k) for k in
                     ("traded", "no-trade", "no-data", "no-release-found", "illiquid", "error")}
    out["excluded"] = [dict(sym=t["sym"], confirm=t["confirm"], why=t["excluded"],
                            pct=t["pct"]) for t in trades if t["excluded"]]
    out["determinism"] = det

    pc = [t["pct"] for t in keep]
    out["head"] = dict(close=stats(pc), level=stats([t["pctLvl"] for t in keep]),
                       close20=stats([t["pct20"] for t in keep]),
                       level20=stats([t["pctLvl20"] for t in keep]),
                       ci=boot(pc), ci20=boot([t["pct20"] for t in keep]),
                       rmean=st.mean(t["rMult"] for t in keep) if keep else None)
    out["buckets"] = {b: stats([t["pct"] for t in keep if t["bucket"] == b])
                      for b in ("trail", "init", "flat", "rev", "sess", "other")}
    out["bucketsLvl"] = {b: stats([t["pctLvl"] for t in keep if t["bucket"] == b])
                         for b in ("trail", "init")}
    out["tps"] = {n: stats([t["pct"] for t in keep if t["nTP"] == n]) for n in range(5)}
    out["cohorts"] = {g: stats([t["pct"] for t in keep if t["grp"] == g]) for g in ("B", "D", "E")}
    out["x2"] = {"yes": stats([t["pct"] for t in keep if t["x2"]]),
                 "no": stats([t["pct"] for t in keep if not t["x2"]])}
    out["dir"] = {d: stats([t["pct"] for t in keep if t["dir"] == d]) for d in ("LONG", "SHORT")}
    first = [t for t in keep if t["seq"] == 1]
    srt = sorted(keep, key=lambda t: t["pct"])
    out["robust"] = {
        "headline": stats(pc),
        "drop best": stats([t["pct"] for t in srt[:-1]]),
        "drop worst": stats([t["pct"] for t in srt[1:]]),
        "drop best 2": stats([t["pct"] for t in srt[:-2]]),
        "first fire only": stats([t["pct"] for t in first]),
        "one per symbol (best)": stats([max(t["pct"] for t in keep if t["sym"] == s)
                                        for s in sorted({t["sym"] for t in keep})]),
    }
    # Ceiling: best price before the trade's own stop, full position out there.
    if any(t.get("path") for t in keep):
        ce = [ceiling(t) for t in keep]
        mfe = [(t.get("path") or {}).get("mfeR") for t in keep]
        bands = [("never in front", lambda m: m is None or m <= 0),
                 ("under +0.5R", lambda m: 0 < m < 0.5), ("+0.5R - 1R", lambda m: 0.5 <= m < 1),
                 ("+1R - 2R", lambda m: 1 <= m < 2), ("+2R - 3R", lambda m: 2 <= m < 3),
                 ("+3R - 4R", lambda m: 3 <= m < 4), ("+4R or more", lambda m: m is not None and m >= 4)]
        out["ceiling"] = dict(
            stats=stats(ce), capture=(st.mean(pc) / st.mean(ce) * 100.0) if st.mean(ce) else None,
            reachable=sum(1 for c in ce if c > 0),
            peakAfterExit=sum(1 for t in keep if (t.get("path") or {}).get("peakAfterExit")),
            bands=[(lab, sum(1 for m in mfe if (m is None and lab == "never in front") or
                             (m is not None and fn(m)))) for lab, fn in bands],
            mfeMedian=st.median([m for m in mfe if m is not None]) if any(m is not None for m in mfe) else None,
            mfeMean=st.mean([m for m in mfe if m is not None]) if any(m is not None for m in mfe) else None)

    # THE LADDER, re-priced off the same component fills - no assumption added.
    def reweigh(t, ws):
        """ws: weight taken at each rung reached (T1..T4); the rest rides to the exit."""
        d = 1 if t["dir"] == "LONG" else -1
        px = [x["close"] for x in t["tps"][:4]]
        take = sum(ws[k] for k in range(len(px)))
        avg = sum(ws[k] * px[k] for k in range(len(px)))
        rem = 1.0 - take
        if rem > 1e-9:
            if t["exitClose"] is None:
                return None
            avg += rem * t["exitClose"]
        return 100.0 * d * (avg - t["blended"]) / t["blended"]

    base = st.mean(t["pct"] for t in keep) if keep else 0.0
    rungs = []
    for k, lab in enumerate(("T1 - 0.60R", "T2 - 1.40R", "T3 - 2.40R", "T4 - 4.00R")):
        ws = [0.25] * 4
        ws[k] = 0.0
        vals = [reweigh(t, ws) for t in keep]
        m = st.mean(v for v in vals if v is not None)
        rungs.append(dict(rung=lab, without=m, worth=base - m, total=(base - m) * len(keep)))
    none = [reweigh(t, [0, 0, 0, 0]) for t in keep]
    rungs.append(dict(rung="No ladder at all", without=st.mean(none), worth=base - st.mean(none),
                      total=(base - st.mean(none)) * len(keep)))
    out["rungs"] = rungs
    schemes = [("No ladder - everything at the exit", [0, 0, 0, 0]),
               ("15 / 15 / 15 / 15", [.15] * 4), ("20 / 20 / 20 / 20", [.20] * 4),
               ("25 / 20 / 15 / 15 - tapered", [.25, .20, .15, .15]),
               ("25 / 25 / 25 / 25 - shipped", [.25] * 4),
               ("30 / 20 / 15 / 10", [.30, .20, .15, .10]), ("30 / 25 / 20 / 15", [.30, .25, .20, .15]),
               ("35 / 25 / 20 / 10", [.35, .25, .20, .10]), ("40 / 20 / 15 / 10", [.40, .20, .15, .10])]
    out["weightings"] = [dict(name=n, runs=round(100 * (1 - sum(w))), **stats([reweigh(t, w) for t in keep]))
                         for n, w in schemes]
    out["cuts"] = {"No adds": stats([t["pct"] for t in keep if t["adds"] == 0]),
                   "At least one add": stats([t["pct"] for t in keep if t["adds"] > 0]),
                   "Long": out["dir"]["LONG"], "Short": out["dir"]["SHORT"],
                   "First fire on the event": stats([t["pct"] for t in keep if t["seq"] == 1]),
                   "Re-entries only": stats([t["pct"] for t in keep if t["seq"] > 1])}
    # RELEASE OR CALL. August's kept prints all landed within 0.30h of the release.
    # September's include prints 1-2 hours later - on the CONFERENCE CALL, not the
    # release (Ciena's 8:30am call, UiPath's and Ambarella's after-hours calls).
    # They pass the pre-set gate and stay in the headline; the split is reported so
    # it is visible rather than folded into one number.
    out["release_vs_call"] = {
        "release": stats([t["pct"] for t in keep if abs(t["printHrs"]) <= 1.0]),
        "call": stats([t["pct"] for t in keep if abs(t["printHrs"]) > 1.0]),
        "call_trades": [dict(sym=t["sym"], printHrs=t["printHrs"], pct=t["pct"])
                        for t in keep if abs(t["printHrs"]) > 1.0],
        "release_ci": boot([t["pct"] for t in keep if abs(t["printHrs"]) <= 1.0]),
        "release_maxHrs": max((abs(t["printHrs"]) for t in keep if abs(t["printHrs"]) <= 1.0), default=None),
        "release_medHrs": st.median([abs(t["printHrs"]) for t in keep if abs(t["printHrs"]) <= 1.0])
        if keep else None}
    top2 = sorted(keep, key=lambda t: -t["pct"])[:2]
    out["top2"] = [dict(sym=t["sym"], pct=t["pct"], release=t["release"], dir=t["dir"]) for t in top2]

    # 1.50 against 1.80 on the SAME September events - paired by symbol and confirm.
    if f180:
        _, _, t180 = load_pass(f180)
        k180 = [t for t in t180 if not t["excluded"] and t["pct"] is not None]
        traded = sorted({t["sym"] for t in keep} | {t["sym"] for t in k180})
        a = {(t["sym"], t["confirm"]): t for t in keep}
        b = {(t["sym"], t["confirm"]): t for t in k180}
        both = sorted(set(a) & set(b))
        per = []
        for s in traded:
            x = sum(t["pct"] for t in keep if t["sym"] == s)
            y = sum(t["pct"] for t in k180 if t["sym"] == s)
            per.append((s, x, y))
        loo = [(s, sum(x - y for s2, x, y in per if s2 != s)) for s, _, _ in per]
        out["atr"] = dict(
            s150=stats([t["pct"] for t in keep]), s180=stats([t["pct"] for t in k180]),
            paired=len(both), only150=len(set(a) - set(b)), only180=len(set(b) - set(a)),
            changedR=sum(1 for k in both if abs(a[k]["R"] - b[k]["R"]) > 1e-9),
            perSymbol=per, total_diff=sum(x - y for _, x, y in per),
            loo_min=min((v for _, v in loo), default=None),
            loo_neg=[s for s, v in loo if v <= 0], file=f180)

    # v10.0's own sample, for context - DIFFERENT events, so not a paired test.
    v10 = [float(r["pct"]) for r in csv.DictReader(open(os.path.join(HERE, "trades-v10.0-2026-09-01.csv"),
                                                        encoding="utf-8"))]
    out["v10"] = dict(stats=stats(v10), ci=boot(v10))
    if keep and v10:
        se = math.sqrt(st.variance(pc) / len(pc) + st.variance(v10) / len(v10)) if len(pc) > 1 else None
        out["v10"]["diff"] = st.mean(pc) - st.mean(v10)
        out["v10"]["se"] = se

    # CSVs in v10's schema, plus the flags this run adds.
    cols = ["sym", "grp", "x2", "release", "seq", "confirm", "dir", "entry", "R", "adds", "capital",
            "blended", "nTP", "exitTime", "exitReason", "exitClose", "exitLevel", "printIso",
            "printHrs", "confHrs", "pct", "pctLvl", "rMult", "pct20", "pctLvl20",
            "tp1c", "tp1l", "tp2c", "tp2l", "tp3c", "tp3l", "tp4c", "tp4l", "excluded"]
    with open(os.path.join(HERE, "trades-v10x-2026-09-29.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t in trades:
            tp = {}
            for k, x in enumerate(t["tps"][:4]):
                tp["tp%dc" % (k + 1)], tp["tp%dl" % (k + 1)] = x["close"], x["level"]
            row = dict(t, **tp)
            w.writerow([("" if row.get(c) is None else
                         (round(row[c], 4) if isinstance(row.get(c), float) else row.get(c))) for c in cols])
    scol = ["sym", "grp", "x2", "expect", "release", "status", "covered", "leadHrs", "nBars",
            "firstBar", "nTrades", "flatFrac", "fail"]
    with open(os.path.join(HERE, "symbols-v10x-2026-09-29.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(scol)
        for s in syms:
            w.writerow([("" if s.get(c) is None else s.get(c)) for c in scol])

    out["trades"] = [{k: v for k, v in t.items() if k not in ("tps",)} for t in keep]
    json.dump(out, open(os.path.join(HERE, "v10x-numbers.json"), "w", encoding="utf-8"),
              indent=1, default=str)

    h = out["head"]["close"]
    print("v10.x  %s  inputs %s" % (out["study"], out["inputs"]))
    print("symbols:", out["counts"])
    print("trades kept %d, excluded %d" % (len(keep), len(out["excluded"])))
    if h.get("n"):
        print("HEADLINE (closes, 25%%): n=%d mean %+.3f%% median %+.3f%% sum %+.2f%% win %.1f%% sd %.3f  CI [%+.3f, %+.3f]"
              % (h["n"], h["mean"], h["median"], h["sum"], h["win"], h["sd"], *out["head"]["ci"]))
    for b, v in out["buckets"].items():
        if v.get("n"):
            print("  %-6s n=%-2d mean %+.3f%% sum %+.2f%%" % (b, v["n"], v["mean"], v["sum"]))
    if det:
        print("determinism vs first attempt (pre-SAIL rows): %d identical, %d differ %s, missing %s" % (det["same"], det["diff"], det["mismatches"], det["missing"]))
    if "atr" in out:
        a = out["atr"]
        print("ATR 1.50 vs 1.80: mean %+.3f vs %+.3f  total diff %+.2f  loo_min %s  loo_neg %s"
              % (a["s150"].get("mean", 0), a["s180"].get("mean", 0), a["total_diff"],
                 a["loo_min"], a["loo_neg"]))


if __name__ == "__main__":
    main()
