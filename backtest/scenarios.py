"""Counterfactual exit scenarios for the v9.0 trade set.

The owner asked for the same mean under specific alternative exit rules, so the ladder
and the trail can be judged against what was actually available on the tape.

INPUTS
  trades-v9.0-2026-08-31.csv     the measured trades
  paths-v9.0-2026-08-31.json     per-trade path facts read off the 1-minute bars:
                                 MFE/MAE in R, the first touch time of T1/T2/T3 and
                                 of the initial stop, the first return to entry after
                                 the trade had been [n]R / [n]R in front, and the last
                                 close in the window.

GEOMETRY, verified against the stored levels on [n] of [n] checks:
  initial stop = entry - dir*R          target n = entry + dir*n*R
  (KSS is the one mismatch and is flagged in the report.)

THE CONFIRM BAR IS EXCLUDED from every touch test. Entry is that bar's close, so its
own high/low cannot stop you out or fill a target. Including it marked CBRS, COHR, CRM
and ADSK as stopped out on the bar they entered on.

TIE-BREAK: when a target and the stop are first touched on the SAME minute, the stop is
taken first. That is the conservative reading - one minute of 1-minute data cannot say
which came first, and assuming the good fill would flatter every scenario.
"""
import csv, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
TP = 0.30


def load():
    tr = list(csv.DictReader(open(os.path.join(HERE, "trades-v9.0-2026-08-31.csv"))))
    paths = json.load(open(os.path.join(HERE, "paths-v9.0-2026-08-31.json")))
    byrow = {}
    for rec in paths:
        for t in rec.get("trades", []):
            byrow[t["row"]] = t
    out = []
    for i, r in enumerate(tr):
        p = byrow.get(i)
        if p is None:
            continue
        d = 1 if r["dir"] == "LONG" else -1
        out.append(dict(
            sym=r["sym"], seq=int(r["seq"]), dir=d, group=r["group"],
            entry=float(r["blended"]), R=float(r["R"]),
            actual=float(r["pctLvl"]), actualClose=float(r["pct"]),
            exitReason=r["exitReason"], exitT=r["exitTime"], nTP=int(r["nTP"]),
            path=p))
    return out


def pct(t, legs):
    """legs = [(weight, price)]; weights must total 1."""
    w = sum(x[0] for x in legs)
    assert abs(w - 1.0) < 1e-9, "weights %.6f" % w
    avg = sum(a * b for a, b in legs)
    return 100.0 * t["dir"] * (avg - t["entry"]) / t["entry"]


def target_px(t, n):
    return t["entry"] + t["dir"] * n * t["R"]


def stop_px(t):
    return t["entry"] - t["dir"] * t["R"]


# ── S1 ── targets and the original stop are the ONLY ways out ────────────────
# No trail, no flat-tape exit, no reversal cut, no session-end close.
#
# CAVEAT, and it is a big one: if the stop never triggers there is nothing left to
# close the remainder, so it rides to the end of the two-day measurement window.
# That is not a tradeable rule, it is "hold and hope". `s1(open_end=False)` gives the
# conservative reading where the untriggered remainder is handed back to the exit the
# indicator actually took, so only the STOPPED trades differ from reality.
def s1(t, open_end=True):
    p, legs, banked = t["path"], [], 0.0
    stopT = p["stopT"]
    for n in (1, 2, 3):
        tt = p["t%dT" % n]
        if tt is None:
            break
        if stopT is not None and tt >= stopT:   # tie -> stop wins
            break
        legs.append((TP, target_px(t, n)))
        banked += TP
    rem = 1.0 - banked
    if rem > 1e-9:
        if stopT is not None:
            legs.append((rem, stop_px(t)))
        elif open_end:
            legs.append((rem, p["lastC"]))
        else:
            return t["actual"] if banked == 0 else None   # caller substitutes actual
    return pct(t, legs)


def s1_conservative(t):
    """Only the trades that DID hit their stop are re-priced; the rest keep the exit
    the indicator actually took. Isolates 'was the trail worth having' from
    'what if I never closed the position'."""
    v = s1(t, open_end=False)
    return t["actual"] if v is None else v


# ── S2 ── perfect foresight: the whole position out at the best price offered ─
def s2(t):
    best = t["entry"] + t["dir"] * t["path"]["mfeR"] * t["R"]
    return pct(t, [(1.0, best)])


# ── S3 ── stop to breakeven once the trade is properly in front ──────────────
# Ladder as normal. Once the trade has been `armR` in front, the stop sits at the
# blended entry, so a trade that gives it all back scratches instead of losing.
def s3(t, armR=1.0):
    p = t["path"]
    # The breakeven stop only ever CHANGES an outcome when two things both happen:
    # the trade got armR in front (so the stop moved to entry), and price later came
    # back to entry (so that stop was hit). In every other case the rule is inert and
    # the trade is exactly what it actually was.
    if p["mfeR"] is None or p["mfeR"] < armR:
        return t["actual"]                       # never armed
    beT = p["beT"] if armR >= 1.0 else p["be05T"]
    if beT is None:
        return t["actual"]                       # armed, but never came back - inert
    legs, banked = [], 0.0
    for n in (1, 2, 3):
        tt = p["t%dT" % n]
        if tt is None or tt >= beT:              # only targets banked BEFORE the scratch
            break
        legs.append((TP, target_px(t, n)))
        banked += TP
    rem = 1.0 - banked
    if rem > 1e-9:
        legs.append((rem, t["entry"]))           # remainder scratched at breakeven
    return pct(t, legs)


def stats(a, label):
    if not a:
        return "%-34s n=0" % label
    w = [x for x in a if x > 1e-9]
    return ("%-34s n=%2d  mean=%+7.3f%%  median=%+7.3f%%  win=%5.1f%%  sum=%+8.2f%%"
            % (label, len(a), st.mean(a), st.median(a), 100.0 * len(w) / len(a), sum(a)))


if __name__ == "__main__":
    T = load()
    print("loaded %d trades with path data\n" % len(T))

    A = [t["actual"] for t in T]
    print(stats(A, "ACTUAL (v9.0, level fills)"))
    print(stats([s1(t) for t in T], "S1a targets+stop, remainder rides"))
    print(stats([s1_conservative(t) for t in T], "S1b targets+stop, else as traded"))
    print(stats([s3(t, 1.0) for t in T], "S3a breakeven stop after +1R"))
    print(stats([s3(t, 0.5) for t in T], "S3b breakeven stop after +0.5R"))
    print(stats([s2(t) for t in T], "S2  perfect exit (CEILING)"))
    print(stats([t["actual"] for t in T if "INITIAL STOP" not in t["exitReason"].upper()],
                "S4  if no trade ever hit its stop"))

    nstop = sum(1 for t in T if t["path"]["stopT"] is not None)
    print("\n   trades whose ORIGINAL stop was touched after entry: %d of %d" % (nstop, len(T)))
    print("   so S1a lets %d trades ride to the end of the 2-day window - that is the"
          % (len(T) - nstop))
    print("   part of S1a that is NOT a tradeable rule. S1b removes it.")
    ch3 = sum(1 for t in T if abs(s3(t, 1.0) - t["actual"]) > 1e-9)
    print("   breakeven-after-+1R changes only %d of %d trades; the rest never armed"
          % (ch3, len(T)))
    print("   or never came back." )

    print("\n-- how far trades actually ran, in units of risk (R) --")
    m = sorted(t["path"]["mfeR"] for t in T)
    for lo, hi, lab in ((-99, 0.5, "under [n]R"), (0.5, 1, "[n]R to [n]R"),
                        (1, 2, "[n]R to [n]R"), (2, 3, "[n]R to [n]R"), (3, 99, "[n]R or better")):
        n = sum(1 for x in m if lo <= x < hi)
        print("   %-16s %2d trades  %4.1f%%" % (lab, n, 100.0 * n / len(m)))
    print("   median peak %.2fR   mean peak %.2fR" % (st.median(m), st.mean(m)))

    print("\n-- was the best price before or after the exit? --")
    def ts(s):
        import datetime as dt
        return int(dt.datetime.strptime(s, "%Y-%m-%dT%H:%M").replace(
            tzinfo=dt.timezone.utc).timestamp())
    after = [t for t in T if t["path"]["mfeT"] and t["path"]["mfeT"] > ts(t["exitT"])]
    print("   peak came AFTER the exit (left on the table): %d of %d" % (len(after), len(T)))
    print("   peak came at or before the exit (rode it back): %d" % (len(T) - len(after)))
