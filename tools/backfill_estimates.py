"""Put the AT-PRINT consensus back on every cached card, from the watcher's own log.

WHY THIS EXISTS. Until 2026-09-30 a card stored only its rendered rows, and a
--rebuild re-asked Finnhub for the estimates. Finnhub revises both rows after a
report - the next quarter's within hours of the guide - and drops a quarter's
row altogether a few weeks later, at which point an old fallback picked the NEXT
quarter's numbers. Rebuilding on 2026-09-29 therefore moved estimates on most
cards (lululemon's Q3 guide was held against $2.36B instead of the $2.58B the
street expected when the guide landed) and gave Marvell's August card Q3's
estimates outright.

Cards built from now on store the estimates they were built with ("est"), and a
rebuild reuses them. This script supplies "est" for the cards built BEFORE that,
using the earliest rendering of each row in watch.log - the FILED block written
when the card was first built, else the earliest REBUILT line. Displayed figures
are rounded, so each estimate is back-solved from the displayed percentage and
the release's exact figure where the two agree, and taken as displayed where
they do not.

    python tools/backfill_estimates.py            # dry run: prints what it would store
    python tools/backfill_estimates.py --write
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib.util
import earnings_read as er

W = os.path.join(HERE, ".watch")
RESULTS = os.path.join(W, "results.json")
LOG = os.path.join(W, "watch.log")

_spec = importlib.util.spec_from_file_location("en", os.path.join(HERE, "earnings-now.py"))
en = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(en)

TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)  (.*)$")
MONEY = re.compile(r"(-?)\$(-?)([\d.]+)([TBMk]?)")
UNIT = {"T": 1e12, "B": 1e9, "M": 1e6, "k": 1e3, "": 1.0}


def money(s):
    m = MONEY.fullmatch(s.strip())
    if not m:
        return None
    v = float(m.group(3)) * UNIT[m.group(4)]
    return -v if (m.group(1) or m.group(2)) else v


def usd(s):
    m = re.fullmatch(r"(-?)\$(-?)([\d.]+)", s.strip())
    if not m:
        return None
    v = float(m.group(3))
    return -v if (m.group(1) or m.group(2)) else v


ROW_EST = re.compile(r"^(?P<lab>EPS[^$]*?|Revenue|Guide[^$]*?)\s{2,}(?P<act>-?\$-?[\d.]+[TBMk]?)\s+vs\s+"
                     r"(?P<est>-?\$-?[\d.]+[TBMk]?)\s+est\s+(?P<tail>.*)$")
ROW_NONE = re.compile(r"^(?P<lab>EPS[^$]*?|Revenue|Guide[^$]*?)\s{2,}.*\(no estimate\)")


def rows_in_log():
    """{ticker: [(time, row_text), ...]} in log order - FILED rows and both sides
    of every REBUILT diff."""
    out = {}
    lines = io.open(LOG, encoding="utf-8", errors="replace").read().splitlines()
    cur = None
    for ln in lines:
        m = TS.match(ln)
        if not m:
            continue
        t, body = m.group(1), m.group(2)
        if body.startswith("FILED  "):
            cur = body[7:].split()[0]
            continue
        if body.startswith("REBUILT "):
            cur = body[8:].split()[0]
            continue
        if cur and body.startswith("       ") and not body.strip().startswith("http"):
            s = body.strip()
            if s.startswith(("- ", "+ ")):
                s = s[2:]
            if s.startswith(("was:", "now:")):
                continue
            out.setdefault(cur, []).append((t, s))
        elif not body.startswith(" "):
            cur = None if not body.startswith(("REBUILT", "FILED")) else cur
    return out


def kind(lab):
    return "eps" if lab.startswith("EPS") else ("rev" if lab == "Revenue" else "guide")


def solve(kind_, act_disp, est_disp, tail, exact):
    """The estimate, as precisely as the display allows."""
    if kind_ == "eps":
        e = usd(est_disp)
        a = usd(act_disp)
        gap = re.search(r"\$([\d.]+)\s+(above|below)", tail)
        if gap and a is not None:
            return round(a - float(gap.group(1)) if gap.group(2) == "above" else a + float(gap.group(1)), 4)
        pm = re.search(r"([+-][\d.]+)%", tail)
        if pm and a not in (None, 0):
            p = float(pm.group(1))
            # a verdict % is (actual - est) / |est|
            for cand in (a / (1 + p / 100.0), a / (1 - p / 100.0)):
                if abs(cand - e) <= 0.0051:
                    return round(cand, 4)
        return e
    e = money(est_disp)
    a = money(act_disp)
    pm = re.search(r"([+-][\d.]+)%", tail)
    if pm and exact and a and er.fmt_money(exact) == er.fmt_money(a):
        cand = exact / (1 + float(pm.group(1)) / 100.0)
        if er.fmt_money(cand) == er.fmt_money(e):
            return round(cand)
    return e


def main(write):
    results = json.load(io.open(RESULTS, encoding="utf-8"))
    logged = rows_in_log()
    cm = en.cik_map()
    changed = 0
    for tk in sorted(results):
        r = results[tk]
        if r.get("est") and not os.environ.get("EPD_REDO_EST"):
            continue
        # the exact figures, for back-solving
        exact = {}
        cik = cm.get(tk)
        if cik and r.get("accession"):
            try:
                _url, text = en.press_release(cik, r["accession"])
                p = er.parse(text or "")
                exact = {"rev": p.get("revenue"), "guide": (p.get("guide") or {}).get("mid")}
            except Exception:
                pass
        first = {}
        for _t, s in logged.get(tk, []):
            m = ROW_EST.match(s)
            if m:
                k = kind(m.group("lab"))
                if k not in first:
                    first[k] = solve(k, m.group("act"), m.group("est"), m.group("tail"), exact.get(k))
                continue
            m = ROW_NONE.match(s)
            if m:
                first.setdefault(kind(m.group("lab")), None)
        est = [[first.get("eps"), first.get("rev")], [None, first.get("guide")]]
        # The calendar label as the 2026-09-29 rebuild found it. Marvell's was the
        # stale-fallback Q3 FY27; its release names the quarter itself, so the
        # label only matters for releases that never do.
        head = (r.get("title") or "").split("  filed ")[0].split()
        cal = " ".join(head[1:3]) if len(head) >= 3 and re.match(r"Q[1-4]$", head[1]) else None
        if tk == "MRVL":
            cal = "Q2 FY27"
        print("%-5s est_now=%s  guide_est=%s  cal=%s" % (tk, est[0], est[1][1], cal))
        r["est"], r["cal_label"] = est, cal
        changed += 1
    if write:
        tmp = RESULTS + ".tmp"
        with io.open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        os.replace(tmp, RESULTS)
        print("wrote %d cards" % changed)
    else:
        print("dry run - %d cards would change (pass --write)" % changed)


if __name__ == "__main__":
    main("--write" in sys.argv)
