"""Score earnings_read against a hand-read answer key.

WHY THIS EXISTS. Extraction quality was being judged by eye, one filing at a
time, which is how three separate wrong numbers reached the chart panel. This
scores every field against values a human read out of the filing itself, so
"it looks right" is replaced by a number.

    set EPD_CORPUS=<dir of TICKER.txt exhibits>
    python tools/score_extraction.py

The answer key ships with the repo (backtest/extraction-truth-*.json). The
press releases themselves do not - they are public, re-fetchable with
tools/earnings-now.py, and large.

A field is RIGHT when it matches within tolerance, MISSED when the key has a
value and the parser returned None, WRONG when both have values and disagree,
EXTRA when the parser invented one the key says is not there. WRONG and EXTRA are
the categories that matter: a missed row is a blank line on the panel, a wrong
row is a number the owner might trade on.
"""
import os, sys, json, io

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import earnings_read as er

# EPD_TRUTH points at a different answer key - each earnings season gets its own,
# hand-read from releases the parser has never seen.
TRUTH = os.environ.get("EPD_TRUTH") or os.path.join(HERE, "..", "backtest",
                                                     "extraction-truth-2026-09-05.json")
BASE = os.environ.get("EPD_CORPUS", os.path.join(HERE, ".watch", "corpus"))
truth = json.load(io.open(TRUTH, encoding="utf-8"))


def close(a, b, rel=0.02, absol=None):
    if a is None or b is None:
        return False
    if absol is not None and abs(a - b) <= absol:
        return True
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= rel


FIELDS = ["eps", "revenue", "growth", "margin", "capex", "guide_lo", "guide_hi", "guide_per",
          "raised", "quarter", "annual"]
# EPD_SHOW_MISSED=1 lists blanks per filing too - a blank row is not a wrong one,
# but a blank where the release plainly states the figure is still worth seeing.
SHOW_MISSED = bool(os.environ.get("EPD_SHOW_MISSED"))
tally = {f: {"right": 0, "wrong": 0, "missed": 0, "blank": 0, "extra": 0} for f in FIELDS}
detail = []
seen = 0

for tk in sorted(truth):
    t = truth[tk]
    fp = os.path.join(BASE, tk + ".txt")
    if not os.path.exists(fp):
        continue
    seen += 1
    p = er.parse(io.open(fp, encoding="utf-8").read())
    # The label is scored as the PANEL prints it: the release's own words, else
    # the exhibit's file name, else the calendar (recorded in the key when the
    # release leaves the year unsaid) - the same precedence the watcher uses.
    up = fp[:-4] + ".url"
    url = io.open(up, encoding="utf-8").read().strip() if os.path.exists(up) else None
    p["quarter"] = er.final_quarter(p, url, t.get("calendar_label"))
    g = p.get("guide") or {}
    got = {"eps": p.get("eps"), "revenue": p.get("revenue"), "growth": p.get("growth"),
           "margin": p.get("margin"), "capex": p.get("capex"),
           "guide_lo": g.get("lo"), "guide_hi": g.get("hi"),
           # The period the PANEL will print for the guide - "Q3", "FY", "6M" -
           # because a right figure under the wrong period is a wrong row: HPE's
           # Q4 guide shown as a full-year one never gets compared to anything.
           "guide_per": er.guide_period_class(p) if g else None,
           "raised": p.get("guide_dir"), "quarter": p.get("quarter"),
           "annual": bool(p.get("annual_only"))}
    # The panel shows non-GAAP where it exists, so that is what gets scored.
    exp = {"eps": t.get("eps_nongaap") if t.get("eps_nongaap") is not None else t.get("eps_gaap"),
           "revenue": t.get("revenue"), "growth": t.get("rev_growth"),
           "margin": t.get("gross_margin") if t.get("gross_margin") is not None else t.get("op_margin"),
           "capex": t.get("capex"), "guide_lo": t.get("guide_rev_lo"),
           "guide_hi": t.get("guide_rev_hi"),
           "guide_per": t.get("guide_period") if "guide_period" in t else got["guide_per"],
           "raised": t.get("guide_raised") if t.get("guide_raised") in ("raised", "lowered") else None,
           "quarter": t.get("quarter_label"),
           "annual": bool(t.get("annual_only"))}
    line = [tk]
    for f in FIELDS:
        a, b = got[f], exp[f]
        if f == "quarter":
            # The key writes the year as the release does - "Q3 FY2026", "Q2
            # 2026", "June Quarter 2026" - and the panel has room only for
            # "Q3 FY26". Compare the quarter number and the last two digits of
            # the year, which is the whole of what the panel asserts.
            def _qy(x):
                import re as _re
                if not x:
                    return None
                m = _re.search(r"Q([1-4])\D{0,6}(?:FY)?\s*((?:19|20)?\d{2})", str(x), _re.I)
                if m:
                    return (m.group(1), m.group(2)[-2:])
                # A fiscal-year-only release is labelled by its year: "FY2026" / "FY26".
                m = _re.match(r"\s*FY\s*((?:19|20)?\d{2})\s*$", str(x), _re.I)
                return ("FY", m.group(1)[-2:]) if m else None
            a, b = _qy(a), _qy(b)
            ok = (a == b)
        elif f in ("raised", "guide_per"):
            ok = (a == b)
        elif f == "annual":
            # scored only when either side says the release is annual
            if not a and not b:
                a = b = None
            ok = (a == b)
        elif f == "growth":
            # A point either way is rounding; beyond that a relative band would
            # pass Aeluma's -52% (the SEQUENTIAL quarter) against a true -56%.
            ok = close(a, b, rel=0.0, absol=1.0)
        elif f == "margin":
            ok = close(a, b, rel=0.02, absol=0.6)
        elif f == "eps":
            ok = close(a, b, rel=0.02, absol=0.02)
        else:
            ok = close(a, b, rel=0.03)
        if b is None and a is None:
            tally[f]["blank"] += 1
        elif b is None and a is not None:
            tally[f]["extra"] += 1
            line.append("%s=EXTRA(%s)" % (f, a))
        elif a is None:
            tally[f]["missed"] += 1
            if SHOW_MISSED:
                line.append("%s=missed(%s)" % (f, b))
        elif ok:
            tally[f]["right"] += 1
        else:
            tally[f]["wrong"] += 1
            line.append("%s: got %s want %s" % (f, a, b))
    if len(line) > 1:
        detail.append(line)

if not seen:
    sys.exit("no filings found under %s - set EPD_CORPUS to a directory of TICKER.txt" % BASE)

print("scored %d filings from %s\n" % (seen, BASE))
print("%-10s %6s %6s %7s %6s %6s" % ("field", "right", "WRONG", "missed", "blank", "extra"))
for f in FIELDS:
    d = tally[f]
    print("%-10s %6d %6d %7d %6d %6d"
          % (f, d["right"], d["wrong"], d["missed"], d["blank"], d["extra"]))

print("\nPER-FILING PROBLEMS")
for line in detail:
    print("  %-6s %s" % (line[0], "; ".join(line[1:])[:170]))
