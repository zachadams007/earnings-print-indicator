"""Re-fetch the press releases the answer key was hand-read from.

The key (backtest/extraction-truth-*.json) ships with the repo; the filings
do not, because they are public, large, and re-fetchable. Without them
score_extraction.py has nothing to score and exits, which is how a scored
extractor quietly stopped being scored.

    python tools/build_corpus.py                 # every ticker in the key
    python tools/build_corpus.py --tickers AVAV  # just one

CUTOFF matters. The key was read on a date; a ticker that has reported AGAIN
since then would otherwise be scored against the wrong quarter. So the filing
chosen is the newest Item 2.02 8-K accepted ON OR BEFORE the key's date, which is
taken from the key's filename.
"""
import argparse, io, json, os, re, sys, time, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("en", os.path.join(HERE, "earnings-now.py"))
en = importlib.util.module_from_spec(_s)
_s.loader.exec_module(en)

TRUTHDIR = os.path.join(HERE, "..", "backtest")
OUT = os.path.join(HERE, ".watch", "corpus")


def newest_2_02_before(cik, cutoff):
    """(accession, filed) of the newest Item 2.02 8-K filed on or before cutoff."""
    txt = en.get("%s/submissions/%s.json" % (en.DATA, cik))
    if not txt:
        return None, None
    r = json.loads(txt)["filings"]["recent"]
    for i, form in enumerate(r.get("form", [])):
        if form != "8-K":
            continue
        if "2.02" not in ((r.get("items") or [""] * len(r["form"]))[i] or ""):
            continue
        if r["filingDate"][i] > cutoff:
            continue
        return r["accessionNumber"][i], r["filingDate"][i]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="")
    ap.add_argument("--key", default="")
    a = ap.parse_args()

    key = a.key or sorted(f for f in os.listdir(TRUTHDIR)
                          if f.startswith("extraction-truth-"))[-1]
    cutoff = re.search(r"(\d{4}-\d{2}-\d{2})", key).group(1)
    truth = json.load(io.open(os.path.join(TRUTHDIR, key), encoding="utf-8"))
    want = [t.strip().upper() for t in a.tickers.split(",") if t.strip()] or sorted(truth)

    cm = en.cik_map()
    os.makedirs(OUT, exist_ok=True)
    print("key %s  cutoff %s  %d tickers -> %s\n" % (key, cutoff, len(want), OUT))
    ok = 0
    for tk in want:
        cik = cm.get(tk)
        if not cik:
            print("  %-6s no CIK" % tk)
            continue
        try:
            # A PINNED ACCESSION BEATS THE CUTOFF RULE. The cutoff comes from the
            # key's filename, but an entry hand-read from a filing that landed
            # AFTER that date - AeroVironment's, read on 2026-09-10 - would be
            # scored against the previous quarter and look like a parser failure.
            pin = (truth.get(tk) or {}).get("source_accession")
            if pin:
                acc, filed = pin, "pinned"
            else:
                acc, filed = newest_2_02_before(cik, cutoff)
            if not acc:
                print("  %-6s no Item 2.02 8-K on or before %s" % (tk, cutoff))
                continue
            url, text = en.press_release(cik, acc)
            if not text:
                print("  %-6s could not read %s" % (tk, acc))
                continue
            io.open(os.path.join(OUT, tk + ".txt"), "w", encoding="utf-8").write(text)
            io.open(os.path.join(OUT, tk + ".url"), "w", encoding="utf-8").write(url or "")
            print("  %-6s %s  %s" % (tk, filed, url))
            ok += 1
        except Exception as e:
            print("  %-6s ERROR %s" % (tk, e))
        time.sleep(0.25)
    print("\n%d/%d written" % (ok, len(want)))


if __name__ == "__main__":
    main()
