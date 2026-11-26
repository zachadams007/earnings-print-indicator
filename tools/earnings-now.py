#!/usr/bin/env python3
"""earnings-now - see an earnings result within a minute of it being released.

THE POINT. Waiting for articles costs 20-30 minutes. The press release itself is
public the moment it is filed: a company reporting results files an 8-K carrying
"Item 2.02 - Results of Operations", with the full press release attached as
EX-99.1. That filing is timestamped and free.

Measured on this project's own backtest data: Marvell's Q2 print candle was at
20:05 UTC on 2026-08-27. Its 8-K was ACCEPTED by EDGAR at 20:05:59 UTC. Same
minute as the move. That is the gap this closes.

WHAT IT DOES
  1. Before the release, pulls the consensus EPS and revenue from Finnhub so
     there is something to judge the number against.
  2. Polls EDGAR for each watched company and fires the moment an 8-K with
     Item 2.02 appears.
  3. Pulls the press release, pulls out the headline numbers, and prints a card
     with beat/miss against consensus - plus the direct link, because the
     extraction is best-effort and the release itself never is.

USAGE
  python tools/earnings-now.py --tickers NVDA,CRM,MRVL
  python tools/earnings-now.py --today            # whatever Finnhub says reports today
  python tools/earnings-now.py --tickers MRVL --replay 2026-08-27   # prove it works

  --interval N   seconds between polls (default 20)
  --once         check once and exit, instead of watching

NOTES
  * SEC asks for a real User-Agent and rate-limits at 10 requests/second. This
    stays far under that and identifies itself properly. Do not remove the header.
  * Finnhub is optional. Without a key you still get the numbers, just no
    consensus to compare them to.
"""
import argparse, json, os, re, sys, time, urllib.request, urllib.error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

UA = os.environ.get("SEC_USER_AGENT", "earnings-now contact@example.com")
SEC = "https://www.sec.gov"
DATA = "https://data.sec.gov"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cik-cache.json")
SECRETS = os.environ.get("EARNINGS_SECRETS_ENV", ".env")

PUSH = False          # set by --push; card() writes to the chart panel when true

BOLD, DIM, GRN, RED, YEL, CYA, OFF = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m")


def get(url, tries=3):
    for k in range(tries):
        try:
            rq = urllib.request.Request(url, headers={
                "User-Agent": UA, "Accept-Encoding": "gzip, deflate", "Host": url.split("/")[2]})
            with urllib.request.urlopen(rq, timeout=25) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                return raw.decode("utf-8", "ignore")
        except Exception as e:
            if k == tries - 1:
                return None
            time.sleep(1.2 * (k + 1))
    return None


_FH_KEYS = []
_FH_I = [0]


def finnhub_keys():
    """Every FINNHUB_API_KEY* in the secrets file. The free tier allows 60 calls a
    minute PER KEY; a bulk rebuild on one key hit that wall and silently lost the
    consensus figures on cards that already had them."""
    if not _FH_KEYS:
        try:
            for line in open(SECRETS, encoding="utf-8", errors="ignore"):
                if line.startswith("FINNHUB_API_KEY"):
                    k = line.split("=", 1)[1].strip().strip('"')
                    if k and k not in _FH_KEYS:
                        _FH_KEYS.append(k)
        except Exception:
            pass
        env = os.environ.get("FINNHUB_API_KEY")
        if env and env not in _FH_KEYS:
            _FH_KEYS.append(env)
    return _FH_KEYS


def _next_key(fallback):
    ks = finnhub_keys()
    if not ks:
        return fallback
    _FH_I[0] = (_FH_I[0] + 1) % len(ks)
    return ks[_FH_I[0]]


def finnhub_key():
    try:
        for line in open(SECRETS, encoding="utf-8", errors="ignore"):
            if line.startswith("FINNHUB_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return os.environ.get("FINNHUB_API_KEY")


def cik_map():
    """ticker -> zero-padded CIK. Cached; the file is ~1MB and rarely changes."""
    if os.path.exists(CACHE) and time.time() - os.path.getmtime(CACHE) < 7 * 86400:
        try:
            return json.load(open(CACHE, encoding="utf-8"))
        except Exception:
            pass
    txt = get(SEC + "/files/company_tickers.json")
    if not txt:
        sys.exit("could not reach SEC to map tickers to CIKs")
    m = {v["ticker"].upper(): "CIK%010d" % v["cik_str"] for v in json.loads(txt).values()}
    try:
        json.dump(m, open(CACHE, "w", encoding="utf-8"))
    except Exception:
        pass
    return m


def consensus(tickers, key):
    """{ticker: (epsEst, revEst)} for anything reporting in the next few days."""
    out = {}
    if not key:
        return out
    from datetime import date, timedelta
    a = date.today() - timedelta(days=1)
    b = date.today() + timedelta(days=6)
    txt = get("https://finnhub.io/api/v1/calendar/earnings?from=%s&to=%s&token=%s" % (a, b, key))
    if not txt:
        return out
    try:
        for e in json.loads(txt).get("earningsCalendar", []):
            if e.get("symbol", "").upper() in tickers:
                out[e["symbol"].upper()] = (e.get("epsEstimate"), e.get("revenueEstimate"), e.get("hour"))
    except Exception:
        pass
    return out


def calendar_facts(ticker, key, when=None):
    """(est_now, est_next, fiscal_label) from ONE earnings-calendar request.

    est_now  = (epsEstimate, revenueEstimate) for the quarter this filing reports
    est_next = the same for the quarter after it - what the street expects of the
               GUIDE, the only free source Finnhub offers for that number
    fiscal_label = "Q2 FY27" as the calendar states it

    ANCHORED ON THE FILING'S DATE, not on today. "The newest row with an actual"
    is right for a filing read the minute it lands and wrong for any re-read later;
    the row nearest the filing date is right both times.

    ONE REQUEST, not two, and on a rotating key - see finnhub_keys().
    """
    none = ((None, None), (None, None), None)
    if not key:
        return none
    from datetime import date, timedelta
    d = when or date.today()
    txt = get("https://finnhub.io/api/v1/calendar/earnings?from=%s&to=%s&symbol=%s&token=%s"
              % (d - timedelta(days=40), d + timedelta(days=400), ticker, _next_key(key)))
    if not txt:
        return none
    try:
        rows = sorted(json.loads(txt).get("earningsCalendar", []) or [], key=lambda e: e["date"])
    except Exception:
        return none
    if not rows:
        return none
    near = [e for e in rows if abs((date.fromisoformat(e["date"]) - d).days) <= 10]
    if not near:
        # The calendar keeps recent and upcoming quarters only, and drops a
        # quarter's row a few weeks after it reports. Guessing a row here - "the
        # newest with an actual, else the earliest" - picked the NEXT quarter and
        # labelled Marvell's August print Q3 FY27 against Q3's estimates. No row
        # for this filing's date means no answer.
        return none
    now = min(near, key=lambda e: abs((date.fromisoformat(e["date"]) - d).days))
    i = rows.index(now)
    nxt = rows[i + 1] if i + 1 < len(rows) else {}
    fl = None
    if now.get("quarter") and now.get("year"):
        fl = "Q%d FY%s" % (int(now["quarter"]), str(now["year"])[-2:])
    return ((now.get("epsEstimate"), now.get("revenueEstimate")),
            (nxt.get("epsEstimate"), nxt.get("revenueEstimate")), fl)


def fiscal_label(ticker, key, when=None):
    """'Q2 FY27' from the earnings calendar. ServiceTitan's release says only
    "fiscal second quarter ended July 31, 2026"; reading the year off that date
    gave FY26, but its fiscal year ends in January, so the quarter is FY27."""
    return calendar_facts(ticker, key, when)[2]


def consensus_pair(ticker, key, when=None):
    """((epsEst, revEst) for the quarter just reported, same for the one guided)."""
    a, b, _ = calendar_facts(ticker, key, when)
    return a, b


# ═══════════════════════════════════════════════════════════════════════════
# WHEN WAS IT ACCEPTED?  (Eastern, always)
# ═══════════════════════════════════════════════════════════════════════════
#
# AND THE SOURCE IS A TRAP. data.sec.gov's acceptanceDateTime carries a "Z", but
# for a filing that has JUST landed the value is EASTERN. AeroVironment's 8-K was
# read two minutes after it appeared and the field said "2026-09-09T16:04:03Z";
# EDGAR's own filing index says Accepted 2026-09-09 16:04:03, and the release
# itself says the call was that day at 4:30 pm Eastern. Hours later the same
# field reads "20:04:03Z" - the true UTC. Stamping the fresh value as UTC and
# converting to local shifted every card FOUR HOURS EARLY: AeroVironment's after
# -the-close print showed as 12:04:03, Shoe Carnival's pre-market release as
# 02:15:14. On a tool whose whole purpose is telling the owner whether the tape moved
# before the filing or after it, a four-hour error is the difference between a
# read and a guess.
#
# So the timestamp is taken from the filing index page, which states Accepted in
# Eastern and does not change, and the submissions field is only a fallback.
ET = ZoneInfo("America/New_York")
_ACCEPTED_RX = re.compile(r"Accepted\s*</div>\s*<div[^>]*>\s*"
                          r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", re.I)
_ACCEPTED_LOOSE = re.compile(r"Accepted[^0-9]{0,40}(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", re.I)


def accepted_et(cik, accession, raw=""):
    """When EDGAR accepted the filing, as a timezone-aware Eastern datetime.

    Authoritative from the filing's index page. Falls back to the submissions
    field read AS EASTERN, because that is what it holds while a filing is fresh
    - which is the only time this tool ever reads it.
    """
    acc = (accession or "").replace("-", "")
    if acc and cik:
        url = "%s/Archives/edgar/data/%d/%s/%s-index.htm" % (
            SEC, int(str(cik)[3:] if str(cik).upper().startswith("CIK") else cik),
            acc, accession)
        page = get(url)
        if page:
            m = _ACCEPTED_RX.search(page) or _ACCEPTED_LOOSE.search(page)
            if m:
                try:
                    return datetime.strptime(m.group(1) + " " + m.group(2),
                                             "%Y-%m-%d %H:%M:%S").replace(tzinfo=ET)
                except Exception:
                    pass
    try:
        return datetime.strptime((raw or "")[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ET)
    except Exception:
        return None


def fmt_et(ts):
    """'4:04:03 PM ET' - the only clock this tool shows."""
    if ts is None:
        return ""
    return ts.astimezone(ET).strftime("%-I:%M:%S %p ET").replace("AM", "AM").replace("PM", "PM") \
        if os.name != "nt" else ts.astimezone(ET).strftime("%#I:%M:%S %p ET")


def recent_8ks(cik, want_item="2.02", limit=4):
    """The newest 8-Ks carrying the results item, newest first.

    [] when the company has none; None when EDGAR could not be read - the two
    mean different things to a caller deciding whether to try again."""
    txt = get("%s/submissions/%s.json" % (DATA, cik))
    if not txt:
        return None
    try:
        r = json.loads(txt)["filings"]["recent"]
    except Exception:
        return None
    out = []
    for i, form in enumerate(r.get("form", [])):
        if form != "8-K":
            continue
        items = (r.get("items") or [""] * len(r["form"]))[i]
        if want_item not in items:
            continue
        out.append({
            "accession": r["accessionNumber"][i],
            "accepted": (r.get("acceptanceDateTime") or [""] * len(r["form"]))[i],
            "filed": r["filingDate"][i],
            "items": items,
            "cik": cik,
        })
        if len(out) >= limit:
            break
    return out


def recent_8k(cik, want_item="2.02"):
    """Newest 8-K carrying the results item. Returns None if there isn't one."""
    got = recent_8ks(cik, want_item, limit=1)
    return got[0] if got else None


def press_release(cik, accession):
    """(url, plain text) of the exhibit carrying the numbers.

    Uses the filing's index.json - the HTML directory listing hides the documents
    for some filers (Autodesk is one), which silently returned nothing.

    Naming is not consistent across companies. Marvell files
    "q227_8kx812026ex-991.htm"; Autodesk files "q227pressrelease.htm" with no
    "ex-99" anywhere in it. So try the convention first, then the words, then fall
    back to the biggest document that is not the 8-K body itself.
    """
    acc = accession.replace("-", "")
    base = "%s/Archives/edgar/data/%d/%s" % (SEC, int(cik[3:]), acc)
    j = get(base + "/index.json")
    names, size = [], {}
    if j:
        try:
            for it in json.loads(j)["directory"]["item"]:
                if it["name"].lower().endswith((".htm", ".html")):
                    names.append(it["name"])
                    try:
                        size[it["name"]] = int(it.get("size") or 0)
                    except Exception:
                        size[it["name"]] = 0
        except Exception:
            names = []
    if not names:                                     # last resort: scrape the listing
        idx = get(base + "/") or ""
        names = [c.split("/")[-1] for c in re.findall(r'href="([^"]*?\.html?)"', idx, re.I)
                 if acc in c]

    def rank(n):
        low = n.lower()
        if re.search(r"index|xbrl|metalinks|filingsummary", low):
            return None                               # never the press release
        if re.match(r"r\d+\.htm", low):
            return None                               # XBRL viewer fragments
        if re.search(r"ex-?99", low):
            return 0
        if re.search(r"press|release|earnings|results|announce", low):
            return 1
        if re.match(r"[a-z]+-\d{8}\.htm", low):
            return 3                                  # the 8-K body itself
        return 2

    # Within a rank, take the LARGEST document. NVIDIA files both the press release
    # and a shorter "CFO commentary" in the same 8-K; the release is the bigger one.
    scored = sorted(((rank(n), -size.get(n, 0), n) for n in names if rank(n) is not None))
    if not scored:
        return None, None
    pick = scored[0][2]
    url = base + "/" + pick
    h = get(url)
    if not h:
        return url, None
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", h, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = t.replace("&#160;", " ").replace("&nbsp;", " ").replace("&amp;", "&")
    t = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), t)
    t = re.sub(r"[ \t\xa0]+", " ", t)
    return url, t.strip()


# Extraction lives in earnings_read.py now - one implementation, shared by the
# terminal card, the chart panel and the watcher, so a fix lands in all three.


def bullets(text, n=6):
    if not text:
        return []
    parts = re.split(r"[\u2022\u25cf]|&#8226;", text[:3500])
    got = [re.sub(r"\s+", " ", p).strip(" :;-") for p in parts[1:]]
    return [g for g in got if 12 < len(g) < 220][:n]


def card(tk, f, url, text, est, est_next=None):
    """The terminal version of what the chart panel shows.

    Both go through earnings_read so there is ONE extractor. The version that
    lived here read "non-GAAP EPS" into the dictionary and then looked up
    "nonGAAP EPS" to compute the surprise, so the beat/miss line never printed
    once - a hyphen's worth of difference, invisible because the card still
    looked full.
    """
    import earnings_read as er
    p = er.parse(text)
    acc = f.get("accepted", "")
    try:
        ts = accepted_et(f.get("cik") or "", f.get("accession") or "", acc)
        age = (datetime.now(ET) - ts).total_seconds() / 60.0
        when = "%s   (%.0f min ago)" % (fmt_et(ts), age)
    except Exception:
        when = acc
    print("\n" + BOLD + "=" * 74 + OFF)
    print(BOLD + "  %-8s EARNINGS FILED" % tk + OFF + "     " + CYA + when + OFF)
    print(BOLD + "=" * 74 + OFF)
    _, rows_ = er.rows(p, est, est_next, ticker=tk)
    for row in rows_:
        lab, _, val = row.partition("|")
        col = GRN if "BEAT" in val or "ABOVE" in val or "RAISED" in val else (
            RED if "MISS" in val or "BELOW" in val or "LOWERED" in val else "")
        print("  %-14s %s%s%s" % (lab.strip() + ":", col, val.strip(), OFF))
    if not rows_:
        print("  " + DIM + "nothing could be read from this release - open the link" + OFF)
    bs = bullets(text)
    if bs:
        print("\n  " + DIM + "headline points" + OFF)
        for b in bs:
            print("    - " + b[:150])
    elif text:
        # Extraction is best effort and formats vary wildly. When the parser finds
        # nothing, show the opening of the release rather than an empty card - the
        # numbers are almost always in the first couple of hundred words.
        print("\n  " + DIM + "opening of the release" + OFF)
        body = re.sub(r"\s+", " ", text)
        # drop EDGAR's own document header - "EX-99.1 2 q227pressrelease.htm EX-99.1 Document"
        body = re.sub(r"^\s*EX-?\d+[\d.]*\s+\d+\s+\S+\s+EX-?\d+[\d.]*\s*(Document)?\s*", "",
                      body, flags=re.I)
        for i in range(0, min(len(body), 660), 105):
            print("    " + body[i:i + 105].strip())
    print("\n  " + YEL + (url or "(document not found - open the filing index on EDGAR)") + OFF)
    print()
    if PUSH:
        try:
            import tv_push
            # Only ever write a result onto the chart it belongs to. Pushing blind
            # is how Marvell's numbers ended up under a Palo Alto chart.
            on = tv_push.chart_symbol()
            if on and on != tk:
                print("  " + YEL + "chart is on %s, not %s - not pushing" % (on, tk) + OFF + "\n")
            else:
                r = tv_push.send("%s  %s" % (tk, when), rows_)
                print("  " + (DIM + "pushed to the chart panel" if r.get("ok")
                              else RED + "panel push failed: " + str(r.get("err"))) + OFF + "\n")
        except Exception as e:
            print("  " + RED + "panel push failed: %s" % e + OFF + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="")
    ap.add_argument("--today", action="store_true")
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--replay", default="", help="show that date's filing, to prove it works")
    ap.add_argument("--push", action="store_true",
                    help="also write the result into the indicator's panel on the chart")
    a = ap.parse_args()

    global PUSH
    PUSH = a.push

    key = finnhub_key()
    cm = cik_map()

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    if a.today and key:
        from datetime import date
        txt = get("https://finnhub.io/api/v1/calendar/earnings?from=%s&to=%s&token=%s"
                  % (date.today(), date.today(), key))
        if txt:
            tickers += [e["symbol"].upper() for e in json.loads(txt).get("earningsCalendar", [])
                        if e.get("symbol", "").upper() in cm]
    tickers = [t for t in dict.fromkeys(tickers)]
    if not tickers:
        sys.exit("give me --tickers NVDA,CRM  or --today")

    unknown = [t for t in tickers if t not in cm]
    tickers = [t for t in tickers if t in cm]
    if unknown:
        print(DIM + "not found at SEC (skipped): " + ", ".join(unknown) + OFF)

    est = consensus(set(tickers), key)
    print(BOLD + "watching %d: %s" % (len(tickers), ", ".join(tickers)) + OFF)
    for t in tickers:
        e = est.get(t)
        if e and e[0] is not None:
            print("  %-6s consensus EPS %-8s %s" % (t, e[0], ("(" + e[2] + ")") if e[2] else ""))
    if not key:
        print(DIM + "  no Finnhub key found - numbers still work, no consensus comparison" + OFF)

    if a.replay:
        for t in tickers:
            f = recent_8k(cm[t])
            if f and f["filed"] == a.replay:
                url, txt = press_release(cm[t], f["accession"])
                en_, nx_ = consensus_pair(t, key)
                card(t, f, url, txt, en_, nx_)
            else:
                print(DIM + "  %s: no Item 2.02 8-K filed on %s" % (t, a.replay) + OFF)
        return

    seen = {}
    for t in tickers:                       # baseline, so only NEW filings fire
        f = recent_8k(cm[t])
        seen[t] = f["accession"] if f else None
    print(DIM + "baseline set. watching every %ds. ctrl-c to stop." % a.interval + OFF)

    while True:
        for t in tickers:
            f = recent_8k(cm[t])
            if f and f["accession"] != seen.get(t):
                seen[t] = f["accession"]
                url, txt = press_release(cm[t], f["accession"])
                try:
                    sys.stdout.write("\a")
                except Exception:
                    pass
                en_, nx_ = consensus_pair(t, key)
                card(t, f, url, txt, en_, nx_)
            time.sleep(0.4)                 # stay well inside SEC's 10/sec
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nstopped.")
