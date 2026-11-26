"""The v10.x out-of-sample roster: significant September 2026 reporters.

Selection follows ROSTER-v9.0-candidates.md's categories:
  B  high-profile tech
  D  high-beta / retail favourites
  E  slow large-cap control group, kept so that the idea that the indicator
     works better on high-profile names keeps being re-tested instead of
     quietly dropped
The 2x-leveraged-ETF label (group C in v9) is applied afterwards as an overlay,
and only where verified.

The CORE of this roster is the v10 candidate list's own drop-outs: AVGO, DELL,
PANW, SNOW, HPE, ZS, MDB, DOCU, PATH, LULU, CHWY, FIVE, AEO, M, NIO, CAG, CPB,
NTAP were all hand-picked for v10 and dropped only because they had not reported
inside its August window. They reported in September.

Dates are the Finnhub calendar's; the harness snaps each job to TradingView's own
release event on the 1-minute chart, so a wrong calendar hour cannot misplace a
measurement - it only sets where the search starts.

Excluded: v10's measured names (none reported again in the window anyway);
funds (PDI, UTF); names that had not reported by 2026-09-29 (MU, FDX, JBL, ACN,
checked against EDGAR).

    python backtest/roster_v10x.py   ->  backtest/roster-v10x.json
"""
import json, os, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.environ.get("EPD_CALENDAR") or os.path.join(HERE, "sep-calendar-2026.json")
OUT = os.path.join(HERE, "roster-v10x.json")

GROUPS = {
    "B": "AVGO ORCL DELL PANW SNOW ADBE HPE CIEN NTAP CRDO MDB ZS IOT GWRE DOCU SAIL "
         "GTLB PATH TTAN NTSK BRZE AMBA PL NAVN",
    "D": "GME CHWY LULU FIVE AEO ASO SIG M VSXY OLLI RH PVH CCL UEC NIO KMX AVAV",
    "E": "COST MDT KR CTAS AZO PAYX GIS DRI LEN CPB CAG CASY CPRT KBH THO COO CNM "
         "SUNB TTC",
}
V10_MEASURED = set(
    "ADI ADSK AFRM AMAT ANF BABA BBY BIDU BJ BURL CBRS COHR CRM CRWV CSCO DE DG DKS "
    "DLTR ESTC FN GPS HD HPQ INFQ KSS LITE LOW LUNR MRVL NBIS NTNX NVDA OKTA ROST S "
    "SJM SMCI SPCE TGT TJX ULTA VEEV WDAY WMT XPEV".split())

# Where the release is expected, in UTC. September is EDT (UTC-4): pre-market
# releases land ~06:00-08:00 ET, after-close ones ~16:00-16:30 ET.
HOUR_UTC = {"bmo": 11, "amc": 20, "dmh": 16, "": 16}

# EVERY SYMBOL IS SENT WITH ITS EXCHANGE. The first run sent bare tickers and
# "SAIL" - SailPoint, on NASDAQ - resolved to Steel Authority of India on the NSE.
# Loading it broke TradingView's data connection, and every symbol after it got
# that stock's 300 stale bars under its own name. A bare ticker is ambiguous
# across the world's exchanges; a qualified one is not.
EXCH = {"NASDAQ": "NASDAQ", "NEW YORK STOCK EXCHANGE": "NYSE", "NYSE MKT": "AMEX",
        "NYSE ARCA": "AMEX", "CBOE": "CBOE"}


def tv_symbol(sym, exch):
    for k, v in EXCH.items():
        if (exch or "").upper().startswith(k):
            return "%s:%s" % (v, sym)
    return None


# EDGAR beats the calendar. Finnhub's date is sometimes an estimate that never
# happened: General Mills was listed for 09-15 and actually filed its Item 2.02
# 8-K on 09-23, so the harness searched the wrong week and found no release;
# Conagra was listed for 09-29 and has not reported at all (last earnings 8-K
# 2026-07-15). Both checked against EDGAR's submissions feed on 2026-09-29.
DATE_FIX = {"GIS": ("2026-09-23", "bmo")}
NOT_REPORTED = {"CAG"}


def main():
    cal = json.load(open(CAL, encoding="utf-8"))
    for r in cal:
        if r.get("sym") in DATE_FIX:
            r["date"], r["hour"] = DATE_FIX[r["sym"]]
    cal = [r for r in cal if r.get("sym") not in NOT_REPORTED]
    first = {}
    for r in cal:
        s, d = r.get("sym"), r.get("date")
        if s and d and "2026-09-01" <= d <= "2026-09-29" and s not in first:
            first[s] = r
    jobs, missing = [], []
    for grp, syms in GROUPS.items():
        for s in syms.split():
            if s in V10_MEASURED:
                continue
            r = first.get(s)
            if not r:
                missing.append(s)
                continue
            day = dt.datetime.strptime(r["date"], "%Y-%m-%d")
            hh = HOUR_UTC.get((r.get("hour") or "").lower(), 16)
            exp = int(day.replace(hour=hh, tzinfo=dt.timezone.utc).timestamp())
            tv = tv_symbol(s, r.get("exch"))
            if not tv:
                missing.append("%s(exchange %r)" % (s, r.get("exch")))
                continue
            jobs.append({"sym": s, "tv": tv, "grp": grp, "earnIso": "%sT%02d:00" % (r["date"], hh),
                         "calHour": r.get("hour") or "", "expectTs": exp,
                         "from": exp - 30 * 86400, "to": exp + 3 * 86400,
                         "minFirstTs": exp - 20 * 86400})
    jobs.sort(key=lambda j: j["expectTs"])
    json.dump(jobs, open(OUT, "w", encoding="utf-8"), indent=1)
    print("%d jobs -> %s" % (len(jobs), OUT))
    if missing:
        print("not on the September calendar (dropped):", " ".join(missing))


if __name__ == "__main__":
    main()
