"""earnings_watch - the earnings panel, running by itself.

The previous version did not run by itself.
Installed once by tools/install-earnings-watch.ps1, it starts at logon, has no
window, survives reboots, and keeps running whether or not TradingView is open.

WHAT IT DOES, in two loops that do NOT block each other:

  THE FILING LOOP (every ~25s)
    Rebuilds today's reporter list from the Finnhub earnings calendar each
    morning, then polls SEC EDGAR for an 8-K carrying Item 2.02. When one lands
    it reads the release, fetches consensus for the quarter just reported AND for
    the quarter being guided, and writes a finished result into results.json.
    Marvell's 8-K was accepted 20:05:59 on the same minute as its print candle -
    that is the gap this closes.

  THE CHART LOOP (every ~2s, on its own thread)
    Reads which symbol the chart is on and renders that company's LATEST results
    card - however old it is - building it from EDGAR on the spot when the watcher
    never caught the filing. Clears the panel only for a symbol that has no
    results filing at all (an ETF, a fund, a foreign issuer).

WHY THE SECOND LOOP EXISTS. A result is true of exactly one company. The previous
build pushed whatever it had just found, so Marvell's numbers sat on a Palo Alto
chart directly beneath rows that WERE about Palo Alto. Keying the panel to the
chart's own symbol makes that impossible, and makes switching symbols show the
right result with no action at all.

WHY IT IS A THREAD. It used to run between sweeps, and a sweep of ~50 names takes
30-50 seconds, so the panel answered a symbol change half a minute late.

A CARD NEVER EXPIRES. Cards used to vanish after 30 hours, and a company whose print
the watcher missed had no card at all. Now the card stays until the company files
its next one.

BAR REPLAY. Each card carries the moment the SEC accepted it. In Replay the
indicator keeps the block hidden until the replay clock reaches that moment, and
says how long ago it was by the replay's clock, not the wall clock.

WHAT IT TOUCHES ON THE CHART. Display-only string inputs in_130..in_138, which
the Pine declares last. Nothing else - no symbol changes, no timeframe changes, no
trading inputs.

    python tools/earnings_watch.py --once      # one pass, print what it would do
    python tools/earnings_watch.py --no-chart  # collect results, never touch TV
    python tools/earnings_watch.py --status    # what is cached right now
"""
import argparse, json, os, sys, time, traceback
from datetime import datetime, timezone, date, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import earnings_read as er                                          # noqa: E402

_en = __import__("importlib").import_module("importlib.util")
_spec = _en.spec_from_file_location("earnings_now", os.path.join(HERE, "earnings-now.py"))
en = _en.module_from_spec(_spec)
_spec.loader.exec_module(en)                        # the EDGAR + Finnhub plumbing

STATE = os.path.join(HERE, ".watch")
RESULTS = os.path.join(STATE, "results.json")
SEEN = os.path.join(STATE, "seen.json")
LOG = os.path.join(STATE, "watch.log")

SNAPS = os.path.join(STATE, "estimates.json")

FILING_EVERY = 25          # seconds between EDGAR sweeps
CHART_EVERY = 2            # seconds between chart-symbol checks
STALE_HOURS = 30           # a filing this recent stays on the sweep list (amended 8-Ks)
MAX_WATCH = 90             # companies per sweep; SEC allows 10 req/s, this is far under
SWEEP_WORKERS = 3          # parallel EDGAR reads in a sweep - ~5 req/s, half SEC's limit
RELIST_EVERY = 2 * 3600    # re-read the earnings calendar this often within a day
RECHECK_MINUTES = 30       # a viewed symbol's newest filing is re-checked this often
NO_CARD_HOURS = 6          # a symbol with no results filing is not re-asked for this long
LATE_MINUTES = 60          # a card built later than this after the filing is "late"
STALL_SECONDS = 6 * 60     # no progress for this long = hung; the watchdog restarts it
PIDFILE = os.path.join(STATE, "watch.pid")
TASK_NAME = "EPD Earnings Watch"


# ── SELF-HEALING ─────────────────────────────────────────────────────────────
# Every blocking call now has a timeout, which removes the hang that took the
# watcher dark from 23 to 29 September. This is the second line of defence, for
# whatever hang comes next: a watchdog thread that notices when the main loop
# has stopped making progress, and restarts the whole thing.
#
# The watchdog only EXITS. The restart belongs to the Task Scheduler, whose
# trigger re-fires every five minutes and - with MultipleInstances=IgnoreNew -
# starts the watcher only when none is running. Restarting from inside the
# process was the first design and it was wrong twice over: a spawned copy
# outlives this process as an orphan the scheduler does not know about, and the
# obvious delay command, `timeout`, refuses to run without a console window and
# returns at once, so the restart request landed while this copy was still alive
# and was ignored. Exiting and letting the scheduler own the restart has neither
# problem, and it also covers the case where the process simply crashes.
import threading                                                    # noqa: E402
import subprocess                                                   # noqa: E402

_progress = {"filings": time.time()}   # one stamp per loop; the chart loop adds its own


def _tick(loop="filings"):
    """Proof of forward motion, stamped by each loop and inside every sweep."""
    _progress[loop] = time.time()


def _start_watchdog():
    def run():
        while True:
            time.sleep(30)
            for loop, at in list(_progress.items()):
                idle = time.time() - at
                if idle < STALL_SECONDS:
                    continue
                log("WATCHDOG: the %s loop made no progress for %d min - exiting so the "
                    "scheduler restarts it" % (loop, idle // 60))
                os._exit(3)
    threading.Thread(target=run, name="watchdog", daemon=True).start()


def _write_pid():
    try:
        with open(PIDFILE, "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass


def _pid_alive(pid):
    try:
        out = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"],
                             capture_output=True, text=True, timeout=10).stdout
        return str(pid) in out
    except Exception:
        return False


def _another_watcher_alive():
    """True when a DIFFERENT watcher process is running and still making progress.

    Two watchers would both sweep EDGAR and both write to the chart panel. This is
    checked once at start-up; a stale PID file from a crashed run is ignored.
    """
    try:
        pid = int(open(PIDFILE).read().strip())
    except Exception:
        return False
    if pid == os.getpid() or not _pid_alive(pid):
        return False
    try:
        return time.time() - os.path.getmtime(SEEN) < 180
    except Exception:
        return False


def log(msg):
    os.makedirs(STATE, exist_ok=True)
    line = "%s  %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(path, obj):
    os.makedirs(STATE, exist_ok=True)
    with _lock:                              # two loops write results.json now
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)                # never leave a half-written results file


# The filing loop and the chart loop share one results dict. Every write to it,
# and every walk over it, holds this lock - json.dump walking the dict while the
# other thread inserts a card raises "dictionary changed size during iteration".
_lock = threading.RLock()

# Each upcoming reporter's consensus, recorded BEFORE it reports - see
# snapshot_estimates. Loaded in main().
SNAPSHOTS = {}


# ── consensus, for the quarter reported AND the quarter being guided ─────────
consensus_pair = en.consensus_pair       # one implementation, in earnings-now.py


def todays_reporters(key, cm):
    """Whoever the calendar says reports today or tomorrow, that SEC files 8-Ks.

    Foreign issuers file a 6-K instead and never appear with a CIK here, which is
    why Alibaba would never show up - that is a real limit, not a bug to chase.

    RETURNS None WHEN THE CALENDAR COULD NOT BE READ - never an empty list. After a
    wake from sleep the network is often not back yet; the old version returned []
    for that, the loop treated [] as "nobody reports today", and it sat on an empty
    watchlist until midnight. That is what "watching 0 today" on a Tuesday evening
    in the log was. None means "try again shortly"; [] means "genuinely nobody".
    """
    out = []
    if not key:
        return out
    txt = en.get("https://finnhub.io/api/v1/calendar/earnings?from=%s&to=%s&token=%s"
                 % (date.today() - timedelta(days=1), date.today() + timedelta(days=1), key))
    if not txt:
        return None
    try:
        for e in json.loads(txt).get("earningsCalendar", []):
            t = (e.get("symbol") or "").upper()
            if t and t in cm and t not in out:
                out.append(t)
    except Exception:
        return None
    return out[:MAX_WATCH]


def snapshot_estimates(key, tickers, budget=12):
    """Record each upcoming reporter's consensus while it is still the pre-print one.

    A card built at the print freezes the estimates it was built with. A card
    built LATE - the PC slept through a 6am print and caught it at 10:48 - would
    otherwise have to ask the calendar after the guide has already moved the
    next quarter's row. The calendar list covers yesterday to tomorrow, so most
    companies are recorded the day before they report. Each is re-recorded at
    most every 20 hours, a few per pass, so this never holds up a sweep.
    """
    now = datetime.now(timezone.utc)
    n = 0
    for tk in tickers:
        s = SNAPSHOTS.get(tk)
        try:
            if s and (now - datetime.fromisoformat(s["taken"])).total_seconds() < 20 * 3600:
                continue
        except Exception:
            pass
        _tick()
        est_now, est_next, fl = en.calendar_facts(tk, key, date.today())
        n += 1
        if any(v is not None for v in est_now + est_next) or fl:
            with _lock:
                SNAPSHOTS[tk] = {"taken": now.isoformat(), "cal_label": fl,
                                 "est": [list(est_now), list(est_next)]}
        if n >= budget:
            break
        time.sleep(0.2)
    return n


def format_filed(ts):
    """The headline's timestamp: day, Eastern clock time, and how long ago.

    EASTERN ONLY. The old line printed both - "16:04:03 UTC / 12:04:03 local" - and both
    halves were wrong, because the value behind them was Eastern already (see
    accepted_et in earnings-now.py). AeroVironment's after-the-close print read
    12:04:03, four hours before it happened.

    THE DATE MATTERS AS MUCH AS THE TIME.

    "TODAY" IS A CALENDAR QUESTION, NOT AN ELAPSED ONE. The old test was
    int(seconds_elapsed / 86400) == 0, so anything under 24 hours old was TODAY -
    and a filing accepted at 4:04 pm on Wednesday still said TODAY at 9 am on
    Thursday. The owner saw it on every card he checked. Comparing calendar dates in
    Eastern is the question he is actually asking.
    """
    et = ts.astimezone(ET)
    now = datetime.now(ET)
    days = (now.date() - et.date()).days
    age = (now - et).total_seconds()
    if days == 0 and age < 3600:
        rel = "%d min ago" % max(1, int(age // 60))
    elif days == 0:
        rel = "TODAY"
    elif days == 1:
        rel = "yesterday"
    elif days < 0:
        rel = "TODAY"                      # clock skew; never claim the future
    else:
        rel = "%d days ago" % days
    return "%s  (%s)" % (format_abs(ts), rel)


def format_abs(ts):
    """'Wed 30 Sep 4:02:22 PM ET' - the moment alone, with no "how long ago".

    v10.3 of the indicator works out the "how long ago" itself, because only the
    chart knows what NOW is: in Bar Replay it is the replay's clock, and a card
    that said "(yesterday)" while the replay sat on the print candle would be
    wrong. The Pine version (f_ago) uses the same rules as format_filed above."""
    et = ts.astimezone(ET)
    return "%s %s" % (et.strftime("%a %d %b"),
                      et.strftime("%#I:%M:%S %p ET") if os.name == "nt"
                      else et.strftime("%-I:%M:%S %p ET"))


def _heartbeat():
    """(alive, human explanation) - is a watcher process actually running now?

    Read from the LOG rather than by asking the task scheduler, because the task
    can report Running while the Python process behind it has died. The log is
    touched on every sweep, so a recent line is proof of life.
    """
    # READ THE FILE THAT MOVES EVERY SWEEP, NOT THE LOG. The log is only written
    # when something HAPPENS - a filing, a panel change - so on a quiet morning
    # it can sit untouched for hours while the watcher is perfectly healthy, and
    try:
        mtime = max(os.path.getmtime(SEEN), os.path.getmtime(LOG))
    except Exception:
        try:
            mtime = os.path.getmtime(LOG)
        except Exception:
            return False, "no state files yet, so it has never started"
    age = time.time() - mtime
    last = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    if age < 180:
        return True, "last swept %ds ago (%s)" % (int(age), last)
    if age < 3600:
        return False, "log has not moved for %d min (last %s)" % (int(age // 60), last)
    return False, "log last written %s, %.1f hours ago" % (last, age / 3600.0)


def refresh_title(r):
    """The stored headline with its "how long ago" recomputed for right now.

    The title is built once, when the filing is found, but the panel may render it days
    later.
    """
    try:
        ts = datetime.fromisoformat(r["accepted"])
        head = (r["title"] or "").split("  filed ")[0]
        return "%s  filed %s" % (head, format_filed(ts))
    except Exception:
        return r.get("title") or ""


def plain_title(r):
    """The headline with the moment but no "how long ago" - v10.3 adds that itself."""
    try:
        ts = datetime.fromisoformat(r["accepted"])
        head = (r["title"] or "").split("  filed ")[0]
        return "%s  filed %s" % (head, format_abs(ts))
    except Exception:
        return r.get("title") or ""


def accepted_ms(r):
    """When the SEC accepted the filing, as epoch milliseconds - or None."""
    try:
        return int(datetime.fromisoformat(r["accepted"]).timestamp() * 1000)
    except Exception:
        return None


def _snapshot_for(tk, ts):
    """The consensus recorded for this company BEFORE this filing, if there is one.

    Taken before the acceptance and no more than four days before it - a snapshot
    from after the print has already absorbed the guide, and one from weeks
    earlier may be the previous quarter's row."""
    s = SNAPSHOTS.get(tk)
    if not s or ts is None:
        return None
    try:
        taken = datetime.fromisoformat(s["taken"])
    except Exception:
        return None
    if not (timedelta(0) < ts - taken < timedelta(days=4)):
        return None
    return tuple(s["est"][0]), tuple(s["est"][1]), s.get("cal_label")


def build_result(tk, f, url, text, key, prior=None):
    p = er.parse(text)
    # ONE calendar request for both the estimates and the fiscal quarter, anchored
    # on the day the filing was accepted - so it asks about the quarter this
    # filing actually reported.
    when = None
    try:
        when = datetime.fromisoformat((f.get("accepted") or "")[:19]).date()
    except Exception:
        pass
    acc = f.get("accepted", "")
    # The owner asked for exactly when it was published, and SECONDS matter here. The
    # whole point of reading EDGAR is that the filing lands in the same MINUTE as
    # the print candle, so "20:32" cannot tell him whether the tape moved before
    # the filing or after it. Marvell's was accepted at 16:05:59 ET; Palo Alto's
    # at 16:32:15 ET, and its move began at 16:26.
    #
    # The time comes from EDGAR's filing index, in Eastern - NOT from the
    # submissions field, which reads Eastern while a filing is fresh and UTC once
    # it has settled, and was being stamped UTC either way.
    ts = en.accepted_et(f.get("cik") or "", f.get("accession") or "", acc)
    late = ts is not None and \
        (datetime.now(timezone.utc) - ts).total_seconds() > LATE_MINUTES * 60
    # THE STREET'S NUMBERS ARE FROZEN AT THE PRINT. Finnhub keeps revising both
    # rows after a report - the next quarter's within hours, as analysts absorb
    # the guide - so a rebuild that re-asked it compared lululemon's Q3 guide with
    # a $2.36B consensus the guide itself had pulled down from $2.58B (-2.3%
    # instead of -10.8%), and flipped Palo Alto's guide from ABOVE to BELOW. A
    # card built once keeps the estimates it was built with; a rebuild (after a
    # parser fix) re-reads the RELEASE, never the consensus.
    #
    # A card built LATE - the watcher was asleep, or the chart asked for a company
    # it never watched - must not ask the calendar for the guided quarter either:
    # by then that row has moved toward the guide (Progress +3.9% within a day,
    # HPE +10.5% within a month, measured 2026-10-01), and a guide held against it
    # reads IN LINE however far it was from what the street expected. The
    # reported quarter's row barely moves (most within 0.5%), so it is still used.
    # A snapshot taken before the print beats both.
    snap = None if (prior and prior.get("est")) else _snapshot_for(tk, ts)
    if prior and prior.get("est"):
        est_now, est_next = tuple(prior["est"][0]), tuple(prior["est"][1])
        fl = prior.get("cal_label")
    elif late and snap:
        est_now, est_next, fl = snap
    else:
        est_now, est_next, fl = en.calendar_facts(tk, key, when)
        if late:
            est_next = (None, None)
    # The headline label: the release's own words, then the exhibit's file name,
    # then the calendar (see earnings_read.final_quarter for why in that order).
    p["quarter"] = er.final_quarter(p, url, fl)
    if ts is not None:
        iso, hhmm = ts.isoformat(), format_filed(ts)
    else:
        iso, hhmm = "", acc[:19]
    title, rows = er.rows(p, est_now, est_next, ticker=tk, when="filed " + hhmm)
    return {"ticker": tk, "accepted": iso, "accession": f.get("accession"), "url": url,
            "title": title, "rows": rows, "parsed": {k: v for k, v in p.items()
                                                     if k != "guide"},
            "est": [list(est_now), list(est_next)], "cal_label": fl,
            "found_at": datetime.now(timezone.utc).isoformat(),
            "late": bool(prior.get("late")) if (prior and prior.get("est")) else late}


def _newest(cik):
    try:
        return en.recent_8k(cik), None
    except Exception as e:
        return None, e


def sweep(key, cm, watch, seen, results):
    """One pass over the watchlist. Returns how many new filings it found.

    The EDGAR reads run SWEEP_WORKERS at a time. One at a time, a pass over ~50
    names took 32-53 seconds on 2026-10-01 - each read is half a second or more -
    and a print landing just after a pass started waited for all of it.
    """
    from concurrent.futures import ThreadPoolExecutor
    new = 0
    with ThreadPoolExecutor(max_workers=SWEEP_WORKERS) as pool:
        for tk, (f, err) in zip(watch, pool.map(lambda t: _newest(cm[t]), watch)):
            _tick()                     # a slow sweep is alive; only a stuck one is not
            if err is not None:
                continue
            new += _consider(tk, f, key, cm, seen, results)
    return new


def _consider(tk, f, key, cm, seen, results):
    """Act on one company's newest results 8-K. Returns 1 if it made a new card."""
    if not f:
        seen.setdefault(tk, None)
        return 0
    if seen.get(tk) == f["accession"]:
        return 0
    first = tk not in seen
    seen[tk] = f["accession"]
    if first:
        return 0                        # baseline only - this filing predates the watch
    with _lock:
        have = results.get(tk)
    if have and have.get("accession") == f["accession"]:
        return 0                        # the chart loop already built this one
    try:
        url, text = en.press_release(cm[tk], f["accession"])
        ok, why = er.looks_like_results(text)
        if not ok:
            # Item 2.02 is "Results of Operations", and companies file things
            # under it that are not a quarterly result. FedEx's newest is a
            # fiscal-year RECAST of prior periods, accepted at 20:09:35 UTC -
            # right in the after-market print window, and carrying
            # quarter-labelled EPS tables. Drawing it as an earnings card
            # would put stale numbers on a live chart.
            log("  %s: SKIPPED, %s  %s" % (tk, why, url or ""))
            return 0
        r = build_result(tk, f, url, text, key)
    except Exception as e:
        log("  %s: could not read the release (%s)" % (tk, e))
        return 0
    with _lock:
        results[tk] = r
    log("FILED  %s  %s" % (tk, r["title"]))
    for row in r["rows"]:
        log("       %s" % row.replace("|", "  "))
    log("       %s" % (url or "(document not found)"))
    time.sleep(0.4)                     # stay well inside SEC's 10 requests/second
    return 1


def rebuild(results, only=None, dry=False):
    """Re-derive every cached card from its own filing, in place.

    A card is BUILT ONCE and then re-rendered from the cache for as long as it is
    fresh, so a parser fix does not reach the panel until that company reports
    again - which for a quarterly filer is three months. When the fix is to a
    wrong NUMBER, three months is not an acceptable wait: the panel keeps showing
    AeroVironment a 26% revenue MISS on a quarter it beat.

    Re-reading is cheap and exact - the accession is stored, so this fetches the
    same document that produced the card in the first place.
    """
    cm = en.cik_map()
    changed = 0
    for tk in sorted(only or results):
        r = results.get(tk)
        if not r or not r.get("accession"):
            continue
        cik = cm.get(tk)
        if not cik:
            log("  %s: no CIK, left alone" % tk)
            continue
        try:
            url, text = en.press_release(cik, r["accession"])
            if not text:
                log("  %s: could not re-read the filing, left alone" % tk)
                continue
            f = {"accession": r["accession"], "accepted": (r.get("accepted") or ""), "cik": cik}
            fresh_r = build_result(tk, f, url or r.get("url"), text, en.finnhub_key(), prior=r)
        except Exception as e:
            log("  %s: rebuild failed (%s), left alone" % (tk, e))
            continue
        fresh_r["found_at"] = r.get("found_at") or fresh_r["found_at"]
        before, after = r.get("title", ""), fresh_r["title"]
        rows_changed = r.get("rows") != fresh_r["rows"]
        if before != after or rows_changed:
            changed += 1
            log("REBUILT %s" % tk)
            log("   was: %s" % before)
            log("   now: %s" % after)
            for x, y in zip(r.get("rows", []), fresh_r["rows"]):
                if x != y:
                    log("        - %s" % x.replace("|", "  "))
                    log("        + %s" % y.replace("|", "  "))
            for y in fresh_r["rows"][len(r.get("rows", [])):]:
                log("        + %s" % y.replace("|", "  "))
        if not dry:
            results[tk] = fresh_r
        time.sleep(0.3)
    return changed


def fresh(r):
    """Filed within STALE_HOURS - such a company stays on the sweep list, so an
    amended 8-K is still caught. It has nothing to do with DISPLAY any more: a card
    is shown until the company files its next one."""
    try:
        t = datetime.fromisoformat(r.get("accepted") or r["found_at"])
        if t.tzinfo is None:
            t = t.replace(tzinfo=ET)
        return (datetime.now(timezone.utc) - t).total_seconds() < STALE_HOURS * 3600
    except Exception:
        return False


def latest_card(tk, cm, key, have=None):
    """(card, why) - the newest quarterly-results card EDGAR has for this company.

    Returns `have` itself when the cached card is still the newest. Walks back past
    Item 2.02 filings that are not a quarter's results (FedEx's fiscal-year recast)
    to the newest one that is.
    """
    cik = cm.get(tk) or cm.get(tk.replace(".", "-"))       # BRK.B on the chart, BRK-B at SEC
    if not cik:
        return None, "not an SEC 8-K filer (an ETF, a fund, a foreign listing or not a stock)"
    filings = en.recent_8ks(cik, "2.02", limit=4)
    if filings is None:
        return have, "EDGAR did not answer"
    if not filings:
        return have, "no earnings 8-K on EDGAR"
    for f in filings:
        if have and have.get("accession") == f["accession"]:
            return have, "up to date"
        url, text = en.press_release(cik, f["accession"])
        if not text:
            continue
        ok, why = er.looks_like_results(text)
        if not ok:
            continue
        return build_result(tk, f, url, text, key), "built"
    return have, "no quarterly results among its last %d earnings 8-Ks" % len(filings)


def push_card(r, replay):
    """Write one card to the chart. Returns tv_push's reply."""
    import tv_push
    ms = accepted_ms(r)
    res = tv_push.send(plain_title(r), r["rows"], str(ms) if ms else "", replay)
    if res.get("ok") and tv_push.TIME_INPUT not in res.get("wrote", []):
        # A build before v10.3 has no clock of its own, so it gets the old title,
        # with "how long ago" worked out here.
        res = tv_push.send(refresh_title(r), r["rows"])
    return res


class ChartLoop:
    """Keeps the panel on the chart's own symbol. Runs on its own thread.

    Silent when TradingView is not running - that is the normal overnight case,
    not an error worth a log line every two seconds.
    """

    def __init__(self, results, cm, key):
        self.results, self.cm, self.key = results, cm, key
        self.last = None            # (symbol, accession, replay) as written to the chart
        self.checked = {}           # symbol -> when its newest filing was last asked for
        self.nothing = {}           # symbol -> when EDGAR last had no card for it

    def run(self):
        while True:
            _tick("chart")
            try:
                self.step()
            except Exception:
                log("chart loop error:\n" + traceback.format_exc())
                time.sleep(10)
            _tick("chart")
            time.sleep(CHART_EVERY)

    def step(self):
        import tv_push
        try:
            st = tv_push.chart_state()
        except SystemExit:
            return                      # TradingView closed, or no chart open
        except Exception:
            return
        sym = st.get("symbol") or ""
        if not st.get("ok") or not sym:
            return
        replay = bool(st.get("replay"))
        with _lock:
            r = self.results.get(sym)
        self.show(sym, r, replay)       # the cached card at once - then make sure it is the newest
        if self.due(sym, r):
            got = self.lookup(sym, r)
            if got is not r:
                self.show(sym, got, replay)

    def due(self, sym, r):
        now = time.time()
        if now - self.checked.get(sym, 0) < RECHECK_MINUTES * 60:
            return False
        if not r and now - self.nothing.get(sym, 0) < NO_CARD_HOURS * 3600:
            return False
        return True

    def lookup(self, sym, r):
        self.checked[sym] = time.time()
        try:
            got, why = latest_card(sym, self.cm, self.key, have=r)
        except Exception as e:
            log("  %s: on-demand lookup failed (%s)" % (sym, e))
            return r
        if got is None:
            self.nothing[sym] = time.time()
            log("  %s: no card - %s" % (sym, why))
            return None
        if got is not r:
            with _lock:
                self.results[sym] = got
            _save(RESULTS, self.results)
            log("BUILT  %s  %s  (on demand%s)" % (sym, got["title"],
                                                  ", late" if got.get("late") else ""))
            for row in got["rows"]:
                log("       %s" % row.replace("|", "  "))
            log("       %s" % (got.get("url") or "(document not found)"))
        return got

    def show(self, sym, r, replay):
        import tv_push
        want = (sym, r.get("accession") if r else None, replay)
        if want == self.last:
            return
        try:
            if r and self.last and self.last[:2] == want[:2]:
                tv_push.set_inputs({tv_push.REPLAY_INPUT: replay})   # only the clock moved
                log("panel clock -> %s" % ("REPLAY" if replay else "live"))
            elif r:
                res = push_card(r, replay)
                if not res.get("ok"):
                    log("panel write refused: %s" % res.get("err"))
                    return
                log("panel -> %s%s" % (sym, "  (replay)" if replay else ""))
            elif self.last is None or self.last[1]:
                tv_push.send("", [], "", replay)    # the previous symbol's card must not linger
                log("panel cleared (%s %s)" % (sym, "has no results filing" if sym in self.nothing
                                               else "- fetching its latest filing"))
        except Exception as e:
            log("panel write failed: %s" % e)
            return
        self.last = want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="one sweep, then exit")
    ap.add_argument("--no-chart", action="store_true", help="collect only, never write to TV")
    ap.add_argument("--status", action="store_true", help="print what is cached and exit")
    ap.add_argument("--tickers", default="", help="watch these instead of the calendar")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-derive every cached card from its filing, then exit")
    ap.add_argument("--dry", action="store_true", help="with --rebuild, show changes only")
    a = ap.parse_args()

    results = _load(RESULTS, {})
    if a.rebuild:
        only = [t.strip().upper() for t in a.tickers.split(",") if t.strip()] or None
        n = rebuild(results, only, dry=a.dry)
        if not a.dry:
            _save(RESULTS, results)
        print("%d card(s) %s" % (n, "would change" if a.dry else "rebuilt"))
        return
    if a.status:
        # LEAD WITH "IS IT ALIVE". This command exists for one moment - the first
        # thing the owner runs after a reboot to find out whether the watcher came back
        # on its own. A list of cached results does not answer that; a cache
        # survives on disk whether or not anything is running.
        alive, why = _heartbeat()
        print("WATCHER: %s" % ("RUNNING - " + why if alive else "NOT RUNNING - " + why))
        print("         start it with:  Start-ScheduledTask -TaskName 'EPD Earnings Watch'"
              if not alive else "")
        print("-" * 62)
        if not results:
            print("nothing cached yet")
        for tk, r in sorted(results.items(), key=lambda kv: kv[1].get("accepted", "")):
            print("%-6s %s%s" % (tk, refresh_title(r).split("  filed ")[-1],
                                 "   (built after the print)" if r.get("late") else ""))
            for row in r.get("rows", []):
                print("        " + row.replace("|", "  "))
        return

    if not a.once and _another_watcher_alive():
        log("another watcher is already running - this copy exits")
        return

    key = en.finnhub_key()
    cm = en.cik_map()
    seen = _load(SEEN, {})
    SNAPSHOTS.update(_load(SNAPS, {}))
    forced = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    watch, watch_day, watch_at, upcoming = [], None, 0.0, []
    next_filing = 0.0

    log("earnings_watch started  (chart writes %s)" % ("OFF" if a.no_chart else "ON"))
    _write_pid()
    if not a.once:
        _start_watchdog()
        if not a.no_chart:
            threading.Thread(target=ChartLoop(results, cm, key).run, name="chart",
                             daemon=True).start()
    while True:
        try:
            _tick()
            now = time.time()
            if now < next_filing:
                time.sleep(min(2.0, next_filing - now))
                continue
            next_filing = now + FILING_EVERY
            today = date.today().isoformat()
            # Rebuild on a new day, and every two hours within it - Finnhub adds
            # and moves reporters during the day, and a list fetched at 6am can
            # miss a 4pm release that was only scheduled at noon.
            if watch_day != today or now - watch_at >= RELIST_EVERY:
                got = forced or todays_reporters(key, cm)
                if got is None:
                    # The calendar could not be read - usually the network is
                    # not back after a wake. Keep the old list and try again on
                    # the next pass, instead of declaring the day empty.
                    if watch_day != today:
                        log("calendar unreachable - retrying (keeping %d already watched)"
                            % len(watch))
                        watch_day = "retry"
                else:
                    changed = (watch_day != today)
                    watch_day, watch_at, upcoming = today, now, list(got)
                    # Anything already caught today stays on the list, so a later
                    # amended 8-K is still picked up.
                    with _lock:
                        recent = [t for t in results if fresh(results[t])]
                    watch = list(dict.fromkeys(got + recent))
                    if changed:
                        log("watching %d today: %s" % (len(watch), ", ".join(watch[:14]) +
                                                       (" ..." if len(watch) > 14 else "")))
            t_sw = time.time()
            if sweep(key, cm, watch, seen, results):
                _save(RESULTS, results)
            _save(SEEN, seen)
            # A sweep is ~7s for ~40 names. Minutes means detection is late,
            # so say so - on 2026-09-30 passes were 3.5 minutes apart with
            # nothing in the log to show it.
            if time.time() - t_sw > 30:
                log("slow sweep: %.0fs for %d names" % (time.time() - t_sw, len(watch)))
            if a.once:
                return
            if upcoming and snapshot_estimates(key, upcoming, budget=12):
                _save(SNAPS, SNAPSHOTS)
        except KeyboardInterrupt:
            log("stopped by hand")
            return
        except Exception:
            log("loop error:\n" + traceback.format_exc())
            time.sleep(20)               # never die on one bad pass


if __name__ == "__main__":
    main()
