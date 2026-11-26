"""tv_push - write text into the indicator's earnings inputs, so it shows on the chart.

Pine Script has no network access, so the indicator can never fetch a filing itself.
This is the way in: the TradingView desktop app exposes a Chrome DevTools port on
127.0.0.1:9222, and the study's inputs can be set through it. The panel then renders
whatever is written here.

Targets in_130..in_138, which are declared LAST in the Pine precisely so that
appending them could not shift any existing stored setting:

    in_130         earnTitle    the headline
    in_131..136    earnRow1..6  "LABEL|VALUE" rows
    in_137         earnTime     when the SEC accepted the filing, epoch ms (v10.3)
    in_138         earnReplay   Bar Replay is running (v10.3)

Used by earnings-now.py --push and by the watcher. Also runnable on its own to test
the plumbing:

    python tools/tv_push.py --title "TEST  demo row" --rows "EPS|1.23" "revenue|4.5B"
    python tools/tv_push.py --clear
    python tools/tv_push.py --state
"""
import argparse, asyncio, json, re, sys, urllib.request

CDP = "http://127.0.0.1:9222"
FIRST_INPUT = 130          # earnTitle; rows follow at 131..136
N_ROWS = 6                 # earnRow1..earnRow6
TIME_INPUT = "in_137"      # earnTime  - absent on builds before v10.3
REPLAY_INPUT = "in_138"    # earnReplay


def find_chart_target():
    """The TradingView chart page, out of however many targets the app exposes."""
    try:
        with urllib.request.urlopen(CDP + "/json", timeout=5) as r:
            targets = json.loads(r.read().decode())
    except Exception as e:
        raise SystemExit("cannot reach TradingView on %s - is the desktop app running?\n  %s"
                         % (CDP, e))
    pages = [t for t in targets
             if t.get("type") == "page" and "tradingview.com/chart" in (t.get("url") or "")]
    if not pages:
        raise SystemExit("TradingView is running but no chart page is open")
    return pages[0]["webSocketDebuggerUrl"]


# __WRITES__ is {"in_130": value, ...} to write, or null to only read the state.
# Written in the order given, which the callers use to set the clock inputs BEFORE
# the headline - so a card never renders for a moment against the wrong clock.
JS = r"""
(function(){
  try{
    var cwc = window.TradingViewApi._chartWidgetCollection;
    var w = cwc.activeChartWidget.value();
    var m = w.model();
    var sym = '';
    try { var n = m.mainSeries().symbolInfo().name; if (typeof n === 'string') sym = n; } catch(e){}
    if(!sym){ try { var s2 = w.symbol(); if (typeof s2 === 'string') sym = s2; } catch(e){} }
    var replay = null;
    try {
      var v = window.TradingViewApi._replayApi.isReplayStarted();
      replay = !!((v && typeof v === 'object' && typeof v.value === 'function') ? v.value() : v);
    } catch(e){}
    var study = m.panes()[0].sourcesByGroup().all().find(function(s){
      try { return /EPD|Earnings Print/i.test(s.title()); } catch(e){ return false; }
    });
    if(!study) return JSON.stringify({ok:false, symbol:sym, replay:replay,
                                      err:'indicator not found on this chart'});
    var c = study.properties().inputs.childs();
    var writes = __WRITES__;
    var wrote = [];
    if (writes) {
      Object.keys(writes).forEach(function(id){
        if (c[id] === undefined) return;
        c[id].setValue(writes[id]);
        wrote.push(id);
      });
    }
    var cur = {};
    ['in_130', 'in_137', 'in_138'].forEach(function(id){
      if (c[id] !== undefined) { try { cur[id] = c[id].value(); } catch(e){} }
    });
    return JSON.stringify({ok:true, symbol:sym, replay:replay, study:study.title(),
                           wrote:wrote, current:cur});
  } catch(e){ return JSON.stringify({ok:false, err:String(e)}); }
})()
"""


# EVERY CALL IS BOUNDED. The first version awaited TradingView's reply with no
# timeout at all. If the page reloaded across a sleep/wake - which it does - the
# reply never came, recv() waited forever, and because the watcher is one loop,
# EVERYTHING stopped: no EDGAR sweeps, no daily watchlist, no log line, nothing
# raised for the error handler to catch. It went dark from 23 to 29 September
# 2026 with the PC awake, and missed Micron, Costco, FedEx, CarMax and AutoZone.
PUSH_TIMEOUT = 12          # seconds for the whole connect-evaluate-reply exchange


async def _push(writes):
    import websockets
    ws_url = find_chart_target()
    js = JS.replace("__WRITES__", json.dumps(writes))
    async with websockets.connect(ws_url, max_size=20 * 1024 * 1024,
                                  open_timeout=6, close_timeout=2) as ws:
        await ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
        await ws.recv()
        await ws.send(json.dumps({
            "id": 2, "method": "Runtime.evaluate",
            "params": {"expression": js, "returnByValue": True, "awaitPromise": False}}))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("id") == 2:
                res = msg.get("result", {}).get("result", {})
                if res.get("type") == "string":
                    return json.loads(res["value"])
                return {"ok": False, "err": json.dumps(msg)[:300]}


async def push(writes):
    """One bounded exchange with the chart. Raises TimeoutError instead of hanging.

    writes: {"in_130": value, ...}, or None to read the chart's state only. A list
    is accepted too, for the old calling convention: values from in_130 upward.
    """
    if isinstance(writes, (list, tuple)):
        writes = {"in_%d" % (FIRST_INPUT + i): v for i, v in enumerate(writes)}
    return await asyncio.wait_for(_push(writes), timeout=PUSH_TIMEOUT)


def set_inputs(writes):
    return asyncio.run(push(writes))


def send(title, rows, when_ms="", replay=None):
    """Write a whole card. rows: up to six 'LABEL|VALUE' strings.

    when_ms is when the SEC accepted the filing, as epoch milliseconds in a string;
    v10.3 uses it to keep the card hidden in Bar Replay until the replay reaches it,
    and to say how long ago it was. Empty means "no clock" - the card simply shows.
    It is ALWAYS written, so a card pushed without one can never inherit the
    previous card's clock.
    """
    vals = list(rows)[:N_ROWS]
    while len(vals) < N_ROWS:
        vals.append("")
    writes = {TIME_INPUT: str(when_ms or "")}
    if replay is not None:
        writes[REPLAY_INPUT] = bool(replay)
    for i, v in enumerate(vals):
        writes["in_%d" % (FIRST_INPUT + 1 + i)] = v
    writes["in_%d" % FIRST_INPUT] = title
    return asyncio.run(push(writes))


_SYM_OK = re.compile(r"[A-Z0-9][A-Z0-9.\-_!:/&^]*")


def _clean_symbol(raw):
    """'NASDAQ:PANW' -> 'PANW', 'ES1!' -> 'ES1'. Anything that is not a plain symbol
    string -> ''. On 2026-10-01 the chart briefly answered "[object Object]" while
    it was loading, and the panel was cleared for a symbol called that."""
    s = (raw or "").strip().upper()
    if not s or not _SYM_OK.fullmatch(s):
        return ""
    return s.split(":")[-1].split("!")[0]


def chart_state():
    """{'ok', 'symbol', 'replay', 'study', 'current'} for the chart as it is now.

    Raises SystemExit when TradingView is not running or has no chart open.
    """
    r = asyncio.run(push(None))
    r["symbol"] = _clean_symbol(r.get("symbol"))
    return r


def chart_symbol():
    """What ticker the chart is showing right now, e.g. 'NASDAQ:PANW' -> 'PANW'.

    The watcher needs this because a result is only ever true of ONE company.
    Last session left Marvell's numbers rendering on a Palo Alto chart, which is
    worse than showing nothing: the panel sits directly under rows that ARE about
    the symbol on screen, so there is no visual cue that one row is about a
    different stock.
    """
    return chart_state().get("symbol") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="")
    ap.add_argument("--rows", nargs="*", default=[])
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--state", action="store_true", help="read the chart, write nothing")
    a = ap.parse_args()
    if a.state:
        print(json.dumps(chart_state(), indent=1))
        return
    if a.clear:
        r = send("", [])
    else:
        r = send(a.title, a.rows)
    if r.get("ok"):
        print("pushed to %s  (%s)" % (r.get("study"), ", ".join(r.get("wrote", []))))
    else:
        print("FAILED: %s" % r.get("err"))
        sys.exit(1)


if __name__ == "__main__":
    main()
