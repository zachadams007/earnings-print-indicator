"""Is the sweep's best setting stable, or does one trade decide it?

The sample is small, and a mean can be moved a long way by one of them.
This drops each trade in turn and asks whether the same multiplier still wins.

The project has been here before: a stop setting looked best across [n] symbols
until SPCX was removed, and the owner spotted that by instinct before any test did.

    python backtest/sweep_robust.py
"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "atr-sweep-v10.0.json")))
mults = ["%.2f" % m for m in d["mults"]]
trades = [t for t in d["trades"] if t["byMult"].get(mults[0]) is not None]
base = mults[0]

print("per-trade CHANGE from x%s" % base)
print("%-9s" % "trade" + "".join("  x%-6s" % m for m in mults[1:]))
delta = collections.defaultdict(list)
for t in trades:
    line = "%-9s" % t["trade"]
    for m in mults[1:]:
        dv = t["byMult"][m] - t["byMult"][base]
        delta[m].append((t["trade"], dv))
        line += "  %+6.2f " % dv
    print(line)

print("\ntotal change vs x%s" % base)
for m in mults[1:]:
    print("  x%-6s %+7.2f%%" % (m, sum(v for _, v in delta[m])))

print("\nLEAVE ONE OUT - total change with each trade removed")
print("%-9s" % "dropped" + "".join("  x%-6s" % m for m in mults[1:]))
flip = collections.defaultdict(list)
for t in trades:
    line = "%-9s" % t["trade"]
    for m in mults[1:]:
        tot = sum(v for k, v in delta[m] if k != t["trade"])
        line += "  %+6.2f " % tot
        if tot <= 0:
            flip[m].append(t["trade"])
    print(line)

print("\nVERDICT")
for m in mults[1:]:
    if flip[m]:
        print("  x%-6s NOT ROBUST - turns negative if any one of these is removed: %s"
              % (m, ", ".join(flip[m])))
    else:
        print("  x%-6s holds up with any single trade removed" % m)

print("\nHOW CONCENTRATED IS THE GAIN?")
for m in mults[1:]:
    vals = sorted(delta[m], key=lambda x: -x[1])
    tot = sum(v for _, v in vals)
    top2 = sum(v for _, v in vals[:2])
    share = (100.0 * top2 / tot) if tot else 0.0
    print("  x%-6s total %+6.2f%%   best two trades %+6.2f%% (%.0f%% of it): %s, %s"
          % (m, tot, top2, share, vals[0][0], vals[1][0]))
