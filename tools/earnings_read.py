"""earnings_read - turn an 8-K press release into the four numbers that matter.

WHAT THE OWNER ASKED FOR, and why this is a module of its own:

A number with no consensus beside it is useless to him. So every row this
produces is a COMPARISON, never a bare figure.

THE FOUR TRAPS, all of them found by running against real filings:

  1. THE FIGURE COMES BEFORE THE LABEL as often as after it. Marvell writes
     "$0.33 GAAP diluted income per share; $0.94 non-GAAP diluted income per
     share", and NVIDIA writes "GAAP and non-GAAP earnings per diluted share were
     $2.46 and $2.22, respectively" - where the non-GAAP figure is the SECOND
     one. Both directions are needed, plus the "respectively" case.

  2. NEIGHBOURING DOLLAR FIGURES ARE NOT PER-SHARE. Okta writes "non-GAAP net
     income of $169 million ... Non-GAAP diluted net income per share was $1.05";
     a backwards search that ignores the word "million" returns $169 as an EPS.

  3. "REVENUE" IS NOT ALWAYS REVENUE. CoreWeave's release says "Revenue backlog
     was approximately $104 billion". Palo Alto's income statement opens with a
     product-revenue line before the total. Anchoring the table lookup on the
     headline figure kills both.

  4. INCOME STATEMENTS ARE NOT ALWAYS TWO COLUMNS. Marvell and HPE print the
     sequential quarter in the middle, so "the second number" is last quarter,
     not last year. The period header is counted to tell them apart.

AND THE ONE THAT MATTERS MOST: "did they raise?" cannot be read from language.
lululemon called a CUT "our revised full-year outlook" in a confident sentence.
So the raise flag is only ever a secondary note; the primary answer is GUIDANCE
MIDPOINT vs THE STREET, which is also the thing that actually moves the stock.

Every parser returns None rather than a guess. A blank row is honest; a wrong row
is the failure mode this whole tool exists to avoid.
"""
import html as _html
import re
from decimal import Decimal, ROUND_HALF_UP

_SCALE = {"billion": 1e9, "billions": 1e9, "b": 1e9,
          "million": 1e6, "millions": 1e6, "m": 1e6,
          "thousand": 1e3, "thousands": 1e3, "k": 1e3,
          "trillion": 1e12, "trillions": 1e12}


def _num(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except Exception:
        return None


def money(v, unit=None, default_scale=1.0):
    n = _num(v)
    if n is None:
        return None
    return n * (_SCALE.get(str(unit).lower().rstrip("."), 1.0) if unit else default_scale)


def _r(x, nd):
    """Round half AWAY from zero. "%.2f" rounds 2.625 to 2.62 (binary floats and
    half-to-even), so FactSet's $2,625M guide top printed as $2.62B."""
    q = Decimal(1).scaleb(-nd)
    return format(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP), "f")


def _s0(x):
    """Signed whole number for the panel: +3, -12, and +0 rather than -0 for a
    change that rounds away (Signet's flat revenue printed as "-0% YoY")."""
    v = int(Decimal(repr(x)).quantize(Decimal(1), rounding=ROUND_HALF_UP)) if x is not None else 0
    return "%+d" % v


def fmt_money(v):
    if v is None:
        return None
    a = abs(v)
    if a >= 1e12:
        return "$%sT" % _r(v / 1e12, 2)
    if a >= 1e9:
        return "$%sB" % _r(v / 1e9, 2)
    if a >= 1e6:
        return "$%sM" % _r(v / 1e6, 0)
    if a >= 1e5:
        return "$%sM" % _r(v / 1e6, 1)
    if a >= 1e3:
        return "$%sk" % _r(v / 1e3, 0)
    return "$%s" % _r(v, 0)


def clean(t):
    """Press releases are full of smart quotes, hard spaces and zero-width joins."""
    if not t:
        return ""
    # UNESCAPE FIRST. Only a handful of entities were being handled by name, so
    # ServiceTitan's dateline - "LOS ANGELES &#x2013; September 8, 2026" - kept
    # its raw en-dash entity, and the "2013" inside it was read as a YEAR: the
    # panel headline said "Q2 FY13". Any release that escapes its dashes could do
    # the same, so the whole entity set is decoded rather than five of them.
    t = _html.unescape(t)
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\u2013", "-"), ("\u2014", "-"), ("\u00a0", " "), ("\u200b", ""),
                 ("\u2212", "-")):
        t = t.replace(a, b)
    t = re.sub(r"&(nbsp|#160|#xa0|#8203);", " ", t, flags=re.I)
    t = re.sub(r"^\s*EX-?[\d.]+\s+\d+\s+\S+\s+(?:EX-?[\d.]+|EARNINGS RELEASE|EXHIBIT [\d.]+)?"
               r"\s*(?:Document)?\s*", "", t, flags=re.I)
    return re.sub(r"[ \t]+", " ", t)


# ═══════════════════════════════════════════════════════════════════════════
# EPS
# ═══════════════════════════════════════════════════════════════════════════
_PS = r"per\s+(?:diluted\s+|basic\s+and\s+diluted\s+|common\s+)?share"
_EPS_LABEL = (r"(?:diluted\s+|basic\s+and\s+diluted\s+)?(?:net\s+)?"
              r"(?:income|earnings|loss|EPS)\s+(?:\(loss\)\s+)?" + _PS)
_BULLET = "•●▪◦‣·"
_SCALE_WORD = re.compile(r"^\s*(million|billion|thousand|trillion)", re.I)
# "Adjusted" is the same thing as non-GAAP; Best Buy uses only that word.
_NG = r"(?:non-?GAAP|adjusted)"
# A guided figure is not a result. Marvell prints "Non-GAAP diluted net income per
# share is expected to be $1.10" in its outlook, ~9k characters in, and reading it
# as the quarter's EPS put a guide on the panel where the actual belonged.
_FWD = re.compile(r"expect|outlook|guidance|forecast|anticipat|will\s+be|"
                  r"assum(?:es|ing)|range\s+of|rais(?:es|ing)|initiat", re.I)
# Accenture leads with full-year EPS of $13.78 beside a quarterly $3.80, and
# AutoZone's prose gives 36-week sales next to the quarter's. A cumulative figure
# on a panel labelled with one quarter is simply a wrong number.
_CUMULATIVE = re.compile(r"full[-\s]year|fiscal\s+year\s+(?:20\d\d|result)|"
                         r"(?:nine|six|twelve|thirty-six|thirty-nine|forty|fifty-two)[-\s]"
                         r"(?:month|week)|year[-\s]to[-\s]date|\bYTD\b|"
                         r"(?:36|39|52|26|24)[-\s]weeks?", re.I)


# Set by parse() while it reads a fiscal-year-only release, where EVERY figure is
# a full-year one and the cumulative guard below would reject all of them.
_ANNUAL_MODE = False


def _is_cumulative(text, at, span=170):
    """True when the nearest preceding period words describe more than a quarter."""
    if _ANNUAL_MODE:
        return False
    back = text[max(0, at - span):at]
    m = None
    for m in _CUMULATIVE.finditer(back):
        pass
    if not m:
        return False
    # A sentence or bullet boundary after the phrase means it governs something else.
    return not re.search("[" + _BULLET + r"]|\.\s", back[m.end():])


def _plausible(v):
    return v is not None and 0 < abs(v) < 100


def _reject(text, at):
    """One gate for both ways a per-share figure can belong to the wrong thing:
    it is GUIDED rather than reported, or it covers a YEAR rather than a quarter."""
    return _is_forward(text, at) or _is_cumulative(text, at)


def _is_forward(text, at):
    lo = max(0, at - 190)
    back = text[lo:at]
    m = None
    for m in _FWD.finditer(back):
        pass
    if m and not re.search("[" + _BULLET + r"]|\.\s", back[m.end():]):
        return True
    return bool(_FWD.search(text[at:at + 55]))


def _eps_respectively(text, non_gaap):
    """'GAAP and non-GAAP earnings per diluted share were $2.46 and $2.22,
    respectively' - NVIDIA's house style, and the second figure is the non-GAAP."""
    rx = (r"GAAP\s+and\s+" + _NG + r"\s+" + _EPS_LABEL +
          r"\s+(?:were|are)\s+\$\s?\(?(-?[\d.]+)\)?\s+and\s+\$\s?\(?(-?[\d.]+)\)?"
          r"[^.]{0,30}respectively")
    m = re.search(rx, text, re.I)
    if m and not _reject(text, m.start()):
        v = _num(m.group(2) if non_gaap else m.group(1))
        if _plausible(v):
            return v
    return None


def _eps_forwards(text, non_gaap):
    """'Non-GAAP diluted net income per share was $1.05' - label, then figure.

    The gap allowance is generous because filers wedge whole clauses in between:
    Braze writes "...per share attributable to Braze common stockholders,
    diluted, was $0.16".
    """
    lab = (_NG + r"\s*(?:\(\d\))?\s*" if non_gaap
           else r"(?<!non-)(?<!non )GAAP\s*(?:\(\d\))?\s*")
    for rx in (lab + _EPS_LABEL + r"(?:[^$;.]|\.(?=\d)){0,95}?\$\s?\(?(-?[\d.]+)\)?",
               lab + r"(?:diluted\s+)?EPS(?:[^$;.]|\.(?=\d)){0,45}?\$\s?\(?(-?[\d.]+)\)?"):
        for m in re.finditer(rx, text, re.I):
            if _SCALE_WORD.match(text[m.end():m.end() + 12]):
                continue                       # "$169 million" is not a per-share figure
            if _reject(text, m.start()):
                continue
            v = _num(m.group(1))
            if _plausible(v):
                return v
    return None


def _eps_backwards(text, non_gaap):
    """The figure sits BEFORE the label. Anchor on the label, walk left."""
    lab = (_NG + r"\s+(?:\(\d\)\s*)?" + _EPS_LABEL if non_gaap
           else r"(?<!non-)(?<!non )GAAP\s+(?:\(\d\)\s*)?" + _EPS_LABEL)
    for m in re.finditer(lab, text, re.I):
        if _reject(text, m.start()):
            continue
        seg = text[max(0, m.start() - 75):m.start()]
        for mm in reversed(list(re.finditer(r"\$\s?\(?(-?[\d.]+)\)?", seg))):
            if _SCALE_WORD.match(seg[mm.end():mm.end() + 12]):
                continue
            v = _num(mm.group(1))
            if _plausible(v):
                return v
    return None


_BARE_LOSS = re.compile(r"(?<!\()\bloss\b(?!\s*\))", re.I)
_GENERIC_LOSS = re.compile(r"(?:income|earnings)\s*(?:/\s*)?\(\s*loss\s*\)", re.I)


def _signed_fig(text, m, g=1):
    """The figure with its sign. A bracket in front of it is the minus sign, and so
    is a label that says "loss" outright. Upexi's "Basic and diluted loss per share
    $ (3.87)" was read as +3.87 - the reader stripped the bracket and ignored the
    word - and an annual loss of $246M went on the panel as a $3.99 beat.
    "Income (loss)" in brackets is a generic caption and says nothing either way.
    """
    v = _num(m.group(g))
    if v is None:
        return None
    pre = text[max(0, m.start(g) - 3):m.start(g)]
    lab = m.group(0)
    if "(" in pre or (_BARE_LOSS.search(lab) and not _GENERIC_LOSS.search(lab)):
        return -abs(v)
    return v


def _eps_plain(text):
    """No GAAP split stated at all: 'Diluted EPS of $2.92'.

    Best Buy's headline reads "Diluted EPS Increased 70% to $1.48", so the figure
    can sit past a percentage clause rather than straight after 'of'.
    """
    for rx in (r"(?:diluted\s+)?EPS\s+(?:of|was|were)\s+\$\s?\(?(-?[\d.]+)\)?",
               r"(?:diluted\s+)?EPS\s+(?:increased|decreased|grew|declined|rose|fell)"
               r"(?:[^$.]|\.(?=\d)){0,40}?\$\s?\(?(-?[\d.]+)\)?",
               _EPS_LABEL + r"\s+(?:of|was|were)\s+\$\s?\(?(-?[\d.]+)\)?",
               r"\$\s?\(?(-?[\d.]+)\)?\s+" + _EPS_LABEL,
               _EPS_LABEL + r"\s*\$\s?\(?(-?[\d.]+)\)?"):
        for m in re.finditer(rx, text, re.I):
            if _SCALE_WORD.match(text[m.end():m.end() + 12]):
                continue
            if _is_forward(text, m.start()) or _is_year_ago(text, m):
                continue
            # "EPS increased from $0.67 to $0.84": the first figure is last year's.
            nxt = re.match(r"\s*\)?\s*to\s+\$\s?\(?(-?[\d.]+)\)?", text[m.end():m.end() + 30])
            if nxt and re.search(r"from\s+\$?\s?\(?$", text[max(0, m.start(1) - 8):m.start(1)], re.I):
                v2 = _num(nxt.group(1))
                if _plausible(v2):
                    return v2
            v = _signed_fig(text, m)
            if _plausible(v):
                return v
    return None


# "compared to a net loss of $30.3 million, or loss of $0.22 per share in 2025" -
# InnovAge's FULL-YEAR 2025 figure, one clause after the current one. Read as the
# quarter it put -$0.22 on a quarter that earned $0.06.
_TAIL_HEAD = (r"\s*\)?\s*(?:per\s+(?:basic\s+and\s+)?(?:fully\s+)?(?:diluted\s+)?share\s*)?"
              r",?\s*(?:in|for)\s+(?:the\s+)?")
_YEAR_AGO_TAIL = re.compile(_TAIL_HEAD + r"(?:(?:19|20)\d{2}\b(?!\s+(?:first|second|third|fourth|"
                            r"fiscal|quarter|Q[1-4]))|prior[-\s]year|same\s+(?:period|quarter)|year[-\s]ago|"
                            r"(?:first|second|third|fourth)\s+quarter\s+last\s+year)", re.I)
# ...or a CUMULATIVE period stated after the figure: Shoe Station's "EPS of $1.04
# in the first six months of 2025" - last year's half-year, read as the quarter.
_CUM_TAIL = re.compile(_TAIL_HEAD + r"(?:first\s+)?(?:six|nine|twelve|twenty[-\s]six|thirty[-\s]nine|"
                       r"fifty[-\s]two)[\s-](?:months|weeks)|" + _TAIL_HEAD +
                       r"(?:year[-\s]to[-\s]date|(?:full|fiscal)\s+year\b|year\s+ended)", re.I)
_YEAR_AGO_HEAD = re.compile(r"(?:compared\s+(?:to|with)|versus|vs\.?|from)\s+(?:a\s+|an\s+)?"
                            r"[^.;$]{0,60}$", re.I)


def _is_year_ago(text, m):
    """True when the figure belongs to the comparison, not the quarter."""
    if _YEAR_AGO_HEAD.search(text[max(0, m.start() - 70):m.start()]):
        return True
    if _YEAR_AGO_TAIL.match(text, m.end()):
        return True
    return not _ANNUAL_MODE and bool(_CUM_TAIL.match(text, m.end()))


# A RANGE IS A GUIDE, NEVER A RESULT. Shoe Station's outlook says "Adjusted EPS of
# $0.75 to $0.90" and the panel showed $0.75 as the quarter's adjusted EPS - a
# BEAT of 118% on a quarter that earned $0.23 against a $0.34 estimate. SailPoint's
# "$0.07 to $0.08" Q3 guide stood in for its $0.09 actual the same way. No company
# reports an actual as a range, so every range is masked before EPS is read.
# ("from $0.67 to $0.84" is a result and is left alone.)
_RANGE = re.compile(r"(?<!from\s)\$\s?\(?-?\d[\d,]*(?:\.\d+)?\)?\s*(?:to|-)\s*\$\s?\(?-?\d[\d,]*(?:\.\d+)?\)?",
                    re.I)


def _mask_ranges(text):
    return _RANGE.sub(" [range] ", text)


def _eps_via_net_income(text, non_gaap):
    """'non-GAAP net income ... was $853 million, or $1.02 per diluted share'.

    Palo Alto, Costco and Mission Produce all use this shape, where the per-share
    figure hangs off a net-income sentence and the GAAP/non-GAAP word is nowhere
    near the words "per share". Nothing else in this module reaches it.
    """
    lab = (_NG + r"\s+net\s+(?:income|loss)" if non_gaap
           else r"(?:^|[\s•])(?:GAAP\s+)?net\s+(?:income|loss)")
    # "per basic and diluted share" (LightPath) and "$0.16 per basic and $0.15 per
    # fully diluted share" (Radiant) are the same statement; the lazy gap walks
    # past the basic figure to the diluted one.
    rx = (lab + r"(?:[^.]|\.(?=\d)){0,150}?\$\s?\(?(-?[\d.]+)\)?\s*\)?\s+per\s+"
          r"(?:basic\s+and\s+)?(?:fully\s+)?(?:diluted\s+)?share")
    for m in re.finditer(rx, text, re.I):
        if _reject(text, m.start()):
            continue
        if _is_year_ago(text, m):
            continue
        # The figure itself must not sit in a "compared to ..." clause: InnovAge's
        # "Net loss ... of $2.5 million, or loss of $0.02 per share, compared to a
        # net loss of $30.3 million, or loss of $0.22 per share in 2025".
        if re.search(r"compared\s+(?:to|with)|versus", text[m.start():m.start(1)], re.I):
            continue
        v = _num(m.group(1))
        if _plausible(v):
            # "(0.35) per diluted share" is a loss - the bracket is the minus sign.
            if re.search(r"\(\s*[\d.]+\s*\)\s*per", m.group(0)) or \
               re.search(r"net\s+loss", m.group(0), re.I):
                v = -abs(v)
            return v
    return None


def _eps_of(text, non_gaap):
    """'Diluted net earnings per share ("EPS") : GAAP of $1.06 ... Non-GAAP of $1.11'.

    HPE splits the label from both figures, so the basis word carries no EPS
    wording of its own. Only trusted when an EPS label sits just before it.
    """
    lab = _NG if non_gaap else r"(?<!non-)(?<!non )GAAP"
    for m in re.finditer(lab + r"\s*(?:\(\d\))?\s*of\s+\$\s?\(?(-?[\d.]+)\)?", text, re.I):
        if not re.search(_EPS_LABEL + r"[^$]{0,140}$", text[max(0, m.start() - 220):m.start()],
                         re.I | re.S):
            continue
        if _reject(text, m.start()):
            continue
        v = _num(m.group(1))
        if _plausible(v):
            return v
    return None


def _eps_nested(text, non_gaap):
    """HPE nests the basis UNDER the label instead of beside it:

        - Diluted net earnings per share ("EPS") :
            - GAAP of $1.06, up $0.85 from the prior-year period
            - Non-GAAP (1) of $1.11, up $0.67 from the prior-year period

    The child bullet never says "per share", so every pattern that needs the
    basis word adjacent to the EPS label misses it, and the panel showed HPE's
    GAAP $1.06 against a $0.94 ADJUSTED consensus - a 13% "beat" measured on the
    wrong basis. The parent supplies the label; the child supplies the figure.
    """
    lab = (_NG if non_gaap else r"(?<!non-)(?<!non )GAAP")
    for m in re.finditer(_EPS_LABEL + r"\s*(?:\(\s*\"?EPS\"?\s*\))?\s*:", text, re.I):
        if _reject(text, m.start()):
            continue
        seg = text[m.end():m.end() + 320]
        # Stop before the next top-level bullet so a neighbouring metric's
        # GAAP/non-GAAP pair cannot be read as this one's.
        nxt = re.search(r"[" + _BULLET[0] + r"]", seg)
        if nxt:
            seg = seg[:nxt.start()]
        mm = re.search(lab + r"\s*(?:\(\d\))?\s*of\s+\$\s?\(?(-?[\d.]+)\)?", seg, re.I)
        if mm:
            v = _num(mm.group(1))
            if _plausible(v):
                return v
    return None


def _eps_on_a_basis(text, non_gaap):
    """Adobe: "Diluted earnings per share was $4.25 on a GAAP basis and $5.96 on
    a non-GAAP basis." The basis trails each figure instead of leading it, so
    every label-first and figure-first pattern picks the GAAP number."""
    rx = (_EPS_LABEL + r"\s+(?:was|were)\s+\$\s?\(?(-?[\d.]+)\)?\s+on\s+an?\s+GAAP\s+basis"
          r"\s+and\s+\$\s?\(?(-?[\d.]+)\)?\s+on\s+an?\s+non-?GAAP\s+basis")
    m = re.search(rx.replace("an? GAAP", "an? (?:GAAP)"), text, re.I)
    if m and not _reject(text, m.start()):
        v = _num(m.group(2) if non_gaap else m.group(1))
        if _plausible(v):
            return v
    return None


def _eps_trailing_basis(text, non_gaap):
    """Jabil: "Core diluted earnings per share (Non-GAAP): $3.16" - the basis
    marker comes AFTER the label, in brackets, before the colon. Also catches
    "Core diluted EPS" on its own, since "core" is that filer's word for
    adjusted."""
    if not non_gaap:
        return None
    for rx in (r"(?:core\s+|adjusted\s+)?" + _EPS_LABEL +
               r"\s*\(\s*" + _NG + r"\s*\)\s*:?\s*\$\s?\(?(-?[\d.]+)\)?",
               r"(?:core|adjusted)\s+" + _EPS_LABEL + r"\s*:?\s*\$\s?\(?(-?[\d.]+)\)?"):
        for m in re.finditer(rx, text, re.I):
            if _reject(text, m.start()):
                continue
            v = _num(m.group(1))
            if _plausible(v):
                return v
    return None


def _results_only(text):
    """The part of the release that reports the quarter, with the outlook cut off.

    Guidance tables are shaped exactly like results tables, so a figure lifted
    from one is indistinguishable from a real number once it reaches the panel.
    """
    _span, at = _outlook_span(text)
    return text[:at] if at > len(text) * 0.2 else text


# Delta heads its blocks "June Quarter 2026 GAAP Financial Results" and "June
# Quarter 2026 Non-GAAP Financial Results"; only "adjusted" was recognised, so
# the GAAP $2.44 went on the panel where the $1.56 the street quotes belonged.
_ADJ_BLOCK = re.compile(r"(?:adjusted|non-?GAAP)\s+(?:financial\s+)?results|"
                        r"\badjusted\b[^.\n]{0,40}(?:quarter|results|basis)", re.I)


def _eps_block_basis(text, non_gaap):
    """Delta heads a whole BLOCK with the basis and then repeats a plain
    "Earnings per share of $X" inside it, once for GAAP and once for adjusted.

    Reading the sentence alone cannot tell the two apart - both are identical -
    so the nearest preceding block heading decides which one this is.
    """
    hits = []
    for m in re.finditer(_EPS_LABEL + r"\s+of\s+\$\s?\(?(-?[\d.]+)\)?", text, re.I):
        if _reject(text, m.start()):
            continue
        v = _num(m.group(1))
        if not _plausible(v):
            continue
        back = text[max(0, m.start() - 900):m.start()]
        b = None
        for b in _ADJ_BLOCK.finditer(back):
            pass
        hits.append((v, b is not None))
    if not hits:
        return None
    want = [v for v, adj in hits if adj == bool(non_gaap)]
    if want:
        return want[0]
    return None


# ── SEPTEMBER 2026 SHAPES ────────────────────────────────────────────────────
# Five ways a release states EPS that nothing above could read, each taken from a
# real September filing and each wrong on the panel before this block existed.

_MONTH_DATE = re.compile(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
                         r"\s+\d{1,2},?\s+(?:19|20)\d{2}", re.I)


def _strip_dates(seg):
    """Remove calendar dates before reading table cells.

    Copart's statement says "Year Ended July 31, 2026" and the cell reader took
    the 31 as an EPS of $31.00. RH and Delta did the same with "MAY 2," and
    "June 30,". A date is never a value, so it is removed before any cell is read.
    """
    seg = _MONTH_DATE.sub(" ", seg)
    return re.sub(r"\b(?:19|20)\d{2}\b", " ", seg)


def _eps_loss_words(text, non_gaap):
    """A LOSS stated in words, with a positive number.

    Hain: "Loss per diluted share was $0.68" and "Adjusted loss per diluted share
    was $0.05". Dave & Buster's: "a $0.36 net loss per diluted share" and "a $0.27
    Adjusted net loss per diluted share". The sign lives in the word "loss", and
    reading the number alone put a loss on the panel as a profit - a BEAT on a
    quarter that lost money. The tables do carry $(0.68), but the prose is read
    first, so the prose has to get the sign right itself.

    "(loss)" in brackets is the generic label on a statement that covers both
    cases, and says nothing about the sign - that comes from the number's own
    brackets. Only a bare "loss" means the quarter lost money.
    """
    adj = r"(?:adjusted|non-?GAAP)\s+"
    pre = adj if non_gaap else r"(?<!adjusted )(?<!non-GAAP )"
    rxs = (
        # "Adjusted loss per diluted share was $0.05"
        pre + r"(?:net\s+)?loss\s+per\s+(?:diluted\s+|basic\s+and\s+diluted\s+)?share"
        r"\s*(?:\(\d\))?\s*(?:was|were|of|:)?\s*\$\s?\(?([\d.]+)\)?",
        # "a $0.27 Adjusted net loss per diluted share"
        r"\$\s?\(?([\d.]+)\)?\s+" + (adj if non_gaap else r"") +
        r"(?:net\s+)?loss\s+per\s+(?:diluted\s+)?share",
    )
    for rx in rxs:
        for m in re.finditer(rx, text, re.I):
            if _reject(text, m.start()):
                continue
            if not non_gaap and re.search(r"adjusted|non-?GAAP",
                                          text[max(0, m.start() - 12):m.end()], re.I):
                continue
            # the year-ago comparison clause - "compared to a net loss of ..."
            if re.search(r"compared\s+(?:to|with)[^.]{0,40}$",
                         text[max(0, m.start() - 60):m.start()], re.I):
                continue
            v = _num(m.group(1))
            if _plausible(v):
                return -abs(v)
    return None


def _eps_and_adjusted(text, non_gaap):
    """Korn Ferry: "Diluted and adjusted diluted earnings per share were $1.32 and
    $1.43". Two figures, one label, no "respectively" - the adjusted one is the
    second."""
    rx = (r"(?:diluted\s+)?(?:earnings|EPS)?\s*and\s+(?:adjusted|non-?GAAP)\s+"
          r"(?:diluted\s+)?(?:net\s+)?(?:earnings|income|EPS)(?:\s+per\s+(?:diluted\s+)?share)?"
          r"\s+(?:were|was|of)\s+\$\s?\(?(-?[\d.]+)\)?\s+and\s+\$\s?\(?(-?[\d.]+)\)?")
    for m in re.finditer(rx, text, re.I):
        if _reject(text, m.start()):
            continue
        v = _num(m.group(2) if non_gaap else m.group(1))
        if _plausible(v):
            return v
    return None


def _eps_paren_adjusted(text, non_gaap):
    """Lennar: "$1.19 ($1.23 excluding mark-to-market losses ...)". The adjusted
    figure is a bracketed aside right after the reported one."""
    rx = (r"\$\s?(-?[\d.]+)\s*\(\s*\$\s?(-?[\d.]+)\s+(?:excluding|before|"
          r"on\s+an\s+adjusted\s+basis|as\s+adjusted|adjusted)")
    for m in re.finditer(rx, text, re.I):
        # must be an EPS sentence, not a revenue or net-income one
        ctx = text[max(0, m.start() - 110):m.start()]
        if not re.search(r"per\s+(?:diluted\s+)?share|EPS", ctx, re.I):
            continue
        if _reject(text, m.start()):
            continue
        v = _num(m.group(2) if non_gaap else m.group(1))
        if _plausible(v):
            return v
    return None


def _eps_moved_to(text, non_gaap):
    """Copart: "Fully diluted earnings per share for the three months ended July 31,
    2026 declined to $0.35 compared to $0.41 last year". A period clause and a verb
    sit between the label and the figure. The FULL-YEAR twin of that sentence
    follows immediately, so the cumulative guard is what picks the quarter."""
    if non_gaap:
        return None
    rx = (r"(?:fully\s+)?(?:diluted\s+)?(?:net\s+)?(?:earnings|income|EPS)\s+per\s+"
          r"(?:diluted\s+|common\s+)?share\s+for\s+the\s+([^$]{0,60}?)\s+"
          r"(?:increased|decreased|declined|rose|fell|grew|was|were)\s+(?:to\s+)?"
          r"\$\s?\(?(-?[\d.]+)\)?")
    for m in re.finditer(rx, text, re.I):
        period = m.group(1)
        if _CUMULATIVE.search(period) or re.search(r"\byear\s+ended\b", period, re.I):
            continue
        if _reject(text, m.start()):
            continue
        v = _num(m.group(2))
        if _plausible(v):
            return v
    return None


def _eps_subrow_diluted(text, non_gaap):
    """GameStop: "Adjusted net income per share / Basic $0.36 $0.31 ... / Diluted
    0.27 0.20 ..." - the per-share label heads a block, and the diluted figure is
    a SUB-ROW beneath it, with the basic figure (a different number) above."""
    lab = (r"(?:adjusted|non-?GAAP)[^$\n]{0,30}" if non_gaap else r"")
    rx = (lab + r"(?:net\s+)?(?:income|earnings|loss)\s*(?:\(loss\))?\s+per\s+"
          r"(?:common\s+)?share[^$]{0,20}?\bBasic\b.{0,90}?\bDiluted\b\s*:?\s*"
          r"\$?\s?(\(?-?[\d.]+\)?)")
    for m in re.finditer(rx, text, re.I | re.S):
        if not non_gaap and re.search(r"adjusted|non-?GAAP",
                                      text[max(0, m.start() - 40):m.start() + 40], re.I):
            continue
        if _reject(text, m.start()):
            continue
        raw = m.group(1)
        v = _num(raw.strip("()"))
        if _plausible(v):
            return -abs(v) if raw.startswith("(") else v
    return None


def read_eps(text):
    """(value, basis). Non-GAAP is preferred because that is what the street quotes.

    Strategies are tried in order of how specific they are, not how convenient.
    The plain (no-basis) reader runs BETWEEN the non-GAAP and GAAP passes, because
    a company that states only one EPS figure means that figure, and forcing it
    into the GAAP slot mislabels it on the panel.
    """
    text = _mask_ranges(text)
    body = _results_only(text)
    head = body[:11000]
    for fn in (_eps_loss_words, _eps_and_adjusted, _eps_paren_adjusted,
                _eps_respectively, _eps_on_a_basis, _eps_trailing_basis, _eps_forwards,
                _eps_nested, _eps_via_net_income, _eps_of, _eps_backwards,
                _eps_subrow_diluted):
        v = fn(head, True)
        if v is not None:
            return v, "non-GAAP"
    v = _eps_from_table(text, non_gaap=True)
    if v is not None:
        return v, "non-GAAP"
    v = _eps_subrow_diluted(body, True)
    if v is not None:
        return v, "non-GAAP"
    v = _eps_block_basis(head, True)
    if v is not None:
        return v, "non-GAAP"
    v = _eps_plain(head)
    if v is not None:
        return v, ""
    for fn in (_eps_loss_words, _eps_and_adjusted, _eps_paren_adjusted, _eps_moved_to,
                _eps_respectively, _eps_on_a_basis, _eps_forwards,
                _eps_nested, _eps_via_net_income, _eps_of, _eps_backwards,
                _eps_subrow_diluted):
        v = fn(head, False)
        if v is not None:
            return v, "GAAP"
    v = _eps_from_table(text)
    if v is not None:
        return v, "GAAP"
    # The statement sits past the prose window in long releases - Shoe Station's
    # "Net income per share: Basic $0.23 ... Diluted $0.23" is 20,000 characters in.
    v = _eps_subrow_diluted(body, False)
    if v is not None:
        return v, "GAAP"
    v = _eps_long_label(body)
    if v is not None:
        return v, "GAAP"
    return None, None


def _eps_long_label(text):
    """Last resort: a per-share LOSS row whose label runs on for a sentence.

    Gloo: "Net loss per share attributable to common stockholders of Gloo Holdings,
    Inc. (Class A and Class B) and units of members of Gloo Holdings, LLC,
    respectively $ (0.25)". Champions Oncology prints only "Net loss per common
    share outstanding basic $ (0.03)". Only a bracketed figure is taken, so the
    sign is never guessed.
    """
    rx = (r"net\s+(?:income\s*\(loss\)|loss)\s+per\s+(?:common\s+)?share[^$]{0,220}?"
          r"\$\s*\(\s*([\d.]+)\s*\)")
    for m in re.finditer(rx, text, re.I):
        if _reject(text, m.start()) or _is_year_ago(text, m):
            continue
        v = _num(m.group(1))
        if _plausible(v):
            return -abs(v)
    return None


# Kestra and Rent the Runway write "Net loss per share attributable to common
# shareholders, basic and diluted $ (0.75)" - a 47-character run between "share"
# and "diluted". SailPoint's reconciliation row is "Adjusted EPS, diluted $ 0.09",
# and Hooker's caption is "Diluted earnings / (loss) per share".
_EPS_ROW = re.compile(r"(?:net\s+)?(?:income|earnings|loss)\s*(?:/\s*)?(?:\(loss\))?\s*" + _PS +
                      r"[^$\n]{0,60}?,?\s*diluted|diluted\s+(?:net\s+)?"
                      r"(?:income|earnings|loss)\s*(?:/\s*)?(?:\(loss\))?\s*" + _PS +
                      r"|\bEPS,?\s*diluted|diluted\s+EPS\b", re.I)


_TABLE_TITLE_TAIL = re.compile(
    r"^[\s,:()a-z]{0,40}?(?:unaudited|three|six|nine|twelve|thirteen|reconciliation|"
    r"months?\s+ended|weeks?\s+ended|quarter\s+ended|year\s+ended|period\s+ended|"
    r"in\s+(?:millions|thousands))",
    re.I)


def _eps_from_table(text, non_gaap=False):
    """EPS off the statement of operations, for releases that never say it in prose.

    With non_gaap=True this looks only inside a NON-GAAP RECONCILIATION table.
    UiPath states its non-GAAP EPS nowhere but there, so the prose pass found the
    GAAP $0.07 and the panel compared it against a $0.15 ADJUSTED consensus - a
    53% "miss" on a quarter that was roughly in line. Wrong basis on both sides of
    a comparison is worse than no comparison.

    UiPath's release names non-GAAP EPS only in its definitions section and puts
    the actual figure in a table roughly 15,000 characters in - past every prose
    window - so without this the panel showed a revenue row and no EPS at all.

    Per-share rows are never scaled by the "(in thousands)" caption above them,
    which is why this deliberately ignores the table's scale.
    """
    for pm in _PERIOD_HDR.finditer(text):
        seg = text[pm.start():pm.end() + 1800]
        if non_gaap:
            # Only trust a row that sits under a non-GAAP heading, and only take
            # the DILUTED line - the basic figure is a different number and sits
            # immediately above it.
            head = text[max(0, pm.start() - 700):pm.start()]
            if not re.search(_NG + r"|\bcore\b", head, re.I):
                continue
        for rm in _EPS_ROW.finditer(seg):
            before = seg[max(0, rm.start() - 60):rm.start()]
            if non_gaap and not re.search(_NG + r"|\bcore\b", before + rm.group(0), re.I):
                continue
            if non_gaap and re.search(r",\s*basic\s*$", rm.group(0), re.I):
                continue
            after = seg[rm.end():rm.end() + 90]
            # A TITLE or a column header, not a row: what follows is period words
            # and dates, and "MAY 2," reads as an EPS of 2.00 if you let it.
            if _TABLE_TITLE_TAIL.match(after):
                continue
            # "(2) $ (1.97)" - the leading (2) is a footnote marker, not a value.
            after = re.sub(r"^[\s:]*\(\d{1,2}\)(?=[\s$])", " ", after)
            after = _strip_dates(after)
            cells = _CELL.findall(after)
            if not cells:
                continue
            v = _num(cells[0])
            if _plausible(v) and abs(v) < 60:
                neg = re.search(r"\(\s*" + re.escape(cells[0]) + r"\s*\)", after)
                # "Net loss per share, basic and diluted $ 0.47" - enGene prints
                # the loss unbracketed and lets the label carry the sign.
                lab = rm.group(0)
                if _BARE_LOSS.search(lab) and not _GENERIC_LOSS.search(lab):
                    neg = True
                return -abs(v) if neg else v
    return None


# ═══════════════════════════════════════════════════════════════════════════
# REVENUE, and the growth rate the owner actually asked for
# ═══════════════════════════════════════════════════════════════════════════
_REV = r"(?:net\s+|total\s+|record\s+|consolidated\s+|worldwide\s+)*(?:revenues?|net\s+sales|sales)"
# Anything that contains one of these words is a different number wearing the
# word "revenue": CoreWeave's $104bn backlog, deferred revenue, RPO.
_NOT_REV = re.compile(r"backlog|deferred|remaining\s+performance|unearned|"
                      r"obligation|bookings|recurring\s+revenue|run[-\s]rate|"
                      # Lennar's "average sales price ... $370,000 to $380,000" was
                      # read as revenue guidance: a house price on the panel as the
                      # company's revenue. A price is never revenue.
                      r"sales\s+price|selling\s+price|\bASP\b|per\s+(?:home|unit)|"
                      # Jersey Mike's "Systemwide sales ... $1.210 billion" is its
                      # franchisees' sales; its own revenue was $208 million.
                      # Upexi's "Digital Asset Revenue Totaled $17.4 Million" is
                      # one line of a $25.0M total.
                      r"system[-\s]?wide|gross\s+(?:merchandise|booking)|digital\s+asset\s+revenue|"
                      # FactSet's non-GAAP table puts "Organic revenues $635,543"
                      # a line below "Revenues $634,698", and the larger won. An
                      # organic, constant-currency or pro-forma figure is a
                      # restatement of revenue, never the reported number.
                      r"organic\s+(?:net\s+)?(?:revenues?|sales)|constant[-\s]currency\s+(?:net\s+)?"
                      r"(?:revenues?|sales)|pro[-\s]?forma\s+(?:net\s+)?(?:revenues?|sales)",
                      re.I)
# Bullets are the real sentence boundary in a press release - full stops are not.
# Reading 190 characters past a revenue bullet walked into the NEXT bullet and
# picked up Docusign's ARR percentage and Marvell's per-share line as if they
# belonged to the revenue sentence.
# "Not the end of a sentence": any character that is not a full stop or a bullet,
# PLUS a full stop that is part of a number. Without the second half, every
# window regex stopped dead at the "." in "$2.739 billion" - which is how
# Marvell's "grew by 37% year-on-year" went unread and a table fallback put
# -47% on the panel instead.
_SENT = r"(?:[^.•●▪◦‣]|\.(?=\d))"

_REV_RX = (
    r"" + _REV + r"\s*[:\-]\s*\$\s?([\d,.]+)\s*(billion|million|trillion)?",
    _REV + r"\s+(?:was|were|of|totaled|reached|came\s+in\s+at)\s+\$\s?([\d,.]+)\s*(billion|million|trillion)?",
    _REV + r"(?:[^.$\u2022\u25cf\u25aa]|\.(?=\d)){0,150}?\bto\s+\$\s?([\d,.]+)\s*(billion|million|trillion)",
    r"(?:revenue|net\s+sales)" + _SENT + r"{0,160}?\$\s?([\d,.]+)\s*(billion|million|trillion)",
)


# A COMPONENT of revenue is not revenue. AeroVironment explains its quarter with
# "due to higher product sales of $15.5 million and higher service revenue of
# $10.3 million" - both the shape "<revenue word> of $N", which the narrow
# pattern matched, putting $15.5 million on the panel as the quarter's revenue.
# What they have in common is a qualifier sitting immediately in front.
_COMPONENT = re.compile(
    r"(?:product|service|contract|subscription|licen[cs]e|software|hardware|usage|"
    r"higher|lower|increase[d]?\s+in|decrease[d]?\s+in|comparable|same[-\s]store|"
    r"organic|international|domestic)\s*$", re.I)
# ...and a sentence that has just said "segment" is about to name one, not the
# total: "From a segment standpoint, Autonomous Systems ("AxS") recorded revenue
# of $346.0 million".
_SEG_CTX = re.compile(r"\bsegment", re.I)


def read_revenue_prose(text):
    """The headline revenue figure, as the company itself leads with it.

    PATTERN ORDER IS DELIBERATE and load-bearing. The tight "<revenue> was $N"
    forms come first and the broad "<revenue> ... $N within 160 characters" form
    last, because the broad one reaches across a whole headline: Navan's reads
    "Revenue grew 35% and Gross Booking Volume (GBV) grew 45% ... New Signed GBV
    Reached a Record $4.0 Billion", and taking the earliest match in the document
    rather than the most specific pattern read that $4.0bn booking figure as a
    $233M company's revenue.
    """
    head = text[:9000]
    for rx in _REV_RX:
        for m in re.finditer(rx, head, re.I):
            if _NOT_REV.search(head[max(0, m.start() - 45):m.end() + 20]):
                continue
            if _is_forward(head, m.start()):      # a guide is not a result
                continue
            if _COMPONENT.search(head[max(0, m.start() - 30):m.start()]):
                continue                          # a product or service line
            if _SEG_CTX.search(head[max(0, m.start() - 150):m.start()]):
                continue                          # a named segment, not the total
            v = money(m.group(1), m.group(2))
            if v is not None and v > 1e5:
                return v
    return None


_PERIOD_HDR = re.compile(
    r"(?:thirteen|fourteen|twelve|sixteen|three|3|12|13|14|16)\s*(?:-|\s)?(?:weeks?|months?)\s+ended"
    # Shoe Station splits its captions across two lines, which flatten to
    # "Thirteen Thirteen Twenty-six Twenty-six Weeks Ended Weeks Ended ...".
    r"|(?:thirteen|fourteen|twelve)\s+(?:thirteen|fourteen|twelve)\s+(?=(?:twenty[-\s]?six|thirty[-\s]?nine)\b)"
    # Micron: "4th Qtr. 3rd Qtr. 4th Qtr. Year Ended September 3, 2026 ...".
    r"|(?:1st|2nd|3rd|4th)\s+Qtr\.?(?=\s+(?:1st|2nd|3rd|4th)\s+Qtr)"
    # Bassett: "Quarter Ended  Nine Months Ended  August 29, 2026 ...".
    r"|(?<![a-z]\s)\bQuarter\s+Ended(?=\s+(?:(?:six|nine|twelve)\s+months|(?:fiscal\s+)?years?\s+ended|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},))"
    # lululemon heads its statement "Second Quarter  First Two Quarters", and UNFI
    # "Fourth Quarter Ended  Fiscal Year Ended". Neither says "three months", so
    # both statements went unread and the panel fell back to the prose's rounded
    # "$2.4 billion" and "$7.6 billion" - a revenue miss mis-sized by half a
    # point. Only taken when a second caption, the years or a date follow at
    # once, so a sentence that merely mentions "the second quarter" is not a table.
    r"|(?:first|second|third|fourth)\s+quarter(?:\s+ended)?(?=\s+(?:first\s+(?:two|three)\s+quarters|"
    r"(?:fiscal\s+)?years?\s+ended|(?:six|nine|twelve)\s+months|(?:19|20)\d{2}\s+(?:19|20)\d{2}\b|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},))",
    re.I)
_DATE_TOK = re.compile(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}",
                       re.I)
# A caption for MORE than a quarter. "Twelve weeks" is AutoZone's quarter, so
# twelve counts only as months.
_CUM_HDR = re.compile(
    r"(?:six|nine|twelve|6|9|12)\s*(?:-|\s)?months?\s+ended|"
    r"(?:twenty[-\s]?four|twenty[-\s]?six|thirty[-\s]?six|thirty[-\s]?nine|forty|fifty[-\s]?two|"
    r"fifty[-\s]?three|24|26|36|39|40|52|53)\s*(?:-|\s)?weeks?\s+ended|"
    r"(?:fiscal\s+)?years?\s+ended|first\s+(?:two|three)\s+quarters|year[-\s]to[-\s]date", re.I)
# A figure, not a caption's "$": Aeluma's "($ in thousands, except per share
# data)" sat between its captions and its dates, and stopping there counted no
# dates at all - the sequential quarter became "last year".
_FIRST_FIG = re.compile(r"\$\s*\(?\s*\d|\b\d{1,3},\d{3}\b|\b\d+\.\d+\b")


def _hdr_end(body, pm):
    """Where the caption block that starts at pm ENDS - at its first figure.

    Captions stack: "Thirteen Thirteen Twenty-six Twenty-six Weeks Ended ..." (Shoe
    Station) or a fiscal-year caption right beside the quarter's. The next table
    begins only after this one's figures do, so the search for it starts there -
    stopping at the sibling caption left the statement with no rows at all.
    """
    f = _FIRST_FIG.search(body, pm.end())
    return max(pm.end() + 20, f.start() if f and f.start() - pm.end() < 400 else 0)


def _hdr_columns(body, pm):
    """(offset, ncol) - where THIS QUARTER's columns sit under a period header.

    offset: columns to skip before the quarter's. Coffee Holding prints "Nine
    months ended July 31, Three months ended July 31, 2026 2025 2026 2025", so the
    first figure in every row is NINE months - $69.4M of sales on a $21.7M quarter.
    ncol: how many columns the quarter owns. Three means the middle one is last
    QUARTER and the year-ago figure is the third (Marvell, HPE, Aeluma).

    When the captions are STACKED ahead of the dates - "Three Months Ended Year
    Ended June 30, 2026 June 30, 2025 June 30, 2026 June 30, 2025" (InnovAge) -
    the cumulative caption owns the last two dates, not none of them. Counting
    zero there fell back to reading the fourth column, last year's FULL YEAR, as
    the year-ago quarter: -69% growth on a quarter that grew 18%.
    """
    before = body[max(0, pm.start() - 90):pm.start()]
    cb = None
    for cb in _CUM_HDR.finditer(before):
        pass
    # Only a caption sitting RIGHT beside this one is a column group: nothing but
    # a month and day may come between. LightPath's prose "full year ended June
    # 30, 2026. Financial Summary: Three Months Ended" is a sentence, not a column.
    if cb and re.fullmatch(r"[\s,()]*(?:[A-Za-z]{3,9}\.?\s+\d{1,2},?)?[\s,()]*(?:unaudited)?[\s,()]*",
                           before[cb.end():], re.I):
        return 2, 2
    tail = body[pm.end():pm.end() + 320]
    cut = _CUM_HDR.search(tail)
    hdr = tail[:cut.start()] if cut else tail[:190]
    n = len(_DATE_TOK.findall(hdr))
    if cut and n == 0:
        rest = tail[cut.end():]
        f = _FIRST_FIG.search(rest)
        blk = rest[:f.start()] if f else rest
        # Predecessor/successor stub periods ("Period from January 16 to June
        # 29, 2025") break the two-dates-per-caption rule; Jersey Mike's became
        # -47% growth against its 26-week column. Keep the plain reading there.
        if not re.search(r"period\s+from", blk, re.I):
            total = len(_DATE_TOK.findall(blk))
            if total >= 4:
                n = total - 2
    return 0, max(2, min(4, n or 2))
_ROWNAME = re.compile(r"\b(?:total\s+(?:net\s+)?(?:revenues?|sales)|net\s+revenues?|"
                      r"net\s+sales|revenues?|sales)\b", re.I)
_CELL = re.compile(r"\(?\$?\s?(\d[\d,]*(?:\.\d+)?)\s*\)?")
_PCT_CELL = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\s*\)?\s*%")
# (RH writes the caption "% OF NET" above the date and "REVENUES" below it.)
_PCT_COLS = re.compile(r"percent(?:age)?\s+of\s+(?:net\s+|total\s+)?(?:revenues?|sales)|"
                       r"%\s+of\s+(?:net|total|revenues?|sales)\b", re.I)
# What may sit between a row label and its first figure: the rest of the label
# ("Sales of marine technology products"), a colon, ", net", a bracketed caption
# or footnote, a "[1]" marker, "- GAAP", the dollar sign. A label is a noun
# phrase; a VERB in it means the "row" is a sentence.
_ROW_LABEL_TAIL = re.compile(r"[A-Za-z&,'/\-\s]{0,70}?(?=[\s:]*(?:\([^()]{0,60}\)\s*)?(?:\[\d\]\s*)?"
                             r"(?:-\s*GAAP\s*)?\(?\s*\$?\s*\d)")
# Verbs only: "Revenue from contracts with customers" and "Revenues attributable
# to ..." are real row labels, so prepositions alone do not make a sentence.
_PROSE_WORD = re.compile(r"\b(?:increas\w*|decreas\w*|grew|grow\w*|was|were|is|are|rose|fell|declin\w*|"
                         r"compar\w*|totaled|totalled|reach\w*|versus|vs|ended|of\s+approximately)\b", re.I)


# Mission Produce's "Segment Sales" table ends each row with the total, so its
# "Total net sales" row reads "$414.3 $15.5 $14.8 $5.4 $450.0" - four segments
# and then the company. The word "Total" ranked it above the real income
# statement and the panel showed one segment's $414.3M as the quarter.
_SEGMENT_TBL = re.compile(r"(?:reportable\s+segment|segment\s+(?:results|information|data|"
                          r"reporting|sales|revenues?|net\s+sales)|results\s+by\s+segment|"
                          r"revenue\s+by\s+(?:segment|business|product\s+line|geograph))", re.I)


def _table_rows(text):
    """Every (current, prior, scale) revenue row found in a quarterly statement.

    Yielded in document order. CoreWeave, Casey's and Best Buy state the quarter's
    revenue ONLY in the statement - the prose leads with backlog, an EPS headline
    or a comparable-sales figure - so without this they show no revenue at all.
    """
    # A guidance table is not a statement of results. Docusign prints
    # "Revenue [1] $886 to $890" under a period header identical in shape to the
    # income statement's, and reading it as the quarter put $886M on the panel as
    # a result and then computed a -99.9% growth rate against it.
    _span, out_at = _outlook_span(text)
    body = text[:out_at] if out_at > len(text) * 0.5 else text

    for gid, pm in enumerate(_PERIOD_HDR.finditer(body)):
        # A SEGMENT TABLE IS NOT THE INCOME STATEMENT. AeroVironment's is headed
        # "Reportable Segment Results" and its columns are AxS | SCDE | Total, so
        # the row reads "... Total Revenue $345,969 $134,521 $480,490": the word
        # "Total" belongs to the COLUMN HEADER but sits directly in front of the
        # row label, and _ROWNAME matched the two as "Total Revenue". That scored
        # rank 0 - the preference meant for a real total - and beat the income
        # statement, putting one segment's revenue on the panel as the quarter's,
        # with a second segment as the "year-ago" column. $346M vs $134.5M, read
        # as +157% growth, when the quarter was $480.5M and +6%.
        if _SEGMENT_TBL.search(body[max(0, pm.start() - 260):pm.start() + 40]):
            continue
        # Most statements print the quarter and the year-to-date side by side
        # under ONE header - "Three Months Ended ... Six Months Ended ..." with
        # four dates. Counting all four made the year-to-date PRIOR column look
        # like the year-ago quarter. Three dates means the MIDDLE column is last
        # quarter, not last year - Marvell and HPE both print it that way.
        offset, ncol = _hdr_columns(body, pm)
        if _CUMULATIVE.search(pm.group(0)):
            continue
        nxt = re.compile(r"(?:thirteen|fourteen|twelve|sixteen|three|thirty-six|thirty-nine|"
                         r"fifty-two|twenty-six|3|12|13|14|16|24|26|36|39|52)\s*(?:-|\s)?"
                         r"(?:weeks?|months?)\s+ended", re.I).search(body, _hdr_end(body, pm))
        stop = min(pm.end() + 700, nxt.start() if nxt else len(body))
        seg = body[pm.start():stop]
        sc = re.findall(r"in\s+(millions|thousands|billions)",
                        body[max(0, pm.start() - 1600):pm.end() + 260], re.I)
        scale = _SCALE[sc[-1].lower()] if sc else 1.0
        for rm in _ROWNAME.finditer(seg):
            if _NOT_REV.search(seg[max(0, rm.start() - 45):rm.end() + 12]):
                continue
            # A statement ROW has its figures right after the label. Jersey
            # Mike's prose "Total revenue increased 10% ... to $208 million ...
            # Royalties and other revenue increased 11% to $138 million" sits
            # under a period header too, and read as a row it made the
            # royalties figure last year's revenue: +51% on a +10% quarter.
            lt = _ROW_LABEL_TAIL.match(seg, rm.end())
            if not lt or _PROSE_WORD.search(lt.group(0)):
                continue
            # "Cost of oncology revenue 7,534" is a cost line wearing the word;
            # Champions Oncology's statement put it on the panel as revenue.
            if re.search(r"cost\s+of\s+(?:[a-z&]+\s+){0,2}$", seg[max(0, rm.start() - 30):rm.start()], re.I):
                continue
            # Dave & Buster's prints a percent-of-revenue column beside every
            # figure: "Total revenues 544.1 100.0 % 557.4 100.0 %". The 100.0 is
            # not last year's revenue.
            row = _PCT_CELL.sub(" ", seg[rm.end():rm.end() + 190])
            cells = [c for c in (_num(x) for x in _CELL.findall(row[:150])) if c is not None]
            cells = cells[offset:]
            # CarMax prints the same column with NO % sign: "Net sales and
            # operating revenues $ 7,877.9 100.0 $ 6,594.7 100.0". A revenue row's
            # share of revenue is always 100.0, which gives the layout away - and
            # without this the year-ago revenue read as 100.0.
            if len(cells) >= 4 and abs(cells[1] - 100.0) < 0.05 and abs(cells[3] - 100.0) < 0.05 \
                    and cells[0] > 100 and cells[2] > 100:
                cells = cells[0::2]
            if len(cells) < 2:
                continue
            yield gid, cells, ncol, scale, rm.group(0), offset


def _promote_to_total(cells, ncol):
    """The subtotal line, when the statement adds its components up unlabelled.

    AeroVironment's income statement reads

        Revenue:  Product sales      329,058   313,533
                  Contract services  151,432   141,143
                                     480,490   454,676

    - the total carries NO row label at all, so no name-based rule can ever find
    it, and the "Revenue" caption matches the first COMPONENT. The panel showed
    $329M for a $480.5M quarter.

    The arithmetic is the label: a pair that equals the running sum of the pairs
    above it is a subtotal. Fires only when the sum actually holds to within a
    rounding unit, so a statement that does not have this shape is untouched.
    """
    pairs = [(cells[i], cells[i + ncol - 1])
             for i in range(0, len(cells) - ncol + 1, ncol)]
    run_c = run_p = 0.0
    for k, (c, pr) in enumerate(pairs):
        if k and run_c and abs(c - run_c) <= max(1.0, abs(run_c) * 0.005)                 and abs(pr - run_p) <= max(1.0, abs(run_p) * 0.005):
            return c, pr
        run_c += c
        run_p += pr
    return None


def read_revenue_table(text):
    """Revenue from the income statement, for releases whose prose never states it.

    Prefers a row literally called "total revenue" over a bare "revenue" row,
    because Palo Alto's statement opens with product revenue and Best Buy's with
    a single segment.
    """
    best = None
    for gid, cells, ncol, scale, name, _off in _table_rows(text):
        tot = _promote_to_total(cells, ncol)
        if tot:
            cells = [tot[0], tot[1]]
        v = cells[0] * scale
        # Aeluma's quarter was $582 thousand; a $1M floor sent it to the prose's
        # rounded "$0.6 million". The lower floor needs a STATED scale, though:
        # Lesaka's caption went unread and "188,321" (thousands) came through
        # as $188 thousand.
        if not ((1e5 if scale != 1.0 else 1e6) <= v <= 5e12):
            continue
        # A RECONCILIATION STATEMENT REPEATS THE NUMBER. Oracle prints
        # "TOTAL REVENUES $19,184  $-  $19,184   $15,903  $-  $15,903" - GAAP,
        # adjustment, non-GAAP, then the same three for last year. Once the
        # em-dash column stops being read as a figure, the cell sitting where the
        # year-ago column should be is the CURRENT quarter again, and the panel
        # reported 0.0% growth on a quarter that grew 21%. A revenue identical to
        # the last cell is a repeated column, never a year that did not move.
        k = min(ncol - 1, len(cells) - 1)
        while k + 1 < len(cells) and abs(cells[k] * scale - v) < 1e-6:
            k += 1
        prior = cells[k] * scale
        # A real year-on-year pair moves by less than 8x either way - American
        # Battery's +407% is the widest in any key. A segment total beside one
        # small segment (Mission Produce: $414.3M vs $15.5M, "26x") is not a year.
        if prior <= 0 or not (0.1 < v / prior < 8):
            continue                       # not a year-on-year pair
        # Segments always sum to the total, so among the revenue rows of one
        # statement the LARGEST is the total - which is what makes this immune to
        # Palo Alto's product line, Okta's subscription line and HPE's Networking
        # line all sitting above the number the owner wants.
        rank = 0 if re.match(r"total", name, re.I) else 1
        # The income statement is the FIRST quarterly table in the release, and
        # its revenue line is the first row of it. HPE repeats "Net revenue" for
        # the nine-month columns further down, so ranging over the whole document
        # and taking the largest put the year-to-date figure on the panel.
        key = (rank, gid, -v)
        if best is None or key < best[0]:
            best = (key, v, prior)
    return (best[1], best[2]) if best else (None, None)


def _table_year_ago(text, anchor):
    """The year-ago figure for a revenue of `anchor`, read off the income statement.

    ANCHORED on the headline number, which is what makes this safe. Palo Alto's
    statement opens with a $738M product-revenue line; anchoring on the $3.41B
    headline skips straight past it to the total. Without the anchor this returns
    a component and the growth rate is nonsense.

    Column count comes from the period header: two dates means the next number is
    last year, three means the middle one is last QUARTER and must be stepped over.
    """
    if anchor is None:
        return None
    for _gid, cells, ncol, scale, _name, _off in _table_rows(text):
        for i, c in enumerate(cells[:3]):
            if abs(c * scale - anchor) / anchor > 0.015:      # not the row we want
                continue
            rest = [x for x in cells[i + 1:i + ncol + 2]
                    if abs(x * scale - anchor) > max(1.0, anchor * 1e-6)]
            # Skipping cells EQUAL to the anchor is what makes a reconciliation
            # statement readable. Oracle's row is "19,184  -  19,184  15,903 ..."
            # (GAAP, adjustment, non-GAAP); the cell where the year-ago column
            # should sit is this quarter over again, and returning it reported
            # 0.0% growth on a quarter that grew 21%.
            if not rest:
                continue
            prior = rest[0] * scale
            if prior > 0 and 0.05 < anchor / prior < 25:
                return prior
    return None


_GROWTH_RX = (
    (r"(?:grew|rose|increased|up|climbed|gained|growth\s+of)\s+(?:by\s+)?([\d.]+)\s*(?:%|percent)", 1),
    (r"\b([\d.]+)\s*(?:%|percent)\s+year[-\s]?(?:over|on)[-\s]?year\s+(?:increase|growth)", 1),
    (r"(?:declined|decreased|fell|down|dropped|decrease\s+of)\s+(?:by\s+)?([\d.]+)\s*(?:%|percent)", -1),
)
def _mentions(seg, v):
    """Does this sentence actually quote the figure? '$12.2 billion', '12,213',
    '$2.739 billion' and '875.7 million' are all the same number written four ways."""
    for cand in ("%.0f" % v, "%.1f" % (v / 1e9), "%.2f" % (v / 1e9), "%.3f" % (v / 1e9),
                 "%.1f" % (v / 1e6), "%.0f" % (v / 1e6), "%.0f" % (v / 1e3)):
        if len(cand.replace(".", "")) >= 2 and cand in seg.replace(",", ""):
            return True
    return False


_YOY = re.compile(r"year[-\s]?(?:over|on)[-\s]?year|prior[-\s]year|from\s+(?:the\s+)?(?:same|prior|"
                  r"a\s+year)|y/y|\byoy\b|a\s+year\s+ago|last\s+year|"
                  r"prior[-\s]year\s+period|same\s+period", re.I)


def read_growth_prose(text, anchor=None):
    """Stated year-on-year REVENUE growth, signed. None when the release is silent.

    Only trusted inside a revenue sentence. HPE's release says "GAAP OP up 464%"
    two lines from its revenue bullet, and lululemon says "comparable sales
    decreased 9%" beside a 4% revenue decline; reading either as revenue growth
    would be worse than showing nothing.
    """
    head = text[:7000]
    # "Net sales of $284.3 million compared to $306.4 million in the second quarter
    # 2025" - both figures, so the rate is arithmetic, not a claim to go looking for.
    # The comparison figure must be LAST YEAR's: Micron's "Revenue of $41.46
    # billion versus $23.86 billion for the prior quarter and $9.30 billion for the
    # same period last year" names three quarters, and the first one after
    # "versus" is the sequential one.
    if anchor:
        for m in re.finditer(_REV + r"\s+(?:of|was|were|totaled)\s+\$\s?([\d,.]+)\s*(billion|million|thousand)?",
                             head, re.I):
            a = money(m.group(1), m.group(2))
            if not a or abs(a - anchor) / anchor > 0.02:
                continue
            if _NOT_REV.search(head[max(0, m.start() - 45):m.start()]):
                continue
            sent = re.split(r"(?<!\d)\.(?!\d)|[•●▪]", head[m.end():m.end() + 240])[0]
            b = None
            y = re.search(r"\$\s?([\d,.]+)\s*(billion|million|thousand)?\s*(?:for|in)\s+the\s+"
                          r"(?:same\s+(?:period|quarter)\s+(?:last|a|of\s+the\s+prior)\s+year|"
                          r"prior[-\s]year\s+(?:period|quarter)|year[-\s]ago\s+(?:period|quarter)|"
                          r"(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?(?:fiscal\s+)?(?:19|20)\d{2})",
                          sent, re.I)
            if y:
                b = money(y.group(1), y.group(2) or m.group(2))
            else:
                y = re.match(r",?\s+(?:compared\s+(?:to|with)|versus|vs\.?)\s+\$\s?([\d,.]+)\s*"
                             r"(billion|million|thousand)?([^$]{0,60})", sent, re.I)
                if y and not re.search(r"(?:prior|previous|preceding|last)\s+quarter|sequential",
                                       y.group(3), re.I):
                    b = money(y.group(1), y.group(2) or m.group(2))
            if b and 0.1 < a / b < 8:
                return 100.0 * (a - b) / b
    for m in re.finditer(_REV + _SENT + r"{0,190}", head, re.I):
        seg = m.group(0)
        if re.search(r"comparable|comp\s+sales|segment|operating\s+profit|"
                     r"gross\s+margin|per\s+share", seg, re.I):
            continue
        if anchor and not _mentions(seg, anchor):
            continue
        for rx, sign in _GROWTH_RX:
            g = re.search(rx, seg, re.I)
            if not g:
                continue
            # Only what comes BEFORE the rate can make it another metric's.
            # General Mills: "Net sales of $4.6 billion were up 1 percent;
            # organic net sales were in line" - the organic clause follows.
            if _NOT_REV.search(seg[:g.end()]):
                continue
            if not _YOY.search(seg) and rx is not _GROWTH_RX[1][0]:
                continue                          # a bare "grew 8%" may be sequential
            v = _num(g.group(1))
            if v is not None and v <= 100000:
                return sign * v
    for m in re.finditer(_REV + r"\s+(increased|decreased|grew|declined|rose|fell)\s+([\d.]+)\s*(?:%|percent)",
                         head, re.I):
        # "Shoe Carnival banner net sales declined 6.5 percent" is one of the
        # company's two store brands; the company's own sales fell 7.2%. A
        # qualifier in front makes it a part, not the whole.
        if re.search(r"(?:banner|brand|segment|channel|store|comparable|same[-\s]store|organic|"
                     r"vinyl|product|service|subscription|division|region)\s*$",
                     head[max(0, m.start() - 30):m.start()], re.I):
            continue
        if _NOT_REV.search(head[max(0, m.start() - 45):m.end()]):
            continue
        v = _num(m.group(2))
        if v is not None:
            return -v if m.group(1).lower() in ("decreased", "declined", "fell") else v
    return None


# ═══════════════════════════════════════════════════════════════════════════
# MARGIN AND CAPEX - the two extras worth a row of their own
#
# These two earn the space because they change what a beat MEANS in the minutes after the print:
#
#   MARGIN separates a beat won on price from a beat won on discounting. A
#   revenue beat with gross margin down 200bps is a different trade from the
#   same beat with margin up.
#
#   CAPEX is the number that moves the AI-infrastructure names the owner actually
#   trades. CoreWeave spent $6.4bn on property and equipment in a quarter
#   against $2.5bn a year earlier; nothing else in that release says as much
#   about the stock.
# ═══════════════════════════════════════════════════════════════════════════
_MARGIN_KIND = r"(?:gross|operating)"

def _kpat(kind):
    """'gross' also matches 'gross profit' - Hain's consolidated line says
    "Adjusted gross profit margin was 22.7%", and skipping it let a segment's
    31.1% further down stand in for the whole company."""
    return r"gross(?:\s+profit)?" if kind == "gross" else kind



def _margin_sane(v):
    """A margin below 3% or above 95% is a misread, not a remarkable quarter.

    Accenture's "Gross margin (gross profit as a percentage of revenues) for the
    quarter was 32.8%" was being read as 0.4 - the parenthetical swallowed the
    real figure and a stray number took its place. Rejecting the band turns that
    class of failure into a blank row instead of a wrong one.
    """
    return v is not None and 3.0 <= v <= 95.0


def _margin_after(text, non_gaap, kind):
    """'Non-GAAP gross margin was 81.7%' - label, then figure."""
    lab = (_NG + r"\s+" if non_gaap else r"(?<!non-)(?<!non )GAAP\s+")
    # "of approximately 80%", "was 35.8% of sales", and Accenture's
    # "Gross margin (gross profit as a percentage of revenues) for the quarter
    # was 32.8%" all sit behind filler this used to trip over.
    filler = r"(?:\([^)]{0,70}\)\s*)?(?:for\s+the\s+quarter\s*)?"
    adv = r"(?:approximately|about|roughly|nearly|around)?\s*"
    rx = (lab + _kpat(kind) + r"\s+margins?\s*(?:\(loss\))?\s*" + filler +
          r"(?:was|were|of|:|is)?\s*(?:both\s+)?" + adv + r"([\d.]+)\s*%")
    for m in re.finditer(rx, text, re.I):
        if _is_forward(text, m.start()):
            continue
        v = _num(m.group(1))
        if v is not None and -200 < v < 100:
            return v
    return None


def _margin_before(text, non_gaap, kind):
    """The figure comes first - Marvell writes '58.9% non-GAAP gross margin'."""
    lab = (_NG + r"\s+" + _kpat(kind) + r"\s+margin" if non_gaap
           else r"(?<!non-)(?<!non )GAAP\s+" + _kpat(kind) + r"\s+margin")
    for m in re.finditer(lab, text, re.I):
        if _is_forward(text, m.start()):
            continue
        seg = text[max(0, m.start() - 40):m.start()]
        got = re.findall(r"([\d.]+)\s*%", seg)
        if got:
            v = _num(got[-1])
            if v is not None and -200 < v < 100:
                return v
    return None


def _margin_both(text, kind):
    """'GAAP and non-GAAP gross margins were both 75.0%' - NVIDIA's phrasing."""
    m = re.search(r"GAAP\s+and\s+" + _NG + r"\s+" + kind +
                  r"\s+margins?\s+were\s+both\s+([\d.]+)\s*%", text, re.I)
    if m and not _is_forward(text, m.start()):
        v = _num(m.group(1))
        if v is not None and -200 < v < 100:
            return v
    return None


def _margin_nested(text, non_gaap, kind):
    """The same nested-bullet shape as _eps_nested, for margins:

        - Gross margins :
            - GAAP of 40.1%, up 1,090 basis points from the prior-year period
            - Non-GAAP (1) of 40.4%, up 1,050 basis points

    Returns (value, bps_change_or_None) - HPE states the year-on-year move in
    basis points right there, which is more precise than differencing two
    rounded percentages.
    """
    lab = (_NG if non_gaap else r"(?<!non-)(?<!non )GAAP")
    for m in re.finditer(kind + r"\s+margins?\s*:", text, re.I):
        if _is_forward(text, m.start()):
            continue
        seg = text[m.end():m.end() + 340]
        nxt = re.search(r"[" + _BULLET[0] + r"]", seg)
        if nxt:
            seg = seg[:nxt.start()]
        mm = re.search(lab + r"\s*(?:\(\d\))?\s*of\s+([\d.]+)\s*%", seg, re.I)
        if not mm:
            continue
        v = _num(mm.group(1))
        if v is None or not (-200 < v < 100):
            continue
        bps = None
        bm = re.search(r"(up|down)\s+([\d,]+)\s*basis\s+points\s+from\s+the\s+prior[-\s]year",
                       seg[mm.end():mm.end() + 130], re.I)
        if bm:
            b = _num(bm.group(2))
            if b is not None:
                bps = b if bm.group(1).lower() == "up" else -b
        return v, bps
    return None, None


_GROSS_PROFIT = re.compile(r"\bgross\s+(?:profit|margin)\b(?!\s*(?:%|percent|yield))", re.I)
_TOT_COST = re.compile(r"\b(?:total\s+)?cost\s+of\s+(?:revenues?|sales|goods\s+sold)\b", re.I)


def _margin_from_table(text, revenue):
    """Gross margin COMPUTED from the income statement, and the year-ago figure.

    This is the reliable path and it is tried first. Prose states gross margin in
    a dozen incompatible shapes - as a dollar subtotal, as a per-unit yield, only
    inside guidance, or not at all - which is why reading it as text scored 1
    right out of 15. Every income statement, by contrast, carries either a gross
    profit line or a total cost-of-revenue line, and revenue is already anchored.

        gross margin = gross profit / revenue
                     = (revenue - total cost of revenue) / revenue

    Both columns come from the same statement, so this quarter and a year ago are
    measured the same way and the change between them is real.
    """
    if not revenue:
        return None, None
    # THE MOST PRECISE STATEMENT WINS. LightPath prints a summary in millions to
    # one decimal ("Revenue $ 21.2 ... Gross Profit $ 8.3") before the statement
    # in dollars, and reading the summary because it came first put 39.2% on the
    # panel for a 39.4% quarter. Every candidate is collected, and the one whose
    # revenue figure is printed to the finest unit is used.
    best = None
    for unit, order, got in _margin_candidates(text, revenue):
        if best is None or unit < best[0]:
            best = (unit, order, got)
    return best[2] if best else (None, None)


def _unit(x, scale):
    """The finest amount a printed figure can express: 21.2 in millions is 100,000."""
    s = repr(float(x))
    dec = 0 if s.endswith(".0") or "." not in s else len(s.split(".")[1])
    return scale * 10.0 ** -dec


def _margin_candidates(text, revenue):
    """(unit, order, (margin, prior)) for every statement that yields a margin."""
    order = 0
    for gid, cells, ncol, scale, name, off in _table_rows(text):
        rev_now = cells[0] * scale
        if abs(rev_now - revenue) / revenue > 0.015:
            continue                      # not the revenue row this quarter used
        rev_prior = cells[min(ncol - 1, len(cells) - 1)] * scale
        unit = _unit(cells[0], scale)
        order += 1
        seg = _table_segment(text, gid)
        if seg is None:
            continue
        for rx, derive in ((_GROSS_PROFIT, False), (_TOT_COST, True)):
            for rm in rx.finditer(seg):
                after = seg[rm.end():rm.end() + 150]
                # A statement ROW puts its figures straight after the label. UNFI's
                # prose "Gross profit in the fourth quarter of fiscal 2026 was
                # $1,050 million" read "2026" as the gross profit - a 26.5% margin
                # on a 13.7% business.
                # (A caption in brackets may come first: Casey's "Cost of goods
                # sold (exclusive of depreciation and amortization, shown
                # separately below) 4,471,203".)
                if not re.match(r"[\s:,]*(?:\([^()]{0,90}\)\s*)?\(?\s*\$?\s*\d", after, re.I):
                    continue
                # A row of RATIOS: General Mills' "Gross margin 33.9 % 33.9 % Flat".
                # The figures are the margins themselves - stripping them as
                # percent cells left stray numbers that computed a -2% margin.
                if not derive and re.match(r"[\s:,]*-?[\d.]+\s*%", after):
                    pcts = [_num(x) for x in re.findall(r"(-?[\d.]+)\s*%", after[:90])][off:]
                    if pcts and _margin_sane(pcts[0]):
                        pr = pcts[ncol - 1] if len(pcts) >= ncol else None
                        yield unit, order, (pcts[0], (pr if pr is not None and _margin_sane(pr)
                                                      else None))
                        break
                    continue
                after = _PCT_CELL.sub(" ", re.sub(r"^[\s:,]*\([^()]{3,90}\)", " ", after))
                raw = _CELL.findall(after)
                nums = [c for c in (_num(x) for x in raw) if c is not None][off:]
                # American Eagle prints a "(Percentage of revenue)" column with no
                # % signs: "Gross profit 672,064 48.7 499,962 38.9". Read straight,
                # 48.7 was last year's gross profit and the margin's year-on-year
                # move came out -1bps on a quarter that expanded 980.
                # CarMax's has no caption above it at all - "2026 % (1) 2025 % (1)"
                # - and read straight, its "Gross profit $ 799.5 10.1 $ 717.7 10.9"
                # made 10.1 last year's gross profit: +5bps on a quarter whose
                # margin fell 74. But a share-of-revenue column is the figure
                # beside it divided by revenue, so it proves itself.
                share_col = len(nums) >= 4 and rev_now and \
                    abs(100.0 * nums[0] * scale / rev_now - nums[1]) < 0.15 and nums[2] >= 100
                if share_col or (_PCT_COLS.search(seg[:600]) and len(nums) >= 4 and nums[1] < 100
                                 and nums[3] < 100 and nums[0] >= 100 and nums[2] >= 100):
                    nums = nums[0::2]
                    raw = raw[:off] + raw[off::2]
                if len(nums) < 2:
                    continue
                gp_now = nums[0] * scale
                gp_prior = nums[min(ncol - 1, len(nums) - 1)] * scale
                # "Gross margin (loss) (3,089,240)" - American Battery's gross
                # LOSS. The bracket is the sign; without it the loss read as a
                # 14% margin.
                neg_now = bool(re.search(r"\(\s*\$?\s*" + re.escape(raw[off]) + r"\s*\)", after)) \
                    if len(raw) > off else False
                if neg_now and not derive:
                    gp_now = -gp_now
                if derive:
                    gp_now, gp_prior = rev_now - gp_now, rev_prior - gp_prior
                m_now = 100.0 * gp_now / rev_now if rev_now else None
                m_prior = (100.0 * gp_prior / rev_prior) if rev_prior else None
                if m_now is None or not (_margin_sane(m_now) or (neg_now and -100 < m_now < 0)):
                    continue
                if m_prior is not None and not _margin_sane(m_prior):
                    m_prior = None
                yield unit, order, (m_now, m_prior)
                break
            else:
                continue
            break                         # one margin per statement - the first row found


def _table_segment(text, want_gid):
    """The slice of the statement that _table_rows yielded rows from."""
    _span, out_at = _outlook_span(text)
    body = text[:out_at] if out_at > len(text) * 0.5 else text
    for gid, pm in enumerate(_PERIOD_HDR.finditer(body)):
        if gid != want_gid:
            continue
        nxt = re.compile(r"(?:thirteen|fourteen|twelve|sixteen|three|thirty-six|thirty-nine|"
                         r"fifty-two|twenty-six|3|12|13|14|16|24|26|36|39|52)\s*(?:-|\s)?"
                         r"(?:weeks?|months?)\s+ended", re.I).search(body, _hdr_end(body, pm))
        stop = min(pm.end() + 2400, nxt.start() if nxt else len(body))
        return body[pm.start():stop]
    return None


def _margin_dollars_then_pct(text, kind):
    """Signet: "Gross margin was $556.5 million, or 35.8% of sales".

    The row leads with a DOLLAR subtotal - Broadcom labels a $20.5bn line "Gross
    margin" outright - so the first number after the label is not a ratio at all.
    The percentage after "or" is.
    """
    rx = (kind + r"\s+margins?\s+(?:was|were|of)\s+\$\s?[\d,.]+\s*"
          r"(?:billion|million|thousand)?\s*,?\s*or\s+([\d.]+)\s*%")
    for m in re.finditer(rx, text, re.I):
        if _is_forward(text, m.start()):
            continue
        v = _num(m.group(1))
        if v is not None and -200 < v < 100:
            return v
    return None


def read_margin(text, revenue=None):
    """(value, kind, basis, prior_or_None). Gross preferred; operating is the
    fallback for companies that do not report a gross margin at all.

    The year-ago figure is only taken when the release states it in the SAME
    sentence - Docusign writes "GAAP gross margin was 79.7% compared to 79.3% in
    the same period last year". Pulling it from anywhere else risks pairing this
    quarter's non-GAAP against last year's GAAP, which would invent a swing.
    """
    head = text[:14000]
    mt, mp = _margin_from_table(text, revenue)
    if mt is not None:
        # Computed off the GAAP statement, so say GAAP. Zscaler headlines an 80%
        # non-GAAP gross margin against a 76.7% GAAP one; an unlabelled 76.7
        # would read as a miss against the number everyone quotes.
        return mt, "gross", "GAAP", mp
    for kind in ("gross", "operating"):
        for basis, ng in (("non-GAAP", True), ("GAAP", False)):
            dv = _margin_dollars_then_pct(head, kind)
            if dv is not None:
                return dv, kind, "", None
            nv, nbps = _margin_nested(head, ng, kind)
            if nv is not None:
                # A stated basis-point move beats differencing two rounded
                # percentages, so it is passed back as the prior directly.
                return nv, kind, basis, (nv - nbps / 100.0) if nbps is not None else None
            for fn in (_margin_both if ng else None, _margin_after, _margin_before):
                if fn is None:
                    continue
                v = fn(head, kind) if fn is _margin_both else fn(head, ng, kind)
                if v is None or not _margin_sane(v):
                    continue
                prior = None
                lab = (_NG if ng else "GAAP") + r"[^.]{0,20}" + kind + r"\s+margins?"
                m = re.search(lab + r"[^.]{0,60}?" + re.escape("%s" % v).replace(".", r"\.") +
                              r"\s*%[^.]{0,45}?compared\s+(?:to|with)\s+([\d.]+)\s*%",
                              head, re.I)
                if m:
                    prior = _num(m.group(1))
                return v, kind, basis, prior
    return None, None, None, None


# Scored against a 22-filing answer key the parser had never seen: margin came
# back right once, wrong twice and blank twelve times. Gross margin is simply not
# stated in a consistent enough shape to put on a live panel, so ONLY the two
# readings that were unambiguous there reach the chart - a nested bullet with a
# stated basis-point move (HPE), and a "$X, or N% of sales" line (Signet). The
# rest stay available to callers but are not shown.


# The rest of a capex row's label before its figures: Palo Alto's "purchases of
# property, equipment, and other assets 103 87", FactSet's "... leasehold
# improvements and capitalized internal-use software (27,401)", Evolution's
# "Capital expenditures for oil and natural gas properties (1,360)", GameStop's
# segment row "Capital expenditures - 1.7 - 1.7" with dashes for nil cells.
_CAPEX_TAIL = re.compile(r"[A-Za-z&,'/\-\s]{0,100}?(?=[\s:]*(?:\([^()]{0,60}\)\s*)?[\-—\s]*"
                         r"\(?\s*\$?\s*\(?\s*\d)")
_CAPEX_PROSE = re.compile(r"\b(?:increas\w*|decreas\w*|were|was|is|are|totaled|compar\w*|versus|"
                          r"of\s+approximately)\b", re.I)
_CAPEX_ROW = re.compile(
    r"(?:purchases?\s+of|additions?\s+to|acquisitions?\s+of)\s+property"
    r"(?:,|\s+and)?\s*(?:plant,?)?\s*(?:and\s+)?equipment"
    r"|capital\s+expenditures?", re.I)


def read_capex(text, revenue=None):
    """(this quarter, year-ago) capital spending in dollars, from the cash-flow
    statement. None when the release carries no cash-flow statement.

    THREE TRAPS, all of them live in the test corpus:

      * Casey's states its capex PLAN in the outlook - "purchase of property and
        equipment is expected to be approximately $800 million" - 4,000
        characters from the actual $655.9M cash-flow row. Forward-looking text is
        rejected here for the same reason it is for EPS.

      * DICK'S labels the row "Gross capital expenditures (8) $ 743 $ 526", where
        the (8) is a FOOTNOTE MARKER. Read as a cell it reported capex of $8M
        against a real $743M.

      * The row is signed negative in a statement of cash flows and usually
        parenthesised, so magnitude is what gets reported. A definition sentence
        carries no figures and is dropped by the two-cell requirement.

    `revenue` is used only as a sanity bound: capital spending above 3x revenue
    or below a thousandth of it is a misread row, not a remarkable quarter.
    """
    _span, out_at = _outlook_span(text)
    body = text[:out_at] if out_at > len(text) * 0.5 else text
    prose = None
    for pm in _PERIOD_HDR.finditer(body):
        # A SEGMENT table's columns are segments, not years. GameStop's "United
        # States Australia Europe Total / As of and for the three months ended"
        # reads "Capital expenditures - 1.7 - 1.7" - Australia, then the total -
        # and the total was taken as last year's: "+0% YoY" on a -59% quarter.
        if _SEGMENT_TBL.search(body[max(0, pm.start() - 260):pm.start() + 40]) or \
                re.search(r"as\s+of\s+and\s+for\s+the\s*$", body[max(0, pm.start() - 40):pm.start()], re.I):
            continue
        tail = body[pm.end():pm.end() + 220]
        cut = _CUM_HDR.search(tail)
        hdr = tail[:cut.start()] if cut else tail[:190]
        offset, ncol = _hdr_columns(body, pm)
        sc = re.findall(r"in\s+(millions|thousands|billions)",
                        body[max(0, pm.start() - 1600):pm.end() + 260], re.I)
        scale = _SCALE[sc[-1].lower()] if sc else 1.0
        # Oracle, Costco, Paychex, General Mills and Ciena state cash flows only
        # for the year or year-to-date, in releases whose income statement is
        # cleanly quarterly. Reporting a nine-month capex figure as "this quarter"
        # overstates it by roughly three times.
        if _CUMULATIVE.search(hdr) or _CUMULATIVE.search(pm.group(0)):
            continue
        # The cash-flow statement must be for a QUARTER. Costco, Paychex and
        # Jabil print theirs year-to-date only, under a header the quarterly
        # income statement's shape otherwise matches; their capex was being
        # reported as the quarter's and overstated two to threefold.
        if not re.search(r"(?:three|thirteen|twelve|fourteen|3|12|13|14)\s*(?:-|\s)?"
                         r"(?:month|week)s?\s+ended|quarter|\bqtr\b|thirteen\s+thirteen", pm.group(0), re.I):
            continue
        # The row must sit close enough to this header to belong to it. A cash
        # flow statement that follows a cumulative one 3,000 characters later was
        # being attributed to the quarterly header above it.
        nxt_hdr = re.compile(r"(?:thirteen|fourteen|twelve|sixteen|three|six|nine|"
                             r"twenty-six|thirty-six|thirty-nine|fifty-two|"
                             r"3|6|9|12|13|14|16|24|26|36|39|52)\s*(?:-|\s)?"
                             r"(?:weeks?|months?)\s+ended", re.I).search(body, _hdr_end(body, pm))
        seg = body[pm.start():min(pm.end() + 2600, nxt_hdr.start() if nxt_hdr else len(body))]
        for rm in _CAPEX_ROW.finditer(seg):
            if _is_forward(seg, rm.start()) or _is_cumulative(seg, rm.start()):
                continue
            after = seg[rm.end():rm.end() + 140]
            # A statement ROW, with its figures straight after the label. Jersey
            # Mike's prose "Capital expenditures in the second quarter were $3
            # million. ... As of June 28, 2026" read "28" as last year's capex
            # and printed -89% on a quarter that fell 40%.
            lt = _CAPEX_TAIL.match(after)
            if not lt or _CAPEX_PROSE.search(lt.group(0)):
                # (Micron: "Investments in capital expenditures, net (2) were
                # $10.77 billion for the fourth quarter of 2026".)
                pm2 = re.match(r"[\s,]*(?:net\s*)?(?:\(\d\)\s*)?(?:in\s+the\s+(?:first|second|third|fourth)\s+"
                               r"quarter\s+|for\s+the\s+quarter\s+)?(?:of|were|was|totaled)\s+"
                               r"(?:approximately\s+)?\$\s?([\d,.]+)\s*(million|thousand|billion)"
                               r"(?!\s+for\s+the\s+(?:full|fiscal)\s+year)",
                               after, re.I)
                if pm2 and prose is None:
                    v = money(pm2.group(1), pm2.group(2))
                    if v and v > 1e4 and not (revenue and not (revenue / 1000.0 < v < revenue * 3)):
                        prose = v
                continue
            # Drop a leading footnote marker: "(8) $ 743 $ 526" -> "$ 743 $ 526".
            after = re.sub(r"^[\s:]*\((\d{1,2})\)(?=\s*\$)", " ", after)
            cells = [c for c in (_num(x) for x in _CELL.findall(after)) if c is not None][offset:]
            if len(cells) < 2:
                continue
            now = cells[0] * scale
            prior = cells[min(ncol - 1, len(cells) - 1)] * scale
            if now <= 1e4:
                continue
            if revenue and not (revenue / 1000.0 < now < revenue * 3):
                continue
            return now, (prior if prior > 1e4 else None)
    # No statement row: a sentence that states the quarter's figure outright is
    # still the figure - with no year-ago beside it, so no growth is claimed.
    return (prose, None) if prose else (None, None)


# ═══════════════════════════════════════════════════════════════════════════
# GUIDANCE
# ═══════════════════════════════════════════════════════════════════════════
_ANNUAL = re.compile(r"full[-\s]?year|fiscal|\bFY\b|year\s+end", re.I)
_OUTLOOK_HEAD = re.compile(
    r"(?:^|[\s•])((?:first|second|third|fourth|full[-\s]year|fiscal|next)[^.\n]{0,70}?"
    r"(?:outlook|guidance)|guidance|business\s+outlook|financial\s+outlook|"
    r"(?:2\d{3})\s+outlook)\b", re.I)
_LEGAL = re.compile(r"forward[-\s]looking\s+statements|safe\s+harbor|about\s+(?:the\s+)?compan|"
                    r"non-?GAAP\s+financial\s+measures", re.I)


# The outlook search must NOT stop at a "Non-GAAP Financial Measures" heading.
# UiPath prints that section BEFORE its Financial Outlook, so treating it as the
# end of the release threw away the guidance entirely and the panel claimed the
# guide was "given on the call" when it was three paragraphs further down.
_LEGAL_OUTLOOK = re.compile(r"forward[-\s]looking\s+statements|safe\s+harbor", re.I)


def _outlook_spans(text):
    """Every candidate outlook section, in document order.

    Guessing WHICH heading is the right one never held up: UiPath's is a bare
    "Financial Outlook", Palo Alto's names the quarter, Salesforce prints the
    full year first and the quarter second. So this hands back all of them and
    lets the caller keep the first that actually contains a quarterly figure.
    """
    stop = _LEGAL_OUTLOOK.search(text)
    body = text[:stop.start()] if stop else text
    # A heading that is just "Outlook" - Rent the Runway's "Outlook For the fiscal
    # third quarter of 2026 ... Revenue of between $87 million and $90 million" -
    # matched nothing above, so the panel said the guide was "given on the call".
    allm = sorted(list(_OUTLOOK_HEAD.finditer(body)) + list(_OUTLOOK_BARE.finditer(body)),
                  key=lambda m: m.start())
    # Skip only the TITLE region, where "Raises Guidance" is a headline and not a
    # section. Skipping the first fifth of the release threw away Progress
    # Software's "2026 Business Outlook" - 12% of the way in, ahead of long
    # reconciliation tables - and the panel said there was no guide at all.
    hits = [m for m in allm if m.start() > 800]
    if not hits:
        hits = allm
    for m in hits:
        yield body[m.start():m.start() + 4500], m.start()


_OUTLOOK_BARE = re.compile(r"(?:^|[\s•])(?:Outlook|OUTLOOK)\s*:?\s+(?=[A-Z])")


def _outlook_span(text):
    """The first outlook section, for callers that only need a position."""
    for span, at in _outlook_spans(text):
        return span, at
    return "", 0


def _scale_hint(text, at):
    seg = text[max(0, at - 2600):at + 600]
    sc = re.findall(r"in\s+(millions|thousands|billions)", seg, re.I)
    return _SCALE[sc[-1].lower()] if sc else None


_GUIDE_RX = (
    _REV + _SENT + r"{0,95}?\$\s?([\d,.]+)\s*(billion|million)?\s*(?:to|-|and)\s*\$?\s?([\d,.]+)\s*(billion|million)?",
    _REV + _SENT + r"{0,70}?expected\s+to\s+be\s+(?:approximately\s+|about\s+)?\$\s?([\d,.]+)\s*(billion|million)?()()",
    r"(?:^|[\s•])revenues?\s*(?:\[\d\])?\s*\$\s?([\d,.]+)()\s*(?:to|-)\s*\$?\s?([\d,.]+)()",
    # A POINT guide. Gloo: "expects third-quarter revenue to be $55 million";
    # Kestra: "increasing its FY27 revenue guidance to $141 million".
    _REV + r"\s+(?:guidance\s+)?(?:to\s+be|to|of|at)\s+(?:approximately\s+|about\s+|roughly\s+)?"
           r"\$\s?([\d,.]+)\s*(billion|million)()()",
)
# A CENTRE and a spread. Micron: "Revenue $61.5 billion ± $1.5 billion" - the
# panel said "no revenue guide in the release" beside the one number its whole
# post-print move turns on.
_PLUSMINUS_RX = re.compile(_REV + r"\s*(?:\(\d\)\s*)?:?\s*\$\s?([\d,.]+)\s*(billion|million)?\s*"
                           r"(?:±|\+/-|\+/−|plus\s+or\s+minus)\s*\$?\s?([\d,.]+)\s*(billion|million|%)?",
                           re.I)
# InnovAge prints its outlook as a Low / High table: "Total revenues $ 1,050 $ 1,085".
_LOWHIGH_RX = re.compile(r"(?:^|[\s•])(?:total\s+)?(?:net\s+)?revenues?\s*(?:\(\d\))?\s*"
                         r"\$\s?([\d,.]+)\s+\$\s?([\d,.]+)", re.I)
# Another metric named between the revenue word and the figure means the figure
# is that metric's. Sunbelt's outlook table reads "Total Revenue 4.5% to 7.5%
# growth 6% to 9% growth Rental Revenue ... Adjusted EBITDA $4.85 billion to $5.05
# billion" - and the EBITDA range went on the panel as a revenue guide.
_GAP_METRIC = re.compile(r"EBITDA|\bincome\b|margin|earnings|\bEPS\b|cash\s+flow|expenditure|"
                         r"\bcapex\b|profit|\bARR\b|bookings|per\s+share|\bshares\b", re.I)


_LINE_UNIT = re.compile(r"\(\s*\$?\s*in\s+(billions|millions|thousands)\s*\)", re.I)
_PERIOD_RX = re.compile(r"(?:first|second|third|fourth)[-\s]+quarter|\bQ[1-4]\b|"
                        r"three\s+months\s+ended\s+\w+\.?\s+\d+|full[-\s]year|"
                        r"six[-\s]month|transition\s+period|(?:first|second)[-\s]half|"
                        r"fiscal\s+(?:year\s+)?(?:ending\s+\w+\s+\d+,\s+)?\d{4}", re.I)


def _period_near(span, at, fallback=None):
    """The period a guided figure belongs to: the NEAREST period words before it.

    Nearest, not the section heading, because one section often guides two
    periods. IBEX: "For fiscal year 2027, revenue ... $700 to $715 million. For
    first quarter fiscal year 2027, revenue ... $168 to $170 million" - reading
    the heading made the quarter's guide a full-year one. And when the nearest
    words are "third quarter of fiscal 2027", the QUARTER names the period; the
    year only dates it.
    """
    back = span[max(0, at - 260):at]
    last = None
    for last in _PERIOD_RX.finditer(back):
        pass
    if not last:
        return fallback
    txt = re.sub(r"\s+", " ", last.group(0)).strip()
    if re.match(r"fiscal", txt, re.I):
        q = None
        for q in re.finditer(r"(?:first|second|third|fourth)[-\s]+quarter|\bQ[1-4]\b",
                             back[max(0, last.start() - 40):last.start()], re.I):
            pass
        if q:
            return re.sub(r"\s+", " ", q.group(0)).strip()
    return txt


def _is_annual_period(period):
    """A period naming a QUARTER is quarterly even when it also says "fiscal":
    Cooper's "Fiscal fourth quarter 2026" is a Q4 guide, not a full-year one."""
    if not period:
        return False
    if re.search(r"quarter|\bQ[1-4]\b|three\s+months", period, re.I):
        return False
    return _ANNUAL.search(period) is not None


def _line_before(span, at, reach=170):
    """The text of the figure's own bullet or sentence, back to its start."""
    seg = span[max(0, at - reach):at]
    cut = max(seg.rfind(c) for c in _BULLET + ".")
    return seg[cut + 1:] if cut >= 0 else seg


def read_guidance(text, revenue=None):
    """Next-period revenue guidance as {'lo','hi','mid','period'}, or None.

    CoreWeave says outright that it guides on the call and prints no figure. None
    is the right answer there, not a failure.

    With `revenue`, a candidate that cannot be a revenue guide (see _sane_guide)
    is passed over HERE, so the search goes on to the real one instead of the
    whole guide being thrown away afterwards.
    """
    found = []
    for span, at in _outlook_spans(text):
        scale = _scale_hint(text, at) or 1.0
        # The section heading's period is only the FALLBACK. Cooper's heading
        # says "Fiscal Year 2026" while the figures sit in a "Fiscal fourth
        # quarter 2026" bullet; the bullet is what the figures belong to.
        span_period = None
        p = _PERIOD_RX.search(span[:500])
        if p:
            span_period = re.sub(r"\s+", " ", p.group(0)).strip()
        hits = []
        for ri, rx in enumerate(_GUIDE_RX):
            for m in re.finditer(rx, span, re.I):
                if _NOT_REV.search(span[max(0, m.start() - 70):m.end()]):
                    continue
                if _GAP_METRIC.search(span[m.start():m.start(1)]):
                    continue
                if _is_year_ago(span, m) or re.search(r"prior[-\s]year|year[-\s]ago|last\s+year",
                                                      span[max(0, m.start() - 40):m.start()], re.I):
                    continue
                # A RESULT verb means a result. FactSet's "GAAP revenues increased
                # 6.3% or $37.8 million to $634.7 million" sits under a results
                # table whose last column is headed "Guidance", and read as a
                # guide it hid the real FY27 range further down.
                if re.search(r"\b(?:increased|decreased|grew|rose|fell|declined|was|were|totaled|"
                             r"reached|came\s+in)\b", span[m.start():m.start(1)], re.I):
                    continue
                # Two or more figures between the label and the range make it the
                # LAST COLUMN of a results row: FactSet's "Revenues $634,698
                # $596,901 6.3% $2,476,256 $2,321,748 6.7% $2,450 - $2,470M" is the
                # year just ENDED and the guide it had for it. A guide row starts
                # with the guide.
                if len(re.findall(r"\d[\d,]*(?:\.\d+)?", span[m.start():m.start(1)])) >= 2:
                    continue
                # A single figure is only a guide when its own sentence looks
                # forward. Culp's "prior-year period sales of $22.7 million" is a
                # result, and so is Lennar's "produced total revenues of $8.0
                # billion" - in a sentence that also says "within our guidance".
                # So the forward word must be a verb of expectation, or "guidance"
                # must sit inside the phrase itself ("revenue guidance to $141M").
                if ri == len(_GUIDE_RX) - 1:
                    sent = _line_before(span, m.start(), 200) + span[m.start():m.start(1)]
                    if not (re.search(r"expect|anticipat|project|forecast|target|outlook", sent, re.I)
                            or re.search(r"guidance", m.group(0), re.I)):
                        continue
                # One unit, written after the range, covers both ends: Cooper's
                # "$1.057 - $1.080 billion" read the first figure in millions and
                # produced a guide a thousand times too small.
                u_lo = m.group(2) or (m.group(4) if m.group(3) else None)
                a, ua, b, ub = m.group(1), u_lo, m.group(3), m.group(4)
                # "raising our annual revenue expectations from a range of $12.10
                # billion to $12.25 billion to a range of $12.15 billion to $12.27
                # billion" - Cintas. The first range is the one being REPLACED.
                nx = re.match(r"\s*(?:billion|million)?\s*,?\s*to\s+(?:a\s+range\s+of\s+|between\s+)?"
                              r"\$\s?([\d,.]+)\s*(billion|million)?\s*(?:to|-|and)\s*\$?\s?([\d,.]+)"
                              r"\s*(billion|million)?", span[m.end():m.end() + 90], re.I)
                if nx and re.search(r"from\s+(?:a\s+range\s+of\s+|between\s+)?\$?\s?$",
                                    span[max(0, m.start(1) - 30):m.start(1)], re.I):
                    a, b = nx.group(1), nx.group(3)
                    ua, ub = nx.group(2) or nx.group(4), nx.group(4)
                hits.append((m.start(), a, ua, b, ub, m))
        if re.search(r"\bLow\b\s+\bHigh\b", span, re.I):
            for m in _LOWHIGH_RX.finditer(span):
                if not re.search(r"\bLow\b\s+\bHigh\b", span[max(0, m.start() - 400):m.start()], re.I):
                    continue
                hits.append((m.start(), m.group(1), None, m.group(2), None, m))
        for m in _PLUSMINUS_RX.finditer(span):
            c, cu, w, wu = _num(m.group(1)), m.group(2), _num(m.group(3)), m.group(4)
            if c is None or w is None:
                continue
            if wu == "%":
                lo, hi = c * (1 - w / 100.0), c * (1 + w / 100.0)
            elif wu and cu and wu.lower() != cu.lower():
                continue                    # "$61.5 billion ± $1.5 million" is not a range
            else:
                lo, hi = c - w, c + w
            if lo <= 0:
                continue
            fmt = lambda x: ("%.6f" % x).rstrip("0").rstrip(".")
            hits.append((m.start(), fmt(lo), cu, fmt(hi), cu, m))
        hits.sort(key=lambda h: h[0])
        span_hits = []
        for pos, a, ua, b, ub, m in hits:
            own = _line_before(span, pos) + span[pos:m.end()]
            # A unit written INTO the row label beats the table caption: UNFI's
            # "Net sales ($ in billions) $31.2 - $31.8" sits under an "(in
            # millions)" caption and printed as $31M on a $31bn company.
            lu = _LINE_UNIT.search(own)
            row_scale = _SCALE[lu.group(1).lower()] if lu else scale
            lo = money(a, ua, row_scale)
            hi = money(b, ub, row_scale) if b else lo
            period = _period_near(span, pos, span_period)
            if lo is None or lo < 1e5:
                continue
            if hi is None or hi < lo:
                hi = lo
            if hi / lo > 3:                 # two unrelated figures, not a range
                hi = lo
            h = _sane_guide({"lo": lo, "hi": hi, "mid": (lo + hi) / 2.0, "period": period}, revenue)
            if h:
                span_hits.append(h)
        # A quarterly guide is the one worth comparing to consensus, so any in
        # this section wins; the section's first annual figure is the fallback.
        for h in span_hits:
            # The size of the figure outranks the words beside it: Progress's
            # "$1,044 - $1,052" full-year row sits under an intro that names "the
            # fiscal fourth quarter", and taking it as the quarter's guide hid the
            # real Q4 row ("$297 - $305") two lines down.
            if revenue and h["mid"] > 2.5 * revenue:
                continue
            if h["period"] and not _is_annual_period(h["period"]) \
                    and not _SIX_MONTH.search(h["period"]):
                return h
        if span_hits:
            found.append(span_hits[0])
    return found[0] if found else None


_SIX_MONTH = re.compile(r"six[-\s]month|transition\s+period|(?:first|second)[-\s]half", re.I)


def read_guidance_growth(text):
    """{'lo','hi','period'} in PERCENT, for a revenue guide stated only as growth.

    Oracle: "Total revenues are expected to grow between 30% and 34%". Sunbelt's
    table: "Total Revenue 4.5% to 7.5% growth 6% to 9% growth" under "Prior
    Outlook  Current Outlook" - the second pair is the one in force. Saying
    "not in the release" beside either of those would be false.
    """
    for span, _at in _outlook_spans(text):
        m = re.search(_REV + r"\s+(?:are|is)\s+(?:now\s+)?expected\s+to\s+(?:grow|increase)\s+"
                      r"(?:between\s+|by\s+)?([\d.]+)\s*%\s*(?:and|to|-)\s*([\d.]+)\s*%", span, re.I)
        if m:
            return {"lo": _num(m.group(1)), "hi": _num(m.group(2)),
                    "period": _period_near(span, m.start(), None)}
        m = re.search(r"total\s+revenues?\s+([\d.]+)\s*%\s*to\s*([\d.]+)\s*%\s+growth"
                      r"(?:\s+([\d.]+)\s*%\s*to\s*([\d.]+)\s*%\s+growth)?", span, re.I)
        if m:
            lo, hi = (m.group(3), m.group(4)) if m.group(3) else (m.group(1), m.group(2))
            return {"lo": _num(lo), "hi": _num(hi), "period": _period_near(span, m.start(), None)}
    return None


# High-precision only. A wrong "RAISED" beside a cut guide is the single worst
# thing this tool could put on the chart, so anything softer gets no flag at all.
_RAISE = re.compile(
    r"(?:rais(?:ing|ed|es)|increas(?:ing|ed|es)|lift(?:ing|ed|s)|boost(?:ing|ed|s))\s+"
    r"(?:its\s+|our\s+|the\s+|full[-\s]year\s+|fiscal[^.]{0,20}?\s+|financial\s+|revenue\s+|"
    r"annual\s+|profit\s+|\w+\s+){0,4}?(?:outlook|guidance|forecast)"
    r"|(?:higher|raised|increased|improved)\s+(?:full[-\s]year\s+|fiscal[^.]{0,20}?\s+)?"
    r"(?:outlook|guidance)"
    r"|(?:outlook|guidance)\s+(?:has\s+been\s+|was\s+)?(?:rais|increas)ed", re.I)
_CUT = re.compile(
    r"(?:lower(?:ing|ed|s)|reduc(?:ing|ed|es)|cut(?:ting|s)?|trimm(?:ing|ed))\s+"
    r"(?:its\s+|our\s+|the\s+|full[-\s]year\s+|fiscal[^.]{0,20}?\s+|financial\s+|revenue\s+|"
    r"annual\s+|\w+\s+){0,4}?(?:outlook|guidance|forecast)"
    r"|(?:outlook|guidance)\s+(?:has\s+been\s+|was\s+)?(?:lower|reduc|cut)ed", re.I)


def _sane_guide(g, revenue):
    """Reject a revenue guide that cannot be one.

    Any revenue guide - next quarter or full year - is at least a third of the
    quarter just reported. Below that the scale is wrong: UNFI's $31.2bn annual
    outlook came through as $31.2M. Try x1000 once; if that lands in a believable
    range keep it, otherwise show no guide rather than a wrong one.
    """
    if not g or not revenue:
        return g
    lo = g["lo"]
    if lo >= revenue / 3.0:
        return g
    if revenue / 3.0 <= lo * 1000 <= revenue * 7:
        return dict(g, lo=g["lo"] * 1000, hi=g["hi"] * 1000, mid=g["mid"] * 1000)
    return None


def read_guide_direction(text):
    """'raised' / 'lowered' / None, from the company's own words only.

    NOT a substitute for comparing the guide to the street. lululemon called a cut
    "our revised full-year outlook" in a confident sentence and this returns None
    for it, which is correct: silence beats a wrong flag.
    """
    stop = _LEGAL.search(text)
    body = text[:stop.start()] if stop else text[:22000]

    def first(rx):
        # A raised SYNERGY target or VOLUME guide is not the company's financial
        # outlook. Mission Produce "raises annualized Calavo synergy outlook"
        # while REAFFIRMING its guidance; Limoneira raised its avocado VOLUME
        # guide. Both went on the panel as "^ RAISED".
        for m in rx.finditer(body):
            if re.search(r"synerg|volume|production|shipment|units?\b|stores?\b|dividend|"
                         r"repurchase|buyback|capital\s+expenditure|capex|tax\s+rate|"
                         r"interest\s+expense|headcount", m.group(0), re.I):
                continue
            return m
        return None

    r, c = first(_RAISE), first(_CUT)
    if r and not c:
        return "raised"
    if c and not r:
        return "lowered"
    if r and c:
        return "raised" if r.start() < c.start() else "lowered"
    return None


# ═══════════════════════════════════════════════════════════════════════════
_YEAR_CAPTION = re.compile(r"(?:fiscal\s+)?years?\s+ended|(?:twelve|12)\s+months\s+ended|"
                           r"(?:fifty[-\s]?two|fifty[-\s]?three|52|53)\s+weeks\s+ended", re.I)


def parse(text):
    """Every figure the panel shows, read from one press release.

    A fiscal-YEAR-only release is read as what it is: its year columns are the
    statement, its year figures are the results, and nothing on it is compared
    with a quarterly consensus (see is_annual_only).
    """
    global _ANNUAL_MODE
    t0 = clean(text)
    annual = is_annual_only(t0)
    q, explicit = read_quarter_ex(t0)
    if annual:
        q, explicit = read_fiscal_year(t0) or q, True
        # The year's captions are the statement's only captions, so they are
        # read the way a quarter's would be.
        t = _YEAR_CAPTION.sub("Three Months Ended", t0)
    else:
        t = t0
    _ANNUAL_MODE = annual
    try:
        p = _parse_body(t)
    finally:
        _ANNUAL_MODE = False
    p["quarter"] = q
    p["quarter_explicit"] = explicit
    p["annual_only"] = annual
    p["guide_growth"] = None if p.get("guide") else read_guidance_growth(t)
    return p


def _parse_body(t):
    eps, basis = read_eps(t)
    rev, prior = read_revenue_table(t)
    if rev is None:
        rev = read_revenue_prose(t)
        prior = _table_year_ago(t, rev)
    else:
        # Cross-check the statement against the prose. Delta reports GAAP revenue
        # of $19.76bn including refinery sales AND a $17.67bn non-GAAP figure its
        # own prose calls "total revenue"; a mismatch that large means one of the
        # two readers grabbed a segment or a per-unit number, so the tiny one is
        # dropped rather than shown.
        pr = read_revenue_prose(t)
        if pr and rev and (pr / rev < 0.15 or rev / pr < 0.15):
            rev = max(pr, rev)
            prior = _table_year_ago(t, rev)
    # Arithmetic on two columns of the same statement beats a claim in the prose:
    # Casey's says "inside same-store sales increased 5.5%" one line from its
    # revenue, and that 5.5% is not revenue growth.
    # Computed FIRST, from the same two columns the revenue row came from, so the
    # growth rate always describes the revenue printed beside it.
    growth = None
    if rev and prior:
        g = 100.0 * (rev - prior) / prior
        # A misidentified prior column shows up as a wild rate, not a subtle one:
        # Oracle came back at +853% and Dave & Buster's at +459% when the year-ago
        # cell was actually a small sub-total. Real quarters live well inside this
        # band - CoreWeave's +112% is the fastest in the whole test set - so
        # anything outside it is a column error and the prose is trusted instead.
        if -80.0 <= g <= 300.0:
            growth = g
    if growth is None:
        growth = read_growth_prose(t, rev)
        prior = None if growth is None else prior
    mv, mkind, mbasis, mprior = read_margin(t, rev)
    capex, capex_prior = read_capex(t, rev)
    if _ANNUAL_MODE:
        # The panel's capex row says "this quarter"; a year's spend is not that.
        capex = capex_prior = None
    return {"eps": eps, "eps_basis": basis, "revenue": rev, "growth": growth,
            "rev_prior": prior, "guide": _sane_guide(read_guidance(t, rev), rev),
            "guide_dir": read_guide_direction(t),
            "margin": mv, "margin_kind": mkind, "margin_basis": mbasis,
            "margin_prior": mprior, "capex": capex, "capex_prior": capex_prior}


_ORDW = r"(first|second|third|fourth)[\s-]+quarter"
# THE FISCAL YEAR CAN COME FIRST. "AeroVironment Announces Fiscal 2027 First
# Quarter Results" puts the year in front of the quarter, so a pattern that only
# looks forwards skipped it and bound to the next four-digit number it could
# find - the DATELINE, "September 9, 2026". The panel then read "Q1 FY26" for a
# quarter the release itself calls fiscal 2027, one row above a guidance line
# correctly labelled FY27. General Mills hyphenates it ("Fiscal 2026
# Fourth-quarter"), which is why the ordinal and "quarter" are joined by [\s-]+.
_QTR_YEAR_FIRST = re.compile(
    r"fiscal\s+(?:year\s+)?((?:19|20)\d{2})[\s-]+(?:%s|Q([1-4]))" % _ORDW, re.I)
# An explicit fiscal marker AFTER the quarter is the next most trustworthy form.
_QTR_FISCAL = re.compile(
    r"\b(?:%s|Q([1-4]))\b[^.]{0,40}?(?:fiscal\s+(?:year\s+)?|FY\s*)((?:19|20)\d{2})" % _ORDW,
    re.I)
# Last resort: any nearby year, because calendar-year filers never write "fiscal".
_QTR_LOOSE = re.compile(
    r"\b(?:%s|Q([1-4]))\b[^.]{0,40}?((?:19|20)\d{2})" % _ORDW, re.I)
# ...but a year that is the tail of a date is a PUBLICATION date, not a fiscal
# year. That is the trap the loose pattern fell into on AeroVironment.
_DATELINE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*$", re.I)
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def read_quarter(text):
    """'Q2 FY2027' as the release labels it, for the panel headline."""
    return read_quarter_ex(text)[0]


def read_quarter_ex(text):
    """(label, explicit). EXPLICIT means the release NAMED the fiscal year -
    "Second Quarter Fiscal 2026", "Fiscal 2027 First Quarter", "Q2 FY26".

    The distinction decides whether the Finnhub calendar may overrule it. The
    calendar numbers a January-year-end company's year by the calendar year it
    ENDS in, while half of those companies name it by the year it STARTS in:
    American Eagle, GameStop, Chewy, RH, lululemon and Dave & Buster's all call
    the quarter to August 2026 "fiscal 2026", and the calendar calls it FY27.
    Letting the calendar win everywhere put Q2 FY27 on six of those panels. When
    the company says the year itself, that is the answer; the calendar is only
    for releases that never say it (ServiceTitan: "fiscal second quarter ended
    July 31, 2026").

    Taken from the TITLE region only. Deeper in, every release compares against
    prior quarters and a match there names the wrong one.
    """
    head = clean(text)[:600]
    best = None

    for m in _QTR_YEAR_FIRST.finditer(head):
        q = _ORD.get((m.group(2) or "").lower()) or _num(m.group(3))
        if q:
            best = (m.start(), 0, int(q), m.group(1)[-2:])
            break

    # EARLIEST WINS, not most-specific. A release states its own quarter in the
    # headline and only then compares against the year-ago one, so position is
    # the signal. Preferring an explicit "fiscal" marker instead read AutoZone's
    # comparison clause - "an increase of 8.4% from the third quarter of fiscal
    # 2025" - and labelled the card Q3 FY25, a year stale.
    for rx, prio in ((_QTR_FISCAL, 1), (_QTR_LOOSE, 2)):
        for m in rx.finditer(head):
            before = head[max(0, m.start(3) - 40):m.start(3)]
            # A date that a quarter ENDED on names the fiscal year; a publication
            # dateline does not. AutoZone's "quarter (12 weeks) ended May 9, 2026"
            # is the former, AeroVironment's "ARLINGTON, VA, September 9, 2026"
            # the latter - and binding to the latter is what produced "Q1 FY26"
            # on a fiscal-2027 quarter.
            if _DATELINE.search(before) and not re.search(r"end(?:ed|ing)\b", before, re.I):
                continue
            q = _ORD.get((m.group(1) or "").lower()) or _num(m.group(2))
            if q:
                cand = (m.start(), prio, int(q), m.group(3)[-2:])
                if best is None or cand[:2] < best[:2]:
                    best = cand
                break

    if not best:
        return None, False
    explicit = best[1] < 2
    if not explicit:
        # A year written straight onto the quarter - "Second Quarter 2026",
        # "Fiscal Third Quarter 2026", "Q2 2026" - is the company NAMING the
        # period, just as "fiscal 2026" would be. GameStop, Chewy, RH and Dave &
        # Buster's all headline that way. Only a year that belongs to a DATE
        # ("quarter ended July 31, 2026") leaves the fiscal year unsaid.
        pos = best[0]
        explicit = bool(re.match(r"(?:(?:fiscal\s+)?(?:first|second|third|fourth)[\s-]+quarter|Q[1-4])"
                                 r"(?:\s+of)?(?:\s+fiscal)?(?:\s+year)?\s+'?(?:19|20)?\d{2}\b",
                                 head[pos:pos + 60], re.I))
    return "Q%d FY%s" % (best[2], best[3]), explicit


_URL_QY = (re.compile(r"q([1-4])[-_]?(?:fy)?((?:20)?[23]\d)(?!\d)", re.I),
           re.compile(r"f(?:y)?((?:20)?[23]\d)[-_]?q([1-4])(?!\d)", re.I))


def label_from_url(url):
    """'Q2 FY27' from the exhibit's own file name, or None.

    Filers name the document after their fiscal calendar: Marvell's is
    "q227_8kx812026ex-991.htm", GameStop's "a991-fy26q2earningsrelease.htm",
    Casey's "q1fy2027earningspressrelea.htm", UNFI's "f26q4earningsrelease.htm".
    That is the company's own name for the quarter, which is what the panel
    headline should say when the release text itself never names the year.
    """
    if not url:
        return None
    name = url.rsplit("/", 1)[-1].lower()
    m = _URL_QY[0].search(name)
    if m:
        return "Q%s FY%s" % (m.group(1), m.group(2)[-2:])
    m = _URL_QY[1].search(name)
    if m:
        return "Q%s FY%s" % (m.group(2), m.group(1)[-2:])
    return None


def final_quarter(p, url=None, calendar=None):
    """The headline label, by precedence: the release's own words, then the
    exhibit's file name, then the earnings calendar, then a date-derived guess.

    The calendar comes LAST among the sources that know the company's fiscal
    naming because it applies one convention to every company; see
    read_quarter_ex for the six retailers that convention mislabelled.
    """
    if p.get("annual_only") or (p.get("quarter") and p.get("quarter_explicit")):
        return p.get("quarter")
    own = re.match(r"Q([1-4])", p.get("quarter") or "")
    u = label_from_url(url)
    if u and (not own or u[1] == own.group(1)):
        return u
    if calendar and (not own or calendar[1:2] == own.group(1)):
        return calendar
    return p.get("quarter") or calendar


# ── FISCAL-YEAR-ONLY RELEASES ────────────────────────────────────────────────
# Some Q4 filers publish ONLY the year. Alliance Entertainment's release is
# "Reports Fiscal 2026 Revenue Up 8% to $1.15 Billion" with every statement
# column a year; American Battery's is "Reports FY 26 Financial Results"; Upexi's
# "Financial Results for Fiscal Year Ended June 30, 2026". Read as a quarter,
# all three compared twelve months of revenue with ONE quarter's consensus -
# Alliance "beat" by 380% - and Upexi's $3.87 annual LOSS per share went up as a
# $3.87 beat.
_FY_RESULTS = re.compile(
    r"fiscal\s+(?:year\s+)?(?:20\d\d|\d\d)\b[^.]{0,60}?(?:results|revenue|sales)|"
    r"\bFY\s?'?\d\d\b[^.]{0,30}?results|"
    r"(?:results|revenue)\s+for\s+(?:the\s+)?(?:fiscal\s+)?year\s+ended|"
    r"full[-\s]year\s+(?:20\d\d\s+)?(?:financial\s+)?results", re.I)
_QTR_WORD = re.compile(r"(?:first|second|third|fourth)[\s-]+(?:fiscal\s+)?quarter|\bQ[1-4]\b|"
                       r"three[\s-]+months|thirteen[\s-]+weeks|quarterly\s+(?:results|revenue)|"
                       r"\bquarter\s+ended", re.I)
_QTR_HDR_ANY = re.compile(r"(?:three|thirteen|fourteen|twelve|3|12|13|14|16)\s*(?:-|\s)?weeks?\s+ended|"
                          r"(?:three|3)\s*(?:-|\s)?months?\s+ended|quarter\s+ended", re.I)


def is_annual_only(text):
    """True when the release reports a fiscal YEAR and no quarter at all."""
    t = clean(text)
    if not _FY_RESULTS.search(t[:700]):
        return False
    if _QTR_WORD.search(t[:3500]):
        return False
    return not _QTR_HDR_ANY.search(t)


def read_fiscal_year(text):
    """'FY26' for a fiscal-year-only release, from its own words."""
    head = clean(text)[:900]
    m = re.search(r"fiscal\s+(?:year\s+)?((?:19|20)\d{2})\b|\bFY\s?'?((?:19|20)?\d{2})\b|"
                  r"fiscal\s+year\s+ended\s+\w+\.?\s+\d{1,2},?\s+((?:19|20)\d{2})", head, re.I)
    if not m:
        return None
    y = next(g for g in m.groups() if g)
    return "FY" + y[-2:]


# ═══════════════════════════════════════════════════════════════════════════
# IS THIS EVEN AN EARNINGS RELEASE?
# ═══════════════════════════════════════════════════════════════════════════
# Item 2.02 is "Results of Operations and Financial Condition", and companies
# file things under it that are NOT a quarterly result. FedEx's newest such
# filing is titled "SUPPLEMENTAL HISTORICAL FINANCIAL INFORMATION" - a recast of
# prior periods after it moved its fiscal year end from May to December. It
# carries quarter-labelled EPS tables for RECAST CALENDAR quarters, and it was
# accepted at 20:09:35 UTC, squarely inside the after-market print window. Every
# surface signal says "earnings release". None of the numbers are current.
_NOT_RESULTS = re.compile(
    r"supplemental\s+historical\s+financial|recast(?:ing)?\s+(?:of\s+)?(?:prior|historical)|"
    r"changed\s+its\s+fiscal\s+year\s+end|transition\s+report\s+on\s+form|"
    r"unaudited\s+pro\s+forma\s+condensed", re.I)


def looks_like_results(text, parsed=None):
    """(ok, reason). False when the filing is not this quarter's results.

    Two independent tests, because either alone lets something through:
      1. Explicit recast / fiscal-year-change language near the top.
      2. Nothing extractable. A card with neither EPS nor revenue tells the owner
         nothing anyway, so refusing to draw it costs him nothing and stops a
         non-result rendering as one.
    """
    t = clean(text or "")
    m = _NOT_RESULTS.search(t[:4000])
    if m:
        return False, "not a results release: %r" % t[m.start():m.start() + 60].strip()
    p = parsed if parsed is not None else parse(t)
    if p.get("eps") is None and p.get("revenue") is None:
        return False, "no EPS and no revenue could be read"
    return True, ""


# ═══════════════════════════════════════════════════════════════════════════
# THE PANEL
# ═══════════════════════════════════════════════════════════════════════════
def _usd(v):
    """-$0.24, not $-0.24."""
    return ("-$%.2f" % -v) if v is not None and v < 0 else "$%.2f" % (v or 0)


def _vs(actual, est, tol_pct=0.25, tol_abs=None):
    """Verdict against consensus, with a band inside which nothing was beaten.

    Marvell came in at $0.9400 against a $0.9428 consensus and the first version
    of this called it a MISS of 0.3%. Nobody trades that, and printing it as a
    miss beside a genuine revenue miss makes the panel harder to read, not easier.
    Half a cent on EPS and a quarter of a percent on revenue are the bands.
    """
    if actual is None or est in (None, 0):
        return None, None
    d = actual - est
    pct = 100.0 * d / abs(est)
    if (tol_abs is not None and abs(d) < tol_abs) or abs(pct) < tol_pct:
        return "IN LINE", pct
    return ("BEAT" if d > 0 else "MISS"), pct




PANEL_ROWS = 6          # in_131..in_136; the Pine renders table rows 10..15

_ORDQ = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4"}


def _period_label(period, annual):
    """A short, honest name for the period being guided.

    The raw phrase is whatever the filer wrote, and some of them are long enough
    to push the numbers off the panel - Docusign's is "Three Months Ended October
    31". The label still has to say WHICH period, without eating the row.
    """
    if not period:
        return "next"
    t = period.strip()
    if annual:
        m = re.search(r"((?:19|20)\d{2})", t)
        return "FY" + m.group(1)[-2:] if m else "FY"
    low = t.lower()
    for word, q in _ORDQ.items():
        if low.startswith(word):
            return q
    m = re.match(r"q([1-4])", low)
    if m:
        return "Q" + m.group(1)
    m = re.search(r"ended\s+(\w+)", t, re.I)
    if m:
        return "to " + m.group(1)[:3].title()
    return t[:10]


_NEXT_Q = {"Q1": "Q2", "Q2": "Q3", "Q3": "Q4", "Q4": "Q1"}


def guide_period_class(p):
    """'Q1'..'Q4', 'FY', '6M', 'next' or None - the period the panel names the guide by.

    THE ARITHMETIC OUTRANKS THE WORDS. A full-year revenue guide is several
    quarters of revenue. Best Buy's unlabelled $42.3bn guide beside a $9.8bn
    quarter is annual whatever it is called; HPE's "$13.9 to $14.8 billion" beside
    a $12.2bn quarter is a QUARTER, even though it sits under a "Fiscal 2026"
    heading - and calling it annual meant it was never compared with the street.
    SailPoint's "$326 to $330" (Q3, in a "Q3'27 | FY'27" table) was mislabelled
    the same way.
    """
    g = p.get("guide")
    if not g:
        return None
    per = g.get("period") or ""
    if _SIX_MONTH.search(per):
        return "6M"
    if p.get("annual_only"):
        return "FY"
    annual = _is_annual_period(per)
    rev = p.get("revenue")
    if rev:
        if g["mid"] > 2.5 * rev:
            annual = True
        elif annual and g["mid"] < 2.0 * rev:
            annual = False
    if annual:
        return "FY"
    low = per.lower().strip()
    for word, q in _ORDQ.items():
        if low.startswith(word):
            return q
    m = re.search(r"\bq([1-4])\b", low)
    if m:
        return "Q" + m.group(1)
    # "Three Months Ended October 31", or no period words at all: the quarter
    # after the one just reported.
    cur = re.match(r"Q([1-4])", p.get("quarter") or "")
    return _NEXT_Q["Q" + cur.group(1)] if cur else "next"


def rows(p, est_now=None, est_next=None, ticker="", when=""):
    """The headline plus the rows the chart panel shows.

    est_now  = (epsEstimate, revenueEstimate) for the quarter just reported
    est_next = (epsEstimate, revenueEstimate) for the quarter being guided

    Ordered by what decides a trade, and TRUNCATED from the bottom, so when a release is
    thin the rows that survive are the ones that matter most: the beat, the growth, the
    guide, the year, then margin and capital spending.
    """
    en = est_now or (None, None)
    nx = est_next or (None, None)
    out = []
    # A fiscal-year-only release has nothing a QUARTERLY consensus can be held
    # against. Its figures are shown as the year's, with no verdict.
    if p.get("annual_only"):
        en, nx = (None, None), (None, None)
    yr = "  full year" if p.get("annual_only") else ""

    if p.get("eps") is not None:
        v, pct = _vs(p["eps"], en[0], tol_pct=0.6, tol_abs=0.005)
        basis = p.get("eps_basis") or ""
        lab = ("EPS " + basis).strip()
        # Consensus is the ADJUSTED number. When the release only gave a GAAP or
        # unlabelled figure, the two sides are not the same measure, and a big
        # apparent beat can be one-off items - lululemon's $2.92 carried $0.86 of
        # tariff refunds against a $1.83 adjusted consensus. Say so on the row.
        note = "" if basis == "non-GAAP" else "  [vs adj est]"
        # A PERCENTAGE OF A FEW CENTS IS NOISE. Inspirato lost $0.24 against a $0.02 estimate and
        # the card said "MISS -1668.6%"; Upexi said "BEAT +3261.8%". Both numbers were
        # arithmetically true and told the owner nothing. When the estimate is under ten cents either
        # way, the gap is shown in dollars and cents instead - "$0.26 below" is something you can
        # read. The same holds across a SIGN CHANGE - beating a -$0.12 estimate with +$3.87 is
        # not "+3,325%" of anything - and for any gap over 200%.
        e = en[0]
        noisy = v and e is not None and (abs(e) < 0.10 or (e < 0) != (p["eps"] < 0)
                                         or abs(pct or 0) > 200)
        if noisy:
            gap = "%s %s %s" % (v, _usd(abs(p["eps"] - e)),
                                "above" if p["eps"] > e else "below") \
                if v != "IN LINE" else "IN LINE"
            out.append("%s|%s vs %s est   %s%s" % (lab, _usd(p["eps"]), _usd(e), gap, note))
        else:
            out.append("%s|%s vs %s est   %s %+.1f%%%s"
                       % (lab, _usd(p["eps"]), _usd(e), v, pct, note) if v else
                       "%s|%s%s   (no estimate)" % (lab, _usd(p["eps"]), yr)
                       if not yr else "%s|%s%s" % (lab, _usd(p["eps"]), yr))

    if p.get("revenue") is not None:
        v, pct = _vs(p["revenue"], en[1])
        g = ("   %s%% YoY" % _s0(p["growth"])) if p.get("growth") is not None else ""
        out.append("Revenue|%s vs %s est  %s %+.1f%%%s"
                   % (fmt_money(p["revenue"]), fmt_money(en[1]), v, pct, g) if v else
                   ("Revenue|%s%s%s" % (fmt_money(p["revenue"]), yr, g) if yr else
                    "Revenue|%s   (no estimate)%s" % (fmt_money(p["revenue"]), g)))
    elif p.get("growth") is not None:
        out.append("Revenue|%s%% YoY" % _s0(p["growth"]))

    g = p.get("guide")
    if g:
        # Only a QUARTERLY guide is held against the next-quarter consensus; a
        # full-year or six-month figure against it is arithmetic nonsense. Which
        # one this is comes from guide_period_class, where the size of the figure
        # can overrule the words around it.
        per = guide_period_class(p)
        annual = per in ("FY", "6M")
        v, pct = (None, None) if annual else _vs(g["mid"], nx[1])
        if per == "FY":
            per = _period_label(g.get("period"), True)
        elif per == "6M":
            per = "6 mo"
        if v:
            out.append("Guide %s|%s vs %s est   %s %+.1f%%"
                       % (per, fmt_money(g["mid"]), fmt_money(nx[1]),
                          {"BEAT": "ABOVE", "MISS": "BELOW"}.get(v, "IN LINE"), pct))
        elif g["hi"] > g["lo"]:
            out.append("Guide %s|%s to %s   (no estimate)"
                       % (per, fmt_money(g["lo"]), fmt_money(g["hi"])))
        else:
            out.append("Guide %s|%s   (no estimate)" % (per, fmt_money(g["lo"])))
    elif p.get("guide_growth"):
        # Oracle guides the quarter only as a growth rate; saying the guide was
        # "not in the release" beside that would be false.
        gg = p["guide_growth"]
        gper = gg.get("period") or ""
        if _is_annual_period(gper):
            lab = _period_label(gper, True)
        else:
            lab = guide_period_class({"guide": {"period": gper, "mid": 0},
                                      "quarter": p.get("quarter")})
        out.append("Guide %s|%+.0f%% to %+.0f%% YoY   (growth only)" % (lab, gg["lo"], gg["hi"]))
    elif not p.get("guide_dir"):
        out.append("Guidance|no $ revenue guide in the release")

    if p.get("guide_dir"):
        d = p["guide_dir"].upper()
        arrow = {"RAISED": "^", "LOWERED": "v"}.get(d, "")
        out.append("FY outlook|%s %s the full-year guide" % (arrow + " " + d, "on"))

    # Margin - does the beat come from price or from discounting?
    if p.get("margin") is not None:
        d = ""
        if p.get("margin_prior") is not None:
            bps = (p["margin"] - p["margin_prior"]) * 100.0
            d = "   %sbps YoY" % _s0(bps)
        out.append("%s margin|%.1f%% %s%s"
                   % (p["margin_kind"].title(), p["margin"], p.get("margin_basis") or "", d))

    # Capital spending - the number that moves the AI-infrastructure names.
    if p.get("capex") is not None:
        d = ""
        if p.get("capex_prior"):
            d = "   %s%% YoY" % _s0(100.0 * (p["capex"] - p["capex_prior"]) / p["capex_prior"])
        out.append("Capex|%s this quarter%s" % (fmt_money(p["capex"]), d))

    head = ticker
    if p.get("quarter"):
        head += "  " + p["quarter"]
    return ("%s  %s" % (head, when)).strip(), out[:PANEL_ROWS]
