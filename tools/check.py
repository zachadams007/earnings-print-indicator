"""Generic Pine pre-deploy audit. Usage: python check.py <path.pine>

Covers the three traps that have each cost a failed deploy:
  1. a real newline inside a string literal   -> CE10017
  2. an unescaped " inside a string literal   -> syntax error, parity stays EVEN
  3. a missing closing parenthesis            -> "Missing closing parenthesis"
"""
import io, re, sys

P = sys.argv[1]
s = io.open(P, encoding="utf-8").read()

ESC = re.compile(r'\\"')


def strip_strings(line):
    out = []; i = 0; ins = False
    while i < len(line):
        c = line[i]
        if ins:
            if c == "\\":
                i += 2; continue
            if c == '"':
                ins = False
            i += 1; continue
        if c == '"':
            ins = True; i += 1; continue
        if c == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        out.append(c); i += 1
    return "".join(out)


odd = []
tips = []
for i, line in enumerate(s.split("\n"), 1):
    if line.lstrip().startswith("//"):
        continue
    code = line.split("//", 1)[0]
    stripped = ESC.sub("", code)
    if stripped.count('"') % 2:
        odd.append(i)
    elif "tooltip=" in code:
        seg = stripped.split("tooltip=", 1)[1]
        if seg.count('"') > 2 and "+" not in seg:
            tips.append((i, seg.count('"')))

depth = 0
negs = []
for i, line in enumerate(s.split("\n"), 1):
    if line.lstrip().startswith("//"):
        continue
    c = strip_strings(line)
    depth += c.count("(") + c.count("[") - c.count(")") - c.count("]")
    if depth < 0:
        negs.append(i); depth = 0

ins = re.findall(r"^\s*(\w+)\s*=\s*(?:input\.\w+|input)\s*\(", s, re.M)
plots = re.findall(r"(?<![A-Za-z_.])plot\w*\s*\(", s)

print("FILE                 :", P.split("\\")[-1])
print("odd-quote lines      :", len(odd), odd[:5])
print("suspect tooltip lines:", len(tips), tips[:5])
print("final bracket depth  :", depth, "(must be 0)")
print("negative-depth lines :", len(negs), negs[:5])
print("input count          :", len(ins))
print("plot-family outputs  :", len(plots), "(Pine ceiling 64)")
print("lines                :", s.count("\n") + 1)
ok = not odd and not tips and depth == 0 and not negs
print("VERDICT              :", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
