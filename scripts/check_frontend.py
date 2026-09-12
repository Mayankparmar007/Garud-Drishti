"""Static consistency check between the frontend files.

Not a unit test -- a lint for the specific way a hand-written dashboard breaks:
an id renamed in the HTML but not in the JS, or a helper used before it is
exported. Both fail silently at runtime as a blank panel, which is the worst
possible failure mode for a screen someone is watching during an event.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "frontend"
html = (HERE / "index.html").read_text(encoding="utf-8")
ctl = (HERE / "control.js").read_text(encoding="utf-8")
com = (HERE / "common.js").read_text(encoding="utf-8")
setup = (HERE / "setup.js").read_text(encoding="utf-8") if (HERE / "setup.js").exists() else ""

ok = True

ids_html = set(re.findall(r'id="([^"]+)"', html))
ids_used = set(re.findall(r"\$\('([^']+)'\)", ctl))
missing = sorted(ids_used - ids_html)
print(f"index.html ids           : {len(ids_html)}")
print(f"control.js ids referenced: {len(ids_used)}")
if missing:
    ok = False
    print(f"  MISSING from HTML      : {missing}")

m = re.search(r"const \{([^}]+)\} = UKSI;", ctl, re.S)
wanted = [w.strip() for w in m.group(1).replace("\n", " ").split(",") if w.strip()]
body = re.search(r"return \{(.*?)\n  \};", com, re.S).group(1)
exported = set(re.findall(r"[,\s](\w+)[,:]", "\n" + body))
notfound = [w for w in wanted if w not in exported]
print(f"UKSI helpers imported    : {len(wanted)}")
if notfound:
    ok = False
    print(f"  NOT exported           : {notfound}")

# Every endpoint the browser calls must exist in the API.
api = (HERE.parent / "uksi" / "api.py").read_text(encoding="utf-8")
routes = set(re.findall(r'@app\.(?:get|post)\("([^"]+)"', api))
called = set(re.findall(r"""(?:getJSON|postJSON)\(\s*['"`]([^'"`?]+)""", ctl + com + setup))
called |= set(re.findall(r"""url:\s*['"`]([^'"`?]+)""", setup))
unknown = sorted(c for c in called if c not in routes and not c.startswith("/frames"))
print(f"API routes defined       : {len(routes)}")
print(f"endpoints called by JS   : {len(called)}")
if unknown:
    ok = False
    print(f"  NOT in api.py          : {unknown}")

print("\n" + ("all consistent" if ok else "PROBLEMS FOUND"))
sys.exit(0 if ok else 1)
