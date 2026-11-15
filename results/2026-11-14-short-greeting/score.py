#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Score probe.py's rows: how many short replies are garbled, per arm.

A reply counts as garbled when a short greeting's answer shows one of the patterns
discussion #33 lists: a word repeated across a line break, a line break inside the
first sentence, a stray tag / heading / code token, or a run-on into unrelated text.
Prints the per-arm table and every flagged reply, so the rule can be checked by eye.

    python3 score.py [rows.jsonl] [--all]     # --all also prints the clean replies
"""
import collections, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
PATH = args[0] if args else os.path.join(HERE, "rows.jsonl")


def garbled(text):
    """Return the reason a reply looks garbled, or None."""
    if text is None or not text.strip():
        return "empty"
    t = text.strip()
    m = re.search(r"(\S+)\s*\n+\s*(\S+)", t)
    if m:
        a, b = m.group(1).strip(".,!?").lower(), m.group(2).strip(".,!?").lower()
        if a and (a == b or b.startswith(a) or a.endswith(b)):
            return "repeat across newline"
    if re.search(r"[A-Za-z\u4e00-\u9fff]\n", text):  # a line break straight after a word, no punctuation
        return "newline mid-sentence"
    if re.search(r"<ds_|<\||</?[a-z_]+>|#{1,6} |[A-Za-z\u4e00-\u9fff]#|/\*|```|\[分类\]|\[判定\]", t):
        return "stray tag/heading/code"
    if re.search(r"[a-z](?:const|function|import|return|def|class)\b", t):
        return "glued code token"
    if re.search(r"[a-z]{2}[A-Z][a-z]", re.sub(r"\b(?:DeepSeek|JavaScript|TypeScript|OpenAI|GitHub|YouTube|PyTorch)\b", "", t)):
        return "glued run-on"
    return None


rows = {}
for line in open(PATH):
    r = json.loads(line)
    if "error" in r and r["key"] in rows and "error" not in rows[r["key"]]:
        continue
    rows[r["key"]] = r

stats = collections.OrderedDict()
flagged, clean = [], []
for r in rows.values():
    s = stats.setdefault(r["arm"], dict(n=0, bad=0, err=0, thought=0, hosts=set()))
    if "error" in r:
        s["err"] += 1
        continue
    s["n"] += 1
    s["hosts"].add(r.get("served_by"))
    s["thought"] += bool(r.get("reasoning"))
    why = garbled(r.get("content"))
    if why:
        s["bad"] += 1
        flagged.append((r, why))
    else:
        clean.append(r)

# "thought" = replies that came with reasoning; in a chat (non-thinking) arm it means the host ignored thinking-off
print(f"{'arm':38s} {'garbled':>12s} {'errors':>6s} {'thought':>7s}  served by")
for arm, s in stats.items():
    pct = f"{100 * s['bad'] / s['n']:.1f}%" if s["n"] else "-"
    print(f"{arm:38s} {s['bad']:4d}/{s['n']:<4d} {pct:>6s} {s['err']:4d} {s['thought']:7d}  {', '.join(sorted(h or '?' for h in s['hosts']))}")

print("\n--- flagged replies")
for r, why in flagged:
    print(f"[{r['arm']}] sys={r['system']!r} user={r['user']!r} ({why})\n    {r['content']!r}")
if "--all" in sys.argv:
    print("\n--- clean replies")
    for r in clean:
        print(f"[{r['arm']}] sys={r['system']!r} user={r['user']!r}\n    {r['content']!r}")
errs = [r for r in rows.values() if "error" in r]
if errs:
    print("\n--- errors")
    for r in errs[:12]:
        print(f"[{r['arm']}] {r['error']}")
