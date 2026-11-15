#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Does DeepSeek-V4.1-Flash-1017 garble short non-thinking replies behind a system prompt?

Re-runs the claim of deepseek-ai/DeepSeek-V4.1-Flash-1017 discussion #33 on hosted
endpoints. DeepSeek's own arms go straight to api.deepseek.com (DEEPSEEK_API_KEY) with
the report's exact request; the other hosts and the pre-1017 weights go through
OpenRouter (OPENROUTER_API_KEY), one pinned provider per arm, no fallback. Arms whose
key is missing are skipped. Writes one JSON line per request to OUT (default rows.jsonl
beside this file) and skips rows already there, so it resumes.

    python3 probe.py            # run every arm
    python3 probe.py --plan     # print the arms and request counts, send nothing
"""
import concurrent.futures, hashlib, json, os, sys, time, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("OUT", os.path.join(HERE, "rows.jsonl"))
OR_URL, DS_URL = "https://openrouter.ai/api/v1/chat/completions", "https://api.deepseek.com/chat/completions"
M1017, MPRE = "deepseek/deepseek-v4.1-flash-1017", "deepseek/deepseek-v4.1-flash"
MDS, DIRECT = "deepseek-flash", "deepseek-api"  # DIRECT as provider tag = DeepSeek's own API, not OpenRouter
KEY_ENV = {True: "DEEPSEEK_API_KEY", False: "OPENROUTER_API_KEY"}  # by "is direct"

SYSTEMS = [
    "You are a helpful assistant.",
    "You are a concise assistant.",
    "You are a friendly customer support agent for an online bookstore.",
    "Answer in the same language as the user.",
    "You are an expert software engineer.",
    "You are a helpful assistant. Keep answers short.",
]
GREETINGS = ["hi", "hello", "hey", "hihi", "thanks", "good morning", "yo", "你好", "在吗", "早上好"]
# Longer system prompts: does the fault go away once the context is past ~50 tokens, as the report says?
PERSONA = ("You are Mira, the in-app assistant for a small team's project tracker. You help people find tickets, "
           "summarise what changed since they last looked, draft status updates, and explain how features of the "
           "tracker work. Be warm but brief. Prefer one or two sentences unless the person asks for detail. Never "
           "invent ticket numbers, dates or names; if you do not know, say so and offer to look it up. When someone "
           "greets you, greet them back and ask what they would like to work on. Use plain language, no jargon, "
           "and match the language the person writes in.")
GUIDE = PERSONA + "\n\nHouse rules:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate([
    "Dates are written as day, month name, year. Times carry a time zone.",
    "A ticket is referred to by its key and title together the first time it comes up.",
    "Status updates lead with what shipped, then what is blocked, then what is next.",
    "Never promise a delivery date on the team's behalf; quote the date on the ticket and who set it.",
    "If two tickets look like duplicates, say so and link both rather than choosing one.",
    "Summaries of long threads name who decided what, and leave out the back and forth.",
    "Questions about permissions go to the workspace admin; say who that is if it is on file.",
    "When asked to draft a message, give one draft, then offer a shorter or a more formal version.",
    "Numbers come with their unit and the period they cover.",
    "If the request is ambiguous between two projects, ask which one before doing anything.",
    "Do not repeat the person's question back to them.",
    "Close by offering one concrete next step, not a list of everything you can do.",
], 1))
LONG_SYSTEMS = [PERSONA, GUIDE]
TASKS = ["Tell me a short joke.", "What is 2+2?", "Name three colors.", "What is the capital of France?"]

# (arm, model, provider tag, thinking, system prompts, user messages, temperature, max_tokens)
ARMS = [
    ("api/deepseek/chat/sys+greet", MDS, DIRECT, False, SYSTEMS, GREETINGS, 0, 60),
    ("1017/novita-fp8/chat/sys+greet", M1017, "novita/fp8", False, SYSTEMS, GREETINGS, 0, 60),
    ("1017/wafer-mxfp4/chat/sys+greet", M1017, "wafer/mxfp4", False, SYSTEMS, GREETINGS, 0, 60),
    ("api/deepseek/chat/nosys+greet", MDS, DIRECT, False, [None], GREETINGS, 0, 60),
    ("api/deepseek/chat/longsys+greet", MDS, DIRECT, False, LONG_SYSTEMS, GREETINGS, 0, 60),
    ("api/deepseek/chat/sys+task", MDS, DIRECT, False, SYSTEMS, TASKS, 0, 60),
    ("api/deepseek/think/sys+greet", MDS, DIRECT, True, SYSTEMS, GREETINGS, 1, 600),
    ("pre1017/novita-fp8/chat/sys+greet", MPRE, "novita/fp8", False, SYSTEMS, GREETINGS, 0, 60),
    ("pre1017/wafer-mxfp4/chat/sys+greet", MPRE, "wafer/mxfp4", False, SYSTEMS, GREETINGS, 0, 60),
]


def rows():
    for arm, model, prov, think, systems, users, temp, mx in ARMS:
        for s in systems:
            for u in users:
                key = hashlib.sha256(json.dumps([arm, s, u]).encode()).hexdigest()[:16]
                yield dict(key=key, arm=arm, model=model, provider=prov, thinking=think, system=s, user=u,
                           temperature=temp, max_tokens=mx)


def call(row, keys):
    direct = row["provider"] == DIRECT
    url, api_key = (DS_URL if direct else OR_URL), keys[KEY_ENV[direct]]
    msgs = ([{"role": "system", "content": row["system"]}] if row["system"] else []) + [
        {"role": "user", "content": row["user"]}]
    body = {"model": row["model"], "messages": msgs, "temperature": row["temperature"],
            "max_tokens": row["max_tokens"]}
    if direct:
        body["thinking"] = {"type": "enabled" if row["thinking"] else "disabled"}
    else:
        body.update(seed=1, reasoning={"enabled": row["thinking"]},
                    provider={"only": [row["provider"]], "allow_fallbacks": False})
    if not row["thinking"]:
        body.update(logprobs=True, top_logprobs=5)
    for attempt in range(4):
        req = urllib.request.Request(url, json.dumps(body).encode(),
                                     {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            d = {"error": {"code": e.code, "message": e.read().decode("utf8", "replace")[:300]}}
        except Exception as e:  # network
            d = {"error": {"code": "net", "message": str(e)[:300]}}
        if "error" in d and "logprobs" in body and attempt == 0:
            body.pop("logprobs"), body.pop("top_logprobs")  # host without logprobs: retry plain
            continue
        if "error" in d and d["error"].get("code") in (429, 502, 503, "net"):
            time.sleep(3 * (attempt + 1))
            continue
        break
    out = dict(row)
    if "error" in d:
        out["error"] = d["error"]
        return out
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    out.update(content=msg.get("content"), reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
               finish=ch.get("finish_reason"), served_by=d.get("provider") or (DIRECT if direct else None),
               upstream_model=d.get("model"), fingerprint=d.get("system_fingerprint"), usage=d.get("usage"))
    lp = (ch.get("logprobs") or {}).get("content")
    if lp:
        out["logprobs"] = [dict(t=x.get("token"), lp=x.get("logprob"),
                                top=[[y.get("token"), y.get("logprob")] for y in x.get("top_logprobs") or []])
                           for x in lp[:24]]
    return out


def main():
    todo = list(rows())
    if "--plan" in sys.argv:
        for arm in dict.fromkeys(r["arm"] for r in todo):
            print(f"{sum(r['arm'] == arm for r in todo):4d}  {arm}")
        print(f"{len(todo):4d}  total")
        return
    keys = {e: os.environ[e] for e in KEY_ENV.values() if os.environ.get(e)}
    if not keys:
        sys.exit("set DEEPSEEK_API_KEY and/or OPENROUTER_API_KEY")
    skipped = [r for r in todo if KEY_ENV[r["provider"] == DIRECT] not in keys]
    todo = [r for r in todo if KEY_ENV[r["provider"] == DIRECT] in keys]
    if skipped:
        print(f"skipping {len(skipped)} requests with no key: {sorted({r['arm'] for r in skipped})}", flush=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            if "error" not in r:
                done.add(r["key"])
    todo = [r for r in todo if r["key"] not in done]
    print(f"{len(done)} done, {len(todo)} to send -> {OUT}", flush=True)
    with open(OUT, "a") as f, concurrent.futures.ThreadPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(lambda row: call(row, keys), todo), 1):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"{i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
