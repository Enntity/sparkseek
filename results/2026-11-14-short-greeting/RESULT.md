<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# The 1017 short-greeting fault, re-run on hosted endpoints (2026-11-14)

`deepseek-ai/DeepSeek-V4.1-Flash-1017` discussion #33 (open since 2026-10-23)
reports that non-thinking mode garbles short replies when a system prompt is
present and the user message is a short greeting. This re-runs that claim
before SparkSeek engine work starts. No Spark was used.

```sh
DEEPSEEK_API_KEY=... OPENROUTER_API_KEY=... python3 probe.py   # 414 requests -> rows.jsonl
python3 score.py --all                                         # table, flagged and clean replies
```

DeepSeek's own arms go straight to `api.deepseek.com` (`deepseek-flash`,
`thinking: disabled|enabled`), with the report's request. The other hosts and
the pre-1017 weights go through OpenRouter, each arm pinned to one provider
with fallback off. Non-thinking arms run at temperature 0, max 60 tokens; the
thinking arm at temperature 1. Six system prompts by ten greetings (seven
English, three Chinese) unless noted.

## Result

| Arm | Weights | Host | Garbled |
|---|---|---|---|
| Non-thinking, system prompt + greeting | API `deepseek-flash` | DeepSeek | 37/60 (62%) |
| Same | 1017 | Novita, FP8 | 36/60 (60%) |
| Same | 1017 | Wafer, MXFP4 | 39/60 (65%) |
| Non-thinking, no system prompt + greeting | API | DeepSeek | 0/10 |
| Non-thinking, system prompt + short task | API | DeepSeek | 0/24 |
| Non-thinking, 128- and 359-token system prompt + greeting | API | DeepSeek | 0/20 |
| Thinking, system prompt + greeting | API | DeepSeek | 6/60 (10%) |
| Non-thinking, system prompt + greeting | pre-1017 | Wafer, MXFP4 | 17/60 (28%) |

The fault is real and it is in the model, not in one host's serving stack.

- The report's minimal repro (`You are a helpful assistant.` / `hello`) returns
  `Hello! How can\n can I help you today?` on all three hosts.
- 28 of the 60 system-plus-greeting replies are byte-identical across the three
  hosts, two weight formats included.
- It is not a near-tie. Where DeepSeek's API garbles, the line-break token has
  probability 1.00; the runners-up are `#`, `<` and `##`, as if a new document
  were starting.
- The break falls 4 to 8 tokens into the reply, at no fixed absolute position.
- It needs both conditions and a short context: no system prompt, a real
  question, or a longer system prompt each give zero.
- Thinking mode is not immune: 6 of 60 final answers carry the same damage
  (`Hi! How can### hi\n\nHi! How can I help you today?`).
- The earlier weights are not a refuge. They fail less often but worse: 11 of
  the 17 bad replies leak a `<ds_safety>` moderation block into the answer.
- DeepSeek's API still serves the fault three weeks after the report.

## Reading the numbers

- The table is `score.py`'s rule, except two rows corrected by eye: thinking is
  6, not the rule's 7 (one clean multi-line Chinese reply is flagged), and
  pre-1017 Wafer was checked reply by reply. Every flagged and clean reply was
  read; the rule's count matches the hand count on the three main arms.
- The Novita pre-1017 arm is void and left out: that host ignored thinking-off
  and reasoned on 51 of 60 requests.
- DeepSeek's API does not say which checkpoint `deepseek-flash` serves. Its
  replies match the pinned 1017 hosts, so it behaves as 1017.
- The long-system-prompt arm is 20 requests on one host; the report's larger
  run gives the same zero.
- 14 of 60 thinking-arm replies came back with no reasoning text.
- Rates here (60 to 65%) are above the report's 47 to 50%. The prompt sets
  differ, so the two are not a like-for-like comparison.

## What it means for SparkSeek

- An exact engine reproduces this fault. A SparkSeek build that returns the
  garbled greeting is correct, and one that returns a clean greeting has
  diverged from the reference. The minimal repro is a cheap conformance
  check.
- Short system-prompt-plus-greeting chat must not be used as a "does the engine
  work" smoke test: it fails on the reference.
- Agent and coding traffic sits behind long system prompts and is outside the
  fault's range on this evidence.
- Staying on 1017 is right; the earlier checkpoint is worse here.
