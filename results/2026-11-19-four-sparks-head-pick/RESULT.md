<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# `start.sh` on four Sparks at the engine pin `fb3b99b6` (2026-11-19)

The first run through `./start.sh` since the pin moved twice: the image
`install/build.sh` built from this repository at `c321a03` and the pinned
engine (Enntity/atlas `fb3b99b68ec5b1c906fe85f61f9ed8865a0549bf`), DeepSeek's
release as published, on the four DGX Sparks of the
[first receipt](../2026-11-17-four-sparks-base/RESULT.md), with the same
clients and prompts. One stream, greedy, a handful of requests, every
repetition listed. It is not a benchmark suite.

```sh
./start.sh                                  # drafts 5, prefill rows 8192, context 131072
python3 prefill.py 32000                    # three times
python3 chat.py 500 "<prose prompt>"        # then the Python and JSON prompts; the three, three times over
./start.sh stop
./start.sh --prefill-rows 16384             # and the same requests again
./start.sh stop
./start.sh --prefill-rows 16384 --drafts 0  # and again
```

The first `./start.sh` built the image (its `checks` stage ran the engine's
unit tests for this model). All three starts passed `start.sh`'s own check
(17*23 answered with 391).

## Result

Cold 32,000-token prefill, three runs each:

| Start | Seconds | tok/s |
|---|---|---|
| defaults (8,192 rows, 5 drafts) | 5.04, 4.52, 4.53 | 6,353, 7,079, 7,057 |
| `--prefill-rows 16384` | 4.95, 4.38, 4.37 | 6,458, 7,304, 7,314 |
| `--prefill-rows 16384 --drafts 0` | 4.65, 4.38, 4.38 | 6,880, 7,299, 7,311 |

Replies, tok/s, three runs each in the order they were made:

| Start | Prose, 500 tokens | Python, 500 tokens | JSON, 400 tokens |
|---|---|---|---|
| defaults | 55.2, 58.6, 58.2 | 100.6, 100.1, 104.3 | 90.7, 98.0, 92.8 |
| `--prefill-rows 16384` | 55.2, 58.4, 58.1 | 100.6, 99.9, 104.3 | 90.3, 97.8, 92.7 |
| `--prefill-rows 16384 --drafts 0` | 46.2, 49.3, 49.2 | 46.4, 49.3, 49.3 | 47.1, 49.4, 49.3 |

- Prefill is timed at the client: one request of 32,000 prompt tokens and one
  generated token, nothing cached. The first request after a start is the
  slower one each time.
- The reply rates are the engine's own (`Done:` lines of rank 0,
  [raw/](raw/)). With the draft head on, a step kept 1.78 tokens on the prose
  prompt, 4.10 to 4.13 on the Python one and 3.52 to 3.60 on the JSON one.
  Time to first token was 122 to 145 ms.
- The first round of the three prompts after a start is the slowest, except
  on the Python prompt with the draft head on.
- After a 32,000-token and a 9,000-token prompt every start continues with
  the same 64 tokens as the engine build the first receipt compared against
  (`raw/*/tokens.txt`).

## Against the first receipt

Same Sparks, clients and prompts, two engine pins apart, one run then and
three now. The pins between are the faster batch attention (`bcef3bef`) and
four decode changes (`fb3b99b6`): a greedy step's token picked without the
whole head, a faster draft pass, the router scoring three rows a pass, and a
verify's rollback moved out of the gap before the draft pass.

| | 2026-11-17, `8c71b62e` | Now, `fb3b99b6` |
|---|---|---|
| Prefill, defaults | 5.45, 4.87, 5.01 s | 5.04, 4.52, 4.53 s |
| Prefill, 16,384 rows, drafts off | 5.36, 4.72, 4.73 s | 4.65, 4.38, 4.38 s |
| Prose, defaults | 53.0 tok/s | 55.2, 58.6, 58.2 |
| Python, defaults | 95.9 | 100.6, 100.1, 104.3 |
| JSON, defaults | 86.5 | 90.7, 98.0, 92.8 |
| Drafts off | 45.7, 45.9, 46.6 | 46.2 to 47.1, then 49.2 to 49.4 |

This is two receipts side by side and not a controlled comparison: the first
receipt has one reply a prompt, and the Sparks' other work differed (below).

## What it shows that the first receipt could not

`--prefill-rows 16384` with the draft head on starts. On 2026-11-17 one Spark
shared about 29 GB of its memory with other work and that rank did not fit.
This time the Spark that shares the most had 22 GB in use before the start,
and its rank started with 14 GB of memory still available.

## What it does not show

- Nothing about several requests at once: the engine takes one sequence at a
  time.
- With `--drafts 0` the engine does not load the draft head, and the pick of
  a greedy step's token rides on the draft head's copy of the head, so that
  start runs every step through the whole head. The engine started by hand
  with the draft head loaded and no drafts proposed ran at 50.4 to 50.8
  tok/s; `start.sh` has no option for that.
- No comparison with other engines: their published figures come from other
  prompts and harnesses.

## The prompts and the clients

The prompts, `prefill.py` and `chat.py` are the first receipt's
([prompts](../2026-11-17-four-sparks-base/RESULT.md#the-prompts),
[`prefill.py`](../2026-11-17-four-sparks-base/prefill.py),
[`chat.py`](../2026-11-17-four-sparks-base/chat.py)).

## Files

- `raw/default/`, `raw/rows16384/`, `raw/rows16384-drafts0/`: each client's
  output (`prefill-32000.txt`, `decode.txt`), the engine's `Done:` lines
  (`done-lines.txt`) and the token comparison (`tokens.txt`), unedited. The
  first two `Done:` lines of each file are the 9,000-token comparison and a
  warm-up greeting; the nine after them are the replies in the table.
