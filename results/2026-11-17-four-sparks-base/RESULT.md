<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# First start through `start.sh` on four Sparks (2026-11-17)

The first run of SparkSeek's own entry point: `./start.sh` on four DGX Sparks
cabled as a ring, with the image built by `install/build.sh` from the pinned
engine (Enntity/atlas `8c71b62ee67ad1b470c90e381cbce81967a687c5`) and
DeepSeek's release as published. It shows that the base version starts and
answers, and records a first speed for each default. It is not a benchmark
suite: one stream, a handful of requests, every repetition listed.

```sh
./start.sh                                  # drafts 5, prefill rows 8192, context 131072
python3 prefill.py 32000                    # three times
python3 chat.py 500 "<prose prompt>"        # then the Python and JSON prompts below
./start.sh --prefill-rows 16384 --drafts 0  # and the same requests again
```

Both starts passed `start.sh`'s own check (17*23 answered with 391).

## Result

| Start | 32,000-token prefill, cold, three runs | Prose, 500 tokens | Python, 500 tokens | JSON, 400 tokens |
|---|---|---|---|---|
| defaults (draft head on, 5 drafts; 8,192 rows) | 5.45 s, 4.87 s, 5.01 s (5,875, 6,567, 6,391 tok/s) | 53.0 tok/s | 95.9 tok/s | 86.5 tok/s |
| `--prefill-rows 16384 --drafts 0` | 5.36 s, 4.72 s, 4.73 s (5,966, 6,784, 6,771 tok/s) | 45.7 tok/s | 45.9 tok/s | 46.6 tok/s |

- Prefill is timed at the client: one request of 32,000 prompt tokens and one
  generated token, nothing cached. The first request after a start is the
  slower one each time.
- The reply rates are the engine's own (`Done:` lines of rank 0,
  [raw/](raw/)), greedy, one request at a time. With the draft head on, a step
  kept 1.78 tokens on the prose prompt, 4.13 on the Python one and 3.58 on
  the JSON one; with it off every step is one token. Time to first token was
  126 to 148 ms.
- After a 32,000-token and a 9,000-token prompt both starts continue with the
  same 64 tokens as the engine build before this tooling
  (`raw/*/tokens.txt`): on these two prompts the draft head changes the
  speed and not the tokens.

## What it does not show

- Nothing about several requests at once: the engine takes one sequence at a
  time.
- `--prefill-rows 16384` with the draft head on did not start here. One of the
  four Sparks shares its memory with other work (about 29 GB), and that rank
  had 5.2 GB left where it keeps 7.5 GB aside. A rank needs about 89 GB before
  its sequence state in that setting, which a Spark with its memory to itself
  has.
- No comparison with other engines: their published figures come from other
  prompts and harnesses.

## The prompts

- Prefill: the recorded sequence of
  [2026-11-15-reference-depth](../2026-11-15-reference-depth/RESULT.md)
  (1,449 tokens), repeated to 32,000 tokens and sent as token ids.
- Prose: "Write a thoughtful essay about why lighthouses still matter in the
  age of satellite navigation. Use plain paragraphs."
- Python: "Write a Python module implementing an LRU cache class with get, put
  and delete, with type hints and docstrings, plus unit tests using unittest."
- JSON: "Return a JSON array of 12 objects describing fictional books, each
  with title, author, year, genre and a list of three tags. Output only JSON."

## Files

- `prefill.py`, `chat.py`: the two clients.
- `raw/default/`, `raw/rows16384-drafts0/`: each client's output
  (`prefill-32000.txt`, `decode.txt`), the engine's `Done:` lines
  (`done-lines.txt`) and the token comparison (`tokens.txt`), unedited.
