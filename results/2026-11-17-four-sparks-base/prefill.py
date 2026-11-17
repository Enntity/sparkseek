#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Time one cold prefill: `prefill.py TOKENS [URL]`.

The prompt is the recorded prompt of results/2026-11-15-reference-depth
(1,449 tokens) repeated to TOKENS tokens, sent as token ids, with one token
generated. Prints seconds and prompt tokens per second, client side.
"""
import json
from pathlib import Path
import sys
import time
import urllib.request

tokens = int(sys.argv[1])
url = sys.argv[2] if len(sys.argv) > 2 else 'http://127.0.0.1:8894'
meta = json.loads((Path(__file__).parent.parent/'2026-11-15-reference-depth'/'meta.json').read_text())
ids = meta['token_ids'] if isinstance(meta, dict) else meta
prompt = ids[:1] + (ids[1:] * (tokens // len(ids) + 2))[:tokens - 1]
body = {'model': 'sparkseek', 'prompt': prompt, 'max_tokens': 1, 'temperature': 0}
request = urllib.request.Request(url + '/v1/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
began = time.time()
with urllib.request.urlopen(request, timeout=7200) as reply:
    usage = json.load(reply).get('usage')
took = time.time() - began
print(f'{tokens} tokens: {took:.2f} s, {tokens / took:.0f} tok/s; usage {usage}')
