#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""One greedy chat completion: `chat.py MAX_TOKENS PROMPT... [--url URL]`.

Prints wall seconds, completion tokens and the start of the reply. The
engine's own rate for the request is the `Done:` line in rank 0's log.
"""
import json
import sys
import time
import urllib.request

args = sys.argv[1:]
url = 'http://127.0.0.1:8894'
if '--url' in args:
    at = args.index('--url')
    url = args[at + 1]
    del args[at:at + 2]
body = {'model': 'sparkseek', 'messages': [{'role': 'user', 'content': ' '.join(args[1:])}],
        'max_tokens': int(args[0]), 'temperature': 0}
request = urllib.request.Request(url + '/v1/chat/completions', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json'})
began = time.time()
with urllib.request.urlopen(request, timeout=3600) as reply:
    answer = json.load(reply)
message = answer['choices'][0]['message']
text = message.get('content') or message.get('reasoning_content') or ''
print(f"{time.time() - began:.1f} s; {answer['usage']['completion_tokens']} tokens: {text[:70]!r}")
