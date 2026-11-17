#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""The image's entrypoint: one rank's `spark serve` command line and environment.

It follows SparkGLM's install/serve.py (AGPL-3.0-only). SparkSeek has one
launch, so it is written out here instead of read from a profile file.
start.sh checks the options with `serve.py --check` before it starts anything.
"""
import ipaddress
import json
import os
from pathlib import Path
import re
import sys

# The pinned engine splits the release's weights four ways, one part a rank.
RANKS = 4
MAX_CONTEXT = 1048576  # the model's window
MAX_DRAFTS = 15        # the most drafts the engine verifies in a step
DEFAULTS = {'SPARKSEEK_SPARKS': '4', 'SPARKSEEK_DRAFTS': '5', 'SPARKSEEK_PREFILL_ROWS': '8192',
            'SPARKSEEK_CONTEXT': '131072', 'SPARKSEEK_PORT': '8894'}
DEVICE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')

# The ring between the four ranks, as SparkGLM's four-Spark profile sets it.
RING = {
    'ATLAS_RDMA_ALLREDUCE': '1',
    'ATLAS_RDMA_ONESHOT': '1',
    'ATLAS_RDMA_ONESHOT_MAX': '8388544',
    'ATLAS_RDMA_ONESHOT_SM_STAGE': '1',
    'ATLAS_RDMA_ONESHOT_STRIPE_MIN': '16384',
    'ATLAS_RDMA_PAIR_CHAIN': '1',
    'ATLAS_RDMA_PAIR_SPIN_US': '30000',
    'ATLAS_RDMA_TOPOLOGY': 'ring',
}
# NCCL only brings the ranks together, over sockets; the model's traffic rides the ring.
NCCL = {
    'NCCL_NET': 'Socket',
    'NCCL_IB_DISABLE': '1',
    'NCCL_DEBUG': 'WARN',
    'NCCL_CUMEM_ENABLE': '0',
    'NCCL_CUMEM_HOST_ENABLE': '0',
    'NCCL_NVLS_ENABLE': '0',
    'NCCL_ALGO': 'Ring',
    'NCCL_PROTO': 'Simple',
}
# On every rank: a decode step's kernels launched with programmatic dependent
# launch, the ranks' commands over the RDMA ring, and Engram row faults read by
# standing workers.
MODEL = {'ATLAS_DSV41_PDL': '1', 'ATLAS_GLM_CMD_RDMA': '1', 'ATLAS_PLE_FAULT_POOL': '1'}
# The release's draft head, a part of it on every rank, proposing --num-drafts
# tokens every step. The engine reads ATLAS_NO_MTP_K_LADDER by presence, so
# both are left out, not set to 0, when drafts are off.
DRAFT = {'ATLAS_DSV41_DRAFT': '1', 'ATLAS_NO_MTP_K_LADDER': '1'}


def number(environ, key, option):
    value = environ.get(key) or DEFAULTS[key]
    if not re.fullmatch(r'0|[1-9][0-9]{0,7}', value):
        raise ValueError(f'{option} must be a whole number, got {value!r}')
    return int(value)


def options(environ):
    """The launch options, checked: Sparks, drafts, prefill rows, context and port."""
    sparks = number(environ, 'SPARKSEEK_SPARKS', '--sparks')
    if sparks == 3:
        raise ValueError('three Sparks are not served yet: they need the weights split three ways, '
                         'and the pinned engine splits them four ways only. SparkSeek serves on four Sparks.')
    if sparks != RANKS:
        raise ValueError(f"--sparks must be {RANKS}: DeepSeek's release does not fit on fewer "
                         '(results/2026-10-26-weights), and the pinned engine serves it on four only')
    drafts = number(environ, 'SPARKSEEK_DRAFTS', '--drafts')
    if drafts > MAX_DRAFTS:
        raise ValueError(f'--drafts must be 0 (the draft head off) to {MAX_DRAFTS}')
    rows = number(environ, 'SPARKSEEK_PREFILL_ROWS', '--prefill-rows')
    if rows not in (8192, 16384):
        raise ValueError('--prefill-rows must be 8192 or 16384')
    context = number(environ, 'SPARKSEEK_CONTEXT', '--context')
    if not 1 <= context <= MAX_CONTEXT:
        raise ValueError(f"--context must be 1 to {MAX_CONTEXT}, the model's window")
    port = number(environ, 'SPARKSEEK_PORT', '--port')
    if not 1024 <= port <= 65535 - (RANKS - 1):
        raise ValueError(f'--port must be 1024 to {65535 - (RANKS - 1)}: rank R listens on the port + R')
    return {'sparks': sparks, 'drafts': drafts, 'prefill_rows': rows, 'context': context, 'port': port}


def ring_rails(environ):
    """This rank's RDMA devices toward the rank below (left) and the rank above
    (right). In a ring every rank has both, on distinct devices."""
    rails, seen = {}, set()
    for side in ('LEFT', 'RIGHT'):
        value = environ.get(f'FABRIC_{side}_HCA', '')
        names = value.split(',')
        if not all(DEVICE.fullmatch(name) for name in names) or len(names) != len(set(names)):
            raise ValueError(f'FABRIC_{side}_HCA must list unique RDMA device names: '
                             'every rank of a ring has a neighbour on each side')
        if seen.intersection(names):
            raise ValueError('Left and right links must use distinct RDMA devices')
        seen.update(names)
        rails[f'ATLAS_RDMA_{side}_RAILS'] = value
    return rails


def launch(environ):
    chosen = options(environ)
    rank = environ.get('NODE_RANK', '')
    if rank not in [str(r) for r in range(RANKS)]:
        raise ValueError(f'NODE_RANK must be 0 to {RANKS - 1}')
    rank = int(rank)
    master = str(ipaddress.IPv4Address(environ.get('MASTER_ADDR', '')))
    master_port = environ.get('MASTER_PORT', '29610')
    if not re.fullmatch(r'[1-9][0-9]{0,4}', master_port) or int(master_port) > 65535:
        raise ValueError('Invalid MASTER_PORT')
    interface = environ.get('FABRIC_INTERFACE', '')
    if not DEVICE.fullmatch(interface):
        raise ValueError('Invalid FABRIC_INTERFACE')
    model = environ.get('MODEL_PATH', '')
    if not Path(model).is_absolute() or any(c in model for c in '\r\n\0'):
        raise ValueError('MODEL_PATH must be an absolute path')
    name = environ.get('SERVED_MODEL_NAME', 'sparkseek')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/-]{0,127}', name):
        raise ValueError('Invalid SERVED_MODEL_NAME')
    # The launch below is the one that was checked on Sparks, so engine
    # switches from the surroundings are dropped. Logging knobs may pass:
    # they change what is reported, not what is computed.
    env = {key: value for key, value in environ.items()
           if not key.startswith('ATLAS_') or key.startswith('ATLAS_LOG_')}
    env.update(RING)
    env.update(ring_rails(environ))
    env.update(NCCL, NCCL_SOCKET_IFNAME=interface)
    env.update(MODEL, ATLAS_DSV41_ROWS=str(chosen['prefill_rows']))
    argv = ['/usr/local/bin/spark', 'serve', model,
            '--port', str(chosen['port'] + rank), '--no-tui', '--model-name', name,
            '--rank', str(rank), '--world-size', str(RANKS), '--ep-size', str(RANKS), '--tp-size', '1',
            '--master-addr', master, '--master-port', master_port,
            '--max-seq-len', str(chosen['context']),
            # One sequence at a time is all the engine supports for this model so far.
            '--max-num-seqs', '1',
            '--max-prefill-tokens', str(chosen['prefill_rows'])]
    if chosen['drafts']:
        env.update(DRAFT)
        argv += ['--speculative', '--num-drafts', str(chosen['drafts'])]
    return argv, env


def release_info(here):
    release = json.loads((here/'release.json').read_text())
    source_path = here/'source-manifest.json'
    if not source_path.is_file():
        source_path = here/'atlas-source.json'
    source = json.loads(source_path.read_text())
    return {'product': release['product'], 'version': release['version'],
            'engine_commit': source['commit'], 'engine_tree': source['tree']}


def print_version(here):
    info = release_info(here)
    print(f"{info['product']} {info['version']}")
    print(f"Atlas {info['engine_commit']}")
    print(f"Engine tree {info['engine_tree']}")


def main():
    here = Path(__file__).resolve().parent
    if sys.argv[1:] == ['--version']:
        print_version(here)
        return 0
    try:
        if sys.argv[1:] == ['--check']:
            options(os.environ)
            return 0
        argv, env = launch(os.environ)
        if not (Path(env['MODEL_PATH'])/'config.json').is_file():
            raise ValueError(f"no checkpoint at {env['MODEL_PATH']} (config.json is missing)")
    except ValueError as error:
        print(f'sparkseek: {error}', file=sys.stderr)
        return 2
    print(json.dumps({'event': 'atlas-recipe-start', **release_info(here), 'rank': env['NODE_RANK'],
                      **options(os.environ)}), flush=True)
    os.execve(argv[0], argv, env)


if __name__ == '__main__':
    sys.exit(main())
