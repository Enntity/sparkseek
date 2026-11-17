# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for start.sh: its options, the three-Spark refusal and the four launches,
with a fake inventory, transport and Docker around the real start.sh and start-node.sh.

In the style of SparkGLM's install/test_fabric.py (AGPL-3.0-only)."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('test_fabric_topology', ROOT/'tests/test_fabric_topology.py')
topology_fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topology_fixtures)
RING, F0, F1 = topology_fixtures.RING, topology_fixtures.F0, topology_fixtures.F1
REPO = 'deepseek-ai/DeepSeek-V4.1-Flash-1017'
REVISION = '1bae0e1088993b9b82de08f59de5be5a34e42170'
OK = {'choices': [{'message': {'content': '391'}, 'finish_reason': 'stop'}]}
# Each host's ports, keyed by its ssh destination: here-b-c-d-here is the ring.
HOSTS = dict(zip(('here', 'b', 'c', 'd'), RING))

SSH = r'''#!/bin/bash
host=$3; shift 3
echo "$host $*" >> "$CALLS/ssh"
if [[ "$*" == "bash -s" ]]; then cat >/dev/null; cat "$INVENTORY/$host"
elif [[ "$*" == "test -f"* && " $MISSING_ON " == *" $host "* ]]; then exit 1
else bash -c "$*"; fi
'''


class Result:
    def __init__(self, run, calls, root):
        self.code, self.out, self.err = run.returncode, run.stdout, run.stderr
        read = lambda name: (calls/name).read_text().splitlines() if (calls/name).exists() else []
        self.docker, self.ssh, self.hf, self.rsync = read('docker'), read('ssh'), read('hf'), read('rsync')
        self.runs = [line for line in self.docker if line.startswith('run ')]
        self.models = (root/'models').exists()
        self.marker = root/'models'/REPO.replace('/', '--')/'.sparkseek-revision'
        self.marker = self.marker.read_text() if self.marker.exists() else None


def run_start(*arguments, env_text='WORKER="b c d"\n', hosts=HOSTS, smoke=OK, downloaded=True,
              elsewhere=False, **extra):
    """start.sh with these arguments and .env. `downloaded` leaves the release under
    MODEL_ROOT with its marker; `elsewhere` puts a copy at {root}/copy instead."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        shutil.copytree(ROOT/'install', root/'install', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy(ROOT/'fabric.py', root/'fabric.py')
        text = (ROOT/'start.sh').read_text()
        real = 'fabric_ports() { bash install/fabric-ports.sh; }'
        assert real in text and 'case $action in\n  up)' in text
        text = text.replace(real, 'fabric_ports() { cat "$INVENTORY/here"; }')
        # The image is taken as built; everything else is the real control flow.
        text = text.replace('case $action in\n  up)', 'build() { :; }\ncase $action in\n  up)')
        (root/'start.sh').write_text(text)
        (root/'.env').write_text(env_text.replace('{root}', str(root)))
        release = root/'copy' if elsewhere else root/'models'/REPO.replace('/', '--')
        if downloaded or elsewhere:
            release.mkdir(parents=True)
            (release/'config.json').write_text('{}')
        if downloaded and not elsewhere:
            (release/'.sparkseek-revision').write_text(REVISION + '\n')
        (root/'inventory').mkdir()
        sysfs = root/'sysfs'
        for host, ports in hosts.items():
            (root/'inventory'/host).write_text(ports + '\n')
            for line in ports.splitlines():
                (sysfs/line.split()[0]/'ports/1').mkdir(parents=True, exist_ok=True)
                (sysfs/line.split()[0]/'ports/1/state').write_text('4: ACTIVE')
        calls, bindir = root/'calls', root/'bin'
        calls.mkdir()
        bindir.mkdir()
        record = '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS/{}"\n'
        scripts = {
            'systemctl': '#!/bin/sh\nexit 0\n',  # a capped host; tests/test_clock_cap.py has the rest
            'ssh': SSH, 'docker': record.format('docker'), 'rsync': record.format('rsync'),
            'hf': record.format('hf') + 'mkdir -p "$6" && echo "{}" > "$6/config.json"\n',
            'curl': '#!/bin/sh\nprintf \'%s\\n\' "$SMOKE_RESPONSE"\n'}
        for name, body in scripts.items():
            (bindir/name).write_text(body)
            (bindir/name).chmod(0o755)
        env = {k: v for k, v in os.environ.items()
               if k not in ('WORKER', 'SPARKS', 'DRAFTS', 'PREFILL_ROWS', 'CONTEXT', 'PORT',
                            'CHECKPOINT_DIR', 'BUILD', 'CLOCK_CAP')}
        env.update(PATH=f'{bindir}:{os.environ["PATH"]}', MODEL_ROOT=str(root/'models'), IMAGE='test-image',
                   FABRIC_SYSFS=str(sysfs), HOME=str(root), CALLS=str(calls), INVENTORY=str(root/'inventory'),
                   MISSING_ON='', SMOKE_RESPONSE=smoke if isinstance(smoke, str) else json.dumps(smoke))
        env.update(extra)
        run = subprocess.run(['bash', str(root/'start.sh'), *arguments], env=env,
                             capture_output=True, text=True, timeout=60)
        return Result(run, calls, root), root


def setting(call, name):
    return call.split(f'-e {name}=')[1].split()[0]


class FourSparks(unittest.TestCase):
    def test_one_command_starts_four_ranks_workers_first(self):
        result, root = run_start()
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual([setting(c, 'NODE_RANK') for c in result.runs], ['3', '2', '1', '0'])
        for host, rank in (('b', 1), ('c', 2), ('d', 3)):
            self.assertIn(f'rank {rank} on {host}: ring', result.out)
        self.assertIn('rank 0 on here: ring', result.out)
        release = root/'models'/REPO.replace('/', '--')
        for call in result.runs:
            for name, value in (('MASTER_ADDR', '192.0.2.17'), ('MASTER_PORT', '29610'),
                                ('MODEL_PATH', str(release)), ('SPARKSEEK_DRAFTS', '5'),
                                ('SPARKSEEK_PREFILL_ROWS', '8192'), ('SPARKSEEK_CONTEXT', '131072'),
                                ('SPARKSEEK_PORT', '8894')):
                self.assertEqual(setting(call, name), value)
            self.assertTrue(call.endswith(' test-image'))
        # Left is each rank's link to the rank below, right to the rank above.
        self.assertEqual([(setting(c, 'FABRIC_LEFT_HCA'), setting(c, 'FABRIC_RIGHT_HCA')) for c in result.runs],
                         [(F1, F0), (F0, F1), (F0, F1), (F1, F0)])
        self.assertEqual([setting(c, 'FABRIC_INTERFACE') for c in result.runs],
                         ['enP2p1s0f1np1', 'enP2p1s0f1np1', 'enP2p1s0f0np0', 'enP2p1s0f0np0'])
        self.assertIn('ready: http://127.0.0.1:8894/v1 model sparkseek', result.out)
        # The pinned release was already there with its marker: nothing fetched.
        self.assertEqual((result.hf, result.rsync), ([], []))

    def test_the_cables_decide_the_ranks_from_the_first_worker_listed(self):
        # d and b are both cabled to this Spark; the one listed first is rank 1.
        result, _ = run_start(env_text='WORKER="d b c"\n')
        self.assertEqual(result.code, 0, result.err)
        for host, rank in (('d', 1), ('c', 2), ('b', 3)):
            self.assertIn(f'rank {rank} on {host}: ring', result.out)
        self.assertEqual({setting(c, 'MASTER_ADDR') for c in result.runs}, {'192.0.2.114'})
        self.assertEqual([setting(c, 'NODE_RANK') for c in result.runs], ['3', '2', '1', '0'])
        rank0 = result.runs[-1]
        self.assertEqual((setting(rank0, 'FABRIC_LEFT_HCA'), setting(rank0, 'FABRIC_RIGHT_HCA')), (F0, F1))

    def test_options_on_the_command_line_reach_every_rank(self):
        result, _ = run_start('--drafts', '0', '--prefill-rows', '16384', '--context', '65536', '--port', '9000')
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(len(result.runs), 4)
        for call in result.runs:
            self.assertEqual([setting(call, 'SPARKSEEK_' + name) for name in
                              ('DRAFTS', 'PREFILL_ROWS', 'CONTEXT', 'PORT')], ['0', '16384', '65536', '9000'])
        self.assertIn('ready: http://127.0.0.1:9000/v1', result.out)

    def test_env_file_sets_options_and_the_command_line_wins(self):
        env_text = 'WORKER="b c d"\nSPARKS=4\nDRAFTS=3\nPREFILL_ROWS=16384\nCONTEXT=34000\nPORT=9100\n'
        result, _ = run_start('up', '--drafts', '7', env_text=env_text)
        self.assertEqual(result.code, 0, result.err)
        for call in result.runs:
            self.assertEqual([setting(call, 'SPARKSEEK_' + name) for name in
                              ('DRAFTS', 'PREFILL_ROWS', 'CONTEXT', 'PORT')], ['7', '16384', '34000', '9100'])

    def test_ready_requires_a_correct_normally_completed_answer(self):
        for content, finish, accepted in [('391', 'stop', True), ('17*23 = 391.', 'stop', True),
                                          ('392', 'stop', False), ('391', 'length', False), (None, 'stop', False)]:
            with self.subTest(content=content, finish=finish):
                result, _ = run_start(smoke={'choices': [{'message': {'content': content}, 'finish_reason': finish}]})
                self.assertEqual(len(result.runs), 4)
                self.assertEqual(result.code == 0, accepted, result.err)
                self.assertEqual('ready: ' in result.out, accepted)
        for smoke in ('{"choices":', {}, {'choices': []}):
            result, _ = run_start(smoke=smoke)
            self.assertNotEqual(result.code, 0)
            self.assertIn('smoke failed: invalid completion response', result.err)

    def test_fabric_only_reads(self):
        result, _ = run_start('fabric', downloaded=False)
        self.assertEqual(result.code, 0, result.err)
        self.assertIn(f'rank 0 on here: ring, enP2p1s0f0np0, left {F1}, right {F0}', result.out)
        self.assertEqual((result.docker, result.models), ([], False))

    def test_a_chain_is_refused_before_anything_starts(self):
        without = topology_fixtures.without
        chain = dict(HOSTS, here=without(RING[0], topology_fixtures.C, topology_fixtures.D),
                     d=without(RING[3], topology_fixtures.A, topology_fixtures.B))
        for action in ('up', 'fabric'):
            result, _ = run_start(action, hosts=chain, downloaded=False)
            self.assertEqual(result.code, 2, result.err)
            self.assertIn('cabled as a chain', result.err)
            self.assertEqual((result.docker, result.models), ([], False))

    def test_stop_removes_every_rank(self):
        result, _ = run_start('stop')
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual(result.docker, ['rm -f atlas-sparkseek-rank0'] +
                         ['rm -f atlas-sparkseek-rank1 atlas-sparkseek-rank2 atlas-sparkseek-rank3'] * 3)


class ThreeSparks(unittest.TestCase):
    def assert_refused(self, result, *messages):
        self.assertEqual(result.code, 2, result.err)
        for message in messages:
            self.assertIn(message, result.err)
        # Before any ssh, Docker call or model directory.
        self.assertEqual((result.ssh, result.docker, result.models), ([], [], False))

    def test_three_sparks_stop_early_with_the_reason(self):
        reason = ('three Sparks are not served yet: they need the weights split three ways, '
                  'and the pinned engine splits them four ways only')
        for action in ('up', 'fabric', 'build', 'download', 'status'):
            for arguments, env_text in ((('--sparks', '3'), 'WORKER="b c"\n'),
                                        (('--sparks', '3'), 'WORKER="b c d"\n'),
                                        ((), 'WORKER="b c"\nSPARKS=3\n')):
                with self.subTest(action=action, arguments=arguments, env_text=env_text):
                    result, _ = run_start(action, *arguments, env_text=env_text, downloaded=False)
                    self.assert_refused(result, reason)

    def test_a_worker_list_of_two_is_three_sparks(self):
        result, _ = run_start(env_text='WORKER="b c"\n', downloaded=False)
        self.assert_refused(result, 'three Sparks are not served yet',
                            'WORKER lists 2 ssh destinations; --sparks 4 needs 3')

    def test_the_worker_list_must_match(self):
        for workers, count in (('b', 1), ('b c d e', 4)):
            result, _ = run_start(env_text=f'WORKER="{workers}"\n', downloaded=False)
            self.assert_refused(result, f'WORKER lists {count} ssh destinations; --sparks 4 needs 3')
            self.assertNotIn('three Sparks', result.err)
        result, _ = run_start(env_text='', downloaded=False)
        self.assertNotEqual(result.code, 0)
        self.assertIn('set WORKER in .env', result.err)

    def test_other_numbers_of_sparks_and_bad_options_stop_early(self):
        for arguments, message in [(('--sparks', '2'), '--sparks must be 4'),
                                   (('--sparks', '1'), '--sparks must be 4'),
                                   (('--sparks', '8'), '--sparks must be 4'),
                                   (('--prefill-rows', '4096'), '--prefill-rows must be 8192 or 16384'),
                                   (('--drafts', '16'), '--drafts must be 0'),
                                   (('--context', '0'), '--context must be 1 to 1048576'),
                                   (('--port', '80'), '--port must be 1024'),
                                   (('--drafts',), '--drafts needs a number'),
                                   (('--mode', 'spark'), './start.sh [options]'),
                                   (('convert',), './start.sh [options]')]:
            with self.subTest(arguments=arguments):
                result, _ = run_start(*arguments, downloaded=False)
                self.assert_refused(result, message)


class Checkpoint(unittest.TestCase):
    def test_download_fetches_the_pinned_release_once_and_copies_it(self):
        result, root = run_start('download', downloaded=False)
        self.assertEqual(result.code, 0, result.err)
        release = root/'models'/REPO.replace('/', '--')
        self.assertEqual(result.hf, [f'download {REPO} --revision {REVISION} --local-dir {release}'])
        for host in 'bcd':
            self.assertTrue(any(line.endswith(f' {release} {host}:{root}/models/') for line in result.rsync), host)
        # The marker is written after every copy; with it, nothing is fetched again.
        self.assertEqual(result.marker, REVISION + '\n')
        self.assertEqual(result.runs, [])

    def test_env_can_name_a_copy_every_spark_already_has(self):
        result, root = run_start(env_text='WORKER="b c d"\nCHECKPOINT_DIR={root}/copy\n', elsewhere=True)
        self.assertEqual(result.code, 0, result.err)
        self.assertEqual({setting(c, 'MODEL_PATH') for c in result.runs}, {str(root/'copy')})
        self.assertEqual((result.hf, result.rsync, result.models), ([], [], False))
        self.assertIn(f'checkpoint {root}/copy on every Spark', result.out)

    def test_a_copy_missing_on_one_spark_stops_before_any_rank_starts(self):
        for missing, name in (('c', 'c'), ('b d', 'b')):
            result, root = run_start(env_text='WORKER="b c d"\nCHECKPOINT_DIR={root}/copy\n', elsewhere=True,
                                     MISSING_ON=missing)
            self.assertEqual(result.code, 2, result.err)
            self.assertIn(f'CHECKPOINT_DIR: {root}/copy/config.json is missing on {name}', result.err)
            self.assertEqual((result.runs, result.hf), ([], []))
        result, _ = run_start(env_text='WORKER="b c d"\nCHECKPOINT_DIR={root}/nowhere\n')
        self.assertEqual(result.code, 2, result.err)
        self.assertIn('is missing on here', result.err)
        result, _ = run_start(env_text='WORKER="b c d"\nCHECKPOINT_DIR=models/copy\n')
        self.assertEqual(result.code, 2, result.err)
        self.assertIn('CHECKPOINT_DIR must be an absolute path', result.err)
        self.assertEqual(result.runs, [])


if __name__ == '__main__':
    unittest.main()
