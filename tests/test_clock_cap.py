# SPDX-License-Identifier: AGPL-3.0-only
"""start.sh refuses to serve until every Spark runs gpu-clock-cap.service.

Adapted from SparkGLM's tests/test_clock_cap.py (AGPL-3.0-only)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests'))
from test_fabric_topology import RING  # noqa: E402


def fake_tree(root, uncapped=''):
    """A start.sh whose ring resolves, with stub systemctl, ssh and docker.
    `uncapped` names the host whose clock is not capped: here, b, c or d."""
    text = (ROOT/'start.sh').read_text()
    real = 'fabric_ports() { bash install/fabric-ports.sh; }'
    assert real in text
    (root/'start.sh').write_text(text.replace(real, 'fabric_ports() { cat "$INVENTORY/here"; }'))
    shutil.copy(ROOT/'fabric.py', root/'fabric.py')
    (root/'install').mkdir()
    for name in ('serve.py', 'fabric-ports.sh', 'checkpoints.json'):
        shutil.copy(ROOT/'install'/name, root/'install'/name)
    (root/'inventory').mkdir()
    for host, ports in zip(('here', 'b', 'c', 'd'), RING):
        (root/'inventory'/host).write_text(ports + '\n')
    bindir = root/'bin'
    bindir.mkdir()
    stubs = {
        'systemctl': f'exit {3 if uncapped == "here" else 0}',
        # A worker answers the fabric inventory and the systemctl probe.
        'ssh': 'host=$3; shift 3\ncase "$*" in *systemctl*) [ "$host" != "$UNCAPPED" ]; exit;; esac\n'
               'cat >/dev/null; cat "$INVENTORY/$host"',
        'docker': f'touch {root}/DOCKER_CALLED',
        # Never let a passing check reach the real download, copy or health wait.
        'hf': 'exit 1', 'rsync': 'exit 1', 'curl': 'exit 1',
    }
    for name, body in stubs.items():
        p = bindir/name
        p.write_text('#!/bin/sh\n'+body+'\n')
        p.chmod(0o755)
    # No IMAGE: right after the check, start.sh asks the absent install/build.sh
    # for the tag and stops there.
    env = dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}', WORKER='b c d', UNCAPPED=uncapped,
               MODEL_ROOT=str(root/'models'), INVENTORY=str(root/'inventory'))
    for key in ('CLOCK_CAP', 'IMAGE', 'SPARKS', 'DRAFTS', 'PREFILL_ROWS', 'CONTEXT', 'PORT', 'CHECKPOINT_DIR'):
        env.pop(key, None)
    return env


class ClockCapCheck(unittest.TestCase):
    def run_up(self, uncapped='', **extra):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = dict(fake_tree(root, uncapped), **extra)
            result = subprocess.run(['bash', str(root/'start.sh'), 'up'], env=env,
                                    capture_output=True, text=True, timeout=30)
            return result, (root/'DOCKER_CALLED').exists(), (root/'models').exists()

    def test_uncapped_host_refuses_before_docker_and_models(self):
        for name in ('here', 'b', 'd'):
            with self.subTest(uncapped=name):
                result, docker, models = self.run_up(name)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(f'not capped on {name}', result.stderr)
                self.assertIn('./start.sh clock-cap', result.stderr)
                self.assertFalse(docker)
                self.assertFalse(models)

    def test_capped_or_skipped_passes_the_check(self):
        for uncapped, extra in [('', {}), ('here', {'CLOCK_CAP': '0'}), ('c', {'CLOCK_CAP': '0'})]:
            with self.subTest(uncapped=uncapped, **extra):
                result, docker, _ = self.run_up(uncapped, **extra)
                self.assertNotIn('not capped', result.stderr)
                self.assertIn('build.sh', result.stderr)  # went on to the next step
                self.assertFalse(docker)


class ClockCapCommand(unittest.TestCase):
    def test_cap_must_be_a_number_before_any_sudo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = fake_tree(root)
            for name in ('sudo', 'ssh'):
                p = root/'bin'/name
                p.write_text(f'#!/bin/sh\ntouch {root}/{name.upper()}_CALLED\n')
                p.chmod(0o755)
            env['GPU_CLOCK_MAX_MHZ'] = '2200; touch /tmp/pwned'
            result = subprocess.run(['bash', str(root/'start.sh'), 'clock-cap'], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('must be a number', result.stderr)
            self.assertFalse((root/'SUDO_CALLED').exists())
            self.assertFalse((root/'SSH_CALLED').exists())

    def test_the_cap_script_is_shipped_and_named_by_start(self):
        self.assertIn('nvidia-smi -lgc 300,$max', (ROOT/'gpu-clock-cap.sh').read_text())
        self.assertIn('sudo bash gpu-clock-cap.sh "$max"', (ROOT/'start.sh').read_text())


if __name__ == '__main__':
    unittest.main()
