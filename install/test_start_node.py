# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for one rank's container launch, with a recording docker.

In the style of SparkGLM's install/test_install.py and test_tp4_modes.py (AGPL-3.0-only)."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
LEFT, RIGHT = 'rocep1s0f1,roceP2p1s0f1', 'rocep1s0f0,roceP2p1s0f0'


def fake_node(root):
    """A checkpoint, RDMA sysfs and a recording docker for start-node.sh."""
    checkpoint, bin_dir, sysfs = root/'models'/'release', root/'bin', root/'sysfs'
    checkpoint.mkdir(parents=True)
    (checkpoint/'config.json').touch()
    bin_dir.mkdir()
    for dev in ('rocep1s0f0', 'roceP2p1s0f0', 'rocep1s0f1', 'roceP2p1s0f1'):
        (sysfs/dev/'ports/1').mkdir(parents=True)
        (sysfs/dev/'ports/1/state').write_text('4: ACTIVE')
    (sysfs/'rocep1s0f9/ports/1').mkdir(parents=True)
    (sysfs/'rocep1s0f9/ports/1/state').write_text('1: DOWN')
    (bin_dir/'docker').write_text(f'#!/bin/sh\necho "$*" >> {root}/docker.log\n')
    (bin_dir/'docker').chmod(0o755)

    def start(*arguments, **replaced):
        (root/'docker.log').unlink(missing_ok=True)
        options = {'--rank': '0', '--leader-address': '192.0.2.17', '--fabric-interface': 'enP2p1s0f0np0',
                   '--left-hca': LEFT, '--right-hca': RIGHT, '--checkpoint': str(checkpoint),
                   '--image': 'img', '--cuda-cache': str(root/'cuda')}
        options.update({'--' + key.replace('_', '-'): value for key, value in replaced.items()})
        command = ['bash', str(HERE/'start-node.sh')]
        for key, value in options.items():
            if value is not None:
                command += [key, value]
        run = subprocess.run([*command, *arguments], capture_output=True, text=True,
                             env=dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', FABRIC_SYSFS=str(sysfs)))
        log = (root/'docker.log').read_text() if (root/'docker.log').exists() else ''
        return run, [line for line in log.splitlines() if line.startswith('run ')]
    return checkpoint.resolve(), start


class NodeLaunch(unittest.TestCase):
    def test_each_ring_rank_starts_the_image_with_its_rails_and_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, start = fake_node(Path(tmp))
            for rank in '0123':
                run, calls = start(rank=rank)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertEqual(len(calls), 1)
                call = calls[0] + ' '
                for want in (f'--name atlas-sparkseek-rank{rank} ', f'-e NODE_RANK={rank} ',
                             '-e MASTER_ADDR=192.0.2.17 ', '-e MASTER_PORT=29610 ',
                             '-e FABRIC_INTERFACE=enP2p1s0f0np0 ',
                             f'-e FABRIC_LEFT_HCA={LEFT} ', f'-e FABRIC_RIGHT_HCA={RIGHT} ',
                             f'-e MODEL_PATH={checkpoint} ', '-e SERVED_MODEL_NAME=sparkseek ',
                             '-e SPARKSEEK_DRAFTS=5 ', '-e SPARKSEEK_PREFILL_ROWS=8192 ',
                             '-e SPARKSEEK_CONTEXT=131072 ', '-e SPARKSEEK_PORT=8894 ',
                             f'--mount type=bind,src={checkpoint},dst={checkpoint},readonly '):
                    self.assertIn(want, call)
                self.assertTrue(call.endswith(' img '), call)

    def test_container_flags_are_the_working_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, start = fake_node(Path(tmp))
            run, calls = start()
            self.assertEqual(run.returncode, 0, run.stderr)
            for flag in ('--gpus all', '--network host', '--ipc host', '--shm-size 32g',
                         '--ulimit memlock=-1:-1', '--cap-add IPC_LOCK', '--cap-add SYS_NICE',
                         '--device /dev/infiniband:/dev/infiniband', '--restart no'):
                self.assertIn(f' {flag} ', calls[0])
            # No engine switch is set here: the image's serve.py owns them.
            self.assertNotIn('ATLAS_', calls[0])
            self.assertNotIn('NCCL_', calls[0])

    def test_options_are_passed_to_the_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, start = fake_node(Path(tmp))
            run, calls = start('--drafts', '0', '--prefill-rows', '16384', '--context', '65536',
                               '--port', '9000', '--name', 'trial')
            self.assertEqual(run.returncode, 0, run.stderr)
            for want in ('--name trial ', '-e SPARKSEEK_DRAFTS=0 ', '-e SPARKSEEK_PREFILL_ROWS=16384 ',
                         '-e SPARKSEEK_CONTEXT=65536 ', '-e SPARKSEEK_PORT=9000 '):
                self.assertIn(want, calls[0])

    def test_a_bad_launch_never_reaches_docker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, start = fake_node(root)
            (root/'empty').mkdir()
            cases = [
                ({'rank': '4'}, 'usage'), ({'rank': None}, 'usage'), ({'image': None}, 'usage'),
                ({'leader_address': None}, 'usage'), ({'checkpoint': None}, 'usage'),
                ({'left_hca': None}, 'needs --left-hca'), ({'right_hca': None}, 'needs --right-hca'),
                ({'fabric_interface': None}, 'needs --fabric-interface'),
                ({'left_hca': 'rocep1s0f9'}, 'missing or down'),
                ({'right_hca': 'absent0'}, 'missing or down'),
                ({'left_hca': 'rocep1s0f1,rocep1s0f1'}, 'duplicate'),
                ({'left_hca': RIGHT}, 'duplicate'),
                ({'right_hca': 'rocep1s0f0;true'}, 'invalid RDMA device list'),
                ({'checkpoint': 'models/release'}, 'absolute path'),
                ({'checkpoint': str(root/'empty')}, 'config.json'),
                ({'drafts': 'five'}, 'whole numbers'), ({'port': '8894;id'}, 'whole numbers'),
            ]
            for replaced, message in cases:
                with self.subTest(**replaced):
                    run, calls = start(**replaced)
                    self.assertEqual(run.returncode, 2, run.stderr)
                    self.assertIn(message, run.stderr)
                    self.assertEqual(calls, [])
            run, calls = start('--world-size', '3')
            self.assertEqual((run.returncode, calls), (2, []))


if __name__ == '__main__':
    unittest.main()
