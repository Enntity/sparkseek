# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the launch contract: one rank's command line and environment.

In the style of SparkGLM's install/test_install.py (AGPL-3.0-only)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve  # noqa: E402

LEFT, RIGHT = 'rocep1s0f1,roceP2p1s0f1', 'rocep1s0f0,roceP2p1s0f0'
MODEL = '/models/deepseek-ai--DeepSeek-V4.1-Flash-1017'


def environment(**changes):
    env = {'NODE_RANK': '0', 'MASTER_ADDR': '192.0.2.17', 'FABRIC_INTERFACE': 'enP2p1s0f0np0',
           'FABRIC_LEFT_HCA': LEFT, 'FABRIC_RIGHT_HCA': RIGHT, 'MODEL_PATH': MODEL, 'PATH': '/usr/bin'}
    env.update(changes)
    return env


class LaunchContract(unittest.TestCase):
    def test_every_rank_gets_the_launch_that_ran_on_four_sparks(self):
        for rank in range(4):
            argv, env = serve.launch(environment(NODE_RANK=str(rank)))
            self.assertEqual(argv, [
                '/usr/local/bin/spark', 'serve', MODEL, '--port', str(8894 + rank), '--no-tui',
                '--model-name', 'sparkseek', '--rank', str(rank), '--world-size', '4',
                '--ep-size', '4', '--tp-size', '1', '--master-addr', '192.0.2.17',
                '--master-port', '29610', '--max-seq-len', '131072', '--max-num-seqs', '1',
                '--max-prefill-tokens', '8192', '--speculative', '--num-drafts', '5'])
            engine = {key: value for key, value in env.items() if key.startswith(('ATLAS_', 'NCCL_'))}
            self.assertEqual(engine, {
                'ATLAS_DSV41_ROWS': '8192', 'ATLAS_DSV41_PDL': '1', 'ATLAS_GLM_CMD_RDMA': '1',
                'ATLAS_PLE_FAULT_POOL': '1', 'ATLAS_DSV41_DRAFT': '1', 'ATLAS_NO_MTP_K_LADDER': '1',
                'ATLAS_RDMA_ALLREDUCE': '1', 'ATLAS_RDMA_ONESHOT': '1',
                'ATLAS_RDMA_ONESHOT_MAX': '8388544', 'ATLAS_RDMA_ONESHOT_SM_STAGE': '1',
                'ATLAS_RDMA_ONESHOT_STRIPE_MIN': '16384', 'ATLAS_RDMA_PAIR_CHAIN': '1',
                'ATLAS_RDMA_PAIR_SPIN_US': '30000', 'ATLAS_RDMA_TOPOLOGY': 'ring',
                'ATLAS_RDMA_LEFT_RAILS': LEFT, 'ATLAS_RDMA_RIGHT_RAILS': RIGHT,
                'NCCL_SOCKET_IFNAME': 'enP2p1s0f0np0', 'NCCL_NET': 'Socket', 'NCCL_IB_DISABLE': '1',
                'NCCL_DEBUG': 'WARN', 'NCCL_CUMEM_ENABLE': '0', 'NCCL_CUMEM_HOST_ENABLE': '0',
                'NCCL_NVLS_ENABLE': '0', 'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple'})
            self.assertEqual(env['PATH'], '/usr/bin')

    def test_one_sequence_at_a_time(self):
        argv, _ = serve.launch(environment())
        self.assertEqual(argv[argv.index('--max-num-seqs') + 1], '1')

    def test_no_drafts_leaves_the_draft_head_and_its_switches_out(self):
        # ATLAS_NO_MTP_K_LADDER counts when present, whatever its value.
        argv, env = serve.launch(environment(SPARKSEEK_DRAFTS='0', ATLAS_DSV41_DRAFT='1',
                                             ATLAS_NO_MTP_K_LADDER='0'))
        self.assertNotIn('--speculative', argv)
        self.assertNotIn('--num-drafts', argv)
        self.assertNotIn('ATLAS_DSV41_DRAFT', env)
        self.assertNotIn('ATLAS_NO_MTP_K_LADDER', env)
        self.assertEqual(argv[-2:], ['--max-prefill-tokens', '8192'])

    def test_options_reach_the_command_line_and_the_environment(self):
        argv, env = serve.launch(environment(
            NODE_RANK='2', SPARKSEEK_DRAFTS='3', SPARKSEEK_PREFILL_ROWS='16384',
            SPARKSEEK_CONTEXT='65536', SPARKSEEK_PORT='9000'))
        value = lambda flag: argv[argv.index(flag) + 1]
        self.assertEqual(value('--num-drafts'), '3')
        self.assertEqual(value('--max-prefill-tokens'), '16384')
        self.assertEqual(env['ATLAS_DSV41_ROWS'], '16384')
        self.assertEqual(value('--max-seq-len'), '65536')
        self.assertEqual(value('--port'), '9002')

    def test_engine_switches_from_the_surroundings_are_dropped_but_logging_passes(self):
        _, env = serve.launch(environment(ATLAS_DSV41_NO_GEMM='1', ATLAS_RDMA_TOPOLOGY='chain',
                                          ATLAS_LOG_LEVEL='debug'))
        self.assertNotIn('ATLAS_DSV41_NO_GEMM', env)
        self.assertEqual(env['ATLAS_RDMA_TOPOLOGY'], 'ring')
        self.assertEqual(env['ATLAS_LOG_LEVEL'], 'debug')

    def test_invalid_cluster_configuration_fails_before_process_launch(self):
        for key, value in [('NODE_RANK', '4'), ('NODE_RANK', ''), ('MASTER_ADDR', 'not-an-address'),
                           ('MASTER_PORT', '0'), ('MASTER_PORT', '70000'),
                           ('FABRIC_INTERFACE', 'eth0\nINJECTED=1'), ('FABRIC_INTERFACE', ''),
                           ('MODEL_PATH', 'relative'), ('MODEL_PATH', ''),
                           ('SERVED_MODEL_NAME', 'bad\nname')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                serve.launch(environment(**{key: value}))

    def test_a_ring_rank_needs_distinct_devices_on_both_sides(self):
        for changes in ({'FABRIC_LEFT_HCA': ''}, {'FABRIC_RIGHT_HCA': ''},
                        {'FABRIC_LEFT_HCA': RIGHT}, {'FABRIC_LEFT_HCA': 'rocep1s0f1,rocep1s0f1'},
                        {'FABRIC_RIGHT_HCA': 'rocep1s0f0;true'}, {'FABRIC_RIGHT_HCA': 'rocep1s0f0,'},
                        {'FABRIC_LEFT_HCA': 'rocep1s0f1,rocep1s0f0'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                serve.launch(environment(**changes))


class Options(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(serve.options({}), {'sparks': 4, 'drafts': 5, 'prefill_rows': 8192,
                                             'context': 131072, 'port': 8894})
        # An empty value is the default, as an unset one.
        self.assertEqual(serve.options({'SPARKSEEK_DRAFTS': ''})['drafts'], 5)

    def test_three_sparks_are_refused_with_the_reason(self):
        with self.assertRaises(ValueError) as refusal:
            serve.options({'SPARKSEEK_SPARKS': '3'})
        message = str(refusal.exception)
        self.assertIn('three Sparks are not served yet', message)
        self.assertIn('split three ways', message)
        self.assertIn('four ways only', message)

    def test_only_four_sparks_serve(self):
        for sparks in ('0', '1', '2', '5', '8'):
            with self.subTest(sparks=sparks), self.assertRaisesRegex(ValueError, '--sparks must be 4'):
                serve.options({'SPARKSEEK_SPARKS': sparks})

    def test_ranges(self):
        for key, good, bad in [
                ('SPARKSEEK_DRAFTS', ('0', '1', '15'), ('16', '-1', 'five', '5.0', '05')),
                ('SPARKSEEK_PREFILL_ROWS', ('8192', '16384'), ('0', '4096', '8193', '32768')),
                ('SPARKSEEK_CONTEXT', ('1', '34000', '1048576'), ('0', '1048577', '128k')),
                ('SPARKSEEK_PORT', ('1024', '8894', '65532'), ('80', '65533', '8894;id'))]:
            for value in good:
                with self.subTest(key=key, value=value):
                    serve.options({key: value})
            for value in bad:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    serve.options({key: value})

    def test_check_command_reports_without_starting_anything(self):
        def check(**options):
            env = {'PATH': os.environ['PATH'], **options}
            return subprocess.run([sys.executable, str(HERE/'serve.py'), '--check'], env=env,
                                  capture_output=True, text=True)
        self.assertEqual(check().returncode, 0)
        self.assertEqual(check(SPARKSEEK_DRAFTS='0', SPARKSEEK_PREFILL_ROWS='16384').returncode, 0)
        refused = check(SPARKSEEK_SPARKS='3')
        self.assertEqual(refused.returncode, 2)
        self.assertIn('sparkseek: three Sparks are not served yet', refused.stderr)
        self.assertEqual(refused.stdout, '')
        self.assertEqual(check(SPARKSEEK_PREFILL_ROWS='4096').returncode, 2)

    def test_a_missing_checkpoint_stops_the_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = environment(MODEL_PATH=tmp, PATH=os.environ['PATH'])
            run = subprocess.run([sys.executable, str(HERE/'serve.py')], env=env,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn('config.json is missing', run.stderr)


class BuildInputs(unittest.TestCase):
    def test_the_image_builds_the_pinned_kernel_target(self):
        pin = json.loads((HERE/'atlas-source.json').read_text())
        dockerfile = (HERE/'Dockerfile').read_text()
        self.assertIn(f"ENV ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL={pin['kernel_target']} "
                      'ATLAS_TARGET_QUANT=nvfp4\n', dockerfile)
        self.assertIn('cargo build --locked --release -j "$BUILD_JOBS" -p spark-server', dockerfile)
        self.assertIn('install -D target/release/spark /build/dist/spark', dockerfile)
        self.assertIn('ENTRYPOINT ["python3", "/opt/atlas/serve.py"]', dockerfile)
        # Everything the Dockerfile copies from install/ is there.
        for line in dockerfile.splitlines():
            if line.startswith('COPY install/'):
                for source in line.split()[1:-1]:
                    self.assertTrue((HERE.parent/source).exists(), source)

    def test_scripts_start_sh_runs_directly_are_executable(self):
        for script in (HERE.parent/'start.sh', HERE/'build.sh', HERE/'start-node.sh'):
            self.assertTrue(os.access(script, os.X_OK), script.name)

    def test_the_build_refuses_any_other_engine_tree(self):
        build = (HERE/'build.sh').read_text()
        self.assertIn('rev-parse HEAD) == "$commit"', build)
        self.assertIn("rev-parse 'HEAD^{tree}') == \"$pin_tree\"", build)
        self.assertIn('fetched engine is not the pinned', build)


if __name__ == '__main__':
    unittest.main()
