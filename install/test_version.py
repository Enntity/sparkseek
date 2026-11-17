# SPDX-License-Identifier: AGPL-3.0-only
"""Version queries must work before configuration or model initialization.

Adapted from SparkGLM's install/test_version.py (AGPL-3.0-only)."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class VersionQueries(unittest.TestCase):
    def test_engine_pin_is_the_four_spark_engine(self):
        pin = json.loads((HERE/'atlas-source.json').read_text())
        self.assertEqual(pin['repository'], 'https://github.com/Enntity/atlas.git')
        self.assertEqual(pin['branch'], 'upstream/deepseek-v41-flash-20261114')
        self.assertEqual(pin['kernel_target'], 'deepseek-v4.1-flash')
        self.assertEqual(pin['commit'], '8c71b62ee67ad1b470c90e381cbce81967a687c5')
        self.assertEqual(pin['tree'], '4d114aa8ed5c84b17d8fbe47084464af4b64e400')
        self.assertIn('Three Sparks are not served yet', pin['note'])

    def test_the_served_checkpoint_is_the_release_as_published(self):
        pin = json.loads((HERE/'checkpoints.json').read_text())['checkpoints']['deepseek-fp8-mxfp4']
        self.assertEqual(pin['repo'], 'deepseek-ai/DeepSeek-V4.1-Flash-1017')
        self.assertEqual(pin['revision'], '1bae0e1088993b9b82de08f59de5be5a34e42170')
        self.assertEqual(pin['world'], [4])

    def test_checkout_and_image_entrypoint_report_pin_without_configuration(self):
        release = json.loads((HERE/'release.json').read_text())
        pin = json.loads((HERE/'atlas-source.json').read_text())
        self.assertEqual(release['product'], 'SparkSeek')
        self.assertRegex(release['version'], r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'install').mkdir()
            shutil.copy(HERE.parent/'start.sh', root/'start.sh')
            for name in ('serve.py', 'release.json', 'atlas-source.json'):
                shutil.copy(HERE/name, root/'install'/name)
            # The version path must not even source a user's environment file.
            (root/'.env').write_text('exit 97\n')
            env = {'PATH': os.environ['PATH']}
            checkout = subprocess.check_output(['bash', str(root/'start.sh'), '--version'], env=env, text=True)
            (root/'install'/'atlas-source.json').rename(root/'install'/'source-manifest.json')
            image = subprocess.check_output(['python3', str(root/'install'/'serve.py'), '--version'], env=env, text=True)
            self.assertEqual(checkout, image)
            self.assertIn(f"{release['product']} {release['version']}\n", image)
            self.assertIn(pin['commit'], image)
            self.assertIn(pin['tree'], image)


if __name__ == '__main__':
    unittest.main()
