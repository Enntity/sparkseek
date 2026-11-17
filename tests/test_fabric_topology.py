# SPDX-License-Identifier: AGPL-3.0-only
"""Four Sparks: how they are cabled is read from their port inventories.

SparkGLM's tests/test_fabric_topology.py (AGPL-3.0-only) on documentation
addresses (RFC 5737)."""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fabric  # noqa: E402

A, B = 'rocep1s0f0 enp1s0f0np0', 'roceP2p1s0f0 enP2p1s0f0np0'
C, D = 'rocep1s0f1 enp1s0f1np1', 'roceP2p1s0f1 enP2p1s0f1np1'
# A ring of four, two rails on each link: 0-1 on .0/28 and .16/28, 1-2 on
# .32/28 and .48/28, 2-3 on .64/28 and .80/28, 3-0 on .96/28 and .112/28.
RING = [
    f'{A} 192.0.2.1/28\n{C} 192.0.2.98/28\n{B} 192.0.2.17/28\n{D} 192.0.2.114/28',
    f'{A} 192.0.2.2/28\n{C} 192.0.2.33/28\n{B} 192.0.2.18/28\n{D} 192.0.2.49/28',
    f'{A} 192.0.2.34/28\n{C} 192.0.2.65/28\n{B} 192.0.2.50/28\n{D} 192.0.2.81/28',
    f'{A} 192.0.2.97/28\n{C} 192.0.2.66/28\n{B} 192.0.2.113/28\n{D} 192.0.2.82/28',
]
F0, F1 = f'{A.split()[0]},{B.split()[0]}', f'{C.split()[0]},{D.split()[0]}'


def without(inventory, *ports):
    """The inventory without the lines of these ports (A, B, C, D)."""
    return '\n'.join(l for l in inventory.splitlines() if not any(l.startswith(p) for p in ports))


class Topology(unittest.TestCase):
    def test_a_closed_loop_is_a_ring_with_every_rank_on_both_sides(self):
        ranks = fabric.topology(RING)
        self.assertEqual({r['topology'] for r in ranks}, {'ring'})
        self.assertEqual([r['host'] for r in ranks], [0, 1, 2, 3])
        # Left is the link to the rank below (rank 3 for rank 0), right to the rank above.
        self.assertEqual([(r['left'], r['right']) for r in ranks],
                         [(F1, F0), (F0, F1), (F0, F1), (F1, F0)])

    def test_rails_on_an_edge_match_in_peer_order(self):
        ranks = fabric.topology(RING)
        # Rank 1's right rails reach rank 2's left rails device for device, by subnet.
        self.assertEqual(ranks[1]['right'].split(','), ['rocep1s0f1', 'roceP2p1s0f1'])
        self.assertEqual(ranks[2]['left'].split(','), ['rocep1s0f0', 'roceP2p1s0f0'])

    def test_the_first_host_leads_and_every_rank_gets_its_socket_interface(self):
        ranks = fabric.topology(RING)
        self.assertEqual({r['leader'] for r in ranks}, {'192.0.2.17'})
        self.assertEqual([r['interface'] for r in ranks],
                         ['enP2p1s0f0np0', 'enP2p1s0f0np0', 'enP2p1s0f1np1', 'enP2p1s0f1np1'])

    def test_rank_order_follows_the_cables_not_the_host_list(self):
        ranks = fabric.topology([RING[0], RING[2], RING[1], RING[3]])
        self.assertEqual([r['host'] for r in ranks], [0, 2, 1, 3])

    def test_an_open_loop_is_a_chain_with_empty_ends(self):
        chain = [without(RING[0], C, D), RING[1], RING[2], without(RING[3], A, B)]
        ranks = fabric.topology(chain)
        self.assertEqual({r['topology'] for r in ranks}, {'chain'})
        self.assertEqual((ranks[0]['left'], ranks[3]['right']), ('', ''))
        self.assertEqual((ranks[0]['right'], ranks[3]['left']), (F0, F1))

    def test_a_chain_must_start_at_an_end(self):
        chain = [RING[1], without(RING[0], C, D), RING[2], without(RING[3], A, B)]
        with self.assertRaisesRegex(ValueError, 'end of the chain'):
            fabric.topology(chain)

    def test_a_missing_link_is_refused(self):
        broken = [without(RING[0], A, B), without(RING[1], A, B, C, D), without(RING[2], A, B), RING[3]]
        with self.assertRaisesRegex(ValueError, 'neither a ring nor a chain'):
            fabric.topology(broken)

    def test_it_takes_four_sparks(self):
        with self.assertRaisesRegex(ValueError, 'four'):
            fabric.topology(RING[:2])

    def test_the_command_prints_one_line_a_rank_and_fails_cleanly(self):
        run = subprocess.run([sys.executable, str(ROOT/'fabric.py'), 'topology', *RING],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout.splitlines()[0],
                         f'0 0 ring enP2p1s0f0np0 192.0.2.17 {F1} {F0}')
        self.assertEqual(len(run.stdout.splitlines()), 4)
        for arguments in (['topology', *RING[:3]], ['topology', *RING[:3], 'dev0 eth0 not-an-address'], []):
            run = subprocess.run([sys.executable, str(ROOT/'fabric.py'), *arguments],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn('fabric: ', run.stderr)


if __name__ == '__main__':
    unittest.main()
