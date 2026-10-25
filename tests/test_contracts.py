import unittest
from dataclasses import replace

import numpy as np

from prairie_fire import ReLUNetwork, compile_contract, expand_contract, judge, summarize
from prairie_fire.experiments import compressed_checks, multilayer_checks, strict_parallel_checks


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.network = ReLUNetwork((np.eye(2),), (np.zeros(2),))
        self.contract = compile_contract(self.network, [0., 0.], [-1., 0.], [2], rank=1)

    def test_projection_preserves_base_gradient(self):
        rng = np.random.default_rng(31)
        network = ReLUNetwork((rng.normal(size=(10, 7)),), (rng.normal(size=10),))
        contract = compile_contract(network, rng.normal(size=7), rng.normal(size=10), [3, 4], rank=2)
        error = contract.response.output - contract.target
        start = 0
        for node in contract.nodes:
            end = start + node.base_features.size
            np.testing.assert_allclose(node.projection.T @ contract.projected_residual,
                                       contract.response.jacobian[:, start:end].T @ error,
                                       atol=1e-11, rtol=1e-11)
            start = end

    def test_discarded_response_prevents_false_acceptance(self):
        decision = judge(self.contract, [summarize(self.contract.nodes[0], [-.1, 1.])])
        # tau * ||delta|| bounds the discarded response; it is not its exact norm.
        self.assertAlmostEqual(decision.loss_upper_bound, .910)
        self.assertGreaterEqual(decision.loss_upper_bound, .905)
        self.assertFalse(decision.accepted)
        projected_only_loss = .5 * .9 ** 2
        self.assertLess(projected_only_loss, self.contract.base_loss)

    def test_zero_update_is_not_strict_descent(self):
        self.assertFalse(judge(self.contract, [summarize(self.contract.nodes[0], [0., 0.])]).accepted)

    def test_complete_output_space_has_no_tail(self):
        contract = compile_contract(self.network, [0., 0.], [-1., 0.], [2], rank=2)
        self.assertLess(contract.nodes[0].tail_bound, 1e-12)
        delta = np.array([-.1, .2])
        decision = judge(contract, [summarize(contract.nodes[0], delta)])
        self.assertAlmostEqual(decision.loss_upper_bound, .5 * np.sum((np.array([1., 0.]) + delta) ** 2))
        self.assertTrue(decision.accepted)

    def test_zero_residual_is_supported(self):
        contract = compile_contract(self.network, [0., 0.], [0., 0.], [2], rank=1)
        self.assertFalse(judge(contract, [summarize(contract.nodes[0], [0., 0.])]).accepted)

    def test_basis_must_preserve_residual(self):
        with self.assertRaisesRegex(ValueError, "full base loss residual"):
            compile_contract(self.network, [0., 0.], [-1., 0.], [2], basis=np.array([[0.], [1.]]))

    def test_invalid_and_nonfinite_inputs_rejected(self):
        for bad in ([float("nan"), 0.], [float("inf"), 0.]):
            with self.assertRaises(ValueError):
                summarize(self.contract.nodes[0], bad)
        with self.assertRaises(ValueError):
            compile_contract(self.network, [0., 0.], [-1., 0.], [1], rank=1)
        with self.assertRaises(ValueError):
            summarize(self.contract.nodes[0], [1.])

    def test_stale_anchor_is_rejected(self):
        old = summarize(self.contract.nodes[0], [-.1, 0.])
        fresh = compile_contract(self.network, [.1, 0.], [-1., 0.], [2], rank=1)
        with self.assertRaisesRegex(ValueError, "stale"):
            judge(fresh, [old])

    def test_changed_model_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "version changed"):
            judge(self.contract, [summarize(self.contract.nodes[0], [-.1, 0.])], model_version="changed")

    def test_missing_or_duplicate_nodes_rejected(self):
        contract = compile_contract(self.network, [0., 0.], [-1., 0.], [1, 1], rank=1)
        first = summarize(contract.nodes[0], [-.1])
        with self.assertRaises(ValueError):
            judge(contract, [first])
        with self.assertRaises(ValueError):
            judge(contract, [first, first])

    def test_negative_or_nonfinite_summary_bounds_rejected(self):
        valid = summarize(self.contract.nodes[0], [-.1, 0.])
        for value in [-1., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                judge(self.contract, [replace(valid, tail_bound=value)])

    def test_current_loss_controls_reused_contract(self):
        summary = summarize(self.contract.nodes[0], [-.1, 0.])
        self.assertTrue(judge(self.contract, [summary]).accepted)
        self.assertFalse(judge(self.contract, [summary], current_loss=.1).accepted)

    def test_basis_expansion_reduces_tail_and_changes_epoch_id(self):
        expanded = expand_contract(self.contract, self.network, [0., 1.])
        self.assertEqual(expanded.rank, 2)
        self.assertLess(expanded.nodes[0].tail_bound, self.contract.nodes[0].tail_bound)
        self.assertNotEqual(expanded.anchor_id, self.contract.anchor_id)
        with self.assertRaises(ValueError):
            judge(expanded, [summarize(self.contract.nodes[0], [-.1, 0.])])
        self.assertIs(expand_contract(self.contract, self.network, [1., 0.]), self.contract)

    def test_compilation_owns_weight_storage(self):
        weights = np.eye(2)
        network = ReLUNetwork((weights,), (np.zeros(2),))
        version = network.version
        weights[:] = 100.
        self.assertEqual(network.version, version)
        np.testing.assert_array_equal(network.forward([1., 2.]), [1., 2.])
        self.assertFalse(network.weights[0].flags.writeable)

    def test_near_boundary_and_random_updates(self):
        result = compressed_checks(seed=20261025, cases=3, trials=8)
        self.assertEqual(result["bound_failures"], 0)
        self.assertEqual(result["gate_crossing_candidates_accepted"], 3)
        self.assertEqual(result["accepted_crossings_with_actual_descent"], 3)

    def test_multilayer_cascaded_switches(self):
        self.assertEqual(multilayer_checks(seed=6621, cases=4, trials=15)["failures"], 0)

    def test_previous_parallel_bound_baseline(self):
        self.assertEqual(strict_parallel_checks(seed=17, cases=20)["failures"], 0)


if __name__ == "__main__":
    unittest.main()
