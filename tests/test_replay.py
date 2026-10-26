import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from prairie_fire import ReLUNetwork
from prairie_fire.replay import FeatureMenu, ReplayTrainer, Selection, select_menu
from prairie_fire.training import TrainingConfig, make_problem


class ReplayTests(unittest.TestCase):
    def trainer(self, *, steps=4):
        return ReplayTrainer(*make_problem(17, samples=3),
                             TrainingConfig(local_steps=steps, max_rounds=6, fusion_interval=2, target_loss=1e-9))

    def test_capsule_prediction_and_gradient_are_exact_across_gate_changes(self):
        trainer = self.trainer()
        capsules = trainer.capsules()
        rng = np.random.default_rng(919)
        for node_id, capsule in enumerate(capsules):
            for scale in (.01, 2., 20.):
                candidate = trainer.features[node_id] + scale * rng.normal(size=trainer.features[node_id].shape)
                features = list(trainer.features)
                features[node_id] = candidate
                z = np.concatenate(features, axis=1)
                predicted = trainer.fusion.forward_batch(z)
                np.testing.assert_allclose(capsule.prediction(candidate), predicted, rtol=1e-12, atol=1e-12)
                dz, _, _ = trainer.fusion.vjp_batch(z, (predicted - trainer.batch.targets) / len(z))
                np.testing.assert_allclose(capsule.feature_gradient(candidate), dz[:, node_id * 3:(node_id + 1) * 3],
                                           rtol=1e-11, atol=1e-12)

    def test_middle_prefix_is_replayed_exactly_and_committed_atomically(self):
        trainer = self.trainer()
        proposal = trainer.propose_replay()
        before = tuple(n.version for n in trainer.branches)
        index = 1
        features = tuple(m.features[index] for m in proposal.menus)
        selection = Selection(proposal.epoch_id, (index,) * 4, trainer._loss(features))
        self.assertLess(selection.loss, trainer.loss)
        self.assertTrue(trainer.commit_replay(proposal, selection))
        self.assertEqual(tuple(n.version for n in trainer.branches), tuple(m.parameter_versions[index] for m in proposal.menus))
        self.assertNotEqual(before, tuple(n.version for n in trainer.branches))
        self.assertEqual(trainer.max_replay_feature_error, 0.)
        self.assertEqual(trainer.ledger.work['replayed_node_steps'], 4 * proposal.menus[0].steps[index])

    def test_failed_replay_keeps_all_original_parameters(self):
        trainer = self.trainer()
        proposal = trainer.propose_replay()
        before = tuple(n.version for n in trainer.branches)
        before_features = trainer.features
        first = proposal.menus[1]
        versions = list(first.parameter_versions)
        versions[1] = "invalid"
        menus = list(proposal.menus)
        menus[1] = replace(first, parameter_versions=tuple(versions))
        corrupt = replace(proposal, menus=tuple(menus))
        selection = Selection(proposal.epoch_id, (1,) * 4, trainer._loss(tuple(m.features[1] for m in proposal.menus)))
        with self.assertRaisesRegex(ArithmeticError, "replay mismatch"):
            trainer.commit_replay(corrupt, selection)
        self.assertEqual(before, tuple(n.version for n in trainer.branches))
        self.assertIs(before_features, trainer.features)
        self.assertNotIn("replay_commit", trainer.ledger.messages)

    def test_old_data_model_and_parent_cannot_be_committed(self):
        for change in ("data", "fusion", "parent", "epoch"):
            trainer = self.trainer()
            proposal = trainer.propose_replay()
            if change == "data":
                trainer.batch = replace(trainer.batch, targets=trainer.batch.targets + .1)
            elif change == "fusion":
                self.assertTrue(trainer.train_fusion())
            elif change == "parent":
                trainer.branches = proposal.final_branches
            else:
                trainer.propose_replay()
            with self.assertRaises(ValueError):
                trainer.choose(proposal)

    def test_smooth_case_only_uploads_endpoint_and_caches_suffix(self):
        trainer = self.trainer(steps=16)
        proposal = trainer.propose_replay()
        selection = trainer.choose(proposal)
        self.assertEqual(selection.indices, tuple(len(m.steps) - 1 for m in proposal.menus))
        self.assertNotIn("replay_extra_menus", trainer.ledger.messages)
        self.assertEqual(trainer.ledger.payload["replay_endpoints"], sum(m.features[-1].nbytes + 40 for m in proposal.menus))
        trainer.commit_replay(proposal, selection)
        suffix_payload = trainer.ledger.payload["replay_suffix_distribution"]
        trainer.propose_replay()
        self.assertEqual(trainer.ledger.payload["replay_suffix_distribution"], suffix_payload)
        self.assertGreater(trainer.ledger.payload["replay_boundary_distribution"], 0)

    def test_training_never_calls_dense_jacobian_and_descends(self):
        trainer = self.trainer()
        with patch.object(ReLUNetwork, "response_at", side_effect=AssertionError("Jacobian called")):
            result = trainer.run()
        self.assertLess(result["final_loss"], result["initial_loss"])
        self.assertEqual(result["accepted_loss_increases"], 0)
        self.assertEqual(result["max_replay_feature_error"], 0.)
        self.assertEqual(result["ledger"]["work"].get("dense_response_oracle_samples", 0), 0)
        self.assertGreater(result["ledger"]["by_category"]["replay_label_distribution"]["payload_bytes"], 0)

    def test_joint_middle_prefix_repairs_conflict_better_than_any_endpoint_subset(self):
        # Both endpoint proposals are individually beneficial, but together they
        # overshoot. Simultaneous half-steps fit exactly; every endpoint subset
        # has loss at least 1/4. Hidden gates stay positive for all candidates.
        fusion = ReLUNetwork((np.concatenate((np.eye(2), np.eye(2)), axis=1), np.eye(2)),
                              (np.full(2, 10.), np.full(2, -9.)))
        directions = (np.array([[-1.5, -.5]]), np.array([[-.5, -1.5]]))
        menus = tuple(FeatureMenu((0, 1, 2), (np.zeros((1, 2)), d / 2, d), ("zero", "middle", "last")) for d in directions)

        def evaluate(indices, options=menus):
            prediction = fusion.forward_batch(np.concatenate([m.features[i] for m, i in zip(options, indices)], axis=1))
            return float(np.sum(prediction ** 2) / 2)

        endpoints = tuple(FeatureMenu((0, 2), (m.features[0], m.features[2]), ("zero", "last")) for m in menus)
        _, endpoint_loss = select_menu(endpoints, lambda indices: evaluate(indices, endpoints), 1.)
        selected, menu_loss = select_menu(menus, evaluate, 1.)
        self.assertAlmostEqual(endpoint_loss, .25)
        self.assertAlmostEqual(menu_loss, 0.)
        self.assertEqual(selected, (1, 1))


if __name__ == "__main__":
    unittest.main()
