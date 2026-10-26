import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from prairie_fire import ReLUNetwork
from prairie_fire.networks import frozen_array
from prairie_fire.replay import FeatureMenu, ReplayTrainer, Selection, select_menu
from prairie_fire.training import TrainingConfig, make_problem


class ReplayTests(unittest.TestCase):
    def trainer(self, *, steps=4, gate=False):
        return ReplayTrainer(*make_problem(17, samples=3),
                             TrainingConfig(local_steps=steps, max_rounds=6, fusion_interval=2, target_loss=1e-9),
                             gate_menu=gate)

    def conflicting_proposal(self, trainer, proposal, *, harmful_node, improving_node):
        """Craft endpoints that jointly conflict while a rollback mix repairs.

        One node gets a large harmful feature change, one node gets a small
        true-gradient improvement, the others keep their committed features.
        The joint endpoint is harmful, but rolling the harmful node back keeps
        the improvement, so the restricted candidate set already contains an
        accepted combination.
        """
        features = trainer.features
        z = np.concatenate(features, axis=1)
        error = trainer.fusion.forward_batch(z) - trainer.batch.targets
        dz, _, _ = trainer.fusion.vjp_batch(z, error / len(z))
        starts = np.cumsum([0] + [f.shape[1] for f in features])
        rng = np.random.default_rng(5)
        crafted = {
            harmful_node: frozen_array(features[harmful_node] + 3. * rng.normal(size=features[harmful_node].shape)),
            improving_node: frozen_array(features[improving_node]
                                         - 1e-2 * dz[:, starts[improving_node]:starts[improving_node + 1]]),
        }
        menus = [FeatureMenu((0, menu.steps[-1]),
                             (trainer.features[i], crafted.get(i, frozen_array(features[i]))),
                             (trainer.branches[i].version, proposal.final_branches[i].version))
                 for i, menu in enumerate(proposal.menus)]
        return replace(proposal, menus=tuple(menus))

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

    def test_gate_repairs_conflict_with_uploaded_candidates_only(self):
        trainer = self.trainer(gate=True)
        proposal = trainer.propose_replay()
        crafted = self.conflicting_proposal(trainer, proposal, harmful_node=0, improving_node=1)
        before_bytes = trainer.ledger.report()["total_logical_bytes"]
        selection = trainer.choose(crafted)
        self.assertLess(selection.loss, trainer.loss)
        self.assertEqual(selection.indices, (0, 1, 0, 0))
        self.assertNotIn("replay_menu_request", trainer.ledger.messages)
        self.assertNotIn("replay_extra_menus", trainer.ledger.messages)
        self.assertEqual(trainer.ledger.work["gate_accepted_without_menu"], 1)
        self.assertEqual(trainer.ledger.work["gate_menu_requests"], 0)
        # A gate pass must not add any wire bytes beyond the proposal upload.
        self.assertEqual(trainer.ledger.report()["total_logical_bytes"], before_bytes)

    def stalled_proposal(self, trainer, proposal):
        """Every checkpoint of every menu equals the committed state, modeling
        a window in which no branch accepted a single local step."""
        menus = [FeatureMenu(menu.steps,
                             tuple(trainer.features[i] for _ in menu.steps),
                             tuple(trainer.branches[i].version for _ in menu.steps))
                 for i, menu in enumerate(proposal.menus)]
        return replace(proposal, menus=tuple(menus))

    def test_stalled_conflict_skips_menu_request_in_gate_arm(self):
        trainer = self.trainer(gate=True)
        proposal = trainer.propose_replay()
        selection = trainer.choose(self.stalled_proposal(trainer, proposal))
        self.assertEqual(selection.indices, (0, 0, 0, 0))
        self.assertEqual(selection.loss, trainer.loss)
        self.assertEqual(trainer.ledger.work["stall_conflicts"], 1)
        self.assertEqual(trainer.ledger.work["gate_accepted_without_menu"], 0)
        self.assertEqual(trainer.ledger.work["gate_menu_requests"], 0)
        self.assertNotIn("replay_menu_request", trainer.ledger.messages)
        self.assertNotIn("replay_extra_menus", trainer.ledger.messages)

    def test_reference_arm_still_requests_menus_on_stall(self):
        trainer = self.trainer()
        proposal = trainer.propose_replay()
        trainer.choose(self.stalled_proposal(trainer, proposal))
        self.assertEqual(trainer.ledger.work["stall_conflicts"], 1)
        self.assertIn("replay_menu_request", trainer.ledger.messages)
        self.assertGreater(trainer.ledger.payload["replay_extra_menus"], 0)

    def test_gate_requests_menus_when_uploaded_candidates_all_fail(self):
        trainer = self.trainer(gate=True)
        proposal = trainer.propose_replay()
        # Every endpoint is harmful and every restricted mix therefore fails,
        # but the real intermediate checkpoint stays available on the nodes.
        rng = np.random.default_rng(11)
        menus = [FeatureMenu((0, menu.steps[1], menu.steps[-1]),
                             (trainer.features[i],
                              menu.features[1],
                              frozen_array(trainer.features[i] + 3. * rng.normal(size=trainer.features[i].shape))),
                             (menu.parameter_versions[0], menu.parameter_versions[1],
                              proposal.final_branches[i].version))
                 for i, menu in enumerate(proposal.menus)]
        crafted = replace(proposal, menus=tuple(menus))
        trainer.choose(crafted)
        self.assertEqual(trainer.ledger.work["stall_conflicts"], 0)
        self.assertEqual(trainer.ledger.work["gate_accepted_without_menu"], 0)
        self.assertEqual(trainer.ledger.work["gate_menu_requests"], 1)
        self.assertIn("replay_menu_request", trainer.ledger.messages)
        self.assertGreater(trainer.ledger.payload["replay_extra_menus"], 0)
        self.assertEqual(trainer.uploaded_menu_epoch, crafted.epoch_id)

    def test_gate_is_byte_neutral_when_no_conflict_occurs(self):
        batch, branches, fusion = make_problem(17)
        config = TrainingConfig(local_steps=16, max_rounds=6, fusion_interval=2, target_loss=1e-9)
        ungated = ReplayTrainer(batch, branches, fusion, config, seed=17).run()
        # Guard the premise: the gated comparison below is only exact while
        # this arm never hits a joint-endpoint conflict.
        self.assertEqual(ungated["ledger"]["work"].get("joint_endpoint_conflicts", 0), 0)
        gated = ReplayTrainer(batch, branches, fusion, config, gate_menu=True, seed=17).run()
        self.assertEqual(gated["final_loss"], ungated["final_loss"])
        self.assertEqual(gated["ledger"]["total_logical_bytes"], ungated["ledger"]["total_logical_bytes"])
        self.assertEqual(gated["accepted_loss_increases"], 0)
        self.assertEqual(gated["max_replay_feature_error"], 0.)
        self.assertEqual(gated["ledger"]["work"].get("gate_accepted_without_menu", 0), 0)
        self.assertEqual(gated["ledger"]["work"].get("gate_menu_requests", 0), 0)


if __name__ == "__main__":
    unittest.main()
