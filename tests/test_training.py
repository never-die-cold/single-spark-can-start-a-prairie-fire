import unittest
from dataclasses import replace

import numpy as np

from prairie_fire import ReLUNetwork, compile_contract, judge, summarize
from prairie_fire.training import FixedBatch, Trainer, TrainingConfig, make_problem


class BackpropTests(unittest.TestCase):
    def test_batch_vjp_matches_directional_finite_difference(self):
        rng = np.random.default_rng(45)
        weights = (rng.normal(size=(4, 2)), rng.normal(size=(3, 4)))
        biases = (np.full(4, .7), rng.normal(size=3))
        net = ReLUNetwork(weights, biases)
        x, upstream = rng.normal(size=(5, 2)), rng.normal(size=(5, 3))
        dx, dw, db = net.vjp_batch(x, upstream)
        directions = tuple(rng.normal(size=w.shape) for w in weights)
        bias_directions = tuple(rng.normal(size=b.shape) for b in biases)
        input_direction = rng.normal(size=x.shape)
        epsilon = 1e-6

        def perturbed(sign):
            model = ReLUNetwork(tuple(w + sign * epsilon * d for w, d in zip(weights, directions)),
                                tuple(b + sign * epsilon * d for b, d in zip(biases, bias_directions)))
            return np.sum(model.forward_batch(x + sign * epsilon * input_direction) * upstream)

        actual = (perturbed(1) - perturbed(-1)) / (2 * epsilon)
        predicted = np.sum(dx * input_direction)
        predicted += sum(np.sum(g * d) for g, d in zip(dw, directions))
        predicted += sum(np.sum(g * d) for g, d in zip(db, bias_directions))
        self.assertAlmostEqual(actual, predicted, places=7)
        np.testing.assert_allclose(net.forward_batch(x), np.array([net.forward(row) for row in x]))

    def test_loss_anchored_response_preserves_branch_parameter_gradient(self):
        batch, branches, fusion = make_problem(31, samples=3)
        features = tuple(n.forward_batch(x) for n, x in zip(branches, batch.inputs))
        z = np.concatenate(features, axis=1)
        error = fusion.forward_batch(z) - batch.targets
        exact_dz, _, _ = fusion.vjp_batch(z, error / len(z))
        start = 0
        for node_id, (branch, inputs, feature) in enumerate(zip(branches, batch.inputs, features)):
            compressed = []
            for j in range(len(z)):
                contract = compile_contract(fusion, z[j], batch.targets[j], [3] * 4, rank=1)
                node = contract.nodes[node_id]
                compressed.append(node.projection.T @ contract.projected_residual / len(z))
            _, expected_dw, expected_db = branch.vjp_batch(inputs, exact_dz[:, start:start + feature.shape[1]])
            _, actual_dw, actual_db = branch.vjp_batch(inputs, compressed)
            for a, b in zip((*actual_dw, *actual_db), (*expected_dw, *expected_db)):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-11)
            start += feature.shape[1]

    def test_batch_shapes_and_nonfinite_cotangents_are_rejected(self):
        net = ReLUNetwork((np.eye(2),), (np.zeros(2),))
        for inputs in ([], [[1.]], [1., 2.]):
            with self.assertRaises(ValueError):
                net.forward_batch(inputs)
        with self.assertRaises(ValueError):
            net.vjp_batch([[1., 2.]], [[np.nan, 0.]])
        with self.assertRaises(ValueError):
            net.vjp_batch([[1., 2.]], [[1.]])


class TransactionTests(unittest.TestCase):
    def trainer(self, **kwargs):
        inputs = (np.array([[1.], [2.]]),) * 2
        batch = FixedBatch(inputs, np.zeros((2, 2)), ("a", "b"))
        branches = tuple(ReLUNetwork((np.ones((1, 1)),), (np.array([.2]),)) for _ in range(2))
        fusion = ReLUNetwork((np.eye(2),), (np.zeros(2),))
        config = TrainingConfig(local_steps=1, initial_rank=2, max_rounds=4, fusion_interval=2, target_loss=1e-9)
        return Trainer(batch, branches, fusion, config, method="dense", **kwargs)

    def test_rejection_is_atomic_and_retry_starts_at_committed_snapshot(self):
        trainer = self.trainer()
        before = tuple(b.version for b in trainer.branches)
        features = trainer.features
        loss = trainer.loss
        rejected = trainer.propose(100.)
        decision = trainer.decide(rejected)
        self.assertFalse(decision.accepted)
        with self.assertRaisesRegex(ValueError, "accepted decision"):
            trainer.commit(rejected, decision)
        self.assertEqual(before, tuple(b.version for b in trainer.branches))
        self.assertIs(features, trainer.features)
        self.assertEqual(loss, trainer.loss)
        good = trainer.propose(.01)
        self.assertEqual(good.parent_parameter_versions, before)
        trainer.commit(good, trainer.decide(good))
        self.assertLess(trainer.loss, loss)
        self.assertTrue(all(a != b.version for a, b in zip(before, trainer.branches)))
        self.assertLessEqual(trainer.max_bound_violation, 1e-12)

    def test_decision_is_bound_to_candidate_not_just_parent(self):
        trainer = self.trainer()
        first, second = trainer.propose(.01), trainer.propose(.02)
        decision = trainer.decide(first)
        self.assertTrue(decision.accepted)
        with self.assertRaisesRegex(ValueError, "does not match"):
            trainer.commit(second, decision)

    def test_commit_and_refresh_invalidate_pending_proposals(self):
        trainer = self.trainer()
        old = trainer.propose(.01)
        trainer.commit(old, trainer.decide(old))
        with self.assertRaisesRegex(ValueError, "snapshot"):
            trainer.decide(old)
        pending = trainer.propose(.01)
        trainer.refresh()
        with self.assertRaisesRegex(ValueError, "epoch"):
            trainer.decide(pending)

    def test_data_labels_order_and_preprocessing_bind_epoch(self):
        for change in ("labels", "order", "preprocessing"):
            trainer = self.trainer()
            pending = trainer.propose(.01)
            if change == "labels":
                trainer.batch = replace(trainer.batch, targets=trainer.batch.targets + 1.)
            elif change == "order":
                trainer.batch = replace(trainer.batch, sample_ids=("b", "a"))
            else:
                trainer.batch = replace(trainer.batch, preprocessing_id="new")
            with self.assertRaisesRegex(ValueError, "changed"):
                trainer.decide(pending)

    def test_fusion_update_invalidates_contract_and_keeps_branch_snapshot(self):
        trainer = self.trainer()
        pending = trainer.propose(.01)
        before = tuple(b.version for b in trainer.branches)
        version = trainer.fusion.version
        self.assertTrue(trainer.train_fusion())
        self.assertIsNone(trainer.contract)
        self.assertNotEqual(version, trainer.fusion.version)
        self.assertEqual(before, tuple(b.version for b in trainer.branches))
        with self.assertRaisesRegex(ValueError, "epoch"):
            trainer.decide(pending)

    def test_reused_contract_compares_against_actual_current_batch_loss(self):
        trainer = self.trainer()
        initial = trainer.loss
        first = trainer.propose(.01)
        trainer.commit(first, trainer.decide(first))
        second = trainer.propose(.01)
        decision = trainer.decide(second)
        self.assertLess(decision.current_loss, initial)
        self.assertEqual(decision.current_loss, trainer.loss)
        self.assertTrue(decision.accepted)
        trainer.commit(second, decision)
        self.assertEqual(max(trainer.commits_by_contract.values()), 2)

    def test_ledger_charges_rejected_work_and_metadata_without_commit_features(self):
        trainer = self.trainer()
        proposal = trainer.propose(100.)
        trainer.decide(proposal)
        ledger = trainer.ledger.report()
        self.assertEqual(ledger["work"]["rejected_proposals"], 1)
        self.assertGreater(ledger["work"]["branch_vjp_samples_including_forward"], 0)
        self.assertNotIn("committed_features", ledger["by_category"])
        self.assertEqual(ledger["total_logical_bytes"], ledger["payload_bytes"] + ledger["metadata_bytes"])
        for item in ledger["by_category"].values():
            self.assertEqual(item["metadata_bytes"], 128 * item["messages"])


class TrainingSafetyTests(unittest.TestCase):
    def test_batch_acceptance_allows_individual_sample_loss_increase(self):
        batch = FixedBatch((np.array([[1.], [2.]]),), np.array([[0.], [3.]]), ("a", "b"))
        branch = ReLUNetwork((np.ones((1, 1)),), (np.zeros(1),))
        fusion = ReLUNetwork((np.ones((1, 1)),), (np.zeros(1),))
        trainer = Trainer(batch, (branch,), fusion, TrainingConfig(local_steps=1), method="dense")
        proposal = trainer.propose(.1)
        decision = trainer.decide(proposal)
        self.assertTrue(decision.accepted)
        before_first = (trainer.features[0][0, 0] - batch.targets[0, 0]) ** 2 / 2
        after_first = (proposal.features[0][0, 0] - batch.targets[0, 0]) ** 2 / 2
        self.assertGreater(after_first, before_first)
        initial = trainer.loss
        trainer.commit(proposal, decision)
        self.assertLess(trainer.loss, initial)

    def test_real_parameter_training_and_fusion_have_monotone_loss(self):
        for method in ("backprop", "dense", "compressed"):
            batch, branches, fusion = make_problem(17, samples=3)
            config = TrainingConfig(max_rounds=6, local_steps=2, fusion_interval=3, target_loss=1e-9)
            trainer = Trainer(batch, branches, fusion, config, method=method)
            result = trainer.run()
            self.assertLess(result["final_loss"], result["initial_loss"])
            self.assertEqual(result["accepted_loss_increases"], 0)
            self.assertLessEqual(result["max_accepted_actual_minus_bound"], 1e-10)
            self.assertGreater(result["ledger"]["work"].get("fusion_updates", 0), 0)
            self.assertNotEqual(tuple(b.version for b in branches), tuple(b.version for b in trainer.branches))

    def test_random_basis_charges_omitted_anchor_residual(self):
        net = ReLUNetwork((np.eye(2),), (np.zeros(2),))
        contract = compile_contract(net, [0., 0.], [0., -1.], [2],
                                    basis=[[1.], [0.]], basis_strategy="random")
        self.assertAlmostEqual(contract.residual_tail_norm, 1.)
        # Leaving e out while using the anchored formula would predict 0.005
        # for this update and falsely accept it. True loss is 0.605.
        summary = summarize(contract.nodes[0], [0., .1])
        decision = judge(contract, [summary])
        self.assertGreaterEqual(decision.loss_upper_bound + 1e-12, .605)
        self.assertFalse(decision.accepted)

    def test_random_space_bound_across_nonlinear_candidates(self):
        rng = np.random.default_rng(506)
        for case in range(4):
            net = ReLUNetwork((rng.normal(size=(5, 4)), rng.normal(size=(6, 5))),
                              (rng.normal(size=5), rng.normal(size=6)))
            base, target = rng.normal(size=4), rng.normal(size=6)
            contract = compile_contract(net, base, target, [2, 2], rank=2,
                                        basis_strategy="random", seed=case)
            for _ in range(20):
                delta = rng.normal(size=4) * 10 ** rng.uniform(-4, 0)
                summaries = [summarize(node, part) for node, part in zip(contract.nodes, np.split(delta, 2))]
                decision = judge(contract, summaries)
                actual = np.sum((net.forward(base + delta) - target) ** 2) / 2
                self.assertLessEqual(actual, decision.loss_upper_bound + 1e-10 * max(1., actual))

    def test_basis_expansion_invalidates_epoch_and_preserves_anchor_snapshot(self):
        batch, branches, fusion = make_problem(47, samples=3)
        trainer = Trainer(batch, branches, fusion, TrainingConfig(local_steps=1))
        old = trainer.propose(.001)
        versions = trainer.contract.anchor_parameter_versions
        self.assertTrue(trainer.expand())
        self.assertEqual(versions, trainer.contract.anchor_parameter_versions)
        with self.assertRaisesRegex(ValueError, "epoch"):
            trainer.decide(old)
        for sample in trainer.contract.samples:
            self.assertEqual(sample.rank, 2)

    def test_batch_owns_inputs_and_validates_sample_identity(self):
        values = np.ones((2, 1))
        batch = FixedBatch((values,), np.zeros((2, 1)), ("a", "b"))
        version = batch.version
        values[:] = 9
        self.assertEqual(batch.version, version)
        self.assertFalse(batch.inputs[0].flags.writeable)
        with self.assertRaises(ValueError):
            replace(batch, sample_ids=("a", "a"))
        with self.assertRaises(ValueError):
            replace(batch, targets=[[0.]])


if __name__ == "__main__":
    unittest.main()
