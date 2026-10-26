"""Exact boundary replay with feature menus and deterministic prefix commits.

Research candidate, not a novelty claim or a real network transport. Each node
owns one branch and replicates only its first fusion-layer columns and the
remaining small fusion suffix. Labels are shared with these trusted nodes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
import platform

import numpy as np

from .networks import Array, ReLUNetwork, frozen_array
from .training import Trainer, TrainingConfig, make_problem, retained_array_bytes, sgd


@dataclass(frozen=True)
class Capsule:
    own_weight: Array
    other_preactivation: Array
    suffix: ReLUNetwork
    targets: Array

    def prediction(self, features: Array) -> Array:
        pre = self.other_preactivation + features @ self.own_weight.T
        return self.suffix.forward_batch(np.maximum(pre, 0.))

    def feature_gradient(self, features: Array) -> Array:
        pre = self.other_preactivation + features @ self.own_weight.T
        hidden = np.maximum(pre, 0.)
        error = self.suffix.forward_batch(hidden) - self.targets
        dh, _, _ = self.suffix.vjp_batch(hidden, error / len(error))
        return (dh * (pre > 0)) @ self.own_weight


@dataclass(frozen=True)
class FeatureMenu:
    steps: tuple[int, ...]
    features: tuple[Array, ...]
    parameter_versions: tuple[str, ...]


@dataclass(frozen=True)
class ReplayProposal:
    epoch_id: str
    data_id: str
    fusion_version: str
    parents: tuple[str, ...]
    capsules: tuple[Capsule, ...]
    menus: tuple[FeatureMenu, ...]
    final_branches: tuple[ReLUNetwork, ...]


@dataclass(frozen=True)
class Selection:
    epoch_id: str
    indices: tuple[int, ...]
    loss: float


def select_menu(menus: tuple[FeatureMenu, ...], evaluate, current_loss: float, *, sweeps: int = 2) -> tuple[tuple[int, ...], float]:
    """Bounded discrete coordinate search; zero is always a rollback candidate.

    Try the full endpoint first, then improve one choice at a time. This is not
    exhaustive joint minimization. It uses uploaded features without asking any
    branch for another candidate. Cache every evaluated combination.
    """
    zeros = (0,) * len(menus)
    cache = {zeros: current_loss}

    def loss(indices):
        if indices not in cache:
            cache[indices] = evaluate(indices)
        return cache[indices]

    selected, value = zeros, current_loss
    endpoint = tuple(len(menu.steps) - 1 for menu in menus)
    if loss(endpoint) < value:
        selected, value = endpoint, loss(endpoint)
    # Coordinated shorter prefixes can beat every endpoint subset. Seed the
    # coordinate search with uniform prefixes to avoid that simple local trap.
    for level in range(1, max(len(menu.steps) for menu in menus) - 1):
        uniform = tuple(min(level, len(menu.steps) - 1) for menu in menus)
        if loss(uniform) < value:
            selected, value = uniform, loss(uniform)
    for _ in range(sweeps):
        previous = selected
        for node_id, menu in enumerate(menus):
            best, best_value = selected, value
            for index in range(len(menu.steps)):
                trial = list(selected)
                trial[node_id] = index
                trial = tuple(trial)
                candidate_value = loss(trial)
                if candidate_value < best_value:
                    best, best_value = trial, candidate_value
            selected, value = best, best_value
        if selected == previous:
            break
    return selected, value


class ReplayTrainer(Trainer):
    def __init__(self, batch, branches, fusion, config, *, menu_steps: tuple[int, ...] | None = None, seed: int = 0):
        if len(fusion.weights) < 2:
            raise ValueError("replay requires an affine/ReLU join and a nonempty suffix")
        if menu_steps is None:
            menu_steps = tuple(sorted({1, max(1, config.local_steps // 4), config.local_steps}))
        if (not menu_steps or tuple(sorted(set(menu_steps))) != menu_steps or menu_steps[0] < 1
                or menu_steps[-1] != config.local_steps
                or any(not isinstance(step, int) or isinstance(step, bool) for step in menu_steps)):
            raise ValueError("menu steps must be increasing positive integers ending at local_steps")
        self.menu_steps = menu_steps
        self.capsule_model_version: str | None = None
        self.active_epoch: str | None = None
        self.uploaded_menu_epoch: str | None = None
        self.proposal_generation = 0
        self.max_replay_feature_error = 0.
        self.selected_prefix_counts: dict[int, int] = {}
        # Reuse only the fixed-batch state, accounting and exact fusion phase.
        # The overridden step never compiles a response/Jacobian contract.
        super().__init__(batch, branches, fusion, config, method="backprop", seed=seed)
        self.ledger.transfer("replay_label_distribution", len(branches) * batch.targets.nbytes, messages=len(branches))

    def capsules(self) -> tuple[Capsule, ...]:
        z = np.concatenate(self.features, axis=1)
        pre = z @ self.fusion.weights[0].T + self.fusion.biases[0]
        suffix = ReLUNetwork(self.fusion.weights[1:], self.fusion.biases[1:])
        starts = np.cumsum([0] + [feature.shape[1] for feature in self.features])
        caps = []
        for i, feature in enumerate(self.features):
            own_weight = frozen_array(self.fusion.weights[0][:, starts[i]:starts[i + 1]])
            caps.append(Capsule(own_weight, frozen_array(pre - feature @ own_weight.T), suffix,
                                frozen_array(self.batch.targets)))
        self.ledger.work["boundary_setup_sample_nodes"] += len(z) * len(caps)
        # Actual wire model sends the common preactivation once per recipient.
        # Each recipient subtracts its own contribution using cached features.
        self.ledger.transfer("replay_boundary_distribution", len(caps) * pre.nbytes, messages=len(caps))
        if self.capsule_model_version != self.fusion.version:
            payload = sum(c.own_weight.nbytes + retained_array_bytes(c.suffix) for c in caps)
            self.ledger.transfer("replay_suffix_distribution", payload, messages=len(caps))
            self.capsule_model_version = self.fusion.version
        return tuple(caps)

    def trajectory(self, branch: ReLUNetwork, inputs: Array, capsule: Capsule, steps: int,
                   checkpoints: tuple[int, ...], *, replaying: bool = False) -> tuple[ReLUNetwork, FeatureMenu]:
        node = branch
        z = node.forward_batch(inputs)
        batch_size = len(z)
        self.ledger.work["branch_forward_samples"] += batch_size
        saved_features = [frozen_array(z)]
        saved_versions = [node.version]
        rate = self.config.learning_rate

        def objective(features):
            self.ledger.work["capsule_forward_samples"] += batch_size
            error = capsule.prediction(features) - capsule.targets
            return float(np.sum(error * error) / (2 * batch_size))

        value = objective(z)
        for step in range(1, steps + 1):
            dz = capsule.feature_gradient(z)
            self.ledger.work["capsule_forward_samples"] += batch_size
            self.ledger.work["capsule_vjp_samples_including_forward"] += batch_size
            gradients = node.vjp_batch(inputs, dz)
            self.ledger.work["branch_vjp_samples_including_forward"] += batch_size
            for _ in range(self.config.max_attempts):
                candidate = sgd(node, gradients, rate)
                candidate_z = candidate.forward_batch(inputs)
                self.ledger.work["branch_forward_samples"] += batch_size
                candidate_value = objective(candidate_z)
                if candidate_value <= value:
                    node, z, value = candidate, candidate_z, candidate_value
                    break
                rate *= .5
                self.ledger.work["local_step_size_reductions"] += 1
            self.ledger.work["replayed_node_steps" if replaying else "proposed_node_steps"] += 1
            if step in checkpoints:
                saved_features.append(frozen_array(z))
                saved_versions.append(node.version)
        return node, FeatureMenu((0,) + checkpoints, tuple(saved_features), tuple(saved_versions))

    def _replay_storage(self, proposal: ReplayProposal, materialized: tuple[ReLUNetwork, ...] = ()) -> None:
        for i, branch in enumerate(self.branches):
            size = retained_array_bytes(branch, self.batch.inputs[i], self.features[i], proposal.capsules[i],
                                        proposal.menus[i], proposal.final_branches[i],
                                        materialized[i] if i < len(materialized) else ())
            self.peak_node_retained[i] = max(self.peak_node_retained[i], size)
        # Coordinator owns menu features, not node parameter snapshots/capsules.
        uploaded = (proposal.menus if self.uploaded_menu_epoch == proposal.epoch_id
                    else tuple(menu.features[-1] for menu in proposal.menus))
        size = retained_array_bytes(self.fusion, self.batch.targets, self.features, uploaded)
        self.peak_coordinator_retained = max(self.peak_coordinator_retained, size)

    def propose_replay(self) -> ReplayProposal:
        self.proposal_generation += 1
        parents = tuple(branch.version for branch in self.branches)
        epoch = sha256(json.dumps([self.batch.version, self.fusion.version, parents,
                                   self.proposal_generation, self.menu_steps]).encode()).hexdigest()
        self.active_epoch = epoch
        caps = self.capsules()
        finals, menus = [], []
        for branch, inputs, capsule in zip(self.branches, self.batch.inputs, caps):
            final, menu = self.trajectory(branch, inputs, capsule, self.config.local_steps, self.menu_steps)
            finals.append(final)
            menus.append(menu)
        proposal = ReplayProposal(epoch, self.batch.version, self.fusion.version, parents, caps, tuple(menus), tuple(finals))
        self._replay_storage(proposal)
        # Upload only the endpoint initially. Intermediate feature checkpoints
        # stay on their node and are requested only if the joint endpoint fails.
        payload = sum(menu.features[-1].nbytes + 40 for menu in menus)
        self.ledger.transfer("replay_endpoints", payload, messages=len(menus))
        self.ledger.work["proposal_attempts"] += 1
        return proposal

    def _check_replay_epoch(self, proposal: ReplayProposal) -> None:
        if proposal.epoch_id != self.active_epoch:
            raise ValueError("stale replay epoch")
        if proposal.data_id != self.batch.version or proposal.fusion_version != self.fusion.version:
            raise ValueError("replay data or fusion version changed")
        if proposal.parents != tuple(branch.version for branch in self.branches):
            raise ValueError("replay parent snapshot changed")
        if len(proposal.menus) != len(self.branches):
            raise ValueError("replay requires all node menus")

    def choose(self, proposal: ReplayProposal) -> Selection:
        self._check_replay_epoch(proposal)
        starts = np.cumsum([0] + [feature.shape[1] for feature in self.features])
        contributions = {}
        losses = {}
        suffix = proposal.capsules[0].suffix

        def evaluate(indices):
            if indices in losses:
                return losses[indices]
            for i, index in enumerate(indices):
                if (i, index) not in contributions:
                    contributions[i, index] = proposal.menus[i].features[index] @ self.fusion.weights[0][:, starts[i]:starts[i + 1]].T
                    self.ledger.work["selector_projection_samples"] += len(self.batch.targets)
            pre = self.fusion.biases[0] + np.sum([contributions[i, index] for i, index in enumerate(indices)], axis=0)
            error = suffix.forward_batch(np.maximum(pre, 0.)) - self.batch.targets
            self.ledger.work["selector_suffix_forward_samples"] += len(error)
            self.ledger.work["selector_combinations"] += 1
            losses[indices] = float(np.sum(error * error) / (2 * len(error)))
            return losses[indices]

        endpoint = tuple(len(menu.steps) - 1 for menu in proposal.menus)
        endpoint_loss = evaluate(endpoint)
        size = retained_array_bytes(self.fusion, self.batch.targets, self.features,
                                    tuple(menu.features[-1] for menu in proposal.menus), tuple(contributions.values()))
        self.peak_coordinator_retained = max(self.peak_coordinator_retained, size)
        if endpoint_loss < self.loss - 1e-12 * max(1., self.loss):
            return Selection(proposal.epoch_id, endpoint, endpoint_loss)
        self.ledger.work["joint_endpoint_conflicts"] += 1
        if len(self.menu_steps) > 1:
            extra = len(self.menu_steps) - 1
            self.ledger.transfer("replay_menu_request", 8 * extra * len(self.branches), messages=len(self.branches))
            payload = sum(sum(z.nbytes + 40 for z in menu.features[1:-1]) for menu in proposal.menus)
            self.ledger.transfer("replay_extra_menus", payload, messages=len(self.branches))
            self.ledger.work["conflict_menu_requests"] += 1
            self.uploaded_menu_epoch = proposal.epoch_id
            self._replay_storage(proposal)
        indices, loss = select_menu(proposal.menus, evaluate, self.loss)
        # Selection buffers also consume coordinator storage while evaluating.
        size = retained_array_bytes(self.fusion, self.batch.targets, self.features,
                                    proposal.menus if self.uploaded_menu_epoch == proposal.epoch_id else (),
                                    tuple(contributions.values()))
        self.peak_coordinator_retained = max(self.peak_coordinator_retained, size)
        return Selection(proposal.epoch_id, indices, loss)

    def commit_replay(self, proposal: ReplayProposal, selection: Selection) -> bool:
        self._check_replay_epoch(proposal)
        if selection.epoch_id != proposal.epoch_id or len(selection.indices) != len(self.branches):
            raise ValueError("selection does not match replay proposal")
        for index, menu in zip(selection.indices, proposal.menus):
            if not isinstance(index, int) or not 0 <= index < len(menu.steps):
                raise ValueError("invalid selected prefix")
        if selection.loss >= self.loss - 1e-12 * max(1., self.loss):
            self.ledger.transfer("replay_abort", messages=len(self.branches))
            self.ledger.work["rejected_proposals"] += 1
            self.active_epoch = None
            return False
        self.ledger.transfer("replay_prepare_choice", 8 * len(self.branches), messages=len(self.branches))
        branches, features = [], []
        for i, index in enumerate(selection.indices):
            menu, capsule = proposal.menus[i], proposal.capsules[i]
            selected_step = menu.steps[index]
            if selected_step == 0:
                node, z = self.branches[i], self.features[i]
            elif selected_step == self.config.local_steps:
                node, z = proposal.final_branches[i], menu.features[index]
            else:
                node, replayed = self.trajectory(self.branches[i], self.batch.inputs[i], capsule,
                                                 selected_step, (selected_step,), replaying=True)
                z = replayed.features[-1]
            error = float(np.max(np.abs(z - menu.features[index])))
            self.max_replay_feature_error = max(self.max_replay_feature_error, error)
            if node.version != menu.parameter_versions[index] or not np.array_equal(z, menu.features[index]):
                self.ledger.transfer("replay_abort", messages=len(self.branches))
                self.active_epoch = None
                raise ArithmeticError("deterministic replay mismatch; nothing committed")
            branches.append(node)
            features.append(z)
        self._replay_storage(proposal, tuple(branches))
        self.ledger.transfer("replay_prepared_digests", 32 * len(branches), messages=len(branches))
        # Re-evaluate the original fusion network once as a numerical check;
        # selected features are already present, so there is no extra upload.
        actual = self._loss(tuple(features))
        if actual >= self.loss or abs(actual - selection.loss) > 1e-10 * max(1., actual):
            self.ledger.transfer("replay_abort", messages=len(branches))
            self.active_epoch = None
            raise ArithmeticError("selected true loss mismatch; nothing committed")
        self.ledger.transfer("replay_commit", len(branches), messages=len(branches))
        self.branches, self.features, self.loss = tuple(branches), tuple(features), actual
        self.accepted_losses.append(actual)
        self.ledger.work["accepted_proposals"] += 1
        for menu, index in zip(proposal.menus, selection.indices):
            selected_step = menu.steps[index]
            self.selected_prefix_counts[selected_step] = self.selected_prefix_counts.get(selected_step, 0) + 1
            self.ledger.work["committed_prefix_node_steps"] += selected_step
        self.active_epoch = None
        return True

    def step(self) -> bool:
        proposal = self.propose_replay()
        return self.commit_replay(proposal, self.choose(proposal))

    def run(self) -> dict:
        result = super().run()
        result["method"] = "replay_select" if len(self.menu_steps) > 1 else "replay_endpoint"
        result["menu_steps"] = list(self.menu_steps)
        result["selected_prefix_counts"] = self.selected_prefix_counts
        result["max_replay_feature_error"] = self.max_replay_feature_error
        result["max_accepted_actual_minus_bound"] = None  # no bound or Jacobian is used
        result.pop("contract_reuse")
        return result


def compare(*, seeds: tuple[int, ...] = (17, 31, 47), max_rounds: int = 1000, include_contract: bool = True,
            learning_rate: float = .08, fusion_learning_rate: float = .04) -> dict:
    # Keep the original target and learning rates; increase the per-message work
    # window. Every arm gets the SAME window and fusion schedule.
    config = TrainingConfig(max_rounds=max_rounds, local_steps=16, fusion_interval=2, target_loss=.01,
                            learning_rate=learning_rate, fusion_learning_rate=fusion_learning_rate)
    runs = []
    for seed in seeds:
        batch, branches, fusion = make_problem(seed)
        variants = ["backprop", "replay_endpoint", "replay_select"]
        if include_contract:
            variants.append("compressed")
        for name in variants:
            # Start each clock immediately before that arm, not while another
            # arm is training. All share immutable initial model objects.
            if name.startswith("replay"):
                trainer = ReplayTrainer(batch, branches, fusion, config,
                                        menu_steps=(16,) if name == "replay_endpoint" else None, seed=seed)
            else:
                trainer = Trainer(batch, branches, fusion, config, method=name, seed=seed)
            report = trainer.run()
            report["seed"] = seed
            report["initial_versions"] = {"data": batch.version, "branches": [n.version for n in branches], "fusion": fusion.version}
            runs.append(report)
            print(f"seed={seed} {name:15} loss={report['final_loss']:.6f} "
                  f"target={report['target_reached']} bytes={report['ledger']['total_logical_bytes']}", flush=True)
    return {
        "schema_version": 1, "experiment": "exact-boundary-replay-and-prefix-selection",
        "implementation": "endpoint-first-lazy-menu-v1",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "config": {f.name: getattr(config, f.name) for f in fields(config)}, "seeds": list(seeds),
        "scope": "same fixed 12-sample regression and 4 branches as M1; single-process honest-node simulation",
        "accounting": "float64 plus 128-byte unicast headers; includes suffix/label copies, boundaries, all candidate menus, selection, prepare/commit and replay work",
        "memory_scope": "retained ndarray storage at observation points; not measured process peak RAM",
        "novelty": "candidate combination only; cached local training, replicated top models and speculative selection have prior art",
        "all_safety_checks_passed": all(r["accepted_loss_increases"] == 0 and r.get("max_replay_feature_error", 0.) == 0.
                                          and (r["max_accepted_actual_minus_bound"] or 0.) <= 1e-10 for r in runs),
        "all_targets_reached": all(r["target_reached"] for r in runs), "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 31, 47])
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--skip-contract", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=.08)
    parser.add_argument("--fusion-learning-rate", type=float, default=.04)
    parser.add_argument("--output", type=Path, default=Path("results/replay-latest.json"))
    args = parser.parse_args()
    result = compare(seeds=tuple(args.seeds), max_rounds=args.rounds, include_contract=not args.skip_contract,
                     learning_rate=args.learning_rate, fusion_learning_rate=args.fusion_learning_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"safety={result['all_safety_checks_passed']} targets={result['all_targets_reached']} report={args.output}")
    if not result["all_safety_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
