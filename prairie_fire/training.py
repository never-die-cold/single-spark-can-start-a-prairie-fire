"""M1: fixed-batch, four-branch training with transactional response contracts.

This is a single-process numerical simulator. Bytes are a declared logical wire
model, not measured traffic; dense Jacobians are still the correctness oracle.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from .contracts import LossContract, NodeSummary, compile_contract, expand_contract, judge, summarize
from .networks import Array, ReLUNetwork, frozen_array


@dataclass(frozen=True)
class FixedBatch:
    inputs: tuple[Array, ...]
    targets: Array
    sample_ids: tuple[str, ...]
    preprocessing_id: str = "identity-float64-v1"

    def __post_init__(self) -> None:
        inputs = tuple(frozen_array(x) for x in self.inputs)
        targets = frozen_array(self.targets)
        ids = tuple(self.sample_ids)
        if targets.ndim != 2 or min(targets.shape) < 1:
            raise ValueError("targets must be a nonempty batch matrix")
        if not inputs or any(x.ndim != 2 or x.shape[0] != len(targets) or x.shape[1] < 1 for x in inputs):
            raise ValueError("node inputs must match the target batch")
        if len(ids) != len(targets) or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("sample IDs must be nonempty, unique and match the batch")
        if not isinstance(self.preprocessing_id, str) or not self.preprocessing_id:
            raise ValueError("preprocessing ID must be nonempty")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "sample_ids", ids)

    @property
    def version(self) -> str:
        digest = sha256(json.dumps([self.sample_ids, self.preprocessing_id]).encode())
        for array in (*self.inputs, self.targets):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class TrainingConfig:
    max_rounds: int = 160
    local_steps: int = 4
    learning_rate: float = .08
    fusion_learning_rate: float = .04
    fusion_interval: int = 8
    initial_rank: int = 1
    refresh_period: int = 4
    max_attempts: int = 8
    target_loss: float = .01

    def __post_init__(self) -> None:
        for name in ("max_rounds", "local_steps", "fusion_interval", "initial_rank", "refresh_period", "max_attempts"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "fusion_learning_rate", "target_loss"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class Ledger:
    """Charge every modeled unicast, including rejected proposals and rebuilds.

    A message has a fixed 128-byte header (three 32-byte digests and four int64
    fields). Payloads use float64. There is no transport/compression/framing.
    Kernel counters include rejected work; VJP includes its own forward pass.
    """

    def __init__(self) -> None:
        self.payload = Counter()
        self.metadata = Counter()
        self.messages = Counter()
        self.work = Counter()

    def transfer(self, category: str, payload_bytes: int = 0, *, messages: int = 1) -> None:
        self.payload[category] += int(payload_bytes)
        self.metadata[category] += messages * 128
        self.messages[category] += messages

    def report(self) -> dict:
        categories = sorted(self.messages)
        payload = sum(self.payload.values())
        metadata = sum(self.metadata.values())
        return {
            "payload_bytes": payload, "metadata_bytes": metadata,
            "total_logical_bytes": payload + metadata,
            "by_category": {key: {"payload_bytes": self.payload[key], "metadata_bytes": self.metadata[key],
                                   "messages": self.messages[key]} for key in categories},
            "work": dict(sorted(self.work.items())),
        }


def retained_array_bytes(*objects: object) -> int:
    """Owned ndarray storage retained by these objects, deduplicated by identity.

    This excludes transient BLAS/NumPy buffers and interpreter overhead, so it is
    deliberately not reported as measured process or device peak memory.
    """
    seen: set[int] = set()

    def count(obj: object) -> int:
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        if is_dataclass(obj) and not isinstance(obj, type):
            return sum(count(getattr(obj, field.name)) for field in fields(obj))
        if isinstance(obj, (tuple, list)):
            return sum(count(value) for value in obj)
        return 0

    return sum(count(obj) for obj in objects)


def sgd(network: ReLUNetwork, gradients: tuple[Array, tuple[Array, ...], tuple[Array, ...]], rate: float) -> ReLUNetwork:
    _, dw, db = gradients
    return ReLUNetwork(tuple(w - rate * g for w, g in zip(network.weights, dw)),
                       tuple(b - rate * g for b, g in zip(network.biases, db)))


@dataclass(frozen=True)
class BatchContract:
    epoch_id: str
    data_id: str
    model_version: str
    anchor_parameter_versions: tuple[str, ...]
    samples: tuple[LossContract, ...]


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    epoch_id: str
    data_id: str
    parent_parameter_versions: tuple[str, ...]
    branches: tuple[ReLUNetwork, ...]
    features: tuple[Array, ...]
    summaries: tuple[tuple[NodeSummary, ...], ...]


@dataclass(frozen=True)
class BatchDecision:
    proposal_id: str
    epoch_id: str
    parent_parameter_versions: tuple[str, ...]
    accepted: bool
    upper_loss: float
    current_loss: float
    tail_penalty: float
    nonlinear_penalty: float


class Trainer:
    """Honest-node simulator with immutable committed parameter snapshots."""

    def __init__(self, batch: FixedBatch, branches: tuple[ReLUNetwork, ...], fusion: ReLUNetwork,
                 config: TrainingConfig, *, method: str = "compressed", basis_strategy: str = "anchored",
                 refresh_policy: str = "adaptive", seed: int = 0):
        self.started = perf_counter()
        if method not in ("backprop", "dense", "compressed"):
            raise ValueError("unknown training method")
        if basis_strategy not in ("anchored", "random") or refresh_policy not in ("adaptive", "periodic"):
            raise ValueError("unknown ablation policy")
        if len(branches) != len(batch.inputs):
            raise ValueError("one branch per node input is required")
        if fusion.weights[0].shape[1] != sum(node.weights[-1].shape[0] for node in branches):
            raise ValueError("fusion input must concatenate node features")
        if fusion.weights[-1].shape[0] != batch.targets.shape[1]:
            raise ValueError("fusion output and targets must agree")
        if config.initial_rank > batch.targets.shape[1]:
            raise ValueError("initial rank exceeds output dimension")
        self.batch, self.branches, self.fusion, self.config = batch, tuple(branches), fusion, config
        self.method, self.basis_strategy, self.refresh_policy, self.seed = method, basis_strategy, refresh_policy, seed
        self.ledger = Ledger()
        self.ledger.transfer("fixed_input_distribution", sum(x.nbytes for x in batch.inputs), messages=len(branches))
        self.features = self._features(self.branches)
        self.ledger.transfer("initial_features", sum(x.nbytes for x in self.features), messages=len(branches))
        self.loss = self._loss(self.features)
        self.initial_loss = self.loss
        self.contract: BatchContract | None = None
        self.generation = 0
        self.age = 0
        self.peak_node_retained = [0] * len(branches)
        self.peak_coordinator_retained = 0
        self.trace: list[dict] = []
        self.repairs: list[dict] = []
        self.commits_by_contract: Counter[str] = Counter()
        self.accepted_losses = [self.loss]
        self.max_bound_violation = 0.0
        self.target_snapshot: dict | None = None
        self._storage()
        self._target(0)

    def _features(self, branches: tuple[ReLUNetwork, ...]) -> tuple[Array, ...]:
        self.ledger.work["branch_forward_samples"] += len(self.batch.targets) * len(branches)
        return tuple(frozen_array(node.forward_batch(x)) for node, x in zip(branches, self.batch.inputs))

    def _loss(self, features: tuple[Array, ...], fusion: ReLUNetwork | None = None) -> float:
        self.ledger.work["fusion_forward_samples"] += len(self.batch.targets)
        error = (fusion or self.fusion).forward_batch(np.concatenate(features, axis=1)) - self.batch.targets
        return float(np.sum(error * error) / (2 * len(error)))

    def _storage(self, proposal: Proposal | None = None) -> None:
        for i, branch in enumerate(self.branches):
            # The old snapshot and candidate are both retained during a trial.
            node_contracts = tuple(c.nodes[i] for c in self.contract.samples) if self.contract else ()
            residuals = tuple(c.projected_residual for c in self.contract.samples) if self.contract else ()
            candidate = (proposal.branches[i], proposal.features[i]) if proposal else ()
            size = retained_array_bytes(branch, self.batch.inputs[i], self.features[i], node_contracts, residuals, candidate)
            self.peak_node_retained[i] = max(self.peak_node_retained[i], size)
        size = retained_array_bytes(self.fusion, self.batch.targets, self.features, self.contract)
        self.peak_coordinator_retained = max(self.peak_coordinator_retained, size)

    def _target(self, round_id: int) -> None:
        if self.target_snapshot is None and self.loss <= self.config.target_loss:
            self.target_snapshot = {"round": round_id, "loss": self.loss,
                                    "elapsed_seconds": perf_counter() - self.started,
                                    "ledger": self.ledger.report()}

    def _install(self, samples: tuple[LossContract, ...], reason: str, *,
                 anchor_versions: tuple[str, ...] | None = None) -> None:
        self.generation += 1
        versions = tuple(b.version for b in self.branches)
        epoch = sha256(json.dumps([self.batch.version, self.fusion.version, versions, self.generation,
                                   [c.anchor_id for c in samples]]).encode()).hexdigest()
        self.contract = BatchContract(epoch, self.batch.version, self.fusion.version, anchor_versions or versions, samples)
        if reason != "tail_expansion":
            self.age = 0
        size = sum(node.projection.nbytes + 8 + c.projected_residual.nbytes
                   for c in samples for node in c.nodes)
        self.ledger.transfer("contract_distribution", size, messages=len(self.branches))
        self.ledger.work["contract_installations"] += 1
        self.repairs.append({"reason": reason, "generation": self.generation, "ranks": [c.rank for c in samples]})
        self._storage()

    def refresh(self, reason: str = "initial") -> None:
        rank = self.batch.targets.shape[1] if self.method == "dense" else self.config.initial_rank
        if self.contract is not None:
            rank = max(rank, max(c.rank for c in self.contract.samples))
        z = np.concatenate(self.features, axis=1)
        sizes = [feature.shape[1] for feature in self.features]
        samples = tuple(compile_contract(self.fusion, z[j], target, sizes, rank=rank,
                                        seed=self.seed + j, basis_strategy=self.basis_strategy)
                        for j, target in enumerate(self.batch.targets))
        self.ledger.work["dense_response_oracle_samples"] += len(samples)
        self.ledger.work["anchor_rebuilds"] += 1
        self._install(samples, reason)

    def expand(self) -> bool:
        if self.contract is None or all(c.rank == len(c.target) for c in self.contract.samples):
            return False
        samples = []
        for contract in self.contract.samples:
            omitted = contract.response.jacobian - contract.basis @ (contract.basis.T @ contract.response.jacobian)
            self.ledger.work["dense_direction_svd_queries"] += 1
            # This query uses the stored dense coordinator oracle, not a free
            # remote JVP. M2 must replace it and charge its actual transport.
            left, singular, _ = np.linalg.svd(omitted, full_matrices=False)
            if len(singular) and singular[0] > 1e-12:
                updated = expand_contract(contract, self.fusion, left[:, 0])
            else:
                updated = contract
            samples.append(updated)
            self.ledger.work["dense_response_oracle_samples"] += int(updated is not contract)
        if all(a is b for a, b in zip(samples, self.contract.samples)):
            return False
        self.ledger.work["basis_expansions"] += 1
        self._install(tuple(samples), "tail_expansion", anchor_versions=self.contract.anchor_parameter_versions)
        return True

    def _check_epoch(self, proposal: Proposal) -> BatchContract:
        contract = self.contract
        if contract is None or proposal.epoch_id != contract.epoch_id:
            raise ValueError("stale proposal epoch")
        if proposal.data_id != self.batch.version or contract.data_id != self.batch.version:
            raise ValueError("data, labels, samples or preprocessing changed")
        if self.fusion.version != contract.model_version:
            raise ValueError("fusion model version changed")
        if proposal.parent_parameter_versions != tuple(b.version for b in self.branches):
            raise ValueError("proposal was trained from an outdated parameter snapshot")
        if len(proposal.branches) != len(self.branches) or len(proposal.features) != len(self.branches):
            raise ValueError("proposal must contain all nodes")
        if len(proposal.summaries) != len(self.batch.targets):
            raise ValueError("proposal must contain the fixed batch")
        return contract

    def propose(self, rate: float | None = None) -> Proposal:
        if self.contract is None:
            self.refresh()
        assert self.contract is not None
        if self.contract.data_id != self.batch.version or self.contract.model_version != self.fusion.version:
            raise ValueError("contract data/model epoch changed")
        rate = self.config.learning_rate if rate is None else rate
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        candidates, candidate_features = [], []
        samples = self.contract.samples
        batch_size, count = len(samples), len(self.branches)
        for i, (snapshot, inputs) in enumerate(zip(self.branches, self.batch.inputs)):
            node = snapshot
            local_rate = rate
            z = node.forward_batch(inputs)
            self.ledger.work["branch_forward_samples"] += batch_size

            def objective(features: Array) -> float:
                value = 0.
                for j, c in enumerate(samples):
                    local = c.nodes[i]
                    delta = features[j] - local.base_features
                    p = local.projection @ delta
                    value += c.projected_residual @ p + count / 2 * (p @ p + local.tail_bound ** 2 * (delta @ delta))
                self.ledger.work["local_objective_sample_evaluations"] += batch_size
                return float(value / batch_size)

            value = objective(z)
            for _ in range(self.config.local_steps):
                dz = []
                for j, c in enumerate(samples):
                    local = c.nodes[i]
                    delta = z[j] - local.base_features
                    p = local.projection @ delta
                    grad = local.projection.T @ (c.projected_residual + count * p)
                    grad += count * local.tail_bound ** 2 * delta
                    dz.append(grad / batch_size)
                gradients = node.vjp_batch(inputs, dz)
                self.ledger.work["branch_vjp_samples_including_forward"] += batch_size
                # Search locally, where no network exchange is needed. This
                # prevents an unstable surrogate step causing a global retry.
                for _ in range(self.config.max_attempts):
                    candidate = sgd(node, gradients, local_rate)
                    candidate_z = candidate.forward_batch(inputs)
                    self.ledger.work["branch_forward_samples"] += batch_size
                    candidate_value = objective(candidate_z)
                    if candidate_value <= value:
                        node, z, value = candidate, candidate_z, candidate_value
                        break
                    local_rate *= .5
                    self.ledger.work["local_step_size_reductions"] += 1
                else:
                    break
            candidates.append(node)
            candidate_features.append(frozen_array(z))
        features = tuple(candidate_features)
        summaries = tuple(tuple(summarize(c.nodes[i], feature[j] - c.nodes[i].base_features)
                                for i, feature in enumerate(features)) for j, c in enumerate(samples))
        self.ledger.work["summary_sample_nodes"] += batch_size * count
        candidate_id = sha256(json.dumps([self.contract.epoch_id, [b.version for b in candidates]]).encode()).hexdigest()
        proposal = Proposal(candidate_id, self.contract.epoch_id, self.batch.version, tuple(b.version for b in self.branches),
                            tuple(candidates), features, summaries)
        self._storage(proposal)
        return proposal

    def decide(self, proposal: Proposal) -> BatchDecision:
        contract = self._check_epoch(proposal)
        self.ledger.transfer("proposal_summaries", sum(s.projected_change.nbytes + 16 for row in proposal.summaries for s in row),
                             messages=len(self.branches))
        decisions = [judge(c, summaries, model_version=self.fusion.version) for c, summaries in zip(contract.samples, proposal.summaries)]
        self.ledger.work["joint_bound_sample_evaluations"] += len(decisions)
        upper = float(np.mean([d.loss_upper_bound for d in decisions]))
        tail = float(np.mean([.5 * (c.residual_tail_norm + d.discarded_response_bound) ** 2
                              for c, d in zip(contract.samples, decisions)]))
        nonlinear = float(np.mean([d.nonlinear_remainder_bound * math.hypot(d.projected_residual_norm,
                                  c.residual_tail_norm + d.discarded_response_bound) + .5 * d.nonlinear_remainder_bound ** 2
                                  for c, d in zip(contract.samples, decisions)]))
        accepted = upper < self.loss - 1e-12 * max(1., self.loss)
        self.ledger.transfer("proposal_decisions", messages=len(self.branches))
        self.ledger.work["proposal_attempts"] += 1
        self.ledger.work["accepted_proposals" if accepted else "rejected_proposals"] += 1
        return BatchDecision(proposal.proposal_id, contract.epoch_id, proposal.parent_parameter_versions, accepted, upper, self.loss, tail, nonlinear)

    def commit(self, proposal: Proposal, decision: BatchDecision) -> None:
        self._check_epoch(proposal)
        if (decision.proposal_id != proposal.proposal_id or decision.epoch_id != proposal.epoch_id
                or decision.parent_parameter_versions != proposal.parent_parameter_versions):
            raise ValueError("decision does not match proposal snapshot")
        if not decision.accepted or decision.current_loss != self.loss:
            raise ValueError("only a fresh accepted decision can be committed")
        # Commit payload supplies the exact features needed for the next true
        # current loss. No candidate feature upload was used for rejected trials.
        self.ledger.transfer("committed_features", sum(x.nbytes for x in proposal.features), messages=len(self.branches))
        actual = self._loss(proposal.features)
        self.max_bound_violation = max(self.max_bound_violation, actual - decision.upper_loss)
        if actual > decision.upper_loss + 1e-10 * max(1., actual) or actual >= self.loss:
            raise ArithmeticError("accepted candidate violated the loss bound; no parameters committed")
        self.branches, self.features, self.loss = proposal.branches, proposal.features, actual
        self.accepted_losses.append(actual)
        self.commits_by_contract[proposal.epoch_id] += 1

    def _backprop_proposal(self, rate: float) -> tuple[tuple[ReLUNetwork, ...], tuple[Array, ...]]:
        branches, features = self.branches, self.features
        batch_size = len(self.batch.targets)
        for step in range(self.config.local_steps):
            # The first step reuses the last uploaded committed features; later
            # local steps upload fresh activations. Final features are charged
            # by the trial/acceptance evaluation, avoiding a duplicate upload.
            if step:
                self.ledger.transfer("backprop_features", sum(x.nbytes for x in features), messages=len(branches))
            z = np.concatenate(features, axis=1)
            error = self.fusion.forward_batch(z) - self.batch.targets
            self.ledger.work["fusion_forward_samples"] += batch_size
            dz, _, _ = self.fusion.vjp_batch(z, error / batch_size)
            self.ledger.work["fusion_vjp_samples_including_forward"] += batch_size
            self.ledger.transfer("backprop_feature_gradients", dz.nbytes, messages=len(branches))
            starts = np.cumsum([0] + [x.shape[1] for x in features])
            branches = tuple(sgd(node, node.vjp_batch(x, dz[:, starts[i]:starts[i + 1]]), rate)
                             for i, (node, x) in enumerate(zip(branches, self.batch.inputs)))
            self.ledger.work["branch_vjp_samples_including_forward"] += batch_size * len(branches)
            features = self._features(branches)
        return branches, features

    def train_fusion(self) -> bool:
        z = np.concatenate(self.features, axis=1)
        batch_size = len(z)
        error = self.fusion.forward_batch(z) - self.batch.targets
        self.ledger.work["fusion_forward_samples"] += batch_size
        gradients = self.fusion.vjp_batch(z, error / batch_size)
        self.ledger.work["fusion_vjp_samples_including_forward"] += batch_size
        for retry in range(self.config.max_attempts):
            candidate = sgd(self.fusion, gradients, self.config.fusion_learning_rate * .5 ** retry)
            loss = self._loss(self.features, candidate)
            self.ledger.work["fusion_line_search_trials"] += 1
            if loss < self.loss - 1e-12 * max(1., self.loss):
                self.fusion, self.loss = candidate, loss
                self.accepted_losses.append(loss)
                self.contract = None
                if self.method != "backprop":
                    self.ledger.transfer("model_epoch_invalidation", messages=len(self.branches))
                self.ledger.work["fusion_updates"] += 1
                return True
        return False

    def step(self) -> bool:
        if self.method == "backprop":
            for retry in range(self.config.max_attempts):
                branches, features = self._backprop_proposal(self.config.learning_rate * .5 ** retry)
                self._storage(Proposal("", "", self.batch.version, (), branches, features, ()))
                self.ledger.transfer("backprop_trial_features", sum(x.nbytes for x in features), messages=len(branches))
                actual = self._loss(features)
                self.ledger.work["proposal_attempts"] += 1
                accepted = actual < self.loss - 1e-12 * max(1., self.loss)
                self.ledger.work["accepted_proposals" if accepted else "rejected_proposals"] += 1
                self.ledger.transfer("proposal_decisions", messages=len(branches))
                if accepted:
                    self.branches, self.features, self.loss = branches, features, actual
                    self.accepted_losses.append(actual)
                    self._storage()
                    return True
            return False
        if self.contract is None:
            self.refresh("data_or_model_epoch")
        elif self.refresh_policy == "periodic" and self.age >= self.config.refresh_period:
            self.refresh("periodic")
        rate = self.config.learning_rate
        reanchored = False
        accepted = False
        for _ in range(self.config.max_attempts):
            proposal = self.propose(rate)
            decision = self.decide(proposal)
            if decision.accepted:
                self.commit(proposal, decision)
                accepted = True
                break
            if self.refresh_policy == "adaptive":
                if decision.tail_penalty > decision.nonlinear_penalty and self.expand():
                    continue
                if self.age > 0 and not reanchored:
                    self.refresh("nonlinear_or_reuse_slack")
                    reanchored = True
                    continue
            self.ledger.work["step_size_reductions"] += 1
            rate *= .5
        self.age += 1
        return accepted

    def run(self) -> dict:
        for round_id in range(1, self.config.max_rounds + 1):
            if self.target_snapshot is not None:
                break
            before = self.ledger.work["proposal_attempts"]
            installs = self.ledger.work["contract_installations"]
            accepted = self.step()
            branch_loss = self.loss
            self._target(round_id)
            fusion_changed = False
            if self.target_snapshot is None and round_id % self.config.fusion_interval == 0:
                fusion_changed = self.train_fusion()
                self._target(round_id)
            self._storage()
            self.trace.append({"round": round_id, "accepted": accepted, "branch_loss": branch_loss, "loss": self.loss,
                               "attempts": self.ledger.work["proposal_attempts"] - before,
                               "contract_installations": self.ledger.work["contract_installations"] - installs,
                               "fusion_changed": fusion_changed,
                               "rank": self.contract.samples[0].rank if self.contract else None})
        ledger = self.ledger.report()
        return {
            "method": self.method, "basis_strategy": self.basis_strategy, "refresh_policy": self.refresh_policy,
            "initial_loss": self.initial_loss, "final_loss": self.loss,
            "target_reached": self.target_snapshot is not None, "at_target": self.target_snapshot,
            "elapsed_seconds": perf_counter() - self.started,
            "accepted_loss_increases": sum(b >= a for a, b in zip(self.accepted_losses, self.accepted_losses[1:])),
            "max_accepted_actual_minus_bound": self.max_bound_violation,
            "ledger": ledger,
            "contract_reuse": {"contracts_with_commits": len(self.commits_by_contract),
                               "max_commits_on_one_contract": max(self.commits_by_contract.values(), default=0),
                               "accepted_commits_per_installation": sum(self.commits_by_contract.values()) /
                               max(1, self.ledger.work["contract_installations"])},
            "retained_arrays": {"node_max_observed_bytes": self.peak_node_retained,
                                "coordinator_max_observed_bytes": self.peak_coordinator_retained,
                                "scope": "retained ndarray storage at observation points; excludes transient buffers, not measured peak RAM"},
            "trace": self.trace, "contract_events": self.repairs,
        }


def make_problem(seed: int, *, samples: int = 12) -> tuple[FixedBatch, tuple[ReLUNetwork, ...], ReLUNetwork]:
    """Small teacher regression, no held-out/generalization performance claim."""
    rng = np.random.default_rng(seed)

    def network(sizes: tuple[int, ...]) -> ReLUNetwork:
        return ReLUNetwork(tuple(rng.normal(size=(b, a)) / np.sqrt(a) for a, b in zip(sizes[:-1], sizes[1:])),
                           tuple(rng.normal(scale=.15, size=b) for b in sizes[1:]))

    inputs = tuple(rng.uniform(-1., 1., size=(samples, 2)) for _ in range(4))
    teachers = tuple(network((2, 5, 3)) for _ in range(4))
    teacher_fusion = network((12, 8, 6))
    targets = teacher_fusion.forward_batch(np.concatenate([n.forward_batch(x) for n, x in zip(teachers, inputs)], axis=1))
    branches = tuple(network((2, 5, 3)) for _ in range(4))
    fusion = network((12, 8, 6))
    batch = FixedBatch(inputs, targets, tuple(f"sample-{i}" for i in range(samples)))
    return batch, branches, fusion


def run_suite(*, seeds: tuple[int, ...] = (17, 31, 47), config: TrainingConfig | None = None) -> dict:
    config = config or TrainingConfig()
    variants = (("backprop", "anchored", "adaptive"), ("dense", "anchored", "adaptive"),
                ("compressed", "anchored", "adaptive"), ("compressed", "random", "adaptive"),
                ("compressed", "anchored", "periodic"), ("compressed", "random", "periodic"))
    runs = []
    for seed in seeds:
        batch, branches, fusion = make_problem(seed)
        initial_versions = {"data": batch.version, "fusion": fusion.version, "branches": [b.version for b in branches]}
        for method, basis, refresh in variants:
            trainer = Trainer(batch, branches, fusion, config, method=method, basis_strategy=basis,
                              refresh_policy=refresh, seed=seed)
            report = trainer.run()
            report.update(seed=seed, initial_versions=initial_versions)
            runs.append(report)
    passed = all(r["accepted_loss_increases"] == 0 and r["max_accepted_actual_minus_bound"] <= 1e-10 for r in runs)
    return {
        "schema_version": 1, "experiment": "M1-fixed-batch-branch-training",
        "implementation": "local-surrogate-backtracking-v1",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "config": {field.name: getattr(config, field.name) for field in fields(config)}, "seeds": list(seeds),
        "problem": {"nodes": 4, "samples": 12, "node_layers": [2, 5, 3], "fusion_layers": [12, 8, 6],
                    "loss": "mean over samples of half squared output-error norm",
                    "schedule": "branch phase then one fusion step every fusion_interval rounds; all variants share initialization and data"},
        "scope": "single-process end-to-end toy regression; exact dense response oracle; honest numerical summaries",
        "accounting": {"wire": "float64 payload plus 128 bytes per unicast; all attempts, commits, invalidations and rebuilt matrices charged",
                       "provisioning": "fixed input distribution included; preinstalled code, initial weights and central labels excluded equally",
                       "queries": "all response and direction queries execute centrally on retained dense oracle; kernel counts included",
                       "rejected_candidates": "no feature upload or true-loss evaluation; rejection can be conservative",
                       "backprop": "fusion frozen in branch phase; true full-network chain rule each local step, then same separate fusion schedule",
                       "time": "local simulator wall time; not a distributed speed benchmark"},
        "does_not_establish": ["originality", "large-model convergence", "real networking", "device memory savings", "energy savings", "high-end GPU equivalence"],
        "all_safety_checks_passed": passed,
        "all_targets_reached": all(r["target_reached"] for r in runs), "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 31, 47])
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--target-loss", type=float, default=.01)
    parser.add_argument("--output", type=Path, default=Path("results/m1-latest.json"))
    args = parser.parse_args()
    result = run_suite(seeds=tuple(args.seeds), config=TrainingConfig(max_rounds=args.rounds, local_steps=args.local_steps,
                                                                  target_loss=args.target_loss))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for run in result["runs"]:
        print(f"seed={run['seed']} {run['method']:10} {run['basis_strategy']:8} {run['refresh_policy']:8} "
              f"loss={run['final_loss']:.6f} target={run['target_reached']} "
              f"bytes={run['ledger']['total_logical_bytes']}")
    print(f"safety={result['all_safety_checks_passed']}; all_targets={result['all_targets_reached']}; report={args.output}")
    if not result["all_safety_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
