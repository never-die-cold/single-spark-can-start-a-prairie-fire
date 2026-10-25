"""Compressed, loss-anchored response contracts and a joint acceptance bound.

All results are real-arithmetic bounds evaluated with ordinary float64. This is
not interval arithmetic, an adversarial verifier, or a distributed transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from collections.abc import Sequence

import numpy as np

from .networks import Array, ReLUNetwork, Response, frozen_array


@dataclass(frozen=True)
class NodeContract:
    node_id: int
    anchor_id: str
    base_features: Array
    projection: Array
    tail_bound: float


@dataclass(frozen=True)
class NodeSummary:
    node_id: int
    anchor_id: str
    projected_change: Array
    tail_bound: float
    radius_inf: float


@dataclass(frozen=True)
class LossContract:
    anchor_id: str
    model_version: str
    base_features: Array
    target: Array
    basis: Array
    projected_residual: Array
    base_loss: float
    response: Response
    nodes: tuple[NodeContract, ...]

    @property
    def rank(self) -> int:
        return self.basis.shape[1]


@dataclass(frozen=True)
class Decision:
    accepted: bool
    loss_upper_bound: float
    current_loss: float
    projected_residual_norm: float
    discarded_response_bound: float
    nonlinear_remainder_bound: float
    radius_inf: float


def _make_basis(residual: Array, jacobian: Array, rank: int, seed: int) -> Array:
    if not 1 <= rank <= residual.size:
        raise ValueError("rank must be between 1 and the output dimension")
    rng = np.random.default_rng(seed)
    columns: list[Array] = []
    if np.linalg.norm(residual) > 0:
        columns.append(residual)
    if rank > 1:
        columns.extend((jacobian @ rng.normal(size=(jacobian.shape[1], rank - 1))).T)
    columns.extend(rng.normal(size=(residual.size, rank)).T)
    basis: list[Array] = []
    for column in columns:
        candidate = column.copy()
        # Reorthogonalize to keep the loss anchor accurate in small examples.
        for _ in range(2):
            for previous in basis:
                candidate -= previous * (previous @ candidate)
        norm = np.linalg.norm(candidate)
        if norm > np.finfo(float).eps * max(1.0, np.linalg.norm(column)) * 64:
            basis.append(candidate / norm)
        if len(basis) == rank:
            return np.column_stack(basis)
    raise ValueError("could not construct requested basis")


def compile_contract(
    network: ReLUNetwork,
    base_features: object,
    target: object,
    block_sizes: Sequence[int],
    *,
    rank: int = 4,
    seed: int = 20261025,
    basis: object | None = None,
) -> LossContract:
    """Compile a single-sample contract using the dense mathematical oracle.

    Frobenius tail bounds are deterministic upper bounds on spectral norms in
    exact arithmetic. They may be too loose for a practical training system.
    """
    base = frozen_array(base_features)
    target_array = frozen_array(target)
    sizes = tuple(block_sizes)
    if not sizes or any(not isinstance(size, (int, np.integer)) or size <= 0 for size in sizes):
        raise ValueError("block sizes must be positive integers")
    if base.ndim != 1 or sum(sizes) != base.size:
        raise ValueError("blocks must partition the feature vector")
    response = network.response_at(base)
    if target_array.shape != response.output.shape:
        raise ValueError("target dimension does not match output")
    residual = response.output - target_array
    q = _make_basis(residual, response.jacobian, rank, seed) if basis is None else frozen_array(basis)
    if q.ndim != 2 or q.shape[0] != residual.size or not 1 <= q.shape[1] <= residual.size:
        raise ValueError("invalid basis dimensions")
    if not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10, rtol=0):
        raise ValueError("basis columns must be orthonormal")
    residual_error = np.linalg.norm(residual - q @ (q.T @ residual))
    if residual_error > 1e-10 * max(1.0, np.linalg.norm(residual)):
        raise ValueError("basis must contain the full base loss residual")
    digest = sha256(network.version.encode("ascii"))
    for array in (base, target_array, q):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    digest.update(str(sizes).encode("ascii"))
    anchor_id = digest.hexdigest()
    nodes: list[NodeContract] = []
    start = 0
    for node_id, size in enumerate(sizes):
        end = start + size
        a = response.jacobian[:, start:end]
        projection = q.T @ a
        tail = float(np.linalg.norm(a - q @ projection, ord="fro"))
        nodes.append(NodeContract(
            node_id, anchor_id, frozen_array(base[start:end]),
            frozen_array(projection), tail,
        ))
        start = end
    return LossContract(
        anchor_id, network.version, base, target_array, frozen_array(q),
        frozen_array(q.T @ residual), float(residual @ residual / 2),
        response, tuple(nodes),
    )


def summarize(node: NodeContract, delta_features: object) -> NodeSummary:
    """Describe a tentative feature update relative to this contract's anchor."""
    delta = frozen_array(delta_features)
    if delta.shape != node.base_features.shape:
        raise ValueError("feature update dimension does not match node contract")
    projected = node.projection @ delta
    tail = node.tail_bound * float(np.linalg.norm(delta))
    radius = float(np.max(np.abs(delta)))
    if not np.all(np.isfinite(projected)) or not math.isfinite(tail):
        raise FloatingPointError("candidate summary overflowed")
    return NodeSummary(node.node_id, node.anchor_id, frozen_array(projected), tail, radius)


def judge(
    contract: LossContract,
    summaries: Sequence[NodeSummary],
    *,
    current_loss: float | None = None,
    model_version: str | None = None,
    acceptance_margin: float = 1e-12,
) -> Decision:
    """Accept only a jointly improving upper bound under a fixed data/model epoch.

    Summaries are trusted numerical inputs, not proofs against malicious nodes.
    Missing nodes must explicitly submit a zero update.
    """
    if model_version is not None and model_version != contract.model_version:
        raise ValueError("fusion model version changed; recompile the contract")
    if not math.isfinite(acceptance_margin) or acceptance_margin < 0:
        raise ValueError("acceptance margin must be finite and nonnegative")
    loss = contract.base_loss if current_loss is None else float(current_loss)
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("current loss must be finite and nonnegative")
    if len(summaries) != len(contract.nodes):
        raise ValueError("provide one summary per node, including unchanged nodes")
    node_ids = [summary.node_id for summary in summaries]
    if sorted(node_ids) != list(range(len(contract.nodes))):
        raise ValueError("node IDs must be unique and complete")
    for summary in summaries:
        if summary.anchor_id != contract.anchor_id:
            raise ValueError("stale or mismatched response contract")
        if summary.projected_change.shape != (contract.rank,) or not np.all(np.isfinite(summary.projected_change)):
            raise ValueError("invalid projected response")
        if any(not math.isfinite(value) or value < 0 for value in (summary.tail_bound, summary.radius_inf)):
            raise ValueError("summary bounds must be finite and nonnegative")
    p = np.sum([summary.projected_change for summary in summaries], axis=0)
    discarded = math.fsum(summary.tail_bound for summary in summaries)
    radius = max(summary.radius_inf for summary in summaries)
    remainder = contract.response.remainder_bound(radius)
    projected_norm = float(np.linalg.norm(contract.projected_residual + p))
    output_bound = math.hypot(projected_norm, discarded) + remainder
    upper = output_bound * output_bound / 2
    if not math.isfinite(upper):
        upper = math.inf
    accepted = upper < loss - acceptance_margin * max(1.0, loss)
    return Decision(accepted, upper, loss, projected_norm, discarded, remainder, radius)


def expand_contract(
    contract: LossContract, network: ReLUNetwork, output_direction: object
) -> LossContract:
    """Append a missing output direction without moving the anchor.

    A changed basis gets a changed ID. Previously generated summaries cannot be
    mixed into the new contract. Computing/querying the direction is not free.
    """
    if network.version != contract.model_version:
        raise ValueError("fusion model version changed")
    direction = frozen_array(output_direction)
    if direction.shape != contract.response.output.shape:
        raise ValueError("output direction has wrong dimension")
    candidate = direction.copy()
    for _ in range(2):
        candidate -= contract.basis @ (contract.basis.T @ candidate)
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-12 * max(1.0, float(np.linalg.norm(direction))):
        return contract
    q = np.column_stack((contract.basis, candidate / norm))
    return compile_contract(
        network, contract.base_features, contract.target,
        [node.base_features.size for node in contract.nodes], basis=q,
    )
