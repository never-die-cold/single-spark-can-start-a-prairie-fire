"""Reproduce the interface checks described in the research discussion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

import numpy as np

from .baselines import disk_backed_xor
from .contracts import compile_contract, judge, summarize
from .networks import ReLUNetwork


def compressed_checks(*, seed: int, cases: int, trials: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    nodes, width, hidden, outputs, rank = 4, 16, 24, 128, 4
    total = nodes * width
    worst_violation = -float("inf")
    max_gradient_error = 0.0
    rejected_old = accepted_crossing = decreased_crossing = 0
    bound_failures = 0
    for case in range(cases):
        w1 = rng.normal(size=(hidden, total)) / np.sqrt(total)
        z0 = rng.normal(size=total)
        margins = rng.uniform(.3, 1.5, hidden) * rng.choice([-1., 1.], hidden)
        margins[0] = 1e-9
        b1 = margins - w1 @ z0
        w2 = rng.normal(size=(outputs, hidden)) / np.sqrt(hidden)
        w2[:, 0] *= 1e-6
        b2 = rng.normal(scale=.1, size=outputs)
        network = ReLUNetwork((w1, w2), (b1, b2))
        target = rng.normal(size=outputs)
        contract = compile_contract(network, z0, target, [width] * nodes, rank=rank, seed=seed + case)
        for node in contract.nodes:
            start = node.node_id * width
            exact = contract.response.jacobian[:, start:start + width].T @ (contract.response.output - target)
            compact = node.projection.T @ contract.projected_residual
            max_gradient_error = max(max_gradient_error, float(np.linalg.norm(exact - compact) / (np.linalg.norm(exact) + 1e-30)))
        for _ in range(trials):
            delta = rng.normal(size=total) * 10 ** rng.uniform(-5, 0)
            summaries = [summarize(node, delta[node.node_id * width:(node.node_id + 1) * width]) for node in contract.nodes]
            decision = judge(contract, summaries)
            actual = float(np.sum((network.forward(z0 + delta) - target) ** 2) / 2)
            violation = actual - decision.loss_upper_bound
            worst_violation = max(worst_violation, violation)
            bound_failures += int(violation > 1e-10 * max(1.0, actual))
        gradient = contract.response.jacobian.T @ (contract.response.output - target)
        delta = -gradient / (np.max(np.abs(gradient)) + 1e-30) * 1e-4
        if w1[0] @ delta >= -margins[0]:
            w1[0] *= -1
            b1[0] = margins[0] - w1[0] @ z0
            network = ReLUNetwork((w1, w2), (b1, b2))
            contract = compile_contract(network, z0, target, [width] * nodes, rank=rank, seed=seed + case)
            gradient = contract.response.jacobian.T @ (contract.response.output - target)
            delta = -gradient / (np.max(np.abs(gradient)) + 1e-30) * 1e-4
        summaries = [summarize(node, delta[node.node_id * width:(node.node_id + 1) * width]) for node in contract.nodes]
        decision = judge(contract, summaries)
        changed = np.any((w1 @ (z0 + delta) + b1 > 0) != (margins > 0))
        rejected_old += int(np.max(np.abs(delta)) > contract.response.strict_region_radius_inf())
        if changed and decision.accepted:
            accepted_crossing += 1
            actual = float(np.sum((network.forward(z0 + delta) - target) ** 2) / 2)
            decreased_crossing += int(actual < contract.base_loss)
    return {
        "simulated_nodes": nodes, "feature_dim_per_node": width,
        "output_dim": outputs, "shared_rank": rank, "cases": cases,
        "arbitrary_update_checks": cases * trials, "bound_failures": bound_failures,
        "max_actual_loss_minus_upper_bound": worst_violation,
        "max_relative_base_gradient_error": max_gradient_error,
        "constructed_near_boundary_candidates_rejected_by_old_rule": rejected_old,
        "gate_crossing_candidates_accepted": accepted_crossing,
        "accepted_crossings_with_actual_descent": decreased_crossing,
        "response_payload_only": {
            "dtype": "float64", "dense_bytes_per_node": outputs * width * 8,
            "compact_matrix_and_tail_bytes_per_node": (rank * width + 1) * 8,
            "proposal_summary_numbers_per_node_per_sample": rank + 2,
            "excluded": ["basis/coordinator state", "contract construction queries", "metadata", "committed features", "batch and sequence multipliers"],
        },
    }


def multilayer_checks(*, seed: int, cases: int, trials: int = 40) -> dict:
    rng = np.random.default_rng(seed)
    worst_violation = -float("inf")
    failures = 0
    for _ in range(cases):
        sizes = [24, 14, 10, 32]
        network = ReLUNetwork(
            tuple(rng.normal(size=(b, a)) / np.sqrt(a) for a, b in zip(sizes[:-1], sizes[1:])),
            tuple(rng.normal(scale=.3, size=b) for b in sizes[1:]),
        )
        base = rng.normal(size=sizes[0])
        response = network.response_at(base)
        for _ in range(trials):
            delta = rng.normal(size=base.size) * 10 ** rng.uniform(-6, 1)
            bound = response.remainder_bound(float(np.max(np.abs(delta))))
            actual = float(np.linalg.norm(network.forward(base + delta) - response.output - response.jacobian @ delta))
            worst_violation = max(worst_violation, actual - bound)
            failures += int(actual > bound + 1e-10 * max(1.0, actual))
    return {"checks": cases * trials, "failures": failures, "max_actual_remainder_minus_bound": worst_violation}


def strict_parallel_checks(*, seed: int, cases: int) -> dict:
    rng = np.random.default_rng(seed)
    failures = 0
    worst_gap = -float("inf")
    nodes, width = 4, 4
    for _ in range(cases):
        sizes = [nodes * width, 12, 10, 3]
        network = ReLUNetwork(
            tuple(rng.normal(size=(b, a)) / np.sqrt(a) for a, b in zip(sizes[:-1], sizes[1:])),
            tuple(rng.normal(scale=.3, size=b) for b in sizes[1:]),
        )
        base = rng.normal(size=sizes[0])
        response = network.response_at(base)
        error = response.output - rng.normal(size=sizes[-1])
        radius = response.strict_region_radius_inf()
        delta = np.zeros_like(base)
        local_bounds = []
        for node in range(nodes):
            part = slice(node * width, (node + 1) * width)
            a = response.jacobian[:, part]
            change = -np.linalg.pinv(a) @ error / nodes
            change *= min(1.0, .8 * radius / (np.max(np.abs(change)) + 1e-30))
            delta[part] = change
            v = a @ change
            local_bounds.append(float(error @ v + nodes * (v @ v) / 2))
        actual_error = error + network.forward(base + delta) - response.output
        loss_change = float((actual_error @ actual_error - error @ error) / 2)
        upper = sum(local_bounds)
        gap = loss_change - upper
        worst_gap = max(worst_gap, gap)
        failures += int(gap > 1e-10 or upper > 1e-10 or loss_change > 1e-10)
    return {"checks": cases, "failures": failures, "max_actual_loss_change_minus_local_bound": worst_gap}


def run(seed: int, *, quick: bool = False) -> dict:
    compressed = compressed_checks(seed=seed, cases=6 if quick else 100, trials=8 if quick else 20)
    multilayer = multilayer_checks(seed=6621, cases=4 if quick else 40, trials=10 if quick else 40)
    parallel = strict_parallel_checks(seed=seed, cases=20 if quick else 1000)
    xor = disk_backed_xor(seed=seed, rounds=400 if quick else 1600)
    passed = (
        compressed["bound_failures"] == 0
        and compressed["max_relative_base_gradient_error"] < 1e-10
        and compressed["gate_crossing_candidates_accepted"] == compressed["accepted_crossings_with_actual_descent"]
        and multilayer["failures"] == 0 and parallel["failures"] == 0
        and xor["max_replica_difference"] == 0
    )
    return {
        "schema_version": 1, "seed": seed, "quick": quick,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "scope": "single-process mathematical/interface validation; XOR is a separate earlier baseline",
        "does_not_establish": ["originality", "distributed training speed", "end-to-end current-protocol convergence", "energy savings", "measured peak memory", "formal floating-point verification"],
        "compressed_contract": compressed, "multilayer_remainder": multilayer,
        "strict_parallel_baseline": parallel, "zeroth_order_xor_baseline": xor,
        "all_property_checks_passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seed, quick=args.quick)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not result["all_property_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
