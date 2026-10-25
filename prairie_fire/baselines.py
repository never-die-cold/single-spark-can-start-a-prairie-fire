"""Earlier proposals retained for reproducibility, not originality claims."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np


def disk_backed_xor(*, seed: int = 20261025, rounds: int = 1600) -> dict:
    """Simulate seeded, scalar-only, block zeroth-order updates on eight replicas.

    The files live in a temporary directory and are removed after this run.
    There is no network transport. Memory mapping is not an RSS benchmark.
    """
    rng = np.random.default_rng(seed)
    x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    y = x[:, 0] * x[:, 1]
    xa = np.column_stack((x, np.ones(len(x))))
    hidden, block, workers = 8, 8, 8
    parameters = 4 * hidden + 1
    initial = rng.normal(0, 0.3, parameters)

    def forward(w, start=0, direction=None, offset=0.0):
        def read(a, b):
            values = np.array(w[a:b], copy=True)
            if direction is not None:
                lo, hi = max(a, start), min(b, start + len(direction))
                if lo < hi:
                    values[lo - a:hi - a] += offset * direction[lo - start:hi - start]
            return values

        activations = np.empty((len(x), hidden))
        for j in range(hidden):
            activations[:, j] = np.tanh(xa @ read(3 * j, 3 * j + 3))
        return activations @ read(3 * hidden, parameters - 1) + read(parameters - 1, parameters)[0]

    def loss(w, start=0, direction=None, offset=0.0):
        return float(np.mean((forward(w, start, direction, offset) - y) ** 2) / 2)

    with tempfile.TemporaryDirectory(prefix="prairie-fire-xor-") as directory:
        models = []
        for index in range(workers):
            weights = np.memmap(Path(directory) / f"node-{index}.bin", mode="w+", dtype=np.float64, shape=(parameters,))
            weights[:] = initial
            models.append(weights)
        begin = loss(models[0])
        dense_reference = initial.copy()
        max_difference = 0.0
        for step in range(rounds):
            start = (step % ((parameters + block - 1) // block)) * block
            stop = min(parameters, start + block)
            transcript = []
            for index, weights in enumerate(models):
                direction_seed = step * workers + index + 10000
                direction = np.random.default_rng(direction_seed).choice([-1., 1.], stop - start)
                score = (loss(weights, start, direction, .01) - loss(weights, start, direction, -.01)) / .02
                transcript.append((direction_seed, score))
            update = np.zeros(stop - start)
            for direction_seed, score in transcript:
                direction = np.random.default_rng(direction_seed).choice([-1., 1.], stop - start)
                update += score * direction / workers
            dense_reference[start:stop] -= .06 * update
            for weights in models:
                local = np.zeros(stop - start)
                for direction_seed, score in transcript:
                    direction = np.random.default_rng(direction_seed).choice([-1., 1.], stop - start)
                    local += score * direction / workers
                weights[start:stop] -= .06 * local
            if step % 100 == 0 or step == rounds - 1:
                max_difference = max(max_difference, max(float(np.max(np.abs(w - dense_reference))) for w in models))
        final_loss = loss(models[0])
        predictions = forward(models[0]).tolist()
        accuracy = float(np.mean(np.sign(predictions) == y))
        # Explicitly close mappings before TemporaryDirectory cleanup on Windows.
        for weights in models:
            weights.flush()
            weights._mmap.close()
    return {
        "parameters": parameters, "simulated_nodes": workers, "block_size": block,
        "rounds": rounds, "initial_loss": begin, "final_loss": final_loss,
        "sign_accuracy": accuracy, "max_replica_difference": max_difference,
        "predictions": predictions,
        "scope": "single-process, disk-backed zeroth-order baseline; known mechanisms",
    }
