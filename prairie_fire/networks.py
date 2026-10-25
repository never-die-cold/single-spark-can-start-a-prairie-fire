"""Small dense ReLU oracle used to validate response and remainder formulas.

The reference implementation deliberately materializes a Jacobian. Its storage
must not be cited as evidence of a low-memory implementation on actual devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def frozen_array(value: object) -> Array:
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("arrays must contain finite numbers")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Response:
    output: Array
    jacobian: Array
    preactivations: tuple[Array, ...]
    sensitivities: tuple[Array, ...]
    output_influences: tuple[Array, ...]

    def remainder_bound(self, radius_inf: float) -> float:
        """Upper bound on ||G(z0+d)-G(z0)-J0 d||_2 for ||d||_inf <= R.

        For each gate, its linearization error is bounded by
        [kappa * R - |preactivation|]_+. The output influences include the
        baseline downstream masks; later gate errors separately account for
        cascaded switching. See docs/design.md for the telescoping argument.
        """
        if not math.isfinite(radius_inf) or radius_inf < 0:
            raise ValueError("radius must be finite and nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            result = sum(
                float(beta @ np.maximum(kappa * radius_inf - np.abs(pre), 0.0))
                for beta, kappa, pre in zip(
                    self.output_influences, self.sensitivities, self.preactivations
                )
            )
        return float(result) if math.isfinite(result) else math.inf

    def strict_region_radius_inf(self) -> float:
        """Conservative old rule, retained as a comparison rather than a gate."""
        radius = math.inf
        for pre, kappa in zip(self.preactivations, self.sensitivities):
            moving = kappa > 0
            if moving.any():
                radius = min(radius, float(np.min(np.abs(pre[moving]) / kappa[moving])))
        return radius


@dataclass(frozen=True)
class ReLUNetwork:
    """Affine/ReLU/.../affine network with owned, read-only float64 arrays."""

    weights: tuple[Array, ...]
    biases: tuple[Array, ...]

    def __post_init__(self) -> None:
        weights = tuple(frozen_array(w) for w in self.weights)
        biases = tuple(frozen_array(b) for b in self.biases)
        if not weights or len(weights) != len(biases):
            raise ValueError("provide equal nonempty weight and bias sequences")
        for index, (weight, bias) in enumerate(zip(weights, biases)):
            if weight.ndim != 2 or min(weight.shape) < 1:
                raise ValueError("each weight must be a nonempty matrix")
            if bias.shape != (weight.shape[0],):
                raise ValueError("bias and weight output dimensions do not match")
            if index and weight.shape[1] != weights[index - 1].shape[0]:
                raise ValueError("adjacent layer dimensions do not match")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "biases", biases)

    @property
    def version(self) -> str:
        digest = sha256()
        for array in (*self.weights, *self.biases):
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        return digest.hexdigest()

    def _input(self, value: object) -> Array:
        array = frozen_array(value)
        if array.shape != (self.weights[0].shape[1],):
            raise ValueError("expected one input feature vector; batching is not implemented")
        return array

    def forward(self, value: object) -> Array:
        result = self._input(value)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            result = weight @ result + bias
            if index < len(self.weights) - 1:
                result = np.maximum(result, 0.0)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("network output overflowed")
        return result

    def response_at(self, value: object) -> Response:
        result = self._input(value)
        jacobian = np.eye(result.size)
        sensitivity = np.ones(result.size)
        preactivations: list[Array] = []
        sensitivities: list[Array] = []
        masks: list[NDArray[np.bool_]] = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            result = weight @ result + bias
            jacobian = weight @ jacobian
            sensitivity = np.abs(weight) @ sensitivity
            if index < len(self.weights) - 1:
                preactivations.append(frozen_array(result))
                sensitivities.append(frozen_array(sensitivity))
                mask = result > 0
                masks.append(mask)
                result = np.maximum(result, 0.0)
                jacobian = mask[:, None] * jacobian
        influences: list[Array] = [np.empty(0)] * len(masks)
        downstream = self.weights[-1].copy()
        for index in range(len(masks) - 1, -1, -1):
            influences[index] = frozen_array(np.linalg.norm(downstream, axis=0))
            downstream = (downstream * masks[index][None, :]) @ self.weights[index]
        return Response(
            frozen_array(result), frozen_array(jacobian), tuple(preactivations),
            tuple(sensitivities), tuple(influences),
        )
