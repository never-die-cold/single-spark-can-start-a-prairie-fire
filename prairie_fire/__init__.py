"""Research reference code; this package is not a distributed training runtime."""

from .contracts import (
    compile_contract,
    expand_contract,
    judge,
    summarize,
)
from .networks import ReLUNetwork

__all__ = ["ReLUNetwork", "compile_contract", "expand_contract", "judge", "summarize"]
__version__ = "0.1.0"
