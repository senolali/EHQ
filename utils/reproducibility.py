"""Reproducibility utilities: seed control and deterministic mode.

Ported verbatim from senolali/RQEval (utils/reproducibility.py).
"""

import os
import random


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def get_reproducibility_info() -> dict:
    """Return current reproducibility configuration."""
    info = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", "not set"),
    }
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        pass
    return info
