import os
import sys
import traceback

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    """
    Hydra entry point
    """

    # Adjust sys.path to include the 'src' directory
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    _SRC_ROOT = os.path.join(_PROJECT_ROOT, "src")
    if _SRC_ROOT not in sys.path:
        sys.path.insert(0, _SRC_ROOT)

    # Lazy import to avoid expensive imports during hydra job submission
    from optimize import run_optimization

    try:
        run_optimization(config)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
