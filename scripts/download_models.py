"""Download and validate default SeedVR2 models for ComfyUI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.downloads import download_weight  # noqa: E402
from src.utils.model_registry import DEFAULT_DIT, DEFAULT_VAE  # noqa: E402
from src.utils.constants import get_base_cache_dir  # noqa: E402


def main() -> int:
    model_dir = Path(get_base_cache_dir())
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading and validating {DEFAULT_DIT} and {DEFAULT_VAE}...")
    if not download_weight(DEFAULT_DIT, DEFAULT_VAE, str(model_dir)):
        print("Model download failed. Run this installer again to resume.", file=sys.stderr)
        return 1
    print(f"Default SeedVR2 models are ready in {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
