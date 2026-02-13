#!/usr/bin/env python3
"""Download A.E.G.I.S Fast Brain model from Hugging Face.

Downloads the quantized Llama-3.1-8B-Instruct GGUF model to the local
models/fast_brain/ directory. Safe to re-run — skips if file exists.
"""

import logging
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# --- Configuration ---
REPO_ID: str = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
FILENAME: str = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
MODEL_DIR: Path = PROJECT_ROOT / "models" / "fast_brain"


def download_fast_brain_model() -> Path:
    """Download the Fast Brain GGUF model if not already present.

    Returns:
        Path to the downloaded model file.
    """
    model_path = MODEL_DIR / FILENAME

    if model_path.exists():
        log.info("Model already exists: %s", model_path)
        return model_path

    log.info("Creating model directory: %s", MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s from %s ...", FILENAME, REPO_ID)
    log.info("This may take a while (~4.9 GB).")

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=MODEL_DIR,
    )

    log.info("Download complete: %s", downloaded_path)
    return Path(downloaded_path)


def main() -> None:
    try:
        path = download_fast_brain_model()
        log.info("Model ready at: %s", path)
    except Exception:
        log.exception("Failed to download model")
        sys.exit(1)


if __name__ == "__main__":
    main()
