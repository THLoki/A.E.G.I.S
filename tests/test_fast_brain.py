#!/usr/bin/env python3
"""A.E.G.I.S Fast Brain — interactive test loop.

Provides both an interactive chat loop for manual testing and a
non-interactive smoke test for validation.

Usage:
    Interactive:  python tests/test_fast_brain.py
    Smoke test:   python tests/test_fast_brain.py --smoke
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path for src imports.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.brains.fast_brain import FastBrain  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def run_interactive_loop(brain: FastBrain) -> None:
    """Run an interactive chat loop with performance metrics."""
    log.info("A.E.G.I.S Fast Brain — Interactive Test")
    log.info("Type 'quit' or 'exit' to stop. Ctrl+C also works.")
    log.info("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            log.info("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            log.info("Goodbye.")
            break

        start = time.monotonic()
        response = brain.generate_response(user_input)
        elapsed = time.monotonic() - start

        print(f"\nA.E.G.I.S: {response}")
        log.info("(Response time: %.2fs)", elapsed)


def run_smoke_test(brain: FastBrain) -> bool:
    """Run a single non-interactive inference as a smoke test.

    Returns:
        True if the test passed, False otherwise.
    """
    test_prompt = "What is 2 + 2? Answer in one word."
    log.info("Smoke test prompt: %s", test_prompt)

    start = time.monotonic()
    response = brain.generate_response(test_prompt)
    elapsed = time.monotonic() - start

    log.info("Response: %s", response)
    log.info("Time: %.2fs", elapsed)

    if response and len(response.strip()) > 0:
        log.info("SMOKE TEST PASSED")
        return True

    log.error("SMOKE TEST FAILED — empty response")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A.E.G.I.S Fast Brain test loop",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a single non-interactive smoke test instead of the chat loop.",
    )
    args = parser.parse_args()

    log.info("Initialising Fast Brain...")
    brain = FastBrain()

    if args.smoke:
        success = run_smoke_test(brain)
        sys.exit(0 if success else 1)
    else:
        run_interactive_loop(brain)


if __name__ == "__main__":
    main()
