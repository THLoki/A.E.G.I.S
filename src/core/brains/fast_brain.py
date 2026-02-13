"""A.E.G.I.S Fast Brain — quantized local LLM inference via llama.cpp.

Provides the FastBrain class that loads a GGUF model fully onto GPU
and exposes a chat-completion interface for fast interactive responses.
"""

import logging
import time
from pathlib import Path

from llama_cpp import Llama

from src.core.prompts import AEGIS_SYSTEM_PROMPT

log = logging.getLogger(__name__)

# --- Default model path ---
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MODEL_PATH: Path = (
    _PROJECT_ROOT / "models" / "fast_brain"
    / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)


class FastBrain:
    """Fast local LLM brain using llama-cpp-python with full GPU offload.

    Loads the quantized GGUF model once at instantiation time and provides
    a ``generate_response`` method for chat-style inference.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
    ) -> None:
        """Initialise the Fast Brain.

        Args:
            model_path: Path to GGUF model. Defaults to the standard
                        models/fast_brain/ location.
            n_gpu_layers: Number of layers to offload to GPU.
                          -1 means all layers (full GPU offload).
            n_ctx: Context window size in tokens.
        """
        self.model_path: Path = model_path or DEFAULT_MODEL_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run 'python scripts/download_models.py' first."
            )

        log.info("Loading Fast Brain model: %s", self.model_path)
        load_start = time.monotonic()

        self.llm = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

        load_elapsed = time.monotonic() - load_start
        log.info("Fast Brain loaded in %.2f seconds.", load_elapsed)

    def generate_response(
        self,
        user_input: str,
        system_prompt: str = AEGIS_SYSTEM_PROMPT,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response to user input.

        Args:
            user_input: The user's message text.
            system_prompt: System-level instruction for the model.
            temperature: Sampling temperature (higher = more creative).
            top_p: Nucleus sampling threshold.
            max_tokens: Maximum tokens to generate.

        Returns:
            The model's response text.

        Note:
            This is a synchronous, blocking call. For async contexts,
            wrap with ``asyncio.to_thread(brain.generate_response, ...)``.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        log.debug("Generating response for: %.80s...", user_input)
        start = time.monotonic()

        result = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        elapsed = time.monotonic() - start
        response_text: str = result["choices"][0]["message"]["content"]
        token_count: int = result["usage"]["completion_tokens"]

        tokens_per_sec = token_count / elapsed if elapsed > 0 else 0.0
        log.info(
            "Generated %d tokens in %.2fs (%.1f t/s)",
            token_count,
            elapsed,
            tokens_per_sec,
        )

        return response_text
