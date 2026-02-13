# A.E.G.I.S - Project Guidelines for Claude

## Project Overview
**A.E.G.I.S** (Advanced Electronic General Intelligence System) is a modular, local-first, privacy-centric AI assistant running on consumer hardware.
* **Goal:** Hybrid Intelligence (Fast Local LLM + Slow "Deep Thought" LLM via AirLLM).
* **Environment:** Linux Headless (Server).
* **Hardware Constraints:** NVIDIA RTX 3070 (8GB VRAM), 16GB DDR4 RAM. **STRICT RESOURCE MANAGEMENT REQUIRED.**
* **Driver:** NVIDIA 580.x (CUDA Toolkit 13.0 compatible). PyTorch uses CUDA 12.4 wheels (forward-compatible with driver CUDA 13.0).

## Workflow & Git Protocol (STRICT)
You must follow this workflow for EVERY task. Do not commit directly to `main`.

1.  **Understand:** Read the current task/ticket (Jira style). If anything is ambiguous, **ASK** before coding.
2.  **Branch:** Create a new branch for the task.
    * Naming convention: `feature/AEGIS-[ID]-[short-description]` or `fix/AEGIS-[ID]-[short-description]`.
    * Example: `feature/AEGIS-1-setup-venv`
3.  **Implement:** Write the code. Keep changes atomic, modular  and focused on the task.
4.  **Test:**
    * Create a simple validation script (e.g., `tests/test_gpu.py`) to verify your changes work.
    * Run the test. **Do not proceed** if tests fail.
5.  **Commit:** Use conventional commits.
    * Example: `feat: add hardware check script`, `fix: adjust swap file size`.
6.  **Pull Request:**
    * Push the branch to origin.
    * Generate a Pull Request description summarizing:
        * What was done.
        * How to test it.
        * Any manual steps required (e.g., `sudo` commands).
    * **STOP** and wait for User Review/Merge.

## Technology Stack
* **Language:** Python 3.13 (venv at `.venv/`). Compatible with 3.10+ code.
* **Package Manager:** `pip` (inside `venv`).
* **Core Libraries:** `torch` (CUDA 12.x), `transformers`, `accelerate`, `airllm`, `chromadb`.
* **Communication:** Telegram Bot API, SMTP/IMAP.

## Critical Architecture Constraints
1.  **Memory First:**
    * **VRAM (8GB):** Reserved for the "Fast Brain" (Quantized 4-bit models) + Context Window.
    * **RAM (16GB):** Extremely limited. Used for OS + App Logic.
    * **Offloading:** "Big Brain" (70B) **MUST** use AirLLM/Disk-Offloading (NVMe SSD Swap).
2.  **No Docker for Inference:**
    * To maximize VRAM efficiency, run LLMs directly on the host (Python venv). Do not suggest Dockerizing the LLM core unless explicitly asked.
3.  **Async by Default:**
    * All I/O (Telegram, Email, LLM generation) must be asynchronous (`asyncio`) to prevent blocking the event loop during long inference times.

## Code Style
* **Modular:** Keep "Sense" (Input), "Brain" (LLM), and "Act" (Tools) separate.
* **Typing:** Use Python Type Hints (`def process(text: str) -> bool:`).
* **Logging:** Use the standard `logging` library. Print to stdout/stderr is acceptable for MVP debugging but structured logging is preferred.
* **Pathing:** Use `pathlib` for file system operations.

## Meta-Learning & Documentation
* **Maintain this file:** If you encounter a recurring issue, a change in hardware constraints, or a new architectural decision during development, **UPDATE THIS `CLAUDE.md` FILE** in your Pull Request.
* **Learn from errors:** If a library version fails or a specific approach proves inefficient on this hardware, document the working solution here to avoid repeating the mistake in future tasks.
* **User Constraints:** Respect any new constraints added by the user to this file immediately.
