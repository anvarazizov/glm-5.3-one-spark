#!/usr/bin/env python3
"""Fix hybrid-Mamba prefix-hit state indexing (vLLM #55600 / PR #55601).

EngineCore may lower ``cache_config.block_size`` to the smallest cacheable
attention-group block. The Mamba/KDA state table is still indexed in
``mamba_block_size`` units. Using the former for a resumed prefix restores the
wrong recurrent state and can index past the table.

This runtime overlay is fail-closed and idempotent. It targets the exact source
shape in the pinned image and refuses to modify a drifted or partial patch.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


TARGET = Path(
    os.environ.get(
        "GLM53_MAMBA_HYBRID_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/"
        "model_states/mamba_hybrid.py",
    )
)
MARK = "            # [glm53-mamba-prefix-seed] Mamba table uses mamba blocks.\n"

ANCHOR = """            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )
"""

PATCHED = """            # [glm53-mamba-prefix-seed] Mamba table uses mamba blocks.
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1)
                // self.cache_config.mamba_block_size
            )
"""


def resumed_state_index(num_computed_tokens: int, mamba_block_size: int) -> int:
    if num_computed_tokens < 1:
        raise ValueError("a resumed prefix must contain at least one token")
    if mamba_block_size < 1:
        raise ValueError("mamba_block_size must be positive")
    return (num_computed_tokens - 1) // mamba_block_size


def verified_state(source: str) -> bool:
    return (
        source.count(ANCHOR) == 0
        and source.count(PATCHED) == 1
        and source.count(MARK) == 1
        and "// self.cache_config.block_size" not in source
    )


def prepare(source: str) -> tuple[str, str]:
    if verified_state(source):
        return source, "already present"
    if MARK in source or "// self.cache_config.mamba_block_size" in source:
        raise ValueError("partial or independently modified Mamba seed fix")
    count = source.count(ANCHOR)
    if count != 1:
        raise ValueError(f"pinned add_request anchor count is {count}, expected 1")
    patched = source.replace(ANCHOR, PATCHED, 1)
    if not verified_state(patched):
        raise ValueError("post-patch verification failed")
    return patched, "patched"


def replace_file(target: Path, source: str) -> None:
    tmp = target.with_name(f".{target.name}.glm53-mamba-seed.tmp")
    try:
        tmp.write_text(source)
        os.chmod(tmp, stat.S_IMODE(target.stat().st_mode))
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    if not TARGET.is_file():
        raise SystemExit(f"missing {TARGET}")
    source = TARGET.read_text()
    try:
        patched, action = prepare(source)
    except ValueError as exc:
        raise SystemExit(f"mamba prefix seed preflight failed: {exc}") from exc
    compile(patched, str(TARGET), "exec")
    if patched != source:
        replace_file(TARGET, patched)
    print(f"{TARGET.name}: mamba prefix seed {action} (vLLM #55600)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
