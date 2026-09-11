#!/usr/bin/env python3
"""Regression tests for the GLM-5.3 hybrid-Mamba prefix-hit seed fix."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_mamba_hybrid_seed.py"
sys.path.insert(0, str(PATCH.parent))

from patch_mamba_hybrid_seed import (  # noqa: E402
    ANCHOR,
    MARK,
    PATCHED,
    prepare,
    resumed_state_index,
    verified_state,
)


FIXTURE = """class MambaHybridModelState:
    def add_request(self, req_index, new_req_data):
        if self._align_mode:
""" + ANCHOR


def run_patch(path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GLM53_MAMBA_HYBRID_PY"] = str(path)
    return subprocess.run(
        [sys.executable, str(PATCH)],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_units() -> None:
    assert resumed_state_index(14_336, 7_168) == 1
    assert (14_336 - 1) // 64 == 223
    assert resumed_state_index(7_168, 7_168) == 0
    assert resumed_state_index(7_169, 7_168) == 1


def test_apply_and_idempotence() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "mamba_hybrid.py"
        target.write_text(FIXTURE)
        first = run_patch(target)
        assert first.returncode == 0, first.stderr
        source = target.read_text()
        assert verified_state(source)
        assert PATCHED in source and MARK in source
        second = run_patch(target)
        assert second.returncode == 0, second.stderr
        assert "already present" in second.stdout
        again, action = prepare(source)
        assert action == "already present" and again == source


def test_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "mamba_hybrid.py"
        target.write_text(FIXTURE.replace("cache_config.block_size", "block_size"))
        result = run_patch(target)
        assert result.returncode != 0
        assert "preflight failed" in result.stderr


def main() -> int:
    test_units()
    test_apply_and_idempotence()
    test_fail_closed()
    print("mamba hybrid prefix seed patch OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
