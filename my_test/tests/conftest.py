"""Optional patched Taisei source used by live-controller integration tests."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATCHED_SOURCE = ROOT / "windows-build" / "wlive_agent_demo" / "src"
REQUIRED_PATCHED_FILES = ("cli.c", "stage.c", "live_ppo.c", "live_ppo_infer.h")


@pytest.fixture(scope="session")
def patched_live_source():
    """Return the local patched source directory or clearly skip its tests."""
    configured = os.environ.get("TAISEI_PATCHED_SOURCE")
    source = Path(configured).expanduser() if configured else DEFAULT_PATCHED_SOURCE
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()

    missing = [name for name in REQUIRED_PATCHED_FILES if not (source / name).is_file()]
    if missing:
        detail = ", ".join(missing)
        pytest.skip(
            "optional live-source checks need patched Taisei source at "
            f"{source} (missing: {detail}); set TAISEI_PATCHED_SOURCE to the "
            "patched source src/ directory to enable them"
        )
    return source
