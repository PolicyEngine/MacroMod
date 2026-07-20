"""Shared pytest configuration for the policyengine-macro integration tests.

Tests marked `slow` run a real model estimation/solve or a full PolicyEngine
import (seconds to many minutes). They are skipped by default so a plain
`pytest` run stays fast and CI-friendly; run them with `pytest --runslow`.
Mirrors the OBR emulator repo's tests/conftest.py.

The live-deployment tests in test_remote_mcp.py are gated separately by the
POLICYENGINE_MACRO_REMOTE_TESTS environment variable.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run slow tests (real model estimation/solve/import)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow test; use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
