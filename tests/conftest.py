import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Exit right after pytest has reported its results.

    On Windows the process can hang while native thread pools (OpenCV / OpenBLAS)
    are torn down during interpreter finalization, after all tests have already
    finished. Skipping that teardown keeps `pytest` from hanging and preserves the
    real exit status.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(getattr(config, "_exitstatus", 0))


def pytest_sessionfinish(session, exitstatus):
    session.config._exitstatus = int(exitstatus)
