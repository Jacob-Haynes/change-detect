import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from changedetect import synthetic  # noqa: E402


@pytest.fixture(scope="session")
def flat_pair():
    return synthetic.make_pair()


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    d = tmp_path_factory.mktemp("samples")
    return d, synthetic.write_samples(d)
