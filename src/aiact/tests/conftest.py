import pytest
from aiact.engine import ClassificationEngine


@pytest.fixture(scope="session")
def engine_fixture():
    return ClassificationEngine()
