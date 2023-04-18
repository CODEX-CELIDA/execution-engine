from glob import glob

import pytest

pytest_plugins = [
    fixture_file.replace("/", ".").replace(".py", "")
    for fixture_file in glob("tests/fixtures/[!__]*.py", recursive=True)
]


def pytest_addoption(parser):  # type: ignore
    """
    Add command line options to pytest.

    See https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    """
    parser.addoption(
        "--run-slow-tests", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--run-recommendation-tests",
        action="store_true",
        default=False,
        help="run recommendation tests",
    )


def pytest_configure(config):  # type: ignore
    """
    Add custom markers to pytest.

    See https://docs.pytest.org/en/latest/example/markers.html#marking-test-functions-and-selecting-them-for-a-run
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "recommendation: mark test as a recommendation test"
    )


def pytest_collection_modifyitems(config, items):  # type: ignore
    """
    Skip slow tests if --run-slow-tests is not given.

    See https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    """
    if not config.getoption("--run-slow-tests"):
        # --run-slow not given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --run-slow-tests option to run")

        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--run-recommendation-tests"):
        # --run-recommendation-tests not given in cli: skip recommendation tests
        skip_recommendation = pytest.mark.skip(
            reason="need --run-recommendation-tests option to run"
        )

        for item in items:
            if "recommendation" in item.keywords:
                item.add_marker(skip_recommendation)


@pytest.fixture
def run_slow_tests(request) -> bool:  # type: ignore
    """
    Fixture to determine if slow tests should be run.
    """
    return request.config.getoption("--run-slow-tests")


@pytest.fixture
def run_recommendation_tests(request) -> bool:  # type: ignore
    """
    Fixture to determine if recommendation tests should be run.
    """
    return request.config.getoption("--run-recommendation-tests")
