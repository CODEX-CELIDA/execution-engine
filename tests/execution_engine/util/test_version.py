import pytest

from execution_engine.util.version import is_version_below


class TestVersionCompare:
    @pytest.mark.parametrize(
        "target_version,comparison_version,expected_result",
        [
            ("v.1.3.1", "v1.4", True),
            ("v1.3.1-SNAPSHOT", "v1.4", True),
            ("v1.4", "v1.3", False),
            ("v2.0", "v1.9.9", False),
            ("v1.2.3", "v1.2.3", False),
            ("v.1.2.3-SNAPSHOT", "v1.2.3", True),
            ("v.1.10", "v1.9", False),
            ("v0.9", "v1.0", True),
        ],
    )
    def test_is_version_below(
        self, target_version, comparison_version, expected_result
    ):
        assert is_version_below(target_version, comparison_version) == expected_result
