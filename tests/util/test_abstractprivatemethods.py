import sys
from typing import final

import pytest

from execution_engine.util import AbstractPrivateMethods


@pytest.mark.skipif(sys.version_info < (3, 11), reason="requires Python 3.11 or higher")
class TestAbstractPrivateMethods:
    def test_prevent_override(self):
        class Parent(metaclass=AbstractPrivateMethods):
            @final
            def private_method(self):
                pass

            def public_method(self):
                pass

        with pytest.raises(
            RuntimeError,
            match="Methods TestAbstractPrivateMethods.test_prevent_override.<locals>.Parent.private_method may not be overriden",
        ):

            class Child(Parent):
                def private_method(self):
                    pass

        # Test that public_method can be overridden without error
        class ChildPublic(Parent):
            def public_method(self):
                pass

        child = ChildPublic()
        child.public_method()  # No error should be raised
