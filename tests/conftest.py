"""
    Dummy conftest.py for xtal2png.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
import os

import pytest
from pymatgen.core import Structure

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def get_disordered_structure():
    return Structure.from_file(
        os.path.join(THIS_DIR, "test_files", "disordered_structure.cif")
    )
