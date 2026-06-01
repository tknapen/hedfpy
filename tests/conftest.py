"""Shared pytest fixtures for the hedfpy test suite.

The ``tk_1`` fixture under ``test/`` ships its pre-converted ``.msg``,
``.gaz`` and ``.gaz.gz`` files next to the ``.edf``. This means the test
suite does **not** need SR Research's ``edf2asc`` binary on ``PATH``:

* ``EDFOperator`` skips the edf->asc conversion when the ``.msg`` already
  exists, and
* ``clean_gaze_information`` skips gaze cleaning when the ``.gaz.gz``
  already exists.

The fixture files are copied into a temporary directory before the
pipeline runs, because hedfpy writes its intermediate and output files
next to the input ``.edf`` and we never want it mutating the committed
fixtures.
"""

# Force a non-interactive matplotlib backend before anything in the test
# suite (or hedfpy itself) imports matplotlib.pyplot. The pipeline's
# summary_plot step instantiates a figure and would otherwise try to talk
# to a Qt display, which hangs (does not raise) on headless or
# X-forwarding-dropped sessions.
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import shutil
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "test"
TK1_STEM = "tk_1_2014-10-06_16.44.11"
TK1_SUFFIXES = (".edf", ".msg", ".gaz", ".gaz.gz")


@pytest.fixture(scope="session")
def tk1_hdf(tmp_path_factory):
    """Run the full hedfpy pipeline on the committed ``tk_1`` fixture.

    Returns the path to the HDF5 file produced by, in order,
    ``add_edf_file``, ``edf_message_data_to_hdf`` and
    ``edf_gaze_data_to_hdf``. Session-scoped: the pipeline takes a few
    seconds, so it runs once and is shared across the suite.
    """
    import hedfpy

    work = tmp_path_factory.mktemp("tk1")
    for suffix in TK1_SUFFIXES:
        src = FIXTURE_DIR / f"{TK1_STEM}{suffix}"
        if not src.exists():
            pytest.skip(f"missing fixture file: {src}")
        shutil.copy(src, work / src.name)

    edf = work / f"{TK1_STEM}.edf"
    hdf = work / "tk1.h5"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ho = hedfpy.HDFEyeOperator(str(hdf))
        ho.add_edf_file(str(edf))
        ho.edf_message_data_to_hdf(alias="test")
        ho.edf_gaze_data_to_hdf(alias="test")

    return hdf
