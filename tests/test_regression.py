"""End-to-end regression test for the hedfpy pipeline.

This test pins the *current* HDF5 output of the full pipeline so that the
ongoing refactor can be verified not to change behaviour. It is a guard,
not a specification: the baseline simply records what hedfpy produces
today on the ``tk_1`` fixture.

Note that the ``tk_1`` fixture's message file uses an older experiment
message format that the current default regexes do not match, so no
``trials`` / ``events`` / ``parameters`` tables are produced. That is
itself the current behaviour and is pinned here deliberately. The
message-grammar refactor gets its own dedicated unit tests against
synthetic message strings.

To regenerate the baseline after an *intentional* behaviour change::

    HEDFPY_REGEN_BASELINE=1 pytest tests/test_regression.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tables

BASELINE_PATH = Path(__file__).parent / "baseline_tk_1.json"

# Column sums are large floating-point numbers; rel tolerance guards
# against real regressions while tolerating cross-version filter noise.
SUM_REL_TOL = 1e-6
SUM_ABS_TOL = 1e-6


def _fingerprint(hdf_path):
    """Compute a structural + numeric fingerprint of an hedfpy HDF5 file.

    ``pandas_tables`` covers the ``blocks`` / ``block_*`` frames written
    via ``pandas`` (fixed format); ``message_tables`` covers the
    PyTables tables written for eyelink/experiment messages.
    """
    fingerprint = {"pandas_tables": {}, "message_tables": {}}

    with pd.HDFStore(str(hdf_path), mode="r") as h5:
        for key in sorted(h5.keys()):
            frame = h5[key]
            column_sums = {}
            for col in frame.columns:
                values = np.asarray(frame[col])
                if np.issubdtype(values.dtype, np.number):
                    column_sums[str(col)] = float(
                        np.nansum(values.astype("float64"))
                    )
            fingerprint["pandas_tables"][key] = {
                "shape": list(frame.shape),
                "columns": [str(c) for c in frame.columns],
                "column_sums": column_sums,
            }

    with tables.open_file(str(hdf_path), mode="r") as h5:
        for node in h5.walk_nodes("/", "Table"):
            fingerprint["message_tables"][node._v_pathname] = {
                "nrows": int(node.nrows),
                "colnames": list(node.colnames),
            }

    return fingerprint


@pytest.fixture(scope="session")
def fingerprint(tk1_hdf):
    return _fingerprint(tk1_hdf)


def test_regenerate_baseline(fingerprint):
    """Rewrite the committed baseline; skipped unless explicitly asked."""
    if not os.environ.get("HEDFPY_REGEN_BASELINE"):
        pytest.skip("set HEDFPY_REGEN_BASELINE=1 to regenerate the baseline")
    BASELINE_PATH.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n"
    )


@pytest.fixture(scope="session")
def baseline():
    if not BASELINE_PATH.exists():
        pytest.skip(
            "baseline missing; run HEDFPY_REGEN_BASELINE=1 pytest to create it"
        )
    return json.loads(BASELINE_PATH.read_text())


def test_message_tables_structure(fingerprint, baseline):
    """Eyelink/experiment message tables: paths, row counts, columns."""
    assert set(fingerprint["message_tables"]) == set(baseline["message_tables"])
    for path, expected in baseline["message_tables"].items():
        got = fingerprint["message_tables"][path]
        assert got["nrows"] == expected["nrows"], path
        assert got["colnames"] == expected["colnames"], path


def test_gaze_tables_structure(fingerprint, baseline):
    """Gaze / blocks frames: paths, shapes, column names."""
    assert set(fingerprint["pandas_tables"]) == set(baseline["pandas_tables"])
    for path, expected in baseline["pandas_tables"].items():
        got = fingerprint["pandas_tables"][path]
        assert got["shape"] == expected["shape"], path
        assert got["columns"] == expected["columns"], path


def test_gaze_tables_numeric(fingerprint, baseline):
    """Per-column sums of the gaze pipeline output stay within tolerance."""
    for path, expected in baseline["pandas_tables"].items():
        got_sums = fingerprint["pandas_tables"][path]["column_sums"]
        for col, expected_sum in expected["column_sums"].items():
            assert got_sums[col] == pytest.approx(
                expected_sum, rel=SUM_REL_TOL, abs=SUM_ABS_TOL
            ), f"{path}:{col}"
