"""Tests for the elective gaze-preprocessing pipeline.

``edf_gaze_data_to_hdf`` accepts a ``steps`` argument selecting which
preprocessing stages run. The default reproduces hedfpy's historical
behaviour and is already pinned by the byte-identical regression test in
test_regression.py; these tests cover the new behaviours: validation,
dependency checking, and the actual column layout when a subset of steps
is requested.
"""

import shutil
import warnings
from pathlib import Path

import pandas as pd
import pytest

import hedfpy
from hedfpy import GAZE_PIPELINE_STEPS

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO_ROOT / "test"
TK1_STEM = "tk_1_2014-10-06_16.44.11"
TK1_SUFFIXES = (".edf", ".msg", ".gaz", ".gaz.gz")


def _run_pipeline_with_steps(tmp_path, steps):
    """Run the full message + gaze pipeline on the tk_1 fixture using
    the given ``steps``, returning the resulting HDF5 path."""
    for suffix in TK1_SUFFIXES:
        src = FIXTURE_DIR / (TK1_STEM + suffix)
        if not src.exists():
            pytest.skip("missing fixture file: %s" % src)
        shutil.copy(src, tmp_path / src.name)

    hdf = tmp_path / "session.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ho = hedfpy.HDFEyeOperator(str(hdf))
        ho.add_edf_file(str(tmp_path / (TK1_STEM + ".edf")))
        ho.edf_message_data_to_hdf(alias="test")
        ho.edf_gaze_data_to_hdf(alias="test", steps=steps)
    return hdf


# --- the constant itself --------------------------------------------------

def test_gaze_pipeline_steps_constant():
    """Sanity check the published default pipeline."""
    assert set(GAZE_PIPELINE_STEPS) == {
        "interpolate", "filter", "regress",
        "normalize", "summary_plot", "time_frequency",
    }


# --- up-front validation (fast: no pipeline run) --------------------------

def test_unknown_step_raises(tmp_path):
    ho = hedfpy.HDFEyeOperator(str(tmp_path / "session.h5"))
    with pytest.raises(ValueError, match="unknown gaze preprocessing step"):
        ho.edf_gaze_data_to_hdf(alias="test", steps=["xyzzy"])


def test_missing_dependency_raises(tmp_path):
    ho = hedfpy.HDFEyeOperator(str(tmp_path / "session.h5"))
    with pytest.raises(ValueError, match="requires"):
        # 'filter' needs 'interpolate'
        ho.edf_gaze_data_to_hdf(alias="test", steps=["filter"])


def test_regress_without_filter_raises(tmp_path):
    ho = hedfpy.HDFEyeOperator(str(tmp_path / "session.h5"))
    with pytest.raises(ValueError, match="requires"):
        ho.edf_gaze_data_to_hdf(
            alias="test", steps=["interpolate", "regress"])


# --- end-to-end with a subset pipeline ------------------------------------

def test_subset_pipeline_skips_regress_and_tf(tmp_path):
    """A steps=interpolate/filter/normalize pipeline writes exactly the
    columns those stages produce, and none of the columns the skipped
    stages would have written."""
    hdf = _run_pipeline_with_steps(
        tmp_path, steps=["interpolate", "filter", "normalize"])

    with pd.HDFStore(str(hdf), mode="r") as h5:
        cols = set(h5["/test/block_0"].columns)

    # interpolate columns
    assert "L_pupil_int" in cols
    assert "L_interpolated_timepoints" in cols
    assert "L_gaze_x_int" in cols
    # filter columns
    assert "L_pupil_lp" in cols
    assert "L_pupil_hp" in cols
    assert "L_pupil_baseline" in cols
    assert "L_pupil_bp" in cols
    # normalize columns (psc, the default normalization)
    assert "L_pupil_lp_psc" in cols
    assert "L_pupil_bp_psc" in cols

    # regress columns -- skipped
    assert "L_pupil_lp_clean" not in cols
    assert "L_pupil_bp_clean" not in cols
    assert "L_pupil_lp_clean_psc" not in cols
    assert "L_pupil_bp_clean_psc" not in cols
    # time_frequency columns -- skipped
    assert not any(c.startswith("L_pupil_filterbank") for c in cols)
