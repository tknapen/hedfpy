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


def test_regress_blinks_without_nideconv_raises_clean_error():
    """When `use_standard_blinksac_kernels=False`, regress_blinks needs
    nideconv; without it, the ImportError should name the optional extra
    and the GitHub install command rather than the bare module name."""
    try:
        import nideconv  # noqa: F401
        pytest.skip("nideconv is installed; covered by the run test")
    except ImportError:
        pass

    import numpy as np
    from hedfpy import EyeSignalOperator

    eso = EyeSignalOperator(
        input_object={
            "timepoints": np.arange(10000, dtype=float),
            "gaze_X": np.zeros(10000),
            "gaze_Y": np.zeros(10000),
            "pupil": np.ones(10000),
        },
        sample_rate=1000.0,
    )
    # The minimum attributes for regress_blinks to reach the nideconv import:
    eso.bp_filt_pupil = np.ones(10000)
    eso.blink_ends = np.array([], dtype=int)
    eso.sac_ends_EL = np.array([], dtype=int)
    with pytest.raises(ImportError, match="nideconv"):
        eso.regress_blinks(use_standard_blinksac_kernels=False)


def test_regress_blinks_with_nideconv_recovers_kernels():
    """End-to-end smoke test of the nideconv-based deconvolution path.

    Skipped when nideconv (an optional, hedfpy[deconv] dependency) is not
    installed -- on supported environments where it is, this exercises
    the full regress_blinks(use_standard_blinksac_kernels=False) flow
    and asserts the resulting kernel arrays have the expected shapes.
    """
    pytest.importorskip("nideconv")

    import numpy as np
    from hedfpy import EyeSignalOperator

    sample_rate = 100.0
    n = 6000  # 60 s at 100 Hz: long enough for the early/late cutoffs
    rng = np.random.default_rng(0)

    eso = EyeSignalOperator(
        input_object={
            "timepoints": np.arange(n, dtype=float),
            "gaze_X": np.zeros(n),
            "gaze_Y": np.zeros(n),
            "pupil": np.ones(n),
        },
        sample_rate=sample_rate,
    )
    # minimum state to reach regress_blinks (skipping interpolate / filter)
    eso.bp_filt_pupil = rng.standard_normal(n) * 0.1
    eso.lp_filt_pupil = eso.bp_filt_pupil.copy()
    eso.baseline_filt_pupil = np.ones(n) * 100.0
    eso.blink_ends = np.array([3000, 3800, 4200, 4800, 5200])
    eso.sac_ends_EL = np.array([3100, 3500, 4000, 4500, 5000, 5500])
    eso.interpolated_x = np.zeros(n)
    eso.interpolated_y = np.zeros(n)

    interval = 7
    eso.regress_blinks(interval=interval, use_standard_blinksac_kernels=False)

    # FIR response is at new_sample_rate = sample_rate / downsample_rate
    expected_response_len = int(interval * eso.new_sample_rate)
    assert eso.blink_response_downsampled.shape == (expected_response_len,)
    assert eso.sac_response_downsampled.shape == (expected_response_len,)

    # kernels are upsampled back to the input sample rate
    expected_kernel_len = int(interval * sample_rate)
    assert eso.blink_kernel.shape == (expected_kernel_len,)
    assert eso.sac_kernel.shape == (expected_kernel_len,)

    # the residual cleaned signal stays at the original length
    assert eso.bp_filt_pupil_clean.shape == (n,)
    assert eso.lp_filt_pupil_clean.shape == (n,)


def test_time_frequency_decomposition_rejects_morlet():
    """The historical morlet TF path has been removed; requesting it now
    raises a clear error rather than blowing up on the missing MNE API."""
    import numpy as np
    from hedfpy import EyeSignalOperator

    eso = EyeSignalOperator(
        input_object={
            "timepoints": np.arange(1000, dtype=float),
            "gaze_X": np.zeros(1000),
            "gaze_Y": np.zeros(1000),
            "pupil": np.ones(1000),
        },
        sample_rate=1000.0,
    )
    eso.interpolated_pupil = np.ones(1000)
    with pytest.raises(ValueError, match="lp_butterworth"):
        eso.time_frequency_decomposition_pupil(tf_decomposition="morlet")
