"""Tests for the trial-based read API of HDFEyeOperator.

These methods were broken: get_time_period_for_trial used the removed
pandas .ix accessor and a trial_end_EL_timestamp column that read_trials
never produces (the experiment software sends no trial-stop message).
They now derive an implicit trial end -- a trial ends where the next one
starts, the last trial of a block at that block's end.

A small session HDF5 is built here from synthetic messages: three trials
with two phases each, inside one recording block.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import hedfpy

# Three stimulus-triggered trials, two phases each, with controlled
# EL timestamps so the expected time periods are obvious.
READ_API_MSG = "\n".join([
    "MSG\t1000.0\tstart_type-stim_trial-0_phase-0",
    "MSG\t1500.0\tstart_type-stim_trial-0_phase-1",
    "MSG\t2000.0\tstart_type-stim_trial-1_phase-0",
    "MSG\t2500.0\tstart_type-stim_trial-1_phase-1",
    "MSG\t3000.0\tstart_type-stim_trial-2_phase-0",
    "MSG\t3500.0\tstart_type-stim_trial-2_phase-1",
    "",
])

# One recording block spanning every trial; last trial ends here.
BLOCK_END = 10000.0


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Build a HDFEyeOperator backed by a small synthetic session HDF5."""
    work = tmp_path_factory.mktemp("read_api")
    (work / "s.edf").write_bytes(b"")
    (work / "s.msg").write_text(READ_API_MSG)

    edf_operator = hedfpy.EDFOperator(str(work / "s.edf"))
    edf_operator.read_all_messages()

    ho = hedfpy.HDFEyeOperator(str(work / "session.h5"))
    # bypass add_edf_file (which would need gaze data) and feed the
    # parsed EDFOperator straight into the message writer
    ho.edf_operator = edf_operator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ho.edf_message_data_to_hdf(alias="x")

    blocks = pd.DataFrame([{
        "block_start_index": 0,
        "block_end_index": 100,
        "block_start_timestamp": 0.0,
        "block_end_timestamp": BLOCK_END,
        "sample_rate": 1000,
        "eye_recorded": "L",
        "screen_x_pix": 1920.0,
        "screen_y_pix": 1200.0,
    }])
    ho.data_frame_to_hdf("x", "blocks", blocks)
    return ho


def test_get_time_period_for_trial_uses_next_trial_start(session):
    assert session.get_time_period_for_trial(0, "x") == (1000.0, 2000.0)
    assert session.get_time_period_for_trial(1, "x") == (2000.0, 3000.0)


def test_last_trial_ends_at_block_end(session):
    start, end = session.get_time_period_for_trial(2, "x")
    assert start == 3000.0
    assert end == BLOCK_END


def test_get_time_period_for_trial_applies_time_extensions(session):
    start, end = session.get_time_period_for_trial(
        1, "x", time_extensions=[-10, 25])
    assert (start, end) == (1990.0, 3025.0)


def test_get_time_period_for_trial_rejects_unknown_trial(session):
    with pytest.raises(ValueError, match="trial 99 not found"):
        session.get_time_period_for_trial(99, "x")


def test_trial_for_time_period_finds_containing_trial(session):
    assert session.trial_for_time_period("x", [1500.0, 1800.0]) == 0
    assert session.trial_for_time_period("x", [2100.0, 2200.0]) == 1


def test_trial_for_time_period_warns_when_spanning_trials(session):
    with pytest.warns(UserWarning, match="spans multiple trials"):
        trial = session.trial_for_time_period("x", [2500.0, 4000.0])
    assert trial == 1


def test_trial_for_time_period_raises_outside_any_trial(session):
    with pytest.raises(Exception, match="No trial found"):
        session.trial_for_time_period("x", [500.0, 600.0])


def test_time_period_for_trial_phases(session):
    # phase 0 == trial start, so [0, 1] spans trial start -> phase 1
    np.testing.assert_array_equal(
        session.time_period_for_trial_phases(0, [0, 1], "x"),
        np.array([1000.0, 1500.0]))
    # a trailing -1 means the implicit trial end
    np.testing.assert_array_equal(
        session.time_period_for_trial_phases(0, [1, -1], "x"),
        np.array([1500.0, 2000.0]))


def test_eye_and_sample_rate_during_trial(session):
    assert session.eye_during_trial(0, "x") == "L"
    assert session.sample_rate_during_trial(0, "x") == 1000.0


def test_screen_dimensions_during_period(session):
    assert session.screen_dimensions_during_period([1500.0, 1800.0], "x") == (
        1920.0, 1200.0)
