"""Tests for the externalized message grammar (MessageGrammar).

The tk_1 regression fixture happens to use an old experiment message
format that the default grammar does not match, so these tests drive the
parser with small synthetic .msg files instead. An EDFOperator is built
offline -- no edf2asc binary is needed -- by writing the .msg file it
would normally produce next to a placeholder .edf.
"""

import re
from pathlib import Path

import pytest

import hedfpy
from hedfpy import MessageGrammar

# A synthetic message file in hedfpy's default (current) format: two
# stimulus-triggered trials, each with two phases, two parameters and one
# key press, plus one reward and one eyelink saccade/fixation/blink. The
# eyelink lines are copied verbatim from the tk_1 fixture.
NEW_FORMAT_MSG = "\n".join([
    "MSG\t1000.0\tstart_type-stim_trial-0_phase-0",
    "MSG\t1000.0\tstart_type-stim_trial-0_phase-1",
    "MSG\t1010.0\ttrial 0 parameter contrast : 0.80",
    "MSG\t1010.0\ttrial 0 parameter answer : 1",
    "MSG\t1020.0\ttrial 0 event <Event(2-KeyDown {'scancode': 17, "
    "'key': 116, 'unicode': u't', 'mod': 0})> at 1234.5",
    "MSG\t1100.0\treward 0 at 1240.0",
    "MSG\t2000.0\tstart_type-stim_trial-1_phase-0",
    "MSG\t2000.0\tstart_type-stim_trial-1_phase-1",
    "MSG\t2010.0\ttrial 1 parameter contrast : 0.40",
    "MSG\t2010.0\ttrial 1 parameter answer : 0",
    "MSG\t2020.0\ttrial 1 event <Event(2-KeyDown {'scancode': 18, "
    "'key': 100, 'unicode': u'd', 'mod': 0})> at 2234.5",
    "ESACC\tL\t24622375\t24622384\t10\t1414.8\t1394.7\t1377.3\t1395.3\t0.56\t66",
    "EFIX\tL\t24622275.0\t24622374.0\t4294967396\t1423.6\t1398.1\t5981",
    "EBLINK\tL\t24626893\t24627020\t128",
    "",
])

# An experiment that announced trials in a format the default grammar
# knows nothing about.
ALT_FORMAT_MSG = "\n".join([
    "MSG\t5000.0\tBEGIN_TRIAL 0",
    "MSG\t6000.0\tBEGIN_TRIAL 1",
    "",
])


def _edf_operator(tmp_path, msg_text, grammar=None):
    """Build an EDFOperator offline from synthetic message text.

    EDFOperator skips the edf2asc conversion when the .msg file already
    exists, so writing it next to an empty placeholder .edf is enough.
    """
    (tmp_path / "synthetic.edf").write_bytes(b"")
    (tmp_path / "synthetic.msg").write_text(msg_text)
    return hedfpy.EDFOperator(str(tmp_path / "synthetic.edf"), grammar=grammar)


# --- the grammar object itself -------------------------------------------

def test_default_equals_plain_construction():
    assert MessageGrammar.default() == MessageGrammar()


def test_all_default_patterns_compile():
    for name, pattern in MessageGrammar.default().to_dict().items():
        # substitute the per-trial placeholders before compiling
        re.compile(pattern.replace("XXX", "0").replace(" X ", " 0 "))


def test_from_dict_keeps_defaults_for_absent_keys():
    grammar = MessageGrammar.from_dict({"sound": r"custom-sound-pattern"})
    assert grammar.sound == r"custom-sound-pattern"
    assert grammar.saccade == MessageGrammar.default().saccade


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(TypeError, match="unknown MessageGrammar field"):
        MessageGrammar.from_dict({"not_a_field": "x"})


def test_shipped_grammar_toml_matches_default():
    try:
        import tomllib  # noqa: F401  (Python 3.11+)
    except ModuleNotFoundError:
        pytest.importorskip("tomli")
    toml_path = Path(hedfpy.__file__).parent / "grammar.toml"
    assert MessageGrammar.from_toml(toml_path) == MessageGrammar.default()


# --- parsing through EDFOperator -----------------------------------------

def test_default_grammar_parses_new_format(tmp_path):
    op = _edf_operator(tmp_path, NEW_FORMAT_MSG)
    op.read_all_messages()

    assert op.nr_trials == 2
    assert len(op.trials) == 2
    assert len(op.trial_phases) == 4          # 2 trials x 2 phases
    assert len(op.parameters) == 2
    assert len(op.events) == 2
    assert len(op.sounds) == 1
    assert len(op.saccades_from_message_file) == 1
    assert len(op.fixations_from_message_file) == 1
    assert len(op.blinks_from_message_file) == 1

    assert op.trials[0]["trial_start_index"] == 0
    assert op.parameters[0]["contrast"] == pytest.approx(0.80)
    assert op.parameters[1]["contrast"] == pytest.approx(0.40)
    assert op.events[0]["up_down"] == "Down"
    assert op.events[0]["key"] == 116


def test_default_grammar_misses_alternate_format(tmp_path):
    """The default grammar should find no trials in an unknown format."""
    op = _edf_operator(tmp_path, ALT_FORMAT_MSG)
    op.read_trials()
    assert op.nr_trials == 0


def test_custom_grammar_parses_alternate_format(tmp_path):
    """A custom grammar parses a format the default grammar cannot."""
    grammar = MessageGrammar.from_dict(
        {"trial_start_stim": r"MSG\t([\d\.]+)\tBEGIN_TRIAL (\d+)"}
    )
    op = _edf_operator(tmp_path, ALT_FORMAT_MSG, grammar=grammar)
    op.read_trials()
    assert op.nr_trials == 2
    assert len(op.trials) == 2


def test_explicit_regex_argument_still_overrides(tmp_path):
    """Passing a regex straight to read_trials keeps working (back-compat)."""
    op = _edf_operator(tmp_path, ALT_FORMAT_MSG)
    op.read_trials(stim_re=r"MSG\t([\d\.]+)\tBEGIN_TRIAL (\d+)")
    assert op.nr_trials == 2
