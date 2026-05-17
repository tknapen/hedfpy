"""Declarative grammar for the experiment messages that hedfpy parses.

SR Research EDF files contain two kinds of ``MSG`` lines:

* **Eyelink-generated** messages -- block boundaries, screen coordinates,
  and the ``ESACC`` / ``EFIX`` / ``EBLINK`` event lines. Their format is
  set by the eye tracker and rarely changes.
* **Experiment-generated** messages -- trial starts, trial phases,
  per-trial parameters, key presses and sounds. Their format is whatever
  *your* experiment software wrote to the tracker, so these are the
  patterns you are most likely to need to override.

A :class:`MessageGrammar` collects every regular expression hedfpy uses,
so they live in one declarative place instead of being scattered as
default arguments through ``EDFOperator``. :meth:`MessageGrammar.default`
reproduces hedfpy's historical patterns exactly; build your own grammar
(directly, via :meth:`from_dict`, or via :meth:`from_toml`) to parse a
different message format.

Placeholders
------------
Some patterns contain a literal placeholder that is substituted with the
trial number before matching, preserving hedfpy's historical behaviour:

* ``trial_phase`` -- ``XXX`` is replaced by the trial number.
* ``parameter``, ``key_unix``, ``key_win``, ``key_unix2`` -- `` X ``
  (space, X, space) is replaced by `` <trial number> ``.

Capture groups are positional and their meaning is fixed; when overriding
a pattern keep the same groups in the same order. The group layout of
each default pattern is documented inline below.
"""

from dataclasses import dataclass, fields

# --- eyelink-generated patterns (format fixed by SR Research) -------------

# block sampling mode: groups -> (sample_rate, eye_recorded)
DEFAULT_BLOCK_SAMPLE = r'MSG\t[\d\.]+\t!MODE RECORD CR (\d+) \d+ \d+ (\S+)'
# screen geometry: groups -> (x0, y0, x1, y1)
DEFAULT_BLOCK_SCREEN = r'MSG\t[\d\.]+\tGAZE_COORDS (\d+.\d+) (\d+.\d+) (\d+.\d+) (\d+.\d+)'
# saccade: groups -> (eye, start, end, duration, start_x, start_y,
#                     end_x, end_y, length, peak_velocity)
DEFAULT_SACCADE = r'ESACC\t(\S+)[\s\t]+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+.?\d+)'
# fixation: groups -> (eye, start, end, duration, x, y, pupil_size)
DEFAULT_FIXATION = r'EFIX\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?\s+(-?\d+\.?\d*)?'
# blink: groups -> (eye, start, end, duration)
DEFAULT_BLINK = r'EBLINK\t(\S+)\s+(-?\d*\.?\d*)\t(-?\d+\.?\d*)\s+(-?\d?.?\d*)?'

# --- experiment-generated patterns (override these per experiment) --------

# pulse-triggered trial start: groups -> (EL_timestamp, trial_nr, exp_time)
DEFAULT_TRIAL_START_PULSE = r'MSG\t([\d\.]+)\tstart_type-pulse_trial-(\d+)_phase-0_key-t_time-(\d.+)'
# stimulus-triggered trial start: groups -> (EL_timestamp, trial_nr)
DEFAULT_TRIAL_START_STIM = r'MSG\t([\d\.]+)\tstart_type-stim_trial-(\d+)_phase-0'
# trial phase ('XXX' -> trial nr): groups -> (EL_timestamp, phase_index)
DEFAULT_TRIAL_PHASE = r'MSG\t([\d\.]+)\tstart_type-stim_trial-XXX_phase-(\d+)'
# per-trial parameter (' X ' -> ' trial_nr '): groups -> (name, value)
DEFAULT_PARAMETER = r'MSG\t[\d\.]+\ttrial X parameter[\t ]*(\S*?)\s+: ([-\d\.]*|[\w]*)'
# key event, unix/pygame (' X ' -> ' trial_nr '): 8 groups
DEFAULT_KEY_UNIX = r"MSG\t([\d\.]+)\ttrial X event \<Event\((\d)-Key(\S*?) {'scancode': (\d+), 'key': ([-+]?\d+)(, 'unicode': u'\S*?',|,) 'mod': (\d+)}\)\> at (\d+.\d)"
# key event, windows/pygame (' X ' -> ' trial_nr '): 7 groups
DEFAULT_KEY_WIN = r"MSG\t([\d\.]+)\ttrial X event \<Event\((\d)-Key(.*?) {'key': (\d+)(, 'unicode': u'l',|,) 'mod': (\d+)}\)\> at (\d+.\d)"
# key event, minimal fallback (' X ' -> ' trial_nr '): 2 groups
DEFAULT_KEY_UNIX2 = r'MSG\t([\d\.]+)\ttrial X event . at (\d+.\d)'
# sound/reward event: groups -> (EL_timestamp, sound_type, exp_timestamp)
DEFAULT_SOUND = r'MSG\t([\d\.]+)\treward (\d+) at (\d+.\d)'


@dataclass
class MessageGrammar:
    """All regular expressions hedfpy uses to parse EDF message files.

    Every field defaults to hedfpy's historical pattern, so
    ``MessageGrammar()`` and :meth:`default` are equivalent and a
    partially-specified grammar (via :meth:`from_dict` / :meth:`from_toml`)
    keeps the defaults for whatever is not overridden.
    """

    # experiment-generated (most likely to need changing):
    trial_start_pulse: str = DEFAULT_TRIAL_START_PULSE
    trial_start_stim: str = DEFAULT_TRIAL_START_STIM
    trial_phase: str = DEFAULT_TRIAL_PHASE
    parameter: str = DEFAULT_PARAMETER
    key_unix: str = DEFAULT_KEY_UNIX
    key_win: str = DEFAULT_KEY_WIN
    key_unix2: str = DEFAULT_KEY_UNIX2
    sound: str = DEFAULT_SOUND
    # eyelink-generated (format fixed by SR Research):
    block_sample: str = DEFAULT_BLOCK_SAMPLE
    block_screen: str = DEFAULT_BLOCK_SCREEN
    saccade: str = DEFAULT_SACCADE
    fixation: str = DEFAULT_FIXATION
    blink: str = DEFAULT_BLINK

    @classmethod
    def default(cls):
        """Return the grammar reproducing hedfpy's historical patterns."""
        return cls()

    @classmethod
    def from_dict(cls, mapping):
        """Build a grammar from a mapping of field name -> pattern.

        Fields that are absent keep their historical default. Unknown
        keys raise ``TypeError`` rather than being silently ignored.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(mapping) - known
        if unknown:
            raise TypeError(
                "unknown MessageGrammar field(s): %s; valid fields are %s"
                % (sorted(unknown), sorted(known))
            )
        return cls(**dict(mapping))

    @classmethod
    def from_toml(cls, path):
        """Load a grammar from a TOML file.

        Absent fields keep their historical default. If the file has a
        ``[grammar]`` table its contents are used, otherwise the keys are
        read from the top level.
        """
        try:
            import tomllib
        except ModuleNotFoundError:  # Python < 3.11
            import tomli as tomllib
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
        if "grammar" in data:
            data = data["grammar"]
        return cls.from_dict(data)

    def to_dict(self):
        """Return this grammar as a plain ``{field: pattern}`` dict."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
