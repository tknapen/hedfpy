# hedfpy
hedfpy is a tool to convert SR Research eyelink eye position and pupil size data to hdf5 format. hedfpy can also preprocess the data to the starting point of most standard pupil size analyses, for example those performed using [FIRDeconvolution](https://github.com/tknapen/FIRDeconvolution "FIRDeconvolution"). 
These are:
- Blink detection (over and above the eyelink's own blink detection mechanisms)
- Blink interpolation (linear/spline)
- (Micro)Saccade detection (possibly over and above the eyelink's own blink detection mechanisms, using Engbert and Mergenthaler, PNAS 2006 algorithm)
- Band-pass filtering of pupil size signals using Butterworth filters. 
- Cleaning up the pupil size signal based on a nuisance GLM which estimates the effects of blinks, (micro)saccades, and gaze position (foreshortening of the pupil). 

## MSG format
for full parsing of the edf file data, hedfpy assumes a specific trial-based experimental format that is communicated to the eye tracker. Specifically, it looks for explicitly formatted messages by means of regular expressions. These messages detail the start and end of trial phases, trials, button press events, sound events and the stimulus parameters for a given trial which are all stored in tabular format in the HDF5 file. The parsing of these messages can be turned off for basic functionality.

The regular expressions used to parse these messages are collected in a `MessageGrammar` (see `hedfpy/message_grammar.py`); the defaults match the message format produced by [exptools2](https://github.com/VU-Cog-Sci/exptools2). To parse a different format, copy and edit the documented `hedfpy/grammar.toml` template and pass your own grammar:

    from hedfpy import HDFEyeOperator, MessageGrammar

    grammar = MessageGrammar.from_toml('my_grammar.toml')
    ho = HDFEyeOperator('session.h5', grammar=grammar)
    ho.add_edf_file('run.edf')

## Dependencies
hedfpy is installed with `pip install .`, which pulls in its Python dependencies (numpy, scipy, pandas, matplotlib, statsmodels, scikit-learn, lmfit, mne, PyTables, h5py).

It additionally needs the HDF5 libraries and the `edf2asc` command-line utility from [SR Research](http://www.sr-research.com "SR Research") available on the `$PATH`.

TODO
1. ~~The regular expressions that parse experiment-generated `MSG` strings are hard-coded in `EDFOperator`.~~ Done: they now live in a configurable `MessageGrammar` (see above).
2. ~~hedfpy processes an edf file according to a fixed set of preprocessing steps.~~ Done: `edf_gaze_data_to_hdf` accepts a `steps=` argument selecting which stages (`interpolate`, `filter`, `regress`, `normalize`, `summary_plot`, `time_frequency`) run; the default `GAZE_PIPELINE_STEPS` reproduces the historical behaviour.







