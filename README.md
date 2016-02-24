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

## Dependencies
numpy, scipy, matplotlib, statsmodels, sklearn, tables, lmfit

Further install requirements: hdf5 libraries, edf2asc command-line utility from [SR Research](http://www.sr-research.com "SR Research")

TODO
1. For now, the `edf_message_data_to_hdf` method of `HDFEyeOperator` and `EDFOperator` searches for both eyelink-generated and experiment-generated `MSG` strings in the edf file's output. The eyelink generated messages are standard, but the experiment-generated ones are not. At the moment, the methods that tease these messages apart dissect the EDF's output into trial timings, trial phase timings, trial parameters, button press and sound occurrences based on regular expressions that are hard-coded in the `EDFOperator`. This needs to be parceled out of the code so that this becomes more generally user-friendly. 
2. hedfpy now processes an edf file according to a fixed set of preprocessing steps (see the code in `HDFEyeOperator`'s `edf_gaze_data_to_hdf`). These need to be made elective at some point. 







