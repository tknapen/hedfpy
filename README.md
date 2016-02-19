# hedfpy
hedfpy is a tool to convert SR Research eyelink eye position and pupil size data to hdf5 format. hedfpy can also preprocess the data to the starting point of most standard pupil size analyses. 
These are:
- Blink detection (over and above the eyelink's own blink detection mechanisms)
- Blink interpolation (linear/spline)
- (Micro)Saccade detection (over and above the eyelink's own blink detection mechanisms, using Engbert and Mergenthaler, PNAS 2006 algorithm)
- Band-pass filtering of pupil size signals using Butterworth filters. 

## MSG format
for full parsing of the edf file data, hedfpy assumes a specific trial-based experimental format that is communicated to the eye tracker. Specifically, it looks for explicitly formatted messages by means of regular expressions. These messages detail the start and end of trial phases, trials, button press events, sound events and the stimulus parameters for a given trial which are all stored in tabular format in the HDF5 file. The parsing of these messages can be turned off for basic functionality.

## Dependencies
numpy, scipy, matplotlib, statsmodels, sklearn, tables, sympy, lmfit

Further install requirements: hdf5 libraries, edf2asc command-line utility from SR Research

TODO
XXX








