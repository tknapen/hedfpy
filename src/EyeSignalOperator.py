#!/usr/bin/env python
# encoding: utf-8

"""@package Operators
This module offers various methods to process eye movement data

Created by Tomas Knapen on 2010-12-19.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.

More details.
"""

import os, sys, subprocess, re
import pickle

import scipy as sp
import numpy as np
import pandas as pd
import numpy.linalg as LA
import matplotlib.pyplot as plt
from math import *
from scipy.signal import butter, lfilter, filtfilt, fftconvolve, resample
import scipy.interpolate as interpolate
import scipy.stats as stats
import mne
from lmfit import minimize, Parameters, Parameter, report_fit

from Operator import Operator

import fir

from IPython import embed as shell

def detect_saccade_from_data(xy_data = None, vel_data = None, l = 5, sample_rate = 1000.0, minimum_saccade_duration = 0.0075):
	"""Uses the engbert & mergenthaler algorithm (PNAS 2006) to detect saccades.
	
	This function expects a sequence (N x 2) of xy gaze position or velocity data. 
	
	Arguments:
		xy_data (numpy.ndarray, optional): a sequence (N x 2) of xy gaze (float/integer) positions. Defaults to None
		vel_data (numpy.ndarray, optional): a sequence (N x 2) of velocity data (float/integer). Defaults to None.
		l (float, optional):determines the threshold. Defaults to 5 median-based standard deviations from the median
		sample_rate (float, optional) - the rate at which eye movements were measured per second). Defaults to 1000.0
		minimum_saccade_duration (float, optional) - the minimum duration for something to be considered a saccade). Defaults to 0.0075
	
	Returns:
		list of dictionaries, which each correspond to a saccade.
		
		The dictionary contains the following items:
			
	Raises:
		ValueError: If neither xy_data and vel_data were passed to the function.
	
	"""
	
	# If xy_data and vel_data are both None, function can't continue
	if xy_data is None and vel_data is None:
		raise ValueError("Supply either xy_data or vel_data")	
		
	#If xy_data is given, process it
	if not xy_data is None:
		xy_data = np.array(xy_data)
		# when are both eyes zeros?
		xy_data_zeros = (xy_data == 0.0001).sum(axis = 1)
	
	# Calculate velocity data if it has not been given to function
	if vel_data is None:
		# # Check for shape of xy_data. If x and y are ordered in columns, transpose array.
		# # Should be 2 x N array to use np.diff namely (not Nx2)
		# rows, cols = xy_data.shape
		# if rows == 2:
		# 	vel_data = np.diff(xy_data)
		# if cols == 2:
		# 	vel_data = np.diff(xy_data.T)
		vel_data = np.zeros(xy_data.shape)
		vel_data[1:] = np.diff(xy_data, axis = 0)
	else:
		vel_data = np.array(vel_data)

	# median-based standard deviation, for x and y separately
	med = np.median(vel_data, axis = 0)
	
	scaled_vel_data = vel_data/np.mean(np.array(np.sqrt((vel_data - med)**2)), axis = 0)
	# normalize and to acceleration and its sign
	if (float(np.__version__.split('.')[1]) == 1.0) and (float(np.__version__.split('.')[1]) > 6):
		normed_scaled_vel_data = LA.norm(scaled_vel_data, axis = 1)
		normed_vel_data = LA.norm(vel_data, axis = 1)
	else:
		normed_scaled_vel_data = np.array([LA.norm(svd) for svd in np.array(scaled_vel_data)])
		normed_vel_data = np.array([LA.norm(vd) for vd in np.array(vel_data)])
	normed_acc_data = np.r_[0,np.diff(normed_scaled_vel_data)]
	signed_acc_data = np.sign(normed_acc_data)
	
	# when are we above the threshold, and when were the crossings
	over_threshold = (normed_scaled_vel_data > l)
	# integers instead of bools preserve the sign of threshold transgression
	over_threshold_int = np.array(over_threshold, dtype = np.int16)
	
	# crossings come in pairs
	threshold_crossings_int = np.concatenate([[0], np.diff(over_threshold_int)])
	threshold_crossing_indices = np.arange(threshold_crossings_int.shape[0])[threshold_crossings_int != 0]
	
	valid_threshold_crossing_indices = []
	
	# if no saccades were found, then we'll just go on and record an empty saccade
	if threshold_crossing_indices.shape[0] > 1:
		# the first saccade cannot already have started now
		if threshold_crossings_int[threshold_crossing_indices[0]] == -1:
			threshold_crossings_int[threshold_crossing_indices[0]] = 0
			threshold_crossing_indices = threshold_crossing_indices[1:]
	
		# the last saccade cannot be in flight at the end of this data
		if threshold_crossings_int[threshold_crossing_indices[-1]] == 1:
			threshold_crossings_int[threshold_crossing_indices[-1]] = 0
			threshold_crossing_indices = threshold_crossing_indices[:-1]
		
		# check the durations of the saccades
		threshold_crossing_indices_2x2 = threshold_crossing_indices.reshape((-1,2))
		raw_saccade_durations = np.diff(threshold_crossing_indices_2x2, axis = 1).squeeze()
	
		# and check whether these saccades were also blinks...
		blinks_during_saccades = np.ones(threshold_crossing_indices_2x2.shape[0], dtype = bool)
		for i in range(blinks_during_saccades.shape[0]):
			if np.sum(xy_data_zeros[threshold_crossing_indices_2x2[i,0]-20:threshold_crossing_indices_2x2[i,1]+20]) > 0:
				blinks_during_saccades[i] = False
	
		# and are they too close to the end of the interval?
		right_times = threshold_crossing_indices_2x2[:,1] < xy_data.shape[0]-30
	
		valid_saccades_bool = ((raw_saccade_durations / float(sample_rate) > minimum_saccade_duration) * blinks_during_saccades ) * right_times
		if type(valid_saccades_bool) != np.ndarray:
			valid_threshold_crossing_indices = threshold_crossing_indices_2x2
		else:
			valid_threshold_crossing_indices = threshold_crossing_indices_2x2[valid_saccades_bool]
		
	saccades = []
	for i, cis in enumerate(valid_threshold_crossing_indices):
		# find the real start and end of the saccade by looking at when the acceleleration reverses sign before the start and after the end of the saccade:
		# sometimes the saccade has already started?
		expanded_saccade_start = np.arange(cis[0])[np.r_[0,np.diff(signed_acc_data[:cis[0]] != 1)] != 0]
		if expanded_saccade_start.shape[0] > 0:
			expanded_saccade_start = expanded_saccade_start[-1]
		else:
			expanded_saccade_start = 0
			
		expanded_saccade_end = np.arange(cis[1],np.min([cis[1]+50, xy_data.shape[0]]))[np.r_[0,np.diff(signed_acc_data[cis[1]:np.min([cis[1]+50, xy_data.shape[0]])] != -1)] != 0]
		# sometimes the deceleration continues crazily, we'll just have to cut it off then. 
		if expanded_saccade_end.shape[0] > 0:
			expanded_saccade_end = expanded_saccade_end[0]
		else:
			expanded_saccade_end = np.min([cis[1]+50, xy_data.shape[0]])
		
		try:
			this_saccade = {
				'expanded_start_time': expanded_saccade_start,
				'expanded_end_time': expanded_saccade_end,
				'expanded_duration': expanded_saccade_end - expanded_saccade_start,
				'expanded_start_point': xy_data[expanded_saccade_start],
				'expanded_end_point': xy_data[expanded_saccade_end],
				'expanded_vector': xy_data[expanded_saccade_end] - xy_data[expanded_saccade_start],
				'expanded_amplitude': np.sum(normed_vel_data[expanded_saccade_start:expanded_saccade_end]) / sample_rate,
				'peak_velocity': np.max(normed_vel_data[expanded_saccade_start:expanded_saccade_end]),

				'raw_start_time': cis[0],
				'raw_end_time': cis[1],
				'raw_duration': cis[1] - cis[0],
				'raw_start_point': xy_data[cis[1]],
				'raw_end_point': xy_data[cis[0]],
				'raw_vector': xy_data[cis[1]] - xy_data[cis[0]],
				'raw_amplitude': np.sum(normed_vel_data[cis[0]:cis[1]]) / sample_rate,
			}
			saccades.append(this_saccade)
		except IndexError:
			pass
		
	
	# if this fucker was empty
	if len(valid_threshold_crossing_indices) == 0:
		this_saccade = {
			'expanded_start_time': 0,
			'expanded_end_time': 0,
			'expanded_duration': 0.0,
			'expanded_start_point': [0.0,0.0],
			'expanded_end_point': [0.0,0.0],
			'expanded_vector': [0.0,0.0],
			'expanded_amplitude': 0.0,
			'peak_velocity': 0.0,

			'raw_start_time': 0,
			'raw_end_time': 0,
			'raw_duration': 0.0,
			'raw_start_point': [0.0,0.0],
			'raw_end_point': [0.0,0.0],
			'raw_vector': [0.0,0.0],
			'raw_amplitude': 0.0,
		}
		saccades.append(this_saccade)

	return saccades

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, show=False, ax=None):

	"""Detect peaks in data based on their amplitude and other features.
	Parameters
	----------
	x : 1D array_like
		data.
	mph : {None, number}, optional (default = None)
		detect peaks that are greater than minimum peak height.
	mpd : positive integer, optional (default = 1)
		detect peaks that are at least separated by minimum peak distance (in
		number of data).
	threshold : positive number, optional (default = 0)
		detect peaks (valleys) that are greater (smaller) than `threshold`
		in relation to their immediate neighbors.
	edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
		for a flat peak, keep only the rising edge ('rising'), only the
		falling edge ('falling'), both edges ('both'), or don't detect a
		flat peak (None).
	kpsh : bool, optional (default = False)
		keep peaks with same height even if they are closer than `mpd`.
	valley : bool, optional (default = False)
		if True (1), detect valleys (local minima) instead of peaks.
	show : bool, optional (default = False)
		if True (1), plot data in matplotlib figure.
	ax : a matplotlib.axes.Axes instance, optional (default = None).
	Returns
	-------
	ind : 1D array_like
		indeces of the peaks in `x`.
	Notes
	-----
	The detection of valleys instead of peaks is performed internally by simply
	negating the data: `ind_valleys = detect_peaks(-x)`

	The function can handle NaN's 
	See this IPython Notebook [1]_.
	References
	----------
	.. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
	Examples
	--------
	>>> from detect_peaks import detect_peaks
	>>> x = np.random.randn(100)
	>>> x[60:81] = np.nan
	>>> # detect all peaks and plot data
	>>> ind = detect_peaks(x, show=True)
	>>> print(ind)
	>>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
	>>> # set minimum peak height = 0 and minimum peak distance = 20
	>>> detect_peaks(x, mph=0, mpd=20, show=True)
	>>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
	>>> # set minimum peak distance = 2
	>>> detect_peaks(x, mpd=2, show=True)
	>>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
	>>> # detection of valleys instead of peaks
	>>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)
	>>> x = [0, 1, 1, 0, 1, 1, 0]
	>>> # detect both edges
	>>> detect_peaks(x, edge='both', show=True)
	>>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
	>>> # set threshold = 2
	>>> detect_peaks(x, threshold = 2, show=True)
	"""

	x = np.atleast_1d(x).astype('float64')
	if x.size < 3:
		return np.array([], dtype=int)
	if valley:
		x = -x
	# find indices of all peaks
	dx = x[1:] - x[:-1]
	# handle NaN's
	indnan = np.where(np.isnan(x))[0]
	if indnan.size:
		x[indnan] = np.inf
		dx[np.where(np.isnan(dx))[0]] = np.inf
	ine, ire, ife = np.array([[], [], []], dtype=int)
	if not edge:
		ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
	else:
		if edge.lower() in ['rising', 'both']:
			ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
		if edge.lower() in ['falling', 'both']:
			ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
	ind = np.unique(np.hstack((ine, ire, ife)))
	# handle NaN's
	if ind.size and indnan.size:
		# NaN's and values close to NaN's cannot be peaks
		ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
	# first and last values of x cannot be peaks
	if ind.size and ind[0] == 0:
		ind = ind[1:]
	if ind.size and ind[-1] == x.size-1:
		ind = ind[:-1]
	# remove peaks < minimum peak height
	if ind.size and mph is not None:
		ind = ind[x[ind] >= mph]
	# remove peaks - neighbors < threshold
	if ind.size and threshold > 0:
		dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
		ind = np.delete(ind, np.where(dx < threshold)[0])
	# detect small peaks closer than minimum peak distance
	if ind.size and mpd > 1:
		ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
		idel = np.zeros(ind.size, dtype=bool)
		for i in range(ind.size):
			if not idel[i]:
				# keep peaks with the same height if kpsh is True
				idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
					& (x[ind[i]] > x[ind] if kpsh else True)
				idel[i] = 0  # Keep current peak
		# remove the small peaks and sort back the indices by their occurrence
		ind = np.sort(ind[~idel])

	if show:
		if indnan.size:
			x[indnan] = np.nan
		if valley:
			x = -x
		_plot(x, mph, mpd, threshold, edge, valley, ax, ind)

	return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
	"""Plot results of the detect_peaks function, see its help."""
	if ax is None:
		_, ax = pl.subplots(1, 1, figsize=(8, 4))

	ax.plot(x, 'b', lw=1)
	if ind.size:
		label = 'valley' if valley else 'peak'
		label = label + 's' if ind.size > 1 else label
		ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
				label='%d %s' % (ind.size, label))
		ax.legend(loc='best', framealpha=.5, numpoints=1)
	ax.set_xlim(-.02*x.size, x.size*1.02-1)
	ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
	yrange = ymax - ymin if ymax > ymin else 1
	ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
	ax.set_xlabel('Data #', fontsize=14)
	ax.set_ylabel('Amplitude', fontsize=14)
	mode = 'Valley detection' if valley else 'Peak detection'
	ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
				 % (mode, str(mph), mpd, str(threshold), edge))

class EyeSignalOperator(Operator):
	"""
	EyeSignalOperator operates on eye signals, preferably sampled at 1000 Hz. 
	This operator is just created by feeding it timepoints,
	eye signals and pupil size signals in separate arrays, on a per-eye basis.
	
	Upon init it creates internal variables self.timepoints, self.raw_gazeXY, self.raw_pupil, self.sample_rate
	and, if available, self.blink_dur, self.blink_starts and self.blink_ends
	
	Its further methods create internal variables storing more derived
	signals that result from further processing.
	"""
	def __init__(self, input_object, **kwargs):
		"""input_object is a dictionary with timepoints, gaze_X, and gaze_Y and pupil keys and timeseries as values"""
		super(EyeSignalOperator, self).__init__(input_object = input_object, **kwargs)
		self.timepoints = np.array(self.input_object['timepoints']).squeeze()
		self.raw_gaze_X = np.array(self.input_object['gaze_X']).squeeze()
		self.raw_gaze_Y = np.array(self.input_object['gaze_Y']).squeeze()
		self.raw_pupil = np.array(self.input_object['pupil']).squeeze()
		
		if hasattr(self, 'eyelink_blink_data'):
			# internalize all blinks smaller than 4 seconds, since these are missing signals to be treated differently
			self.blink_dur_EL = np.array(self.eyelink_blink_data['duration']) 
			self.blink_starts_EL = np.array(self.eyelink_blink_data['start_timestamp'])[self.blink_dur_EL<4000] - self.timepoints[0]
			self.blink_ends_EL = np.array(self.eyelink_blink_data['end_timestamp'])[self.blink_dur_EL<4000] - self.timepoints[0]
			self.blink_dur_EL = np.array(self.eyelink_blink_data['duration'])[self.blink_dur_EL<4000]		
		
		if hasattr(self, 'eyelink_sac_data'):
			self.sac_dur_EL = np.array(self.eyelink_sac_data['duration']) 
			self.sac_starts_EL = np.array(self.eyelink_sac_data['start_timestamp']) - self.timepoints[0]
			self.sac_ends_EL = np.array(self.eyelink_sac_data['end_timestamp']) - self.timepoints[0]
		
		if not hasattr(self, 'sample_rate'): # this should have been set as a kwarg, but if it hasn't we just assume a standard 1000 Hz
			self.sample_rate = 1000.0

	
	def interpolate_blinks(self, method='linear', lin_interpolation_points=[[-200],[200]], spline_interpolation_points=[[-0.15,-0.075], [0.075,0.15]], coalesce_period=500, threshold_level = 0.01):
		"""
		interpolate_blinks interpolates blink periods with method, which can be spline or linear.
		spline_interpolation_points is a 2 by X list detailing the data points around the blinks
		(in s offset from blink start and end) that should be used for fitting the interpolation spline.

		The coalesce_period and threshold_level arguments serve to coalesce closely spaced blinks, 
		but only when eyelink-detected blinks are not provided. 
		The interval within which blinks have to fall in order to be grouped together is defined 
		by the coalesce_period, threshold_level determines the level at which data is regarded as 'missing data'.  

		The results are stored in self.interpolated_pupil, self.interpolated_x and self.interpolated_y
		without affecting the self.raw_... variables

		After calling this method, additional interpolation may be performed by calling self.interpolate_blinks2()
		"""
		self.logger.info('Interpolating blinks using interpolate_blinks')
		# set all missing data to 0:
		self.raw_pupil[self.raw_pupil<threshold_level] = 0
		
		# blinks to work with -- preferably eyelink!
		if hasattr(self, 'eyelink_blink_data'):
			for i in range(len(self.blink_starts_EL)):
				self.raw_pupil[self.blink_starts_EL[i]:self.blink_ends_EL[i]] = 0 # set all eyelink-identified blinks to 0:
		else:
			self.blinks_indices = pd.rolling_mean(np.array(self.raw_pupil == 0, dtype = float), int(coalesce_period)) > 0
			self.blinks_indices = np.array(self.blinks_indices, dtype=int)
			self.blink_starts = self.timepoints[:-1][np.diff(self.blinks_indices) == 1]
			self.blink_ends = self.timepoints[:-1][np.diff(self.blinks_indices) == -1]
			# now make sure we're only looking at the blnks that fall fully inside the data stream
			try:
				if self.blink_starts[0] > self.blink_ends[0]:
					self.blink_ends = self.blink_ends[1:]
				if self.blink_starts[-1] > self.blink_ends[-1]:
					self.blink_starts = self.blink_starts[:-1]
			except:
				print('probably not enough blinks to do full checks in this recording')
		
		# we do not want to start or end with a 0:
		import copy
		self.interpolated_pupil = copy.copy(self.raw_pupil[:])
		self.interpolated_x = copy.copy(self.raw_gaze_X)
		self.interpolated_y = copy.copy(self.raw_gaze_Y)
		self.interpolated_pupil[:coalesce_period*2] = max(np.percentile(self.interpolated_pupil[:int(self.sample_rate*2.5)], 90), np.percentile(self.interpolated_pupil, 50))
		self.interpolated_pupil[-coalesce_period:] = max(np.percentile(self.interpolated_pupil[-int(self.sample_rate*2.5):], 90), np.percentile(self.interpolated_pupil, 50))
		self.interpolated_x[:coalesce_period*2] = np.percentile(self.interpolated_x[:int(self.sample_rate*2.5)], 50)
		self.interpolated_x[-coalesce_period:] = np.percentile(self.interpolated_x[-int(self.sample_rate*2.5):], 50)
		self.interpolated_y[:coalesce_period*2] = np.percentile(self.interpolated_y[:int(self.sample_rate*2.5)], 50)
		self.interpolated_y[-coalesce_period:] = np.percentile(self.interpolated_y[-int(self.sample_rate*2.5):], 50)
		
		# detect zero edges (we just created from blinks, plus missing data):
		zero_edges = np.arange(self.interpolated_pupil.shape[0])[np.diff((self.interpolated_pupil<1))]
		if zero_edges.shape[0] == 0:
			pass
		else:
			zero_edges = zero_edges[:int(2 * np.floor(zero_edges.shape[0]/2.0))].reshape(-1,2)
		
		self.blink_starts = zero_edges[:,0]
		self.blink_ends = zero_edges[:,1]
		
		# check for neighbouring blinks (coalesce_period, default is 500ms), and string them together:
		start_indices = np.ones(self.blink_starts.shape[0], dtype=bool)
		end_indices = np.ones(self.blink_ends.shape[0], dtype=bool)
		for i in range(self.blink_starts.shape[0]):
			try:
				if self.blink_starts[i+1] - self.blink_ends[i] <= coalesce_period:
					start_indices[i+1] = False
					end_indices[i] = False
			except IndexError:
				pass
		
		# these are the blink start and end samples to work with:
		if sum(start_indices) > 0:
			self.blink_starts = self.blink_starts[start_indices]
			self.blink_ends = self.blink_ends[end_indices]
		else:
			self.blink_starts = None
			self.blink_ends = None
		self.blink_starts = self.blink_starts[self.blink_starts>coalesce_period]
		self.blink_ends = self.blink_ends[self.blink_starts>coalesce_period]
		
		# do actual interpolation:
		if method == 'spline':
			points_for_interpolation = np.array(np.array(spline_interpolation_points) * self.sample_rate, dtype = int)
			for bs, be in zip(self.blink_starts, self.blink_ends):
				samples = np.ravel(np.array([bs + points_for_interpolation[0], be + points_for_interpolation[1]]))
				sample_indices = np.arange(self.raw_pupil.shape[0])[np.sum(np.array([self.timepoints == s for s in samples]), axis = 0)]
				spline = interpolate.InterpolatedUnivariateSpline(sample_indices,self.raw_pupil[sample_indices])
				self.interpolated_pupil[sample_indices[0]:sample_indices[-1]] = spline(np.arange(sample_indices[1],sample_indices[-2]))
				spline = interpolate.InterpolatedUnivariateSpline(sample_indices,self.raw_gaze_X[sample_indices])
				self.interpolated_x[sample_indices[0]:sample_indices[-1]] = spline(np.arange(sample_indices[1],sample_indices[-2]))
				spline = interpolate.InterpolatedUnivariateSpline(sample_indices,self.raw_gaze_Y[sample_indices])
				self.interpolated_y[sample_indices[0]:sample_indices[-1]] = spline(np.arange(sample_indices[1],sample_indices[-2]))
		elif method == 'linear':
			if self.blink_starts != None:
				points_for_interpolation = np.array([self.blink_starts, self.blink_ends], dtype=int).T + np.array(lin_interpolation_points).T
				for itp in points_for_interpolation:
					self.interpolated_pupil[itp[0]:itp[-1]] = np.linspace(self.interpolated_pupil[itp[0]], self.interpolated_pupil[itp[-1]], itp[-1]-itp[0])
					self.interpolated_x[itp[0]:itp[-1]] = np.linspace(self.interpolated_x[itp[0]], self.interpolated_x[itp[-1]], itp[-1]-itp[0])
					self.interpolated_y[itp[0]:itp[-1]] = np.linspace(self.interpolated_y[itp[0]], self.interpolated_y[itp[-1]], itp[-1]-itp[0])
					
	def interpolate_blinks2(self, lin_interpolation_points = [[-100],[100]]):
		
		"""
		interpolate_blinks2 performs linear interpolation around peaks in the rate of change of
		the pupil size.
		
		The results are stored in self.interpolated_pupil, self.interpolated_x and self.interpolated_y
		without affecting the self.raw_... variables.
		
		This method is typically called after an initial interpolation using self.interpolateblinks(),
		consistent with the fact that this method expects the self.interpolated_... variables to already exist.
		"""
		
		self.pupil_diff = (np.diff(self.interpolated_pupil) - np.diff(self.interpolated_pupil).mean()) / np.diff(self.interpolated_pupil).std()
		self.peaks = detect_peaks(self.pupil_diff, mph=10, mpd=500, threshold=None, edge='rising', kpsh=False, valley=False, show=False, ax=False)[:-1] # last peak might not reflect blink...
		if self.peaks != None:
			points_for_interpolation = np.array([self.peaks, self.peaks], dtype=int).T + np.array(lin_interpolation_points).T
			for itp in points_for_interpolation:
				self.interpolated_pupil[itp[0]:itp[-1]] = np.linspace(self.interpolated_pupil[itp[0]], self.interpolated_pupil[itp[-1]], itp[-1]-itp[0])
				self.interpolated_x[itp[0]:itp[-1]] = np.linspace(self.interpolated_x[itp[0]], self.interpolated_x[itp[-1]], itp[-1]-itp[0])
				self.interpolated_y[itp[0]:itp[-1]] = np.linspace(self.interpolated_y[itp[0]], self.interpolated_y[itp[-1]], itp[-1]-itp[0])
			
	def filter_pupil(self, hp = 0.01, lp = 4.0):
		"""
		band_pass_filter_pupil band pass filters the pupil signal using a butterworth filter of order 3. 
		
		The results are stored in self.lp_filt_pupil, self.hp_filt_pupil and self.bp_filt_pupil
		
		This method is typically called after self.interpolateblinks() and, optionally, self.interpolateblinks2(),
		consistent with the fact that this method expects the self.interpolated_... variables to exist.
		"""
		self.logger.info('Band-pass filtering of pupil signals, hp = %2.3f, lp = %2.3f'%(hp, lp))

		# band-pass filtering of signal, high pass first and then low-pass
		# High pass:
		hp_cof_sample = hp / (self.interpolated_pupil.shape[0] / self.sample_rate / 2)
		bhp, ahp = sp.signal.butter(3, hp_cof_sample, btype='high')
		self.hp_filt_pupil = sp.signal.filtfilt(bhp, ahp, self.interpolated_pupil)
		# Low pass:
		lp_cof_sample = lp / (self.interpolated_pupil.shape[0] / self.sample_rate / 2)
		blp, alp = sp.signal.butter(3, lp_cof_sample)
		self.lp_filt_pupil = sp.signal.filtfilt(blp, alp, self.interpolated_pupil)
		# Band pass:
		self.bp_filt_pupil = sp.signal.filtfilt(blp, alp, self.hp_filt_pupil)
		
		# we may also add a baseline variable which contains the baseine 
		# by doing 3rd order savitzky-golay filtering, with a width of ~100 s
		# we dan use this baseline signal for correlations of phasic and tonic pupil responses, for example
		
		# self.baseline_filt_pupil = savitzky_golay(self.interpolated_pupil, self.sample_rate / (hp * 0.25), 3)
		self.baseline_filt_pupil = self.lp_filt_pupil - self.bp_filt_pupil
		
	def zscore_pupil(self, dtype = 'bp_filt_pupil'):
		"""
		zscore_pupil z-scores the low-pass filtered pupil size data and the band-pass filtered pupil size data.
		
		The results are stored in self.bp_filt_pupil_zscore and self.lp_filt_pupil_zscore.
		
		This method is typically called after self.filter_pupil(), consistent with the fact that it
		expects the self.[...]_filt_pupil variables to exist.
		"""
		
		# self.bp_filt_pupil_clean_zscore = (self.bp_filt_pupil_clean - self.bp_filt_pupil_clean.mean()) / self.bp_filt_pupil_clean.std()
		# self.bp_filt_pupil_zscore = (self.bp_filt_pupil - self.bp_filt_pupil.mean()) / self.bp_filt_pupil.std()
		# self.lp_filt_pupil_zscore = (self.lp_filt_pupil - self.lp_filt_pupil.mean()) / self.lp_filt_pupil.std()
		# self.baseline_filt_pupil_zscore = (self.baseline_filt_pupil - self.baseline_filt_pupil.mean()) / self.baseline_filt_pupil.std()
		
		exec('self.' + str(dtype) + '_zscore = (self.' + str(dtype) + ' - np.mean(self.' + str(dtype) + ')) / np.std(self.' + str(dtype) + ')')
		
	def percent_signal_change_pupil(self, dtype = 'bp_filt_pupil'):
		"""
		percent_signal_change_pupil takes percent signal change of the dtype pupil signal, and internalizes it as a dtype + '_psc' self variable.
		"""
		
		exec('self.' + str(dtype) + '_psc = ((self.' + str(dtype) + ' / np.mean(self.baseline_filt_pupil[2000:-2000])) * 100) - 100' )
		
		
	def dt_pupil(self, dtype = 'bp_filt_pupil'):
		"""
		dt_pupil takes the temporal derivative of the dtype pupil signal, and internalizes it as a dtype + '_dt' self variable.
		"""
		
		exec('self.' + str(dtype) + '_dt = np.r_[0, np.diff(self.' + str(dtype) + ')]' )		

	def time_frequency_decomposition_pupil(self, 
										   minimal_frequency = 0.0025, 
										   maximal_frequency = 0.1, 
										   nr_freq_bins = 7, 
										   n_cycles = 1, 
										   cycle_buffer = 3, 
										   tf_decomposition='lp_butterworth'): 
		"""time_frequency_decomposition_pupil has two options of time frequency decomposition on the pupil  data: 1) morlet wavelet transform from mne package 
			or 2) low-pass butterworth filters. Before tf-decomposition the minimal frequency in the data is compared to the input minimal_frequency using np.fft.fftfreq. 
			
			1) Morlet wavelet transform. Interpolated pupil data is z-scored and zero-padded to avoid edge artifacts during wavelet transformation. After morlet 
			transform, zero-padding is removed and transformed data is saved in a DataFrame self.band_pass_filter_bank_pupil with wavelet frequencies as columns.
			
			2) Low-pass butterworth filters. Low-pass cutoff samples are calculated for each frequency in frequencies. Low-pass filtering is performed and saved in 
			lp_filter_bank_pupil. Note: low-pass filtered signals are not yet band-pass here, thus, filtered signals with higher frequency cutoffs share lower frequency 
			information at that point. band_pass_signals calculates the difference between subsequent lp_filter_bank_pupil signals to make independent filter bands and 
			vstacks the lowest frequency to the datamatrix. Lastly, band_pass_signals are saved in a df in self.band_pass_filter_bank_pupil with low-pass frequencies as columns. 
			"""
		
		# check minimal frequency
		min_freq_in_data = np.fft.fftfreq(self.timepoints.shape[0], 1.0/self.sample_rate)[1] 
		if minimal_frequency < min_freq_in_data and minimal_frequency != None:
			self.logger.warning("""time_frequency_decomposition_pupil: 
									requested minimal_frequency %2.5f smaller than 
									data allows (%2.5f). """%(minimal_frequency, min_freq_in_data))

		if minimal_frequency == None:
			minimal_frequency = min_freq_in_data

		# use minimal_frequency for bank of logarithmically frequency-spaced filters
		frequencies = np.logspace(np.log10(maximal_frequency), np.log10(minimal_frequency), nr_freq_bins)
		self.logger.info('Time_frequency_decomposition_pupil, with filterbank %s'%str(frequencies))
	
		if tf_decomposition == 'morlet': 
			#z-score self.interpolated_pupil before morlet decomposition of pupil signal 
			interpolated_pupil_z = ((self.interpolated_pupil - np.mean(self.interpolated_pupil))/self.interpolated_pupil.std())
			#zero-pad runs to avoid edge-artifacts 
			zero_padding_samples = int((1/minimal_frequency)*self.sample_rate*cycle_buffer)
			padded_interpolated_pupil_z = np.zeros((interpolated_pupil_z.shape[0] + 2*(zero_padding_samples)))
			padded_interpolated_pupil_z[zero_padding_samples:-zero_padding_samples] = interpolated_pupil_z	
			#filtered signal is real part of Morlet-transformed signal
			padded_band_pass_filter_bank_pupil = np.squeeze(np.real(mne.time_frequency.cwt_morlet(padded_interpolated_pupil_z[np.newaxis,:], self.sample_rate, frequencies, use_fft=True, n_cycles=n_cycles, zero_mean=True)))
			#remove zero-padding and save as dataframe with frequencies as index
			self.band_pass_filter_bank_pupil = pd.DataFrame(np.array([padded_band_pass_filter_bank_pupil[i][zero_padding_samples:-zero_padding_samples] for i in range(len(padded_band_pass_filter_bank_pupil))]).T,columns=frequencies)
		
		elif tf_decomposition == 'lp_butterworth': 
			lp_filter_bank_pupil=np.zeros((len(frequencies), self.interpolated_pupil.shape[0]))
			lp_cof_samples = [freq / (self.interpolated_pupil.shape[0] / self.sample_rate / 2) for freq in frequencies]
			for i, lp_cutoff in enumerate(lp_cof_samples): 
				blp, alp = sp.signal.butter(3, lp_cutoff) 
				lp_filt_pupil = sp.signal.filtfilt(blp, alp, self.interpolated_pupil)
				lp_filter_bank_pupil[i,0:self.interpolated_pupil.shape[0]]=lp_filt_pupil
			#calculate band passes from the difference between subsequent low pass frequencies (except the last frequency, this one is directly added to the df as the lowest freq in data) 
			band_pass_signals = np.vstack((np.array(lp_filter_bank_pupil[:-1]) - np.array(lp_filter_bank_pupil[1:]), lp_filter_bank_pupil[-1]))
			self.band_pass_filter_bank_pupil = pd.DataFrame(band_pass_signals.T, columns=frequencies)
		
		else: 
			print('you did not specify a tf-decomposition')
	
	def regress_blinks(self, interval = 7):
		"""
		"""
		self.logger.info('Regressing blinks, saccades and gaze position of pupil signals')
		# params:
		
		x = np.linspace(0, interval, interval * self.sample_rate, endpoint=False)
				
		# events:
		blinks = self.blink_ends
		blinks = blinks[blinks>25]
		blinks = blinks[blinks<((self.timepoints[-1]-self.timepoints[0]))-interval]
		
		if blinks.size == 0:
			blinks = np.array([0], dtype = int)
		else:
			blinks = blinks.astype(int)
		
		sacs = self.sac_ends_EL
		sacs = sacs[sacs>25]
		sacs = sacs[sacs<((self.timepoints[-1]-self.timepoints[0]))-interval]
		sacs = sacs.astype(int)

		def double_gamma(params, x): 
			a1 = params['a1']
			sh1 = params['sh1']
			sc1 = params['sc1']
			a2 = params['a2']
			sh2 = params['sh2']
			sc2 = params['sc2']
			return a1 * sp.stats.gamma.pdf(x, sh1, loc=0.0, scale = sc1) + a2 * sp.stats.gamma.pdf(x, sh2, loc=0.0, scale = sc2)
	
		# use standard values:
		standard_blink_parameters = {'a1':-0.604, 'sh1':8.337, 'sc1':0.115, 'a2':0.419, 'sh2':15.433, 'sc2':0.178}
		blink_kernel = double_gamma(standard_blink_parameters, x)

		standard_sac_parameters = {'a1':-0.175, 'sh1': 6.451, 'sc1':0.178, 'a2':0.0, 'sh2': 1, 'sc2': 1}
		sac_kernel = double_gamma(standard_sac_parameters, x)
				
		# blink and saccade regressors:
		blink_reg = np.zeros(self.bp_filt_pupil.shape[0])
		blink_reg[blinks] = 1
		blink_reg_conv = sp.signal.fftconvolve(blink_reg, blink_kernel, 'full')[:-(len(blink_kernel)-1)]
		sac_reg = np.zeros(self.bp_filt_pupil.shape[0])
		sac_reg[sacs] = 1
		sac_reg_conv = sp.signal.fftconvolve(sac_reg, sac_kernel, 'full')[:-(len(sac_kernel)-1)]

		# high-pass filter the eye position data to remove slow drifts
		hp_cof_sample = 0.005 / (self.interpolated_x.shape[0] / self.sample_rate / 2)
		bhp, ahp = sp.signal.butter(3, hp_cof_sample, btype='high')
		interpolated_x_hp = sp.signal.filtfilt(bhp, ahp, self.interpolated_x)
		interpolated_y_hp = sp.signal.filtfilt(bhp, ahp, self.interpolated_y)

		# we add x and y gaze position for foreshortening, and add intercept
		regs = [blink_reg_conv, sac_reg_conv, interpolated_x_hp, interpolated_y_hp, np.ones(sac_reg_conv.shape[-1]) ]
		print([r.shape for r in regs])

		# GLM:
		design_matrix = np.matrix(np.vstack([reg for reg in regs])).T
		betas = np.array(((design_matrix.T * design_matrix).I * design_matrix.T) * np.matrix(self.bp_filt_pupil).T).ravel()
		explained = np.sum(np.vstack([betas[i]*regs[i] for i in range(len(betas))]), axis=0)
		
		rsq = sp.stats.pearsonr(self.bp_filt_pupil, explained)
		self.logger.info('Nuisance GLM Results, pearsons R (p) is %1.3f (%1.4f)'%(rsq))
		self.logger.info('Nuisance GLM Results, Blink, Saccade beta %3.3f, %3.3f)'%(betas[0], betas[1]))
		self.logger.info('Nuisance GLM Results, Gaze x, y, intercept beta %3.3f, %3.3f, %3.3f )'%(betas[2], betas[3], betas[4]))

		# clean data are residuals:
		self.bp_filt_pupil_clean = self.bp_filt_pupil - explained
		
		# final timeseries:
		self.lp_filt_pupil_clean = self.bp_filt_pupil_clean + self.baseline_filt_pupil
		self.bp_filt_pupil_clean = self.bp_filt_pupil_clean + self.baseline_filt_pupil.mean()
		
	
		
		
		
	
		