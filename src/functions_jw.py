#!/usr/bin/env python
# encoding: utf-8

"""
functions_jw.py
Created by Jan Willem de Gee on 2012-06-19.
Copyright (c) 2012 Jan Willem de Gee. All rights reserved.
"""
from __future__ import division

import os, sys, datetime, pickle
import subprocess, logging, time

import numpy as np
import numpy
import numpy.random as random
import pandas as pd
import scipy as sp
import scipy.stats as stats
import bottleneck as bn
import sympy
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches
import seaborn as sns
from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import statsmodels.formula.api as sm
import mne
# import hddm
# import kabuki
# import pypsignifit as psi
from itertools import combinations as comb

from IPython import embed as shell


sns.set(style='ticks', font='Arial', font_scale=1, rc={
	'axes.linewidth': 0.25, 
	'axes.labelsize': 7, 
	'axes.titlesize': 7, 
	'xtick.labelsize': 6, 
	'ytick.labelsize': 6, 
	'legend.fontsize': 6, 
	'xtick.major.width': 0.25, 
	'ytick.major.width': 0.25,
	'text.color': 'Black',
	'axes.labelcolor':'Black',
	'xtick.color':'Black',
	'ytick.color':'Black',} )
sns.plotting_context()



def figure(ax_type=1):
	fig = plt.figure(num=None, figsize=(8.27, 11.69))
	if ax_type==1:
		ax = plt.subplot2grid((4, 3), (0, 0),rowspan=1, colspan=1)
	if ax_type==2:
		ax = plt.subplot2grid((4, 3), (0, 0),rowspan=1, colspan=2)
	if ax_type==3:
		ax = plt.subplot2grid((4, 3), (0, 0),rowspan=3, colspan=3)
	if ax_type==4:
		ax = plt.subplot2grid((4, 3), (0, 0),rowspan=3, colspan=3)
	return fig, ax

def common_dist(array_a, array_b, bins=25):
	
	"""Finds the common underlying distribution
	
	Parameters
	----------
	array_a : 1D array_like
		data.
	array_b : 1D array_like
		data.
	bins : postive number (default = 25)
		number of bins used.
	Returns
	-------
	indices_a : 1D array_like
		indices of values in 'array_a' to keep.
	indices_b : 1D array_like
		indices of values in 'array_b' to keep.
		
	Jan Willem de Gee, april 2015
	"""
	
	minimum = min(min(array_a),min(array_b))
	maximum = max(max(array_a),max(array_b))
	full_range = maximum - minimum
	edges = np.linspace(minimum-(full_range/20.0), maximum+(full_range/20.0), bins+1)
	indices_a = np.ones(len(array_a), dtype=bool)
	indices_b = np.ones(len(array_b), dtype=bool)
	for i in range(bins):
		ind_a = (array_a >= edges[i]) * (array_a <= edges[i+1])
		ind_b = (array_b >= edges[i]) * (array_b <= edges[i+1])
		sum_a = sum(ind_a)
		sum_b = sum(ind_b)
		if sum_a<sum_b:
			indices_b[np.random.choice(np.where(ind_b)[0], sum_b-sum_a, replace=False)] = False
		if sum_a>sum_b:
			indices_a[np.random.choice(np.where(ind_a)[0], sum_a-sum_b, replace=False)] = False
	return indices_a, indices_b


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
	try:
		import matplotlib.pyplot as plt
	except ImportError:
		print('matplotlib is not available.')
	else:
		if ax is None:
			_, ax = plt.subplots(1, 1, figsize=(8, 4))

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
		# plt.grid()
		plt.show()

def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def orthogonal_projection(timeseries, to_project_out):
	
	# take the norm of vector you want to project out:
	prj = to_project_out/np.linalg.norm(to_project_out)
	
	# project out the reference vector
	return timeseries - (np.dot(timeseries, prj)*prj)
	
def chunks(l, n):
	return [l[i:i+n] for i in range(0, len(l), n)]
def chunks_mean(l, n):
	return [np.mean(l[i:i+n]) for i in range(0, len(l), n)]
def chunks_std(l, n):
	return [np.std(l[i:i+n]) for i in range(0, len(l), n)]

def number_chunks(data, n_chuncks):
	m = float(len(data))/n_chuncks
	return [data[int(m*i):int(m*(i+1))] for i in range(n_chuncks)]

def number_chunks_mean(data, n_chuncks):
	m = float(len(data))/n_chuncks
	return [sp.mean(data[int(m*i):int(m*(i+1))]) for i in range(n_chuncks)]
def movingaverage(interval, window_size):

	window = np.ones(int(window_size))/float(window_size)
	moving_average = np.repeat(np.nan, len(interval))
	moving_average[(window_size/2):-(window_size/2)] = np.convolve(interval, window, 'full')[window_size-1:-window_size]

	return(moving_average)

def smooth(x, window_len=11, window='hanning'):
	if x.ndim != 1:
		raise ValueError, "smooth only accepts 1 dimension arrays."
	if x.size < window_len:
		raise ValueError, "Input vector needs to be bigger than window size."
	if window_len<3:
		return x
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
	s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:  
		w=eval('numpy.'+window+'(window_len)')
	y=np.convolve(w/w.sum(),s,mode='same')
	return y[window_len:-window_len+1]

def zscore_2d(array, rep):
	for i in range(rep):
		axis = i%2
		array = preprocessing.scale(array, axis=axis, with_mean=True, with_std=True)
	return array
	
# -----------------
# linear algebra  -
# -----------------
def magnitude(v):
	return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))

def add(u, v):
	return [ u[i]+v[i] for i in range(len(u)) ]

def sub(u, v):
	return [ u[i]-v[i] for i in range(len(u)) ]

def dot(u, v):
	return sum(u[i]*v[i] for i in range(len(u)))

def normalize(v):
	vmag = magnitude(v)
	return [ v[i]/vmag  for i in range(len(v)) ]

# -----------------
# pupil measures  -
# -----------------
def pupil_scalar_peak(data, time_start, time_end):
	pupil_scalars = np.array([ max(data[i,time_start:time_end]) for i in range(data.shape[0])])
	return pupil_scalars

def pupil_scalar_mean(data, time_start, time_end):
	pupil_scalars = np.array([ bn.nanmean(data[i,time_start:time_end]) for i in range(data.shape[0])])
	return pupil_scalars

def pupil_scalar_lin_projection(data, time_start, time_end, template):
	pupil_scalars = np.array([ np.dot(template, data[i,time_start:time_end])/np.dot(template,template) for i in range(data.shape[0])])
	return pupil_scalars

def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
	"""Kernel Density Estimation with Scikit-learn"""
	
	kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
	kde_skl.fit(x[:, np.newaxis])
	# score_samples() returns the log-likelihood of the samples
	log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
	return np.exp(log_pdf)


def bootstrap(data, nrand=10000, full_output = False, threshold = 0.0):
	"""data is a subjects by value array
	bootstrap returns the p-value of the difference with zero, and optionally the full distribution.
	"""

	permutes = np.random.randint(0, data.shape[0], size = (data.shape[0], nrand))
	bootstrap_distro = data[permutes].mean(axis = 0)
	p_val = 1.0 - ((bootstrap_distro > threshold).sum() / float(nrand))
	if full_output:
		return p_val, bootstrap_distro
	else:
		return p_val

def permutationTest(group1, group2, nrand=1000, tail=0, normalize=False):

	"""
	non-parametric permutation test (Efron & Tibshirani, 1998)

	tail = 0 (test A~=B), 1 (test A>B), -1 (test A<B)

	"""

	if normalize:
		means = np.vstack((group1, group2)).mean(axis=0)
		group1 = group1 - means
		group2 = group2 - means

	a = group1
	b = group2
	ntra = len(a)
	ntrb = len(b) 
	meana = np.mean(a)
	meanb = np.mean(b)
	alldat = np.concatenate((a,b))

	triala = np.zeros(nrand)
	trialb = np.zeros(nrand)

	indices = np.arange(alldat.shape[0])

	for i in range(nrand):
		random.shuffle(indices)
		triala[i] = np.mean(alldat[indices[:ntra]])
		trialb[i] = np.mean(alldat[indices[ntra:]])

	if tail == 0:
		p_value = sum(abs(triala-trialb)>=abs(meana-meanb)) / float(nrand)
	else:
		p_value = sum((tail*(triala-trialb))>=(tail*(meana-meanb))) / float(nrand)

	return(meana-meanb, p_value)

def permutationTest_SDT(target_indices, hit_indices, fa_indices, group_indices, nrand=5000, tail=0):

	"""
	non-parametric permutation test (Efron & Tibshirani, 1998)

	tail = 0 (test A~=B), 1 (test A>B), -1 (test A<B)

	"""

	# shell()

	import numpy as np
	import numpy.random as random

	d_a, c_a = SDT_measures(target_indices[group_indices], hit_indices[group_indices], fa_indices[group_indices])
	d_b, c_b = SDT_measures(target_indices[-group_indices], hit_indices[-group_indices], fa_indices[-group_indices]) 

	trial_c_a = np.zeros(nrand)
	trial_c_b = np.zeros(nrand)
	trial_d_a = np.zeros(nrand)
	trial_d_b = np.zeros(nrand)
	for i in range(nrand):
		random.shuffle(group_indices)
		trial_d_a[i], trial_c_a[i] = SDT_measures(target_indices[group_indices], hit_indices[group_indices], fa_indices[group_indices])
		trial_d_b[i], trial_c_b[i] = SDT_measures(target_indices[-group_indices], hit_indices[-group_indices], fa_indices[-group_indices]) 

	if tail == 0:
		p_value_c = sum(abs(trial_c_a-trial_c_b)>=abs(c_a-c_b)) / float(nrand)
		p_value_d = sum(abs(trial_d_a-trial_d_b)>=abs(d_a-d_b)) / float(nrand)
	else:
		p_value_c = sum((tail*(trial_c_a-trial_c_b))>=(tail*(c_a-c_b))) / float(nrand)
		p_value_d = sum((tail*(trial_d_a-trial_d_b))>=(tail*(d_a-d_b))) / float(nrand)

	return(p_value_d, p_value_c)

def permutationTest_correlation(a, b, tail=0, nrand=10000):
	"""
	test whether 2 correlations are significantly different. For permuting single corr see randtest_corr2
	function out = randtest_corr(a,b,tail,nrand, type)
	tail = 0 (test A~=B), 1 (test A>B), -1 (test A<B)
	type = 'Spearman' or 'Pearson'
	"""

	import numpy as np
	import numpy.random as random

	ntra = a.shape[0]
	ntrb = b.shape[0]
	# truecorrdiff = sp.stats.pearsonr(a[:,0],a[:,1])[0] - sp.stats.pearsonr(b[:,0],b[:,1])[0]
	truecorrdiff = sp.stats.spearmanr(a[:,0],a[:,1])[0] - sp.stats.spearmanr(b[:,0],b[:,1])[0]
	alldat = np.vstack((a,b))
	corrdiffrand = np.zeros(nrand)
	indices = np.arange(alldat.shape[0])

	for irand in range(nrand):
		random.shuffle(indices)
		# randa = sp.stats.pearsonr(alldat[indices[:ntra],0],alldat[indices[:ntra],1])[0]
		# randb = sp.stats.pearsonr(alldat[indices[ntra:],0],alldat[indices[ntra:],1])[0]
		randa = sp.stats.spearmanr(alldat[indices[:ntra],0],alldat[indices[:ntra],1])[0]
		randb = sp.stats.spearmanr(alldat[indices[ntra:],0],alldat[indices[ntra:],1])[0]
		corrdiffrand[irand] = randa - randb
	
	if tail == 0:
		p_value = sum(abs(corrdiffrand) >= abs(truecorrdiff)) / float(nrand)
	else:
		p_value = sum(tail*(corrdiffrand) >= tail*(truecorrdiff)) / float(nrand)

	return(truecorrdiff, p_value)


def roc_analysis(group1, group2, nrand=1000, tail=1):

	import scipy as sp
	import numpy as np
	import random
	from scipy.integrate import cumtrapz

	x = group1
	y = group2
	nx = len(x)
	ny = len(y)

	z = np.concatenate((x,y)) 
	c = np.sort(z) 

	det = np.zeros((c.shape[0],2))
	for ic in range(c.shape[0]):
		det[ic,0] = (x > c[ic]).sum() / float(nx)
		det[ic,1] = (y > c[ic]).sum() / float(ny)

	t1 = np.sort(det[:,0])
	t2 = np.argsort(det[:,0])

	roc = np.vstack(( [0,0],det[t2,:],[1,1] ))
	t1 = sp.integrate.cumtrapz(roc[:,0],roc[:,1])
	out_i = t1[-1]

	# To get the p-value:

	trialx = np.zeros(nrand)
	trialy = np.zeros(nrand)
	alldat = np.concatenate((x,y))

	fprintf = []
	randi = []
	for irand in range(nrand):
		if not np.remainder(irand,1000):
			fprintf.append('randomization: %d\n' + str(irand))
		
		t1 = np.sort(np.random.rand(nx+ny))
		ind = np.argsort(np.random.rand(nx+ny))
		ranx = z[ind[0:nx]]
		rany = z[ind[nx+1:-1]]
		randc = np.sort( np.concatenate((ranx,rany)) )
	
		randet = np.zeros((randc.shape[0],2))
		for ic in range(randc.shape[0]):
			randet[ic,0] = (ranx > randc[ic]).sum() / float(nx)
			randet[ic,1] = (rany > randc[ic]).sum() / float(ny)
		
		t1 = np.sort(randet[:,0])
		t2 = np.argsort(randet[:,0])
	
		ranroc = np.vstack(( [0,0],randet[t2,:],[1,1] ))
		t1 = sp.integrate.cumtrapz(ranroc[:,0],ranroc[:,1])
	
		randi.append(t1[-1])
	
	randi = np.array(randi)

	if tail == 0: # (test for i != 0.5)
		out_p = (abs(randi-0.5) >= abs(out_i-0.5)).sum() / float(nrand)
	if (tail == 1) or (tail == -1): # (test for i > 0.5, and i < 0.5 respectively)
		out_p = (tail*(randi-0.5) >= tail*(out_i-0.5)).sum() / float(nrand)
	
	if (float(1) - out_p) < out_p:
		out_p = float(1) - out_p
	
	return(out_i, out_p)





def SDT_measures(target, hit, fa):

	"""
	Computes d' and criterion
	"""

	import numpy as np
	import scipy as sp
	import matplotlib.pyplot as plt
	
	target = np.array(target, dtype=bool)
	hit = np.array(hit, dtype=bool)
	fa = np.array(fa, dtype=bool)
	
	hit_rate = (np.sum(hit) + 1) / (float(np.sum(target)) + 1)
	fa_rate = (np.sum(fa) + 1) / (float(np.sum(-target)) + 1)
	hit_rate_z = stats.norm.isf(1-hit_rate)
	fa_rate_z = stats.norm.isf(1-fa_rate)

	d = hit_rate_z - fa_rate_z
	c = -(hit_rate_z + fa_rate_z) / 2.0

	return(d, c)

def corr_matrix(C, dv='cor'):
	
	C = np.asarray(C)
	p = C.shape[1]
	P_corr = np.zeros((p, p), dtype=np.float)
	P_p = np.zeros((p, p), dtype=np.float)
	for i in range(p):
		# P_corr[i, i] = 1
		for j in range(i, p):
			if dv=='cor':
				corr, p_value = sp.stats.pearsonr(C[:, i], C[:, j])
			if dv=='var':
				corr = np.mean((np.var(C[:, i]), np.var(C[:, j])))
				p_value = 1
			if dv=='cov':
				# corr = (np.var(C[:, i]+C[:, j]) - np.var(C[:, i]) - np.var(C[:, j])) / 2.0
				corr = np.cov(C[:, i], C[:, j])[0][1]
				p_value = 1
			if dv=='mean':
				corr = np.mean((np.mean(C[:, i]), np.mean(C[:, j])))
				p_value = 1
			if dv=='snr':
				corr = np.mean(( np.mean(C[:, i])/np.std(C[:, i]), np.mean(C[:, j])/np.std(C[:, j]) ))
				p_value = 1
			P_corr[i, j] = corr
			P_corr[j, i] = corr
			P_p[i, j] = p_value
			P_p[j, i] = p_value

	return P_corr, P_p


def corr_matrix_partial(C):
	"""
	
	Partial Correlation in Python (clone of Matlab's partialcorr)

	This uses the linear regression approach to compute the partial 
	correlation (might be slow for a huge number of variables). The 
	algorithm is detailed here:

	    http://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression

	Taking X and Y two variables of interest and Z the matrix with all the variable minus {X, Y},
	the algorithm can be summarized as

	    1) perform a normal linear least-squares regression with X as the target and Z as the predictor
	    2) calculate the residuals in Step #1
	    3) perform a normal linear least-squares regression with Y as the target and Z as the predictor
	    4) calculate the residuals in Step #3
	    5) calculate the correlation coefficient between the residuals from Steps #2 and #4; 

	    The result is the partial correlation between X and Y while controlling for the effect of Z
	
	Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
	for the remaining variables in C.
	
	
	Parameters
	----------
	C : array-like, shape (n, p)
	    Array with the different variables. Each column of C is taken as a variable
	
	
	Returns
	-------
	P : array-like, shape (p, p)
	    P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
	    for the remaining variables in C.
	"""
	
	C = np.asarray(C)
	p = C.shape[1]
	P_corr = np.zeros((p, p), dtype=np.float)
	P_p = np.zeros((p, p), dtype=np.float)
	for i in range(p):
		P_corr[i, i] = 1
		for j in range(i+1, p):
			idx = np.ones(p, dtype=np.bool)
			idx[i] = False
			idx[j] = False
			beta_i = sp.linalg.lstsq(C[:, idx], C[:, j])[0]
			beta_j = sp.linalg.lstsq(C[:, idx], C[:, i])[0]
			
			res_j = C[:, j] - C[:, idx].dot( beta_i)
			res_i = C[:, i] - C[:, idx].dot(beta_j)
			
			corr, p_value = sp.stats.pearsonr(res_i, res_j)
			P_corr[i, j] = corr
			P_corr[j, i] = corr
			P_p[i, j] = p_value
			P_p[j, i] = p_value
	
	return P_corr, P_p



def lin_regress_resid(Y,X,cat=False):
	# shell()
	
	# X = X - np.mean(X)
	# Y = Y - np.mean(Y)
	
	d = {
		'X' : pd.Series(X),
		'Y' : pd.Series(Y),
		# 'Z' : pd.Series(Z),
		}
	data = pd.DataFrame(d)
	
	model = sm.ols(formula='Y ~ X', data=data)
	fitted = model.fit()
	residuals = fitted.resid

	return np.array(residuals)

def pcf3(X,Y,Z):
	"""
	Compute a tuple of the partial correlation coefficients
	r_XY|z , r_XZ|y, r_YZ|x 
	"""
	xbar = sp.mean(X)
	ybar = sp.mean(Y)
	zbar = sp.mean(Z)
	xvar = sp.svar(X)
	yvar = sp.svar(Y)
	zvar = sp.svar(z)
	# computes pairwise simple correlations.
	rxy  = corr(X,Y, xbar=xbar, xvar= xvar, ybar = ybar, yvar = yvar)
	rxz  = corr(X,Z, xbar=xbar, xvar= xvar, zbar = zbar, zvar = zvar)
	ryz  = corr(Y,Z, ybar=ybar, yvar= yvar, zbar = zbar, zvar = zvar)
	rxy_z = rxy -rxz*ryz/sqrt((1 -rxz**2)*(1-ryz**2))
	rxz_y = rxz -rxy*rzy/sqrt((1-rxy**2) *(1-rzy**2))
	ryz_x = ryz -rxy*rxz/sqrt((1-rxy**2) *(1-rxz**2))
	return [(rxy_z, rxz_y, ryz_x)]


# import rpy2.robjects as robjects
# from rpy2.robjects import pandas2ri
# pandas2ri.activate()
# import pandas.rpy.common as com
 
def rm_ANOVA(df):
	
	"""
	Perform a repeated measures ANOVA using Rpy2 with a pandas DataFrame as input.

	Parameters
	----------
	df : pandas DataFrame
	                The DataFrame should have the factors to test as columns and the
	                corresponding data of each subject per row. The first column should
	                contain the subject no.s

	Returns
	-------
	anova_tables : list
	                A list containing pandas DataFrames for each tested effect

	Example
	--------
	The input DataFrame should have a structure like this:

	+-------------+------------+----------+----------+
	| subject_nr  | var 1      | var 2    | var N    |
	+=============+============+==========+==========+
	| 1           | 334.5      | 450.6    | ...      |
	+-------------+------------+----------+----------+
	| 2           | 545        | 354      | ...      |
	+-------------+------------+----------+----------+
	| ...         | ...        | ...      | ...      |
	+-------------+------------+----------+----------+
	"""
	
	
	# Convert grouping var to vector for R (int types screw up analysis)
	df[df.columns[0]] = df[df.columns[0]].astype(object)
	
	# Get ready to fire data at R
	r_long_table = com.convert_to_r_dataframe(df)

	# Get independent vars. Assume [0] is subject_nr and [-1] is dependent var, so leave away
	factors = "*".join(df.columns[2:])

	# Specify the LM forumula
	formula = robjects.r("as.formula(col2 ~ ({0}) + Error(col1/({0})))".format(factors))

	# Perform the analysis and get the summary of it
	aov_analysis = robjects.r["aov"](formula, data=r_long_table)
	anova_summary = robjects.r["summary"](aov_analysis)

	# Convert R dataframes back to pandas.DataFrames and put them in a list
	anova_tables = []
	for entry in anova_summary:
		for table in entry:
			anova_tables.append(com.convert_robj(table))
	
	# Return all 
	return anova_tables
		
		
def pupil_IRF(timepoints, s=1.0/(10**26), n=10.1, tmax=930):

	"""
	Default settings (above): canocial pupil impulse fucntion [ref].

	"""

	# sympy variable:
	t = sympy.Symbol('t')

	# function:
	y = ( (s) * (t**n) * (math.e**((-n*t)/tmax)) )

	# derivative:
	y_dt = y.diff(t)

	# lambdify:
	y = sympy.lambdify(t, y, "numpy")
	y_dt = sympy.lambdify(t, y_dt, "numpy")

	# evaluate:
	y = y(timepoints)
	y_dt = y_dt(timepoints)

	return (y / np.std(y), y_dt / np.std(y_dt))



def pupil_IRF_dn(timepoints, s=1.0/(10**26), n=10.1, tmax=930):

	"""
	Default settings (above): canocial pupil impulse fucntion [ref].

	"""

	y = ( (s) * (timepoints**n) * (math.e**((-n*timepoints)/tmax)) )
	y = y / np.std(y)

	y2 = ( (s) * (timepoints**(n-0.1)) * (math.e**((-(n-0.1)*timepoints)/tmax)) )
	y2 = y2 / np.std(y2)

	y3 = y - y2

	return y3 / np.std(y3)


def psychometric_curve(contrasts, answers, corrects, yesno=False):

	contrasts_unique = np.unique(contrasts)

	if yesno:
	
		ind = ((answers == 1)*(corrects == 1)) + ((answers == 0)*(corrects == 0))
		contrasts = contrasts[ind]
		answers = answers[ind]
		corrects = corrects[ind]
	
		# psychometric curve settings
		# gumbel_l makes the fit a weibull given that the x-data are log-transformed
		# logistic and gauss (core = ab) deliver similar results
		nafc = 1
		sig = 'logistic'
		core = 'ab'
		cuts = [0.50]
		constraints = ( 'flat', 'flat', 'Beta(2,20)' )
		# constraints = ( 'flat', 'flat', 'flat' )
		# constraints = ( 'unconstrained', 'unconstrained', 'Uniform(0.05,0.1)' )

		# separate out the per-test orientation answers
		corrects_grouped = [answers[contrasts == c] for c in contrasts_unique]
		# and sum them...
		corrects_grouped_sum = [c.sum() for c in corrects_grouped]
		# we also need the amount of samples
		nr_samples = [float(c.shape[0]) for c in corrects_grouped]
		# now we construct the tuple that goes into the pypsignifit package
		fit_data = zip(contrasts_unique, corrects_grouped_sum, nr_samples )
	
		# and we fit the data
		pf = psi.BootstrapInference(fit_data, sigmoid=sig, nafc=nafc, core=core, cuts=cuts, priors=constraints, gammaislambda=True)
	
	else:
		# psychometric curve settings
		# gumbel_l makes the fit a weibull given that the x-data are log-transformed
		# logistic and gauss (core = ab) deliver similar results
		nafc = 2
		sig = 'logistic'
		core = 'ab'
		cuts = [0.75]
		constraints = ( 'unconstrained', 'unconstrained', 'Uniform(0,0.1)' )
		# constraints = ( 'unconstrained', 'unconstrained', 'Uniform(0.05,0.1)' )

		# separate out the per-test orientation answers
		corrects_grouped = [corrects[contrasts == c] for c in contrasts_unique]
		# and sum them...
		corrects_grouped_sum = [c.sum() for c in corrects_grouped]
		# we also need the amount of samples
		nr_samples = [float(c.shape[0]) for c in corrects_grouped]
		# now we construct the tuple that goes into the pypsignifit package
		fit_data = zip(contrasts_unique, corrects_grouped_sum, nr_samples)

		# and we fit the data
		pf = psi.BootstrapInference(fit_data, sigmoid=sig, nafc=nafc, core=core, cuts=cuts, priors=constraints)
	# and let the package do a bootstrap sampling of the resulting fits
	pf.sample()

	# psi.GoodnessOfFit(pf)

	return (pf, corrects_grouped)

def hist_q(rt, bins=10, quantiles=[10,30,50,70,90], ax=None, xlim=None, ylim=None, quantiles_color='k', alpha=1):

	nr_observations = len(rt) / 100.0 * np.diff(quantiles)
	break_points = np.percentile(rt, quantiles)
	widths = np.diff(break_points)
	total_width = max(rt) - min(rt)
	heights = nr_observations/(widths/total_width*bins)

	if ax is None:
		ax = plt.gca()
	quantile_hist = ax.hist(rt, bins=bins, color='k', alpha=0.25, lw=0)
	for i in range(heights.shape[0]):
		rect = patches.Rectangle((break_points[i], 0), widths[i], heights[i], fill=False, color=quantiles_color, linewidth=2, alpha=alpha)
		ax.add_patch(rect)
	if xlim:
		ax.set_xlim(xlim)
	if ylim:
		ax.set_ylim(ylim)
	ax2 = ax.twiny()
	ax2.set_xlim(ax.get_xlim())
	ax2.set_xticks([round(bp, 2) for bp in break_points])
	ax2.set_xticklabels([round(bp, 2) for bp in break_points], rotation=45)
	# ax2.set_xlabel('{} quantiles'.format(quantiles))
	return quantile_hist

def hist_q2(rt, bins=10, quantiles=[10,30,50,70,90], corrects=None, ax=None, xlim=None, ylim=None, quantiles_color='k', alpha=1, extra_xticks=None):

	if ax is None:
		ax = plt.gca()

	nr_observations = len(rt[corrects]) / 100.0 * np.diff(quantiles)
	break_points = np.percentile(rt[corrects], quantiles)
	widths = np.diff(break_points)
	total_width = max(rt[corrects]) - min(rt[corrects])
	heights = nr_observations/(widths/total_width*bins)

	# quantile_hist = sns.distplot(rt[corrects], bins=bins, hist=True, norm_hist=True, rug=True, color='g')
	quantile_hist = ax.hist(rt[corrects], bins=bins, color='g', alpha=0.5, lw=0)
	for j in range(heights.shape[0]):
		rect = patches.Rectangle((break_points[j], 0), widths[j], heights[j], fill=False, color=quantiles_color, linewidth=1, alpha=alpha)
		ax.add_patch(rect)
	
	nr_observations = len(rt[-corrects]) / 100.0 * np.diff(quantiles)
	break_points2 = np.percentile(rt[-corrects]*-1.0, quantiles)
	widths = np.diff(break_points2)
	total_width = max(rt[-corrects]*-1.0) - min(rt[-corrects]*-1.0)
	heights = nr_observations/(widths/total_width*bins)

	quantile_hist = ax.hist(rt[-corrects]*-1.0, bins=bins, color='r', alpha=0.5, lw=0)
	for j in range(heights.shape[0]):
		rect = patches.Rectangle((break_points2[j], 0), widths[j], heights[j], fill=False, color=quantiles_color, linewidth=1, alpha=alpha)
		ax.add_patch(rect)
	if xlim:
		ax.set_xlim(xlim)
	if ylim:
		ax.set_ylim(ylim)
	else:
		ax.set_ylim(ymin=0)
	if extra_xticks:
		ax2 = ax.twiny()
		ax2.set_xlim(ax.get_xlim())
		ax2.set_xticks([round(bp, 2) for bp in list(np.concatenate((break_points, break_points2)))])
		ax2.set_xticklabels([round(bp, 2) for bp in list(np.concatenate((break_points, break_points2)))], rotation=45)
	# ax2.set_xlabel('{} quantiles'.format(quantiles))
	return quantile_hist

def quantile_plot(conditions, rt, corrects, subj_idx, quantiles=[10,30,50,70,90], ax=None, fmt='o', color='k'):
	
	conditions = np.array(conditions)
	rt = np.array(rt)
	corrects = np.array(corrects, dtype=bool)
	subj_idx = np.array(subj_idx)
	
	
	# mean:
	accuracy = np.array([np.array([np.mean(corrects[(conditions==c) & (subj_idx==s)]) for c in np.unique(conditions)]) for s in np.unique(subj_idx)]).mean(axis=0)
	break_points_correct = np.array([np.array([np.percentile(rt[(corrects) & (conditions==c) & (subj_idx==s)], quantiles) for c in np.unique(conditions)]) for s in np.unique(subj_idx)]).mean(axis=0)
	break_points_error = np.array([np.array([np.percentile(rt[(-corrects) & (conditions==c) & (subj_idx==s)], quantiles) for c in np.unique(conditions)]) for s in np.unique(subj_idx)]).mean(axis=0)
	break_points_error = break_points_error[::-1,:]
	
	# sem:
	accuracy_sem = sp.stats.sem(np.array([np.array([np.mean(corrects[(conditions==c) & (subj_idx==s)]) for c in np.unique(conditions)]) for s in np.unique(subj_idx)]), axis=0)
	break_points_correct_sem = sp.stats.sem(np.array([np.array([np.percentile(rt[(corrects) & (conditions==c) & (subj_idx==s)], quantiles) for c in np.unique(conditions)]) for s in np.unique(subj_idx)]), axis=0)
	break_points_error_sem = sp.stats.sem(np.array([np.array([np.percentile(rt[(-corrects) & (conditions==c) & (subj_idx==s)], quantiles) for c in np.unique(conditions)]) for s in np.unique(subj_idx)]), axis=0)
	break_points_error_sem = break_points_error_sem[::-1,:]
	
	if not ax:
		fig = plt.figure(figsize=(3,3))
		ax = fig.add_subplot(111)
	
	for i in range(len(quantiles)):
		x = np.concatenate((1-accuracy, accuracy[::-1]))
		y = np.concatenate((break_points_error[:,i], break_points_correct[:,i][::-1]))
		ax.plot(x, y, lw=0.5, color=color)
	for c in range(len(np.unique(conditions))):
		ax.errorbar(np.repeat(accuracy[c], len(quantiles)), break_points_correct[c], xerr=np.repeat(accuracy_sem[c], len(quantiles)), yerr=break_points_correct_sem[c], fmt=fmt, color=color, ms=4, markeredgecolor='w', markeredgewidth=0.25, capsize=0, elinewidth=0.5)
		ax.errorbar(np.repeat(1-accuracy[c], len(quantiles)), break_points_error[c], xerr=np.repeat(accuracy_sem[c], len(quantiles)), yerr=break_points_correct_sem[c], fmt=fmt, color=color, ms=4, markeredgecolor='w', markeredgewidth=0.25, capsize=0, elinewidth=0.5)
	ax.set_ylabel('RT (s)')
	ax.set_xlabel('Response proportion')
	plt.ylim(0.5,2)
	sns.despine(offset=10, trim=True)
	plt.tight_layout()
	
	return ax


def correlation_plot(X, Y, ax=False, dots=True, line=False, stat='pearson', rasterized=False):
	
	if not ax:
		fig = plt.figure(figsize=(3,3))
		ax = fig.add_subplot(111)
	slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
	
	if stat == 'spearmanr':
		r_value, p_value = stats.spearmanr(X,Y)
	
	(m,b) = sp.polyfit(X,Y,1)
	if dots:
		ax.plot(X, Y, 'o', color='k', marker='o', markeredgecolor='w', markeredgewidth=0.5, rasterized=rasterized) #s=20, zorder=2, linewidths=2)
	x_line = np.linspace(ax.axis()[0], ax.axis()[1], 100)
	regression_line = sp.polyval([m,b],x_line)
	if line:
		ax.plot(x_line, regression_line, color='k', zorder=3)
		# ax.text(ax.axis()[0] + ((ax.axis()[1] - ax.axis()[0]) / 10.0), ax.axis()[3] - ((ax.axis()[3]-ax.axis()[2]) / 5.0), 'r = ' + str(round(r_value, 3)) + '\np = ' + str(round(p_value, 4)), size=6, color='k')
	ax.set_title('r = ' + str(round(r_value, 3)) + '  ;  p = ' + str(round(p_value, 4)), size=6)
	sns.despine(offset=10, trim=True)
	plt.tight_layout()

	return ax

def correlation_plot2(X, Y, labelX, labelY, xlim, ylim):

	d = {
	labelX : pd.Series(X),
	labelY : pd.Series(Y),
	}
	data = pd.DataFrame(d)

	color = sns.color_palette()[2]
	g = sns.jointplot(labelX, labelY, data=data, xlim=xlim, ylim=ylim, kind="reg", color=color, size=7)

	return g

def sdt_barplot(hit, fa, miss, cr, p1, p2, type_plot=1, values=True):

	"""
	Type_plot = 1: SDT categories
	Type_plot = 2: Yes vs No, Correct vs Incorrect
	"""

	hit_mean = np.mean(hit)
	fa_mean = np.mean(fa)
	miss_mean = np.mean(miss)
	cr_mean = np.mean(cr)

	hit_sem = stats.sem(hit)
	fa_sem = stats.sem(fa)
	miss_sem = stats.sem(miss)
	cr_sem = stats.sem(cr)

	yes_mean = np.mean(np.concatenate((hit, fa)))
	no_mean = np.mean(np.concatenate((miss, cr)))
	correct_mean = np.mean(np.concatenate((hit, cr)))
	incorrect_mean = np.mean(np.concatenate((fa, miss)))

	yes_sem = stats.sem(np.concatenate((hit, fa)))
	no_sem = stats.sem(np.concatenate((miss, cr)))
	correct_sem = stats.sem(np.concatenate((hit, cr)))
	incorrect_sem = stats.sem(np.concatenate((fa, miss)))

	y_axis_swap = False
	if hit_mean < 0:
		if fa_mean < 0:
			if miss_mean < 0:
				if cr_mean < 0:
					y_axis_swap = True

	if type_plot == 1:
		MEANS = (hit_mean, miss_mean, fa_mean, cr_mean)
		SEMS = (hit_sem, miss_sem, fa_sem, cr_sem)
	if type_plot == 2:
		MEANS = (yes_mean, no_mean, correct_mean, incorrect_mean)
		SEMS = (yes_sem, no_sem, correct_sem, incorrect_sem)	
	
	if values == False:
		sig1 = 'n.s.'
		if p1 <= 0.05:
			sig1 = '*'
		if p1 <= 0.01:
			sig1 = '**'
		if p1 <= 0.001:
			sig1 = '***'
		
		sig2 = 'n.s.'
		if p2 <= 0.05:
			sig2 = '*'
		if p2 <= 0.01:
			sig2 = '**'
		if p2 <= 0.001:
			sig2 = '***'
	else:
		sig1 = round(p1,5)
		sig2 = round(p2,5)

	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

	N = 4
	ind = np.linspace(0,2,4)  # the x locations for the groups
	bar_width = 0.30	   # the width of the bars
	spacing = [0.30, 0, 0, -0.30]

	# FIGURE 1
	fig = plt.figure(figsize=(4,3))
	ax = fig.add_subplot(111)
	if type_plot == 1:
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b','r','b'][i], alpha = [1,.5,.5,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('H', 'M','FA', 'CR') )
	if type_plot == 2:
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b','k','k'][i], alpha = [1,1,.5,.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('Yes', 'No','Corr.', 'Incorr.') )
	ax.set_xticks( (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width) )
	ax.yaxis.set_major_locator(MultipleLocator(0.5))
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='y', which='major', labelsize=10)
	maxvalue = max( np.vstack(MEANS) + np.vstack(SEMS) )
	minvalue = min( np.vstack(MEANS) - np.vstack(SEMS) )
	if y_axis_swap == True:
		ax.set_ylim(ymin=minvalue-(minvalue/8.0), ymax=0)
	if y_axis_swap == False:
		ax.set_ylim(ymin=0, ymax=maxvalue+(maxvalue/4.0))
	left = 0.2
	top = 0.915
	bottom = 0.2
	plt.subplots_adjust(top=top, bottom=bottom, left=left)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)

	# STATS:

	if y_axis_swap == False:
		X = (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width)
		if values == True:
			label_diff(ax, 0,1,sig1,X,MEANS, SEMS, values = True)
			label_diff(ax, 2,3,sig2,X,MEANS, SEMS, values = True)
		if values == False:
			label_diff(ax, 0,1,sig1,X,MEANS, SEMS)
			label_diff(ax, 2,3,sig2,X,MEANS, SEMS)

	return(fig)

def sdt_barplot_bpd(subject, hit, fa, miss, cr, p1, p2, type_plot = 1, values = False):

	# Type_plot = 1: SDT categories
	# Type_plot = 2: Yes vs No, Correct vs Incorrect

	# hit = -HIT
	# fa = -FA
	# miss = -MISS
	# cr = -CR
	# p1 = 0.05
	# p2 = 0.05

	hit_mean = sp.mean(hit)
	fa_mean = sp.mean(fa)
	miss_mean = sp.mean(miss)
	cr_mean = sp.mean(cr)

	hit_sem = stats.sem(hit)
	fa_sem = stats.sem(fa)
	miss_sem = stats.sem(miss)
	cr_sem = stats.sem(cr)

	yes_mean = sp.mean(np.concatenate((hit, fa), axis=0))
	no_mean = sp.mean(np.concatenate((miss, cr), axis=0))
	correct_mean = sp.mean(np.concatenate((hit, cr), axis=0))
	incorrect_mean = sp.mean(np.concatenate((fa, miss), axis=0))

	yes_sem = stats.sem(np.concatenate((hit, fa), axis=0))
	no_sem = stats.sem(np.concatenate((miss, cr), axis=0))
	correct_sem = stats.sem(np.concatenate((hit, cr), axis=0))
	incorrect_sem = stats.sem(np.concatenate((fa, miss), axis=0))

	y_axis_swap = False
	if hit_mean < 0:
		if fa_mean < 0:
			if miss_mean < 0:
				if cr_mean < 0:
					y_axis_swap = True

	if type_plot == 1:
		MEANS = (hit_mean, miss_mean, fa_mean, cr_mean)
		SEMS = (hit_sem, miss_sem, fa_sem, cr_sem)
	if type_plot == 2:
		MEANS = (yes_mean, no_mean, correct_mean, incorrect_mean)
		SEMS = (yes_sem, no_sem, correct_sem, incorrect_sem)	
	
	if values == False:
		sig1 = 'n.s.'
		if p1 <= 0.05:
			sig1 = '*'
		if p1 <= 0.01:
			sig1 = '**'
		if p1 <= 0.001:
			sig1 = '***'
		
		sig2 = 'n.s.'
		if p2 <= 0.05:
			sig2 = '*'
		if p2 <= 0.01:
			sig2 = '**'
		if p2 <= 0.001:
			sig2 = '***'
	else:
		sig1 = round(p1,5)
		sig2 = round(p2,5)

	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

	N = 4
	ind = np.linspace(0,2,4)  # the x locations for the groups
	bar_width = 0.30	   # the width of the bars
	spacing = [0.30, 0, 0, -0.30]

	# FIGURE 1
	fig = plt.figure(figsize=(4,3))
	ax = fig.add_subplot(111)
	if type_plot == 1:
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b','r','b'][i], alpha = [1,.5,.5,1][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('H', 'M','FA', 'CR') )
	if type_plot == 2:
		for i in range(N):
			ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['r','b','k','k'][i], alpha = [1,1,.5,.5][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
		simpleaxis(ax)
		spine_shift(ax)
		ax.set_xticklabels( ('Yes', 'No','Corr.', 'Incorr.') )
	ax.set_xticks( (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width) )
	# ax.yaxis.set_major_locator(MultipleLocator(0.25))
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='y', which='major', labelsize=10)
	maxvalue = max( np.vstack(MEANS) + np.vstack(SEMS) )
	minvalue = min( np.vstack(MEANS) - np.vstack(SEMS) )
	diff = maxvalue - minvalue
	ax.set_ylim(ymin=minvalue-(diff/16.0), ymax=maxvalue+(diff/4.0))
	left = 0.20
	top = 0.915
	bottom = 0.20
	plt.subplots_adjust(top=top, bottom=bottom, left=left)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)

	# STATS:

	if y_axis_swap == False:
		X = (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width)
		if values == True:
			label_diff(ax, 0,1,sig1,X,MEANS, SEMS, values = True)
			label_diff(ax, 2,3,sig2,X,MEANS, SEMS, values = True)
		if values == False:
			label_diff(ax, 0,1,sig1,X,MEANS, SEMS)
			label_diff(ax, 2,3,sig2,X,MEANS, SEMS)

	return(fig)
def GLM_betas_barplot(subject, beta1, beta2, beta3, beta4, p1, p2):

	import numpy as np
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	from matplotlib.ticker import MultipleLocator
	
	def simpleaxis(ax):
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.get_xaxis().tick_bottom()
		ax.get_yaxis().tick_left()

	def spine_shift(ax, shift = 10):
		for loc, spine in ax.spines.iteritems():
			if loc in ['left','bottom']:
				spine.set_position(('outward', shift)) # outward by 10 points
			elif loc in ['right','top']:
				spine.set_color('none') # don't draw spine
			else:
				raise ValueError('unknown spine location: %s'%loc)

	def label_diff(i,j,text,X,Y,Z, values = False):

		# i = 2
		# j = 3
		# text = '***'
		# X = (ind[0]+width, ind[1], ind[2], ind[3]-width)
		# MEANS = MEANS
		# SEMS = SEMS

		middle_x = (X[i]+X[j])/2
		max_value = max(MEANS[i]+SEMS[i], MEANS[j]+SEMS[j])
		min_value = min(MEANS[i]-SEMS[i], MEANS[j]-SEMS[j])
		dx = abs(X[i]-X[j])

		props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':10,'shrinkB':10,'lw':2}
		# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
		# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
		ax.annotate('', xy=(X[i],max_value), xytext=(X[j],max_value), arrowprops=props)

		if values == False:
			if text == 'n.s.':
				kwargs = {'zorder':10, 'size':16, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.0/10))), **kwargs)
			if text != 'n.s.':
				kwargs = {'zorder':10, 'size':24, 'ha':'center'}
				ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(0.60/10))), **kwargs)
		if values == True:
			kwargs = {'zorder':10, 'size':12, 'ha':'center'}
			ax.annotate('p = ' + str(text), xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.15/10))), **kwargs)


	beta1_mean = sp.mean(beta1)
	beta2_mean = sp.mean(beta2)
	beta3_mean = sp.mean(beta3)
	beta4_mean = sp.mean(beta4)
	# beta5_mean = sp.mean(beta5)

	beta1_sem = stats.sem(beta1)
	beta2_sem = stats.sem(beta2)
	beta3_sem = stats.sem(beta3)
	beta4_sem = stats.sem(beta4)
	# beta5_sem = stats.sem(beta5)

	MEANS = (beta1_mean, beta2_mean, beta3_mean, beta4_mean)
	SEMS = (beta1_sem, beta2_sem, beta3_sem, beta4_sem)

	sig1 = 'n.s.'
	if p1 <= 0.05:
		sig1 = '*'
	if p1 <= 0.01:
		sig1 = '**'
	if p1 <= 0.001:
		sig1 = '***'

	sig2 = 'n.s.'
	if p2 <= 0.05:
		sig2 = '*'
	if p2 <= 0.01:
		sig2 = '**'
	if p2 <= 0.001:
		sig2 = '***'
	
	# sig3 = 'n.s.'
	# if p3 <= 0.05:
	#	 sig3 = '*'
	# if p3 <= 0.01:
	#	 sig3 = '**'
	# if p3 <= 0.001:
	#	 sig3 = '***'
	#	 
	# sig4 = 'n.s.'
	# if p4 <= 0.05:
	#	 sig4 = '*'
	# if p4 <= 0.01:
	#	 sig4 = '**'
	# if p4 <= 0.001:
	#	 sig4 = '***'
	#	 
	# sig5 = 'n.s.'
	# if p5 <= 0.05:
	#	 sig5 = '*'
	# if p5 <= 0.01:
	#	 sig5 = '**'
	# if p5 <= 0.001:
	#	 sig5 = '***'
	#	 
	# sig6 = 'n.s.'
	# if p6 <= 0.05:
	#	 sig6 = '*'
	# if p6 <= 0.01:
	#	 sig6 = '**'
	# if p6 <= 0.001:
	#	 sig6 = '***'

	my_dict = {'edgecolor' : 'k', 'ecolor': 'k', 'linewidth': 0, 'capsize': 0, 'align': 'center'}

	# N = 5
	# ind = np.linspace(0,2.6667,5)  # the x locations for the groups
	# bar_width = 0.30	   # the width of the bars
	# spacing = [0.30, 0, 0, -0.30, -.30]

	N = 4
	ind = np.linspace(0,2,4)  # the x locations for the groups
	bar_width = 0.30	   # the width of the bars
	spacing = [0.30, 0, 0, -0.30]

	# FIGURE 1
	fig = plt.figure(figsize=(4,3))
	ax = fig.add_subplot(111)
	for i in range(N):
		ax.bar(ind[i]+spacing[i], MEANS[i], width = bar_width, yerr = SEMS[i], color = ['k','k','k','k'][i], alpha = [0.80, 0.80, 0.80, 0.80][i], edgecolor = 'k', ecolor = 'k', linewidth = 0, capsize = 0, align = 'center')
	simpleaxis(ax)
	spine_shift(ax)
	ax.set_xticklabels( ('Onset', 'Choice','Down', 'Up') )
	ax.set_xticks( (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width) )
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='y', which='major', labelsize=10)
	maxvalue = max( np.vstack(MEANS) + np.vstack(SEMS) )
	minvalue = min( np.vstack(MEANS) - np.vstack(SEMS) )
	diffvalue = maxvalue - minvalue
	ax.set_ylim(ymin=minvalue-(diffvalue/20.0), ymax=maxvalue+(diffvalue/4.0))
	ax.yaxis.set_major_locator(MultipleLocator(1.0))
	left = 0.2
	top = 0.915
	bottom = 0.2
	plt.subplots_adjust(bottom=bottom, top=top, left=left)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)
	# 
	# if p1 < 0.05:
	#	 ax.text(ind[0]+spacing[0],0,sig1, size=24)
	# if p2 < 0.05:
	#	 ax.text(ind[1]+spacing[1],0,sig2, size=24)
	# if p3 < 0.05:
	#	 ax.text(ind[2]+spacing[2],0,sig3, size=24)
	# if p4 < 0.05:
	#	 ax.text(ind[3]+spacing[3],0,sig4, size=24)

	# STATS:

	X = (ind[0]+bar_width, ind[1], ind[2], ind[3]-bar_width)

	label_diff(0,1,sig1,X,MEANS,SEMS)
	label_diff(2,3,sig2,X,MEANS,SEMS)

	return(fig)
def sdt_barplot2(groups, p_values=None, type_plot=0):

	"""

	group order: (1) hit, (2) miss, (3) fa, (4) cr
	type_plot == 1 --> seperate SDT
	type_plot == 2 --> answer & correct


	"""

	import numpy as np
	import scipy as sp
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	from matplotlib.ticker import MultipleLocator

	MEANS = np.zeros(len(groups))
	SEMS = np.zeros(len(groups))
	for i in range(len(groups)):
		MEANS[i] = np.mean(groups[i])
		SEMS[i] = stats.sem(groups[i])

	if p_values!=None:
		P_VALUES = np.zeros(len(p_values))
		for i in range(len(p_values)):
			P_VALUES[i] = round(p_values[i],4)

	# shell()

	####################################
	# PLOTTING VARIABLES: #
	####################################

	group_means = [[MEANS[0], MEANS[1]], [MEANS[2], MEANS[3]]]
	group_sems = [[SEMS[0], SEMS[1]], [SEMS[2], SEMS[3]]]

	if type_plot == 0:
		colors = [['red', 'red'], ['blue', 'blue']]
	if type_plot == 1:
		colors = [['red', 'blue'], ['0.75', '0.75']]

	group_labels = ['H', 'M', 'FA', 'CR']
	location_bars = (0.3, 0.7, 1.3, 1.7)
	location_p = (0.5, 1.5)

	N = len(group_means)
	ind = np.arange(N)  # the x locations for the groups
	margin = 0.1
	width = (1.-2.*margin)/N
	my_dict = {'edgecolor':'k', 'ecolor':'k', 'linewidth':0, 'capsize':0}

	max_value = max(MEANS+SEMS)
	min_value = max(MEANS-SEMS)
	if min_value>0:
		min_value = 0
	
	####################################
	# FIGURE: #
	####################################

	ax = plt.gca()
	for num, vals in enumerate(group_means):
		xdata = ind+margin+(num*width)
		rects = ax.bar(xdata, vals, width, yerr=group_sems[num], color=colors[num], **my_dict)
	if p_values!=None:
		for i in range(len(P_VALUES)):
			ax.text( location_p[i], max_value*1.05, '%d'%P_VALUES[i], ha='center', va='bottom', size=8)
	ax.set_ylim(ymin=min_value*1.10, ymax=max_value*1.10)
	simpleaxis(ax)
	spine_shift(ax)
	ax.set_xticks(location_bars)
	ax.set_xticklabels(group_labels)
	ax.tick_params(axis='x', which='major', labelsize=6)
	ax.tick_params(axis='y', which='major', labelsize=6)
	plt.gca().spines["bottom"].set_linewidth(.5)
	plt.gca().spines["left"].set_linewidth(.5)

	return(rects)

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()

def spine_shift(ax, shift=10):
	for loc, spine in ax.spines.iteritems():
		if loc in ['left','bottom']:
			spine.set_position(('outward', shift)) # outward by 10 points
		elif loc in ['right','top']:
			spine.set_color('none') # don't draw spine
		else:
			raise ValueError('unknown spine location: %s'%loc)

def label_diff(ax,i,j,text,X,Y,Z, values = False):

	middle_x = (X[i]+X[j])/2
	max_value = max(Y[i]+Z[i], Y[j]+Z[j])
	min_value = min(Y[i]-Z[i], Y[j]-Z[j])
	dx = abs(X[i]-X[j])

	props = {'connectionstyle':'bar','arrowstyle':'-','shrinkA':10,'shrinkB':10,'lw':2}
	# ax.annotate(text, xy=(X[i],y+0.4), zorder=10) 
	# ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
	ax.annotate('', xy=(X[i],max_value), xytext=(X[j],max_value), arrowprops=props)

	if values == False:
		if text == 'n.s.':
			kwargs = {'zorder':10, 'size':16, 'ha':'center'}
			ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.0/10))), **kwargs)
		if text != 'n.s.':
			kwargs = {'zorder':10, 'size':24, 'ha':'center'}
			ax.annotate(text, xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(0.60/10))), **kwargs)
	if values == True:
		kwargs = {'zorder':10, 'size':12, 'ha':'center'}
		ax.annotate('p = ' + str(text), xy=(middle_x,max_value + ((plt.axis()[3] - plt.axis()[2])*(1.15/10))), **kwargs)


class behavior(object):
	def __init__(self, data):
		
		"""
		data should be in pandas dataframe format, columns for:
		"""
		self.data = data
		self.subjects = np.unique(data.subj_idx)
		self.nr_subjects = len(self.subjects)
		
		self.data['hit'] = ((self.data['choice'] == 1) & (self.data['correct'] == 1))
		self.data['fa'] = ((self.data['choice'] == 1) & (self.data['correct'] == 0))
		
	def choice_fractions(self, split_by=False, split_target=0):
		
		if split_by:
			split_ind = eval('self.data.' + split_by) == split_target
		else:
			split_ind = np.ones(len(self.data), dtype=bool)
		
		# split_ind = (self.data.pupil_high == 1)
		
		c_correct = np.array([sum((self.data.correct==1) & (self.data.subj_idx==i) & split_ind) / float(sum((self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)])
		c_error = np.array([sum((self.data.correct==0) & (self.data.subj_idx==i) & split_ind) / float(sum((self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)])
		
		c0 = np.array([sum((self.data.choice==0) & (self.data.subj_idx==i) & split_ind) / float(sum((self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)])
		c1 = np.array([sum((self.data.choice==1) & (self.data.subj_idx==i) & split_ind) / float(sum((self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)])
		
		c0_correct = np.array([ sum( (self.data.choice==0) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind ) / float(sum( (self.data.subj_idx==i) & split_ind) ) for i in range(self.nr_subjects)])
		c0_error = np.array([ sum( (self.data.choice==0) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind ) / float(sum( (self.data.subj_idx==i) & split_ind) ) for i in range(self.nr_subjects)])
		c1_correct = np.array([ sum( (self.data.choice==1) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind ) / float(sum( (self.data.subj_idx==i) & split_ind) ) for i in range(self.nr_subjects)])
		c1_error = np.array([ sum( (self.data.choice==1) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind ) / float(sum( (self.data.subj_idx==i) & split_ind) ) for i in range(self.nr_subjects)])
		
		# c0_c = c0_correct / (c0_correct + c0_error)
		# c0_e = c0_error / (c0_correct + c0_error)
		#
		# c1_c = c1_correct / (c1_correct + c1_error)
		# c1_e = c1_error / (c1_correct + c1_error)
		
		# c0_correct = np.array([ sum( (self.data.choice==0) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind) / float(sum((self.data.stimulus==0) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)])
		# c0_error = np.array([ sum( (self.data.choice==0) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind) / float(sum((self.data.stimulus==0) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)])
		# c1_correct = np.array([ sum( (self.data.choice==1) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind) / float(sum((self.data.stimulus==1) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)])
		# c1_error = np.array([ sum( (self.data.choice==1) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind) / float(sum((self.data.stimulus==1) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)])
		
		return (c_correct, c_error, c0, c1, c0_correct, c0_error, c1_correct, c1_error)
	
	def behavior_measures(self, split_by=False, split_target=0):
		
		if split_by:
			split_ind = np.array(eval('self.data.' + split_by) == split_target, dtype=bool)
		else:
			split_ind = np.ones(len(self.data), dtype=bool)
		
		rt = np.array([np.median(self.data.rt[(self.data.subj_idx==s) & split_ind]) for i, s in enumerate(self.subjects)])
		acc = np.array([np.mean(self.data.correct[(self.data.subj_idx==s) & split_ind]) for i, s in enumerate(self.subjects)])
		d = np.array([SDT_measures(self.data.stimulus[(self.data.subj_idx==s) & split_ind], self.data.hit[(self.data.subj_idx==s) & split_ind], self.data.fa[(self.data.subj_idx==s) & split_ind])[0] for i, s in enumerate(self.subjects)])
		c = np.array([SDT_measures(self.data.stimulus[(self.data.subj_idx==s) & split_ind], self.data.hit[(self.data.subj_idx==s) & split_ind], self.data.fa[(self.data.subj_idx==s) & split_ind])[1] for i, s in enumerate(self.subjects)])
		
		return rt, acc, d, c 
		
	def rt_kernel_densities(self, x_grid, bandwidth=0.05, split_by=False, split_target=0):
		
		if split_by:
			split_ind = np.array(eval('self.data.' + split_by) == split_target, dtype=bool)
		else:
			split_ind = np.ones(len(self.data), dtype=bool)
		
		x_grid = np.linspace(x_grid[0], x_grid[1], x_grid[2])
		
		c0_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / sum(kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth)) * (sum((self.data.choice==0) & (self.data.subj_idx==i) & split_ind) / sum((self.data.subj_idx==i) & split_ind)) * 100.0 for i in range(self.nr_subjects)]
		c1_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / sum(kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth)) * (sum((self.data.choice==1) & (self.data.subj_idx==i) & split_ind) / sum((self.data.subj_idx==i) & split_ind)) * 100.0 for i in range(self.nr_subjects)]
		c_correct_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.correct==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (1.0 / sum((self.data.correct==1) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		c_error_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.correct==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (1.0 / sum((self.data.correct==0) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		c0_correct_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (1.0 / sum((self.data.choice==0) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		c0_error_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (1.0 / sum((self.data.choice==0) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		c1_correct_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (1.0 / sum((self.data.choice==1) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		c1_error_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (1.0 / sum((self.data.choice==1) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]

		# c0_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (sum((self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		# c1_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (sum((self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		# c_correct_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.correct==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (sum((self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		# c_error_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.correct==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (sum((self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		# c0_correct_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (sum((self.data.stimulus==0) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		# c0_error_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (sum((self.data.stimulus==1) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		# c1_correct_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (sum((self.data.stimulus==0) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		# c1_error_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) / (sum((self.data.stimulus==1) & (self.data.subj_idx==i) & split_ind)) for i in range(self.nr_subjects)]
		
		# c0_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) for i in range(self.nr_subjects)]
		# c1_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) for i in range(self.nr_subjects)]
		# c_correct_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.correct==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) for i in range(self.nr_subjects)]
		# c_error_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.correct==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) for i in range(self.nr_subjects)]
		# c0_correct_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) for i in range(self.nr_subjects)]
		# c0_error_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==0) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) for i in range(self.nr_subjects)]
		# c1_correct_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) for i in range(self.nr_subjects)]
		# c1_error_pdf = [kde_sklearn(np.array(self.data.rt[(self.data.choice==1) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind]), x_grid, bandwidth=bandwidth) for i in range(self.nr_subjects)]
		
		
		return (c0_pdf, c1_pdf, c_correct_pdf, c_error_pdf, c0_correct_pdf, c0_error_pdf, c1_correct_pdf, c1_error_pdf)

	def rt_mean_var(self, split_by=False, split_target=0):
		
		if split_by:
			split_ind = np.array(eval('self.data.' + split_by) == split_target, dtype=bool)
		else:
			split_ind = np.ones(len(self.data), dtype=bool)
		
		rt_correct = [np.mean(np.array(self.data.rt[(self.data.correct==1) & (self.data.subj_idx==i) & split_ind])) for i in range(self.nr_subjects)]
		rt_error = [np.mean(np.array(self.data.rt[(self.data.correct==0) & (self.data.subj_idx==i) & split_ind])) for i in range(self.nr_subjects)]
		rt_correct_std = [np.std(np.array(self.data.rt[(self.data.correct==1) & (self.data.subj_idx==i) & split_ind])) for i in range(self.nr_subjects)]
		rt_error_std = [np.std(np.array(self.data.rt[(self.data.correct==0) & (self.data.subj_idx==i) & split_ind])) for i in range(self.nr_subjects)]
		
		return (rt_correct, rt_error, rt_correct_std, rt_error_std)
		
	def pupil_bars(self, pupil='pupil', split_by=False, split_target=False):
		
		if split_by:
			split_ind = np.array(eval('self.data.' + split_by) == split_target, dtype=bool)
		else:
			split_ind = np.ones(len(self.data), dtype=bool)
		p = np.array(eval('self.data.' + pupil))
		
		correct = [np.mean(np.array(p[np.array((self.data.correct==1) & (self.data.subj_idx==i) & split_ind)])) for i in range(self.nr_subjects)]
		error = [np.mean(np.array(p[np.array((self.data.correct==0) & (self.data.subj_idx==i) & split_ind)])) for i in range(self.nr_subjects)]
		c0 = [np.mean(np.array(p[np.array((self.data.choice==0) & (self.data.subj_idx==i) & split_ind)])) for i in range(self.nr_subjects)]
		c1 = [np.mean(np.array(p[np.array((self.data.choice==1) & (self.data.subj_idx==i) & split_ind)])) for i in range(self.nr_subjects)]
		c0_correct = [np.mean(np.array(p[np.array((self.data.choice==0) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind)])) for i in range(self.nr_subjects)]
		c0_error = [np.mean(np.array(p[np.array((self.data.choice==0) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind)])) for i in range(self.nr_subjects)]
		c1_correct = [np.mean(np.array(p[np.array((self.data.choice==1) & (self.data.correct==1) & (self.data.subj_idx==i) & split_ind)])) for i in range(self.nr_subjects)]
		c1_error = [np.mean(np.array(p[np.array((self.data.choice==1) & (self.data.correct==0) & (self.data.subj_idx==i) & split_ind)])) for i in range(self.nr_subjects)]
		
		return (correct, error, c0, c1, c0_correct, c0_error, c1_correct, c1_error)
		
	
	
	
	
	
	
	
	

		
	
	
		
		
		
	
	
	
		
	
	
	
		
		
		
		
		
		
		
		
		
		
		
		
		
