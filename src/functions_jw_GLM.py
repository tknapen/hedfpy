from __future__ import division

import numpy as np
import scipy as sp
import sympy
import matplotlib.pyplot as plt
import statsmodels.api as sm

import math

from IPython import embed as shell

class GeneralLinearModel(object):
	"""Design represents the design matrix of a given run"""
	def __init__(self, input_object, event_object, sample_dur=2, new_sample_dur=1):
		
		# variables:
		self.input_object = input_object
		self.event_object = event_object
		self.downsample_ratio = int(new_sample_dur / sample_dur)
		self.sample_dur = sample_dur
		self.new_sample_dur = new_sample_dur
		self.timepoints = np.linspace(0, input_object.shape[0]*self.downsample_ratio*self.sample_dur, input_object.shape[0])
		self.raw_design_matrix = []
		
	def configure(self, IRF='pupil', IRF_params=None, regressor_types=['stick'], demean=False, basis_set=False, normalize_sustained=False):
		
		self.basis_set = basis_set
		
		# resample input_object:
		if self.downsample_ratio != 1:
			self.resample_input_object()
		else:
			self.working_data_array = self.input_object
		
		# create raw regressors, and add them to self.raw_design_matrix:
		for i, reg in enumerate(self.event_object):
			if regressor_types[i] == 'stick':
				self.add_stick_regressor(np.atleast_2d(reg))
			if regressor_types[i] == 'box':
				self.add_box_regressor(np.atleast_2d(reg), normalize_sustained)
			if regressor_types[i] == 'upramp':
				self.add_upramp_regressor(np.atleast_2d(reg), normalize_sustained)
			if regressor_types[i] == 'downramp':
				self.add_downramp_regressor(np.atleast_2d(reg), normalize_sustained)
		
		# create IRF:
		if IRF == 'pupil':
			self.IRF = self.IRF_pupil(dur=IRF_params['dur'], s=IRF_params['s'], n=IRF_params['n'], tmax=IRF_params['tmax'])
		elif IRF == 'BOLD':
			self.IRF = self.HRF(dur=IRF_params['dur'])
		else:
			self.IRF = IRF
		
		# convolve raw regressors with IRF to obtain the full design matrix:
		self.convolve_with_IRF()
		
		if demean:
			self.demean()
			# self.z_score()
			# self.psc()
		
	def resample_input_object(self):
		"""resample_input_object takes a timeseries of data points and resamples them according to the ratio between sample duration and the new sample duration."""
		
		self.working_data_array = sp.signal.decimate(self.input_object, int(self.downsample_ratio))
		
	def convolve_with_IRF(self):
		"""convolve_wit_IRF convolves the designMatrix with the specified IRF (sampled according to resample_ratio)"""
		
		# print
		# print len(self.IRF)
		
		self.design_matrix = np.zeros([len(self.raw_design_matrix)*len(self.IRF), self.timepoints.shape[0]])
		i = 0
		for reg in self.raw_design_matrix:
			for IRF in self.IRF:
				self.design_matrix[i,:] = (sp.signal.fftconvolve(reg, IRF, 'full'))[:-(IRF.shape[0]-1)]
				i += 1
	
	def demean(self):
		"""demeans design matrix"""
		
		self.working_data_array = self.working_data_array - self.working_data_array.mean()
		for i in range(self.design_matrix.shape[0]):
			self.design_matrix[i,:] = (self.design_matrix[i,:] - self.design_matrix[i,:].mean())
	
	def z_score(self):
		"""z-scores design matrix"""
		
		# print 'z-scoring!'
		
		self.working_data_array = (self.working_data_array - self.working_data_array.mean()) / self.working_data_array.std()
		for i in range(self.design_matrix.shape[0]):
			self.design_matrix[i,:] = (self.design_matrix[i,:] - self.design_matrix[i,:].mean()) / self.design_matrix[i,:].std()
	
	def psc(self):
		"""percent signal chances design matrix"""
		
		self.working_data_array = ((self.working_data_array / np.median(self.working_data_array)) * 100) - 100
		for i in range(self.design_matrix.shape[0]):
			self.design_matrix[i,:] = ((self.design_matrix[i,:] / np.median(self.design_matrix[i,:])) * 100) - 100
	
	def execute(self):
		
		# print 'fitting model'
		
		self.design_matrix = np.mat(self.design_matrix).T
		
		# GLM:
		self.betas = np.array(((self.design_matrix.T * self.design_matrix).I * self.design_matrix.T) * np.mat(self.working_data_array).T).ravel()
		
		# predicted signal:
		self.predicted = np.sum(np.vstack([np.array(self.design_matrix).T[i]*b for i, b in enumerate(self.betas)]), axis=0)
		
		# residuals:
		self.residuals = self.working_data_array - self.predicted
		
	# --------------------------------------
	# A variety of regressor shapes:       -
	# --------------------------------------
	
	def add_stick_regressor(self, regressor):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressor_values = np.zeros(self.timepoints.shape[0])
		for event in regressor:
			start_time = np.floor((event[0]+event[1])/self.new_sample_dur)
			regressor_values[start_time] = event[2]
		self.raw_design_matrix.append(regressor_values)
	
	def add_box_regressor(self, regressor, normalize_sustained=False):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressor_values = np.zeros(self.timepoints.shape[0])
		for event in regressor:
			start_time = np.floor(event[0]/self.new_sample_dur)
			end_time = np.floor((event[0]+event[1])/self.new_sample_dur)
			dur = end_time - start_time
			if normalize_sustained:
				height = event[2] / float(dur)
			else:
				height = event[2]
			regressor_values[start_time:end_time] = height
		self.raw_design_matrix.append(regressor_values)
	
	def add_upramp_regressor(self, regressor, normalize_sustained=False):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressor_values = np.zeros(self.timepoints.shape[0])
		for event in regressor:
			start_time = np.floor(event[0]/self.new_sample_dur)
			end_time = np.floor((event[0]+event[1])/self.new_sample_dur)
			dur = end_time - start_time
			if normalize_sustained:
				height = np.linspace(0, (event[2]*2/float(dur)), dur)
			else:
				height = np.linspace(0, event[2]*2, dur)
			regressor_values[start_time:end_time] = height
		self.raw_design_matrix.append(regressor_values)
		
	def add_downramp_regressor(self, regressor, normalize_sustained=False):
		"""
		regressors are vectors identical to custom EV files in FSL
		"""
		regressor_values = np.zeros(self.timepoints.shape[0])
		for event in regressor:
			start_time = np.floor(event[0]/self.new_sample_dur)
			end_time = np.floor((event[0]+event[1])/self.new_sample_dur)
			dur = end_time - start_time
			if normalize_sustained:
				height = np.linspace((event[2]*2/float(dur)), 0, dur)
			else:
				height = np.linspace(event[2]*2, 0, dur)
			regressor_values[start_time:end_time] = height
		self.raw_design_matrix.append(regressor_values)
	
	# --------------------------------------
	# Impulse Response Functions (IRF)     -
	# --------------------------------------
	
	def HRF(self, dur=25, a1=6.0, a2=12.0, b1=0.9, b2=0.9, c=0.35):
		
		# parameters:
		timepoints = np.arange(0, dur, self.new_sample_dur)
		
		# sympy variable:
		t = sympy.Symbol('t')
		
		# function:
		d1 = a1 * b1
		d2 = a2 * b2
		y = ( (t/(d1))**a1 * sympy.exp(-(t-d1)/b1) - c*(t/(d2))**a2 * sympy.exp(-(t-d2)/b2) )
		
		# derivative:
		y_dt = y.diff(t)
		
		# lambdify:
		y = sympy.lambdify(t, y, "numpy")
		y_dt = sympy.lambdify(t, y_dt, "numpy")
		
		# evaluate:
		y = y(timepoints)
		y_dt = y_dt(timepoints)
		
		if self.IRF_dt:
			return [y/np.std(y), y_dt/np.std(y_dt)]
		else:
			return [y/np.std(y)]

	def IRF_pupil(self, dur=4, s=1.0/(10**26), n=10.1, tmax=.930):
		"""
		Canocial pupil impulse fucntion [ref]: 
		"""
		
		# for n in np.linspace(4,12,10):
		# 	for tmax in np.linspace(0.5, 1.3, 10):
		
		
		# parameters:
		timepoints = np.arange(0, dur, self.new_sample_dur)

		# sympy variable:
		t = sympy.Symbol('t')

		# function:
		y = ( (s) * (t**n) * (math.e**((-n*t)/tmax)) )

		# derivative:
		y_dt = y.diff(t)

		# lambdify:
		y = sympy.lambdify(t, y, "numpy")
		y_dt = sympy.lambdify(t, y_dt, "numpy")

		# evaluate and normalize:
		y = y(timepoints)
		y = y/np.std(y)
		y_dt = y_dt(timepoints)
		y_dt = y_dt/np.std(y_dt)

		# dispersion:
		y_dn = ( (s) * (timepoints**(n-0.01)) * (math.e**((-(n-0.01)*timepoints)/tmax)) )
		y_dn = y_dn / np.std(y_dn)
		y_dn = y - y_dn
		y_dn = y_dn / np.std(y_dn)
	
		# plt.plot(timepoints, y, color='k', lw=0.5, alpha=0.5)
		
		# plt.figure()
		# plt.plot(y)
		# plt.plot(y_dt)
		# plt.plot(y_dn)
		# plt.show()
		
		if self.basis_set:
			return [y, y_dt]
		else:
			return [y]