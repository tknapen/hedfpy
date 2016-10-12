#!/usr/bin/env python
# encoding: utf-8
"""
CommandLineOperator.py

Created by Tomas Knapen on 2010-09-23.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys, subprocess, shutil
import tempfile, logging
import re

import scipy as sp
import numpy as np
import matplotlib.pylab as pl

from Operator import *
from log import *

from IPython import embed as shell

### Execute program in shell:
def ExecCommandLine(cmdline):
	tmpf = tempfile.TemporaryFile()
	try:
		retcode = subprocess.call( cmdline, shell=True, bufsize=0, stdout = tmpf, stderr = tmpf)
	finally:
		tmpf.close()
		if retcode > 0:
			raise ValueError( 'Process: '+cmdline+' returned error code: '+str(retcode) )
	return retcode

class CommandLineOperator( Operator ):
	def __init__(self, input_object, cmd, **kwargs):
		"""
		CommandLineOperator can take a Nii file as input but will use only the variable input_file_name
		"""
		super(CommandLineOperator, self).__init__(input_object = input_object, **kwargs)

		if self.input_object.__class__.__name__ == 'NiftiImage':
			self.input_file_name = self.input_object.filename
			self.logger.info(self.__repr__() + ' initialized with ' + os.path.split(self.input_file_name)[-1])
		elif self.input_object.__class__.__name__ == 'str':
			self.input_file_name = self.input_object
			self.logger.info(self.__repr__() + ' initialized with file ' + os.path.split(self.input_file_name)[-1])
			if not os.path.isfile(self.input_file_name):
				self.logger.warning('input_file_name is not a file at initialization')
		elif self.input_object.__class__.__name__ == 'list':
			self.inputList = self.input_object
			self.logger.info(self.__repr__() + ' initialized with files ' + str(self.inputList))
		self.cmd = cmd

	def configure(self):
		"""
		placeholder for configure
		to be filled in by subclasses
		"""
		self.runcmd = self.cmd + ' ' + self.input_file_name

	def execute(self, wait = True):
		"""
		placeholder for execute
		to be filled in by subclasses
		"""
		self.logger.debug(self.__repr__() + 'executing command \n' + self.runcmd)
		# print self.runcmd
		# subprocess.call( self.runcmd, shell=True, bufsize=0, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
		if not wait:
			self.runcmd + '&'
		ExecCommandLine(self.runcmd)

class EDF2ASCOperator( CommandLineOperator ):
	"""
	EDF2ASCOperator provides the tools to convert an edf file to a pair of output files,
	one containing the gaze samples (.gaz) and another containing all the messages/events (.msg).
	It requires edf2asc command-line executable, which is assumed to be on the $PATH.
	Missing values are imputed as 0.0001, time is represented as a floating point number for 2000Hz sampling.
	"""
	def __init__(self, input_object, **kwargs):
		super(EDF2ASCOperator, self).__init__(input_object = input_object, cmd = 'edf2asc', **kwargs)

	def configure(self, gazeOutputFileName = None, messageOutputFileName = None, settings = ' -t -ftime -y -z -v '):
		"""
		configure creates commands self.gazcmd and self.msgcmd which,
		when executed on the command line, convert the edf 2 an asc file,
		taking either the sample data or the event data, respectively.
		it also creates self.runcmd which can be used to run both above
		commands in succession, and which will be executed when calling
		'execute' (as per CommandLineOperator behavior)
		"""
		if gazeOutputFileName == None:
			self.gazeOutputFileName = os.path.splitext(self.input_file_name)[0] + '.gaz'
		else:
			self.gazeOutputFileName = gazeOutputFileName
		if messageOutputFileName == None:
			self.messageOutputFileName = os.path.splitext(self.input_file_name)[0] + '.msg'
		else:
			self.messageOutputFileName = messageOutputFileName
		standardOutputFileName = os.path.splitext(self.input_file_name)[0] + '.asc'

		self.intermediatecmd = self.cmd
		self.intermediatecmd += settings

		self.gazcmd = self.intermediatecmd + '-s -miss 0.0001 -vel "'+self.input_file_name+'"; mv ' + '"' + standardOutputFileName.replace('|', '\|') + '" "' + self.gazeOutputFileName.replace('|', '\|') + '"'
		self.msgcmd = self.intermediatecmd + '-e "'+self.input_file_name+'"; mv ' + '"' + standardOutputFileName.replace('|', '\|') + '" "' + self.messageOutputFileName.replace('|', '\|') + '"'

		self.runcmd = self.gazcmd + '; ' + self.msgcmd
