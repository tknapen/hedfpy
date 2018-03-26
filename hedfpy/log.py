#!/usr/bin/env python
# encoding: utf-8
"""
log.py

Created by Tomas Knapen on 2010-10-03.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import os, sys
from subprocess import *
import logging, logging.handlers, logging.config


# logging...
logFormat = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%y-%m-%d_%H-%M-%S")
logging_handlers = [ logging.StreamHandler( sys.stdout ) ]
logging_levels = [ logging.DEBUG ]

def addLoggingHandler( handler, loggingLevel = logging.DEBUG ):
	logging_handlers.append(handler)
	logging_levels.append(loggingLevel)

def loggingLevelSetup():
	for (handler, level) in zip(logging_handlers, logging_levels):
		handler.setLevel(level)
		handler.setFormatter(logFormat)
