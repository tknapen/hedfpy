#!/usr/bin/env python
# encoding: utf-8
"""
Operator.py

Created by Tomas Knapen on 2010-09-17.
Copyright (c) 2010 __MyCompanyName__. All rights reserved.
"""

import sys
import tempfile
import logging
import numpy as np

# logging...
logFormat = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%y-%m-%d_%H-%M-%S")
logging_handlers = [logging.StreamHandler(sys.stdout)]
logging_levels = [logging.DEBUG]


def addLoggingHandler(handler, loggingLevel=logging.DEBUG):
    logging_handlers.append(handler)
    logging_levels.append(loggingLevel)


def loggingLevelSetup():
    for (handler, level) in zip(logging_handlers, logging_levels):
        handler.setLevel(level)
        handler.setFormatter(logFormat)


class Operator(object):
    def __init__(self, input_object, **kwargs):
        self.input_object = input_object
        for k, v in kwargs.items():
            setattr(self, k, v)

        # setup logging for this operator.
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        loggingLevelSetup()
        for handler in logging_handlers:
            self.logger.addHandler(handler)

    def configure(self):
        """
        placeholder for configure
        to be filled in by subclasses
        """
        pass

    def execute(self):
        """
        placeholder for execute
        to be filled in by subclasses
        """
