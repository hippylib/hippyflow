
# Copyright (c) 2020-2022, The University of Texas at Austin 
# & Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYflow package. For more information see
# https://github.com/hippylib/hippyflow/
#
# hIPPYflow is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import numpy as np
import os

class npToDolfinOperator:

	def __init__(self,npArray):
		"""
		"""
		assert len(npArray.shape) == 2
		self.matrix = npArray

		self.domain_help = None 
		self.range_help = None
		# assert that the dolfin mpi world size is one
		pass

	def init_vector(self,x,dim):
		"""
		"""
		if dim == 0:
			x.init(self.matrix.shape[0])
		elif dim == 1:
			x.init(self.matrix.shape[1])
		else:
			raise

	def mult(self,x,y):
		"""
		"""
		y.zero()
		y.set_local(self.matrix@x.get_local())

	def transpmult(self,x,y):
		"""
		"""
		y.zero()
		y.set_local(self.matrix.T@x.get_local())

		