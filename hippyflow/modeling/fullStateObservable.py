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

import hippylib as hp
import dolfin as dl 
import numpy as np 

from .observable import LinearStateObservable

class StateSpaceIdentityOperator:
	"""
	This class defines an identity operator on the state space
	"""
	def __init__(self, M, use_mass_matrix = True):
		"""
		Constructor:
			:code: `M`: mass matrix of the state function space 
			:code: `use_mass_matrix`: boolean of whether mass matrix is used in the transpose or not. 
		"""
		self.M = M
		self.use_mass_matrix = use_mass_matrix

	def mpi_comm(self):
		return self.M.mpi_comm()

	def init_vector(self, v, dim):
		return self.M.init_vector(v, dim)

	def mult(self, u, y):
		y.zero()
		y.axpy(1.0, u)

	def transpmult(self, x, p):
		"""
		"""
		if self.use_mass_matrix:
			self.M.transpmult(x, p)
		else:
			p.zero()
			p.axpy(1.0, x)

