
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
import warnings

class LowRankRectangularOperator:
	"""
	This class model the action of a low rank operator :math:`A = U s V^T`.
	Here :math:`s` is a diagonal matrix, and the columns of U and V for bases for 
	the input and output of some localized linear operator

	.. note:: This class only works in serial!
	"""
	def __init__(self,U,s,V,U_init_vector = None,V_init_vector = None):
		"""
		Construct the low rank rectangular operator given :code:`U`, :code:`s`, and :code:`V`.
		"""
		warnings.warn('Experimental Class! Be Wary, do not try to use in parallel!')
		self.U = U
		self.s = s
		self.V = V
		self.U_init_vector = U_init_vector
		self.V_init_vector = V_init_vector

	def init_vector(self, x, dim):
		"""
		Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of the operator.
		"""
		if dim == 0:
			assert self.U_init_vector is not None
			self.U_init_vector(x)
		elif dim == 1:
			assert self.V_init_vector is not None
			self.V_init_vector(x)
		else:
			raise ValueError('dim must be 0 or 1')


	def mult(self,x,y):
		"""
		Compute :math:`y = Ax = U s V^T x`
		"""
		Vtx = self.V.dot_v(x)
		sVtx = self.s*Vtx   #elementwise mult
		y.zero()
		self.U.reduce(y, sVtx)

	def transpmult(self,x,y):
		"""
		Compute :math:`y = Ax = V s U^T x`
		"""
		Utx = self.U.dot_v(x)
		sUtx = self.s*Utx   #elementwise mult
		y.zero()
		self.V.reduce(y, sUtx)