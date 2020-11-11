# Copyright (c) 2020, The University of Texas at Austin 
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

from hippylib import *
import dolfin as dl
import numpy as np


class PriorPreconditionedProjector:
	"""
	Prior preconditioned projector operator
	:math:`UU^TC^{-1}`
	"""
	def __init__(self,U,Cinv,my_init_vector):
		"""
		Constructor
			- :code:`U` - Array object must have :code:`dot_v` method implemented
			- :code:`Cinv` - Covariance operator must have :code:`mult` method implemented
			- :code:`my_init_vector` - lambda function for initialization of vectors
		"""
		self.U = U
		self.Cinv = Cinv
		self.my_init_vector = my_init_vector

		self.Cinvx = dl.Vector()
		self.my_init_vector(self.Cinvx,0)

		pass


	def init_vector(self,x,dim):
		"""
        Initialize :code:`x` to be compatible with the range (:code:`dim=0`) or domain (:code:`dim=1`) of :code:`PriorPreconditionedProjector`.
        """
		self.my_init_vector(x,dim)


	def mult(self,x,y):
		"""
        Compute :math:`y = UU^TC^{-1} x`
        """
		self.Cinv.mult(x,self.Cinvx)
		UtCinvx = self.U.dot_v(self.Cinvx)
		y.zero()
		self.U.reduce(y,UtCinvx)






