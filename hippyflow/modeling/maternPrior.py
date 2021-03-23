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


import dolfin as dl
import numpy as np
from hippylib import *

def BiLaplacian2D(Vh_parameter,gamma = 0.1,delta = 0.1, theta0 = 2.0, theta1 = 0.5,\
			 alpha = np.pi/4,mean = None, robin_bc = False):
	"""
	Return 2D BiLaplacian prior given function space and coefficients for Matern covariance
	"""
	anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
	anis_diff.theta0 = theta0
	anis_diff.theta1 = theta1
	anis_diff.alpha = alpha	

	return BiLaplacianPrior(Vh_parameter, gamma, delta, anis_diff,mean = mean,robin_bc = robin_bc)


def Laplacian2D(Vh_parameter,gamma = 0.1,delta = 0.1, theta0 = 2.0, theta1 = 0.5,alpha = np.pi/4,mean = None):
	"""
	Return 2D Laplacian prior given function space and coefficients for Matern covariance
	"""
	anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
	anis_diff.theta0 = theta0
	anis_diff.theta1 = theta1
	anis_diff.alpha = alpha	

	return LaplacianPrior(Vh_parameter, gamma, delta, mean = mean)