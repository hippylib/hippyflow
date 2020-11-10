# This file is part of the hIPPYflow package
#
# hIPPYflow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or any later version.
#
# hIPPYflow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# If not, see <http://www.gnu.org/licenses/>.
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu


import dolfin as dl
import numpy as np
from hippylib import *

def BiLaplacian2D(Vh_parameter,gamma = 0.1,delta = 0.1, theta0 = 2.0, theta1 = 0.5, alpha = np.pi/4,mean = None):
	anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
	anis_diff.theta0 = theta0
	anis_diff.theta1 = theta1
	anis_diff.alpha = alpha	

	return BiLaplacianPrior(Vh_parameter, gamma, delta, anis_diff,mean = mean)


def Laplacian2D(Vh_parameter,gamma = 0.1,delta = 0.1, theta0 = 2.0, theta1 = 0.5, alpha = np.pi/4,mean = None):
	anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
	anis_diff.theta0 = theta0
	anis_diff.theta1 = theta1
	anis_diff.alpha = alpha	

	return LaplacianPrior(Vh_parameter, gamma, delta, anis_diff,mean = mean)