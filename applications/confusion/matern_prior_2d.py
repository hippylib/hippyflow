
import dolfin as dl
import numpy as np
from hippylib import *

def matern_prior_2d(Vh_parameter,gamma = 0.1,delta = 0.1, theta0 = 2.0, theta1 = 0.5, alpha = np.pi/4,mean = None):
	anis_diff = dl.CompiledExpression(ExpressionModule.AnisTensor2D(), degree = 1)
	anis_diff.theta0 = theta0
	anis_diff.theta1 = theta1
	anis_diff.alpha = alpha	

	return BiLaplacianPrior(Vh_parameter, gamma, delta, anis_diff,mean = mean)
