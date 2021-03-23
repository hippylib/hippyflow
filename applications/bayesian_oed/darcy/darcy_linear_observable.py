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
import ufl

path_to_hippylib = '../../hippylib/'
import sys, os
sys.path.append( os.environ.get('HIPPYLIB_PATH',path_to_hippylib))
from hippylib import *

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
from hippyflow import LinearStateObservable

def darcy_linear_observable(mesh,sqrt_n_obs = 10,output_folder ='confusion_setup/',\
									 verbose = False,seed = 0):

	########################################################################
	#####Set up the mesh and finite element spaces#########################

	Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
	Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
	Vh = [Vh2, Vh1, Vh2]
	para_dim = Vh[PARAMETER].dim()
	if verbose:
		print( "Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()) )

	########################################################################
	#####Set up the forward problem#########################
	def u_boundary(x, on_boundary):
	    return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

	u_bdr = dl.Expression("x[1]", degree=1)
	u_bdr0 = dl.Constant(0.0)
	bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
	bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)

	f = dl.Constant(0.0)

	def pde_varf(u,m,p):
	    return dl.exp(m)*dl.inner(dl.nabla_grad(u), dl.nabla_grad(p))*dl.dx - f*p*dl.dx

	pde = PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=True)

	########################################################################
	# Construct the linear observable
	# Targets only on the bottom
	x1 = np.arange(0.1,1.,0.1)
	x2 = np.arange(0.1,1.,0.1)
	X1,X2 = np.meshgrid(x1, x2)
	x_2d = np.array([X1.flatten('F'),X2.flatten('F')])
	targets = x_2d.T
	ntargets = targets.shape[0]

	if verbose:
		print( "Number of observation points: {0}".format(ntargets) )

	B = assemblePointwiseObservation(Vh[STATE], targets)

	observable = LinearStateObservable(pde,B)

	return observable

