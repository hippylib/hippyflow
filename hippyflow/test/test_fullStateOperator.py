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

import unittest 
import dolfin as dl
import ufl
import numpy as np


import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

def u_boundary(x, on_boundary):
	return on_boundary and ( x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

class TestFullStateDerivativeSubspace(unittest.TestCase):
	def setUp(self):
		dl.parameters["ghost_mode"] = "shared_facet"
		ndim = 2
		nx = 10
		ny = 10
		self.mesh = dl.UnitSquareMesh(nx, ny)

		self.observable = self.buildObservable(self.mesh)

		self.prior = hf.BiLaplacian2D(self.Vh[hp.PARAMETER],gamma = 0.1, delta = 1.0)


	
	def buildObservable(self,mesh):
		self.rank = dl.MPI.rank(mesh.mpi_comm())
			
		Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
		Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
		self.Vh = [Vh2, Vh1, Vh2]
		# Initialize Expressions
		f = dl.Constant(0.0)
			
		u_bdr = dl.Expression("x[1]", degree=1)
		u_bdr0 = dl.Constant(0.0)
		bc = dl.DirichletBC(self.Vh[hp.STATE], u_bdr, u_boundary)
		bc0 = dl.DirichletBC(self.Vh[hp.STATE], u_bdr0, u_boundary)
		
		def pde_varf(u,m,p):
			return ufl.exp(m)*ufl.inner(ufl.grad(u), ufl.grad(p))*ufl.dx - f*p*ufl.dx

		self.pde = hp.PDEVariationalProblem(self.Vh, pde_varf, bc, bc0, is_fwd_linear=True)

		u_trial = dl.TrialFunction(self.Vh[hp.STATE])
		u_test = dl.TestFunction(self.Vh[hp.STATE])

		M = dl.PETScMatrix()
		dl.assemble(u_trial*u_test*dl.dx, tensor=M)

		B = hf.StateSpaceIdentityOperator(M)
		return hf.LinearStateObservable(self.pde,B)


		

		
	def testSerializedBatchMethods(self):
		"""
		Test the agreement of the two different routines for computing active subspace
		"""
		AS_parameters = hf.ActiveSubspaceParameterList()
		AS_parameters['observable_constructor'] = self.buildObservable
		AS_parameters['observable_kwargs'] = {}
		AS_parameters['save_and_plot'] = False
		AS_parameters['serialized_sampling'] = False
		AS_parameters['store_Omega'] = True
		AS_parameters['rank'] = 64
		my_collective = hf.NullCollective()

		AS = hf.ActiveSubspaceProjector(self.observable,self.prior,collective = my_collective,parameters = AS_parameters)
		# Construct the subspace via batching
		AS.construct_input_subspace()
		d_batch_in = AS.d_GN
		AS.d_GN = None
		AS_parameters['serialized_sampling'] = True
		AS_parameters['ms_given'] = True
		AS.construct_input_subspace()
		d_serialized_in = AS.d_GN
		input_d_error = np.linalg.norm(d_batch_in - d_serialized_in)
		assert input_d_error < 1e-12

		AS_parameters['serialized_sampling'] = False
		AS.construct_input_subspace()
		d_batch_out = AS.d_NG
		AS_parameters['serialized_sampling'] = True
		AS_parameters['ms_given'] = True
		AS.construct_input_subspace()
		d_serialized_out = AS.d_NG
		output_d_error = np.linalg.norm(d_batch_in - d_serialized_in)
		
		assert output_d_error < 1e-12


if __name__ == '__main__':
	dl.parameters["ghost_mode"] = "shared_facet"
	unittest.main()
