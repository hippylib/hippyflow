# Copyright (c) 2020-2024, The University of Texas at Austin 
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
import shutil
import dolfin as dl
import ufl
import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix 


import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp

sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

sys.path.append('./')
from setupPoissonControlProblem import setupPoissonPDEProblem, poisson_control_settings


class TestdataGenerator(unittest.TestCase):
    def setUp(self):
        # Set up the mesh and function spaces
        self.nx = 16
        self.ny = 16
        self.n_wells_per_side = 5
        self.n_control = self.n_wells_per_side**2

        self.mesh = dl.UnitSquareMesh(self.nx, self.ny)
        Vh_STATE = dl.FunctionSpace(self.mesh, 'CG', 1)
        Vh_PARAMETER = dl.FunctionSpace(self.mesh, 'CG', 1)
        Vh_CONTROL = dl.VectorFunctionSpace(self.mesh, "R", degree=0, dim=self.n_control)
        self.Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE, Vh_CONTROL]

        # Set up the control problem
        control_pde_settings = poisson_control_settings()
        control_pde_settings['nx'] = self.nx
        control_pde_settings['ny'] = self.ny
        control_pde_settings['N_WELLS_PER_SIDE'] = self.n_wells_per_side
        control_pde_settings['LINEAR'] = True
        self.pde, self.prior, self.control_dist = setupPoissonPDEProblem(self.Vh, control_pde_settings)

        # Set up the observable
        B = hp.assemblePointwiseObservation(self.Vh[0], self.mesh.coordinates())
        self.IdentityOperator = hf.StateSpaceIdentityOperator(B, use_mass_matrix=True)
        self.observable = hf.LinearStateObservable(self.pde, self.IdentityOperator)

        # Set up the data generator
        datagen_settings = {}
        datagen_settings['rM'] = 64
        datagen_settings['rZ'] = 16
        datagen_settings['oversample'] = 10
        datagen_settings['verbose'] = True

        self.datagen = hf.DataGenerator(self.observable, self.prior, self.control_dist, datagen_settings)
        

    def test_generate_data(self):        
        self._test_generate()
        self._test_two_step_generate()


    def _test_generate(self):
        # Construct input decoder
        kle_parameters = hf.KLEParameterList()
        kle_parameters["rank"] = 64
        kle_parameters["save_and_plot"] = False

        kle_constructor = hf.KLEProjector(self.prior, parameters=kle_parameters)
        input_d, input_decoder, input_encoder = kle_constructor.construct_input_subspace("prior")

        input_decoder = hf.mv_to_dense(input_decoder)

        # Construct output decoder
        # Sample data
        x = self.observable.generate_vector()
        noise = dl.Vector()
        self.prior.init_vector(noise, "noise")

        state_dim = x[hp.STATE].get_local().shape[0]
        u_data = np.zeros((128, state_dim))	

        for i in range(128):
            hp.parRandom.normal(1.0, noise)
            self.prior.sample(noise, x[hp.PARAMETER])
            self.observable.solveFwd(x[hp.STATE], x)
            u_data[i] = x[hp.STATE].get_local()

        # Compute output decoder from data
        u_trial = dl.TrialFunction(self.Vh[hp.STATE])
        u_test = dl.TestFunction(self.Vh[hp.STATE])
        M = dl.assemble( dl.inner(u_trial, u_test) * dl.dx ) 
        pod_constructor = hf.PODProjectorFromData(self.Vh, M)

        output_d, output_decoder, output_encoder, u_shift = pod_constructor.construct_subspace(u_data, 64, shifted=False)

        self.datagen.generate(64, derivatives=(1,1), output_decoder=output_decoder, input_decoder=input_decoder)
        shutil.rmtree('data')
        # Trying again now to see if it works without the reduced bases
        print('Computing Jacobians via randomized SVD')
        self.datagen.generate(64, derivatives=(1,1), output_decoder=None, input_decoder=None)        


    def _test_two_step_generate(self):
        self.datagen.two_step_generate(64, derivatives=(1,1), pod_rank=64)


    def tearDown(self):
        shutil.rmtree('data')


if __name__ == '__main__':
    unittest.main()
