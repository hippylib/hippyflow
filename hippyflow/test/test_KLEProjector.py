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


class MassPreconditionedCovarianceOperator:
	def __init__(self, C, M):
		"""
		Linear operator representing the mass matrix preconditioned
		covariance matrix :math:`M C M`
		"""
		self.C = C 
		self.M = M 
		self.mpi_comm = self.M.mpi_comm()


		self.Mx = dl.Vector(self.mpi_comm)
		self.CMx = dl.Vector(self.mpi_comm)
		self.M.init_vector(self.Mx, 0)
		self.M.init_vector(self.CMx, 0)

	def init_vector(self,x,dim):
		self.M.init_vector(x,dim)

	def mult(self, x, y):
		self.M.mult(x, self.Mx)
		self.C.mult(self.Mx, self.CMx)
		self.M.mult(self.CMx, y)


class TestKLEProjectorFromData(unittest.TestCase):
    def setUp(self):
        nx = 16
        ny = 16
        self.rank = 256

        # Setup mesh and function spaces
        self.mesh = dl.UnitSquareMesh(nx, ny)
        Vh2 = dl.FunctionSpace(self.mesh, 'Lagrange', 2)
        Vh1 = dl.FunctionSpace(self.mesh, 'Lagrange', 1)
        self.Vh = [Vh2, Vh1, Vh2]

        # Setup prior
        self.prior = hf.BiLaplacian2D(self.Vh[hp.PARAMETER],gamma = 0.1, delta = 1.0)

        # Setup sample vector
        self.m = dl.Function(self.Vh[hp.PARAMETER]).vector()

        # Extract the mass matrix as a csr matrix
        M_mat = dl.as_backend_type(self.prior.M).mat()
        row, col, val = M_mat.getValuesCSR()
        self.M_csr = csr_matrix((val, col, row))

        # Setup subspace constructor
        parameters = hf.KLEParameterList()
        parameters["rank"] = self.rank
        parameters["save_and_plot"] = False
        self.kle_constructor = hf.KLEProjector(self.prior, parameters=parameters)

    
    def test_kle(self):
        """
        Test the KLE constructor
            - Orthogonality check for decoder and encoder
            - Check KLE satisfies eigenvalue problem 
        """      
        self._check_mass()
        self._check_prior()
        self._check_identity()

    
    def _check_mass(self):
        fro_tol = 1e-10

        d, decoder, encoder = self.kle_constructor.construct_input_subspace("mass")

        # Check orthogonality
        m = decoder.dot_mv(encoder)
        orth_error = np.linalg.norm(m - np.eye(self.rank), 'fro')/np.sqrt(self.rank)
        self.assertAlmostEqual(orth_error, 0.0, delta=fro_tol)

        # Check encoder
        Mdecoder = hp.MultiVector(self.m, self.rank)
        hp.MatMvMult(self.prior.M, decoder, Mdecoder)
        Mdecoder_dense = hf.mv_to_dense(Mdecoder)
        encoder_dense = hf.mv_to_dense(encoder)

        encoder_error = np.linalg.norm(Mdecoder_dense - encoder_dense, 'fro') / np.linalg.norm(Mdecoder_dense, 'fro')
        self.assertAlmostEqual(encoder_error, 0.0, delta=fro_tol)

        # Check eigenvalue problem
        eig_tol = 1e-4
        # Initialize operator and multivector
        KLE_Operator = MassPreconditionedCovarianceOperator(self.kle_constructor.C,self.prior.M)
        MCMx = hp.MultiVector(self.m, self.rank)
        Mx = hp.MultiVector(self.m, self.rank)

        MCMx.zero()
        Mx.zero()

        # Apply operators
        hp.MatMvMult(KLE_Operator, decoder, MCMx)
        hp.MatMvMult(self.prior.M, decoder, Mx)

        MCMx_dense = hf.mv_to_dense(MCMx)
        Mx_dense = hf.mv_to_dense(Mx)
        d.reshape(-1, 1)

        eig_error = np.linalg.norm(MCMx_dense - Mx_dense*d.T, 'fro') / np.linalg.norm(MCMx_dense, 'fro')
        self.assertAlmostEqual(eig_error, 0.0, delta=eig_tol)


    def _check_prior(self):
        """
        Test the KLE constructor with orthogonality mode "prior"
            - M-orthogonality check for decoder and encoder
        """
        fro_tol = 1e-10

        d, decoder, encoder = self.kle_constructor.construct_input_subspace("prior")

        # Check orthogonality
        m = decoder.dot_mv(encoder)
        orth_error = np.linalg.norm(m - np.eye(self.rank), 'fro')/np.sqrt(self.rank)
        self.assertAlmostEqual(orth_error, 0.0, delta=fro_tol)

        # Check encoder
        Mdecoder = hp.MultiVector(self.m, self.rank)
        hp.MatMvMult(self.prior.M, decoder, Mdecoder)
        Mdecoder_dense = hf.mv_to_dense(Mdecoder)
        encoder_dense = hf.mv_to_dense(encoder)

        # Apply covariance eigenvalues
        d.reshape(-1, 1)
        encoder_dense = encoder_dense * d.T

        encoder_error = np.linalg.norm(Mdecoder_dense - encoder_dense, 'fro') / np.linalg.norm(Mdecoder_dense, 'fro')
        self.assertAlmostEqual(encoder_error, 0.0, delta=fro_tol)

        # Check eigenvalue problem
        eig_tol = 1e-4
        # Intialize multivector
        Ax = hp.MultiVector(self.m, self.rank)
        Mx = hp.MultiVector(self.m, self.rank)

        Ax.zero()
        Mx.zero()

        # Apply operators
        hp.MatMvMult(self.prior.A, decoder, Ax)
        hp.MatMvMult(self.prior.M, decoder, Mx)


        Ax_dense = hf.mv_to_dense(Ax)
        Mx_dense = hf.mv_to_dense(Mx)

        # Apply eigenvalues
        eigenvals = np.sqrt(1/d)
        eigenvals.reshape(-1, 1)
        eig_error = np.linalg.norm(Ax_dense - Mx_dense*eigenvals.T, 'fro') / np.linalg.norm(Ax_dense, 'fro')
        self.assertAlmostEqual(eig_error, 0.0, delta=eig_tol)


    def _check_identity(self):
        fro_tol = 1e-10

        d, decoder, encoder = self.kle_constructor.construct_input_subspace("identity")

        # Check orthogonality
        m = decoder.dot_mv(encoder)
        orth_error = np.linalg.norm(m - np.eye(self.rank), 'fro')/np.sqrt(self.rank)
        self.assertAlmostEqual(orth_error, 0.0, delta=fro_tol)

        # Check encoder
        decoder_dense = hf.mv_to_dense(decoder)
        encoder_dense = hf.mv_to_dense(encoder)

        encoder_error = np.linalg.norm(decoder_dense - encoder_dense, 'fro') / np.linalg.norm(decoder_dense, 'fro')
        self.assertAlmostEqual(encoder_error, 0.0, delta=fro_tol)

        # Check eigenvalue problem
        eig_tol = 1e-4
        # Initialize multivector
        Rinvx = hp.MultiVector(self.m, self.rank)

        Rinvx.zero()

        # Get Rsolver operator
        Rinv = hp.Solver2Operator(self.prior.Rsolver)

        # Apply operators
        hp.MatMvMult(Rinv, decoder, Rinvx)

        Rinvx_dense = hf.mv_to_dense(Rinvx)
        d.reshape(-1, 1)

        eig_error = np.linalg.norm(Rinvx_dense - decoder_dense*d.T, 'fro') / np.linalg.norm(Rinvx_dense, 'fro')
        self.assertAlmostEqual(eig_error, 0.0, delta=eig_tol)


if __name__ == "__main__":
    unittest.main()