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

class CollectiveOperator:
	def __init__(self, local_op, collective, mpi_op = 'sum'):
		assert hasattr(local_op,'mult')
		self.local_op = local_op
		self.collective = collective
		self.mpi_op = mpi_op

	def mult(self, x,y):
		self.local_op.mult(x,y)
		self.collective.allReduce(y, self.mpi_op)

	def transpmult(self,x,y):
		assert hasattr(self.local_op, 'transpmult')
		self.local_op.transpmult(x,y)
		self.collective.allReduce(y,self.mpi_op)

	def init_vector(self,x,dim):
		self.local_op.init_vector(x,dim)

