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

