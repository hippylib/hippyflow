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


from hippylib import *
import dolfin as dl
import numpy as np


class PriorPreconditionedProjector:
	"""
	"""
	def __init__(self,U,Cinv,my_init_vector):
		"""
		"""
		self.U = U
		self.Cinv = Cinv
		self.my_init_vector = my_init_vector

		self.Cinvx = dl.Vector()
		self.my_init_vector(self.Cinvx,0)

		pass


	def init_vector(self,x,dim):
		"""

		"""
		self.my_init_vector(x,dim)


	def mult(self,x,y):
		"""
		
		"""
		self.Cinv.mult(x,self.Cinvx)
		UtCinvx = self.U.dot_v(self.Cinvx)
		y.zero()
		self.U.reduce(y,UtCinvx)






