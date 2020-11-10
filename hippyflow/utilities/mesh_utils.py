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


def read_serial_write_parallel_mesh(mesh_file,verbose = True):
	world_size = dl.MPI.comm_world.size
	mesh=dl.Mesh()
	with dl.XDMFFile(mesh_file, 'r') as fid:
		fid.read(mesh)

	with dl.XDMFFile('p'+str(world_size)+mesh_file, 'w') as fid:
	   fid.write(mesh)

	if verbose:
		print('Succesfully read and wrote '+mesh_file+' for '+ str(world_size)+' processes')

