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


def read_serial_write_parallel_mesh(mesh_file,verbose = True):
	"""
	This function reads an XDMF mesh file in serial and writes 
	it to be partitioned in parallel
		- :code:`mesh_file` - The file to be read
		- :code:`verbose` - Boolean for printing
	"""
	world_size = dl.MPI.comm_world.size
	mesh=dl.Mesh()
	with dl.XDMFFile(mesh_file, 'r') as fid:
		fid.read(mesh)

	with dl.XDMFFile('p'+str(world_size)+mesh_file, 'w') as fid:
	   fid.write(mesh)

	if verbose:
		print('Succesfully read and wrote '+mesh_file+' for '+ str(world_size)+' processes')

