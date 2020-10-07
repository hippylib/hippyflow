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

