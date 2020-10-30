import os
import numpy as np

# gammas =   [0.5,1.0,2.0,4.0]
# deltas =   [0.5,1.0,2.0,4.0]



# gds = [(1.0,2.0)]

# gds = [(0.1,0.1),(0.5,0.5),(2.0,0.5),(2.0,2.0)]

# gds = [(0.1,0.1),(1.0,0.1),(0.1,1.0),(1.0,1.0)]

gds = [(1.0,1.0)]

# for gamma in gammas:
# 	for delta in deltas:
# 		gds.append((gamma,delta))

# mesh_files = ['meshes/half_dome_chop_16.xdmf','meshes/half_dome_chop_22.xdmf']

mesh_files = ['meshes/half_dome_chop_16.xdmf']

for (gamma,delta) in gds:
	for mesh_file in mesh_files:
		print(80*'#')
		print(('Running for gd = '+str((gamma,delta))+' mesh_file = '+mesh_file).center(80))
		os.system('mpirun -n 4 python ice_problem_setup.py -ninstance 4 -gamma '+str(gamma)+' -delta '+str(delta)+' -mesh '+mesh_file)

