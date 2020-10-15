import os
import numpy as np

# gammas =   [0.5,1.0,2.0,4.0]
# deltas =   [0.5,1.0,2.0,4.0]

gammas = [1.0]
deltas = [2.0]

# gds = [(1.0,2.0)]

gds = [(0.5,2.0),(0.5,0.5),(2.0,0.5),(2.0,2.0)]

# for gamma in gammas:
# 	for delta in deltas:
# 		gds.append((gamma,delta))

nxnys = [(32,32),(64,64),(96,96),(128,128),(160,160),(192,192)]
# nxnys = [(160,160),(192,192)]

for (gamma,delta) in gds:
	for nx,ny in nxnys:
		print(80*'#')
		print(('Running for gd = '+str((gamma,delta))+' nx,ny = '+str((nx,ny))).center(80))
		os.system('mpirun -n 4 python confusion_problem_setup.py -ninstance 4 -gamma '+str(gamma)+' -delta '+str(delta)+' -nx '+str(nx)+' -ny '+str(ny))

