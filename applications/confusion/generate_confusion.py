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


import os
import numpy as np

gds = [(0.1,1.)]

nxnys = [(64,64),(192,192)]

for (gamma,delta) in gds:
	for nx,ny in nxnys:
		print(80*'#')
		print(('Running for gd = '+str((gamma,delta))+' nx,ny = '+str((nx,ny))).center(80))
		os.system('mpirun -n 4 python confusion_problem_setup.py -ninstance 4 -gamma '+str(gamma)+' -delta '+str(delta)+' -nx '+str(nx)+' -ny '+str(ny))