# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import numpy as np
from mpi4py import MPI

def splitCommunicators(comm_world, n_subdomain, n_instances):
    """
    This function takes an MPI comm_world communicator and creates a communicator grid
    based on both mesh parallelism and sample instance parallelism.

        - :code:`comm_world` - MPI comm world object
        - :code:`n_subdomain` - integer number of subdomains
        - :code:`n_instances` - integer number of sampling instances

    Rows correspond to sampling instances
    Columns correspond to mesh subdomain collectives across samples
    Color corresponds to row index, and key corresponds to column index
    """
    mpi_rank = comm_world.rank
    world_size = comm_world.size
    assert world_size == n_subdomain*n_instances

    color = np.floor(mpi_rank/n_subdomain)
    key = np.remainder(mpi_rank,n_subdomain) 
    mesh_constructor_comm = comm_world.Split(color = color,key = key)
    collective_comm = comm_world.Split(color = key,key = color)
    return mesh_constructor_comm, collective_comm


def checkFunctionSpaceConsistentPartitioning(Vh, collective):
    """
    This function checks consistent partitioning for a function space

        - :code:`Vh` - function space 
        - :code:`collective` - MPI collective 
    """
    v = dl.interpolate(dl.Constant(float(Vh.mesh().mpi_comm().rank)),Vh)
    if collective.rank() == 0:
        root_v = dl.interpolate(dl.Constant(float(Vh.mesh().mpi_comm().rank)),Vh)
    else:
        root_v = dl.interpolate(dl.Constant(0.),Vh)
    collective.bcast(root_v.vector(),root = 0)
    diff = v.vector() - root_v.vector()
    tests_passed_here = diff.norm("l2") < 1e-10
    tests_passed_everywhere = False
    tests_passed_everywhere = dl.MPI.comm_world.allreduce(tests_passed_here, op = MPI.LAND)
    return tests_passed_everywhere

def checkMeshConsistentPartitioning(mesh, collective):
    """
    This function checks consistent partitioning for a mesh

        - :code:`mesh` - mesh
        - :code:`collective` - MPI collective 
    """

    V1 = dl.FunctionSpace(mesh,"DG", 0)
    t1 = checkFunctionSpaceConsistentPartitioning(V1 , collective)
    
    V2 = dl.FunctionSpace(mesh,"CG", 1)
    t2 = checkFunctionSpaceConsistentPartitioning(V2, collective)
    return t1 and t2
