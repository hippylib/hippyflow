# Copyright (c) 2020-2022, The University of Texas at Austin 
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

import numpy as np

from hippylib import MultiVector

def mv_to_dense_local(multivector):
    """
    This function converts a MultiVector object to a numpy array
        - :code:`multivector` - hippylib MultiVector object
    """
    multivector_shape = (multivector[0].get_local().shape[0],multivector.nvec())
    out_array = np.zeros(multivector_shape)
    for i in range(multivector_shape[-1]):
        out_array[:,i] = multivector[i].get_local()

    return out_array


def mv_to_dense(multivector):
    """
    This function converts a MultiVector object to a numpy array
        - :code:`multivector` - hippylib MultiVector object
    """
    multivector_shape = (multivector[0].gather_on_zero().shape[0],multivector.nvec())
    out_array = np.zeros(multivector_shape)
    for i in range(multivector_shape[-1]):
        out_array[:,i] = multivector[i].gather_on_zero()

    return out_array

def dense_to_mv_local(dense_array,dl_vector):
    """
    This function converts a numpy array to a MultiVector
        - :code:`dense_array` - numpy array to be transformed
        - :code:`dl_vector` - type :code:`dolfin.Vector` object to be used in the 
            MultiVector object constructor
    """
    # This function actually makes no sense
    temp = MultiVector(dl_vector,dense_array.shape[-1])
    for i in range(temp.nvec()):
        temp[i].set_local(dense_array[:,i])
    return temp

# def dense_to_mv(dense_array,dl_vector):
#     """
#     This function converts a numpy array to a MultiVector
#         - :code:`dense_array` - numpy array to be transformed
#         - :code:`dl_vector` - type :code:`dolfin.Vector` object to be used in the 
#             MultiVector object constructor
#     """
#     ndof,nvec = dense_array.shape
#     temp = MultiVector(dl_vector,nvec)
#     for i in range(temp.nvec()):
#         temp[i].set(dense_array[:,i],np.arange(ndof))
#     return temp




