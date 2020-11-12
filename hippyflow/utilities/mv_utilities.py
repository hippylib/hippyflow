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

import numpy as np

from hippylib import MultiVector


def mv_to_dense(multivector):
    """
    This function converts a MultiVector object to a numpy array
        - :code:`multivector` - hippylib MultiVector object
    """
    multivector_shape = (multivector[0].get_local().shape[0],multivector.nvec())
    as_np_array = np.zeros(multivector_shape)
    for i in range(multivector_shape[-1]):
        temp = multivector[i].get_local()
        # print('For iteration i ||get_local|| = ', np.linalg.norm(temp))
        as_np_array[:,i] = temp

    return as_np_array

def dense_to_mv(dense_array,dl_vector):
    """
    This function converts a numpy array to a MultiVector
        - :code:`dense_array` - numpy array to be transformed
        - :code:`dl_vector` - type :code:`dolfin.Vector` object to be used in the 
            MultiVector object constructor
    """
    temp = MultiVector(dl_vector,dense_array.shape[-1])
    for i in range(temp.nvec()):
        temp[i].set_local(dense_array[:,i])
    return temp