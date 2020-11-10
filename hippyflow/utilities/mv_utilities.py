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

import numpy as np

from hippylib import *


def mv_to_dense(multivector):
    multivector_shape = (multivector[0].get_local().shape[0],multivector.nvec())
    as_np_array = np.zeros(multivector_shape)
    for i in range(multivector_shape[-1]):
        temp = multivector[i].get_local()
        # print('For iteration i ||get_local|| = ', np.linalg.norm(temp))
        as_np_array[:,i] = temp

    return as_np_array

def dense_to_mv(dense_array,dl_vector):
    temp = MultiVector(dl_vector,dense_array.shape[-1])
    for i in range(temp.nvec()):
        temp[i].set_local(dense_array[:,i])
    return temp