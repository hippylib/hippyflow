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