import numpy as np


def SignNumpy(x):
    return np.float32(2.*np.greater_equal(x,0)-1.)