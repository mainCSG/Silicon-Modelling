
from ..qutils.math import expectation_value

import numpy as np

def delta_g(gparams, wavefunc, e_field):

    avg_e = expectation_value(gparams, wavefunc, np.square(e_field))
    
    mu_2 = 2.2* (1e-9)**2 #(nm^2/V^2)

    delta_g = mu_2 * avg_e

    return np.real(delta_g)



