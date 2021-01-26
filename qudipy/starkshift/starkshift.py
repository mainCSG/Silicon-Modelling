
from ..qutils.math import expectation_value

import numpy as np

def delta_g(gparams, wavefunc, e_field):

    #Calcualte the weighted average of the electric field over the wavefunction
    avg_e = expectation_value(gparams, wavefunc, np.square(e_field))
    
    # Multiply by the ratio that was found in https://doi.org/10.1038/nnano.2014.216
    mu_2 = 2.2* (1e-9)**2 #(nm^2/V^2)
    delta_g = mu_2 * avg_e

    # Return the calculated value of delta_g
    return np.real(delta_g)



