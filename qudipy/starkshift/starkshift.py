# From module
from ..qutils.math import expectation_value
from ..qutils.solvers import solve_schrodinger_eq

# From external libraries
import numpy as np

class StarkShift:
    def __init__(self, gparams, consts):
        self.gparams = gparams
        self.consts = consts

    def delta_g(self, e_interp, c_vals, wavefuncs=None):

        if wavefuncs==None:
            _, wavefuncs = solve_schrodinger_eq(self.consts, self.gparams, n_sols=1)

        c_vals_delta_g = []

        for c_val in c_vals:
        # Return the potential interpolated (or calculated) at that particular value
            v_vec = c_val
            new_e = e_interp(v_vec)

            delta_g_list = []
            for wavefunc in wavefuncs:

                #Calcualte the weighted average of the electric field over the wavefunction
                avg_e = expectation_value(self.gparams, wavefunc, np.square(new_e))
            
                # Multiply by the ratio that was found in https://doi.org/10.1038/nnano.2014.216
                mu_2 = 2.2* (1e-9)**2 #(nm^2/V^2)
                delta_g = mu_2 * avg_e
                delta_g_list.append(np.real(delta_g))

            
            c_vals_delta_g.append(c_val + delta_g_list)

        # Return the calculated value of delta_g

        return np.array(c_vals_delta_g)

    def find_wavefunctions(self, pot_interp, approx_location, c_vals):
        pass
        # wavefuncs = []
        # num_dots = len(approx_location)
        # initial_v_vec = []
        # for i in range(len(c_vals)):
        #     setattr(self, 'c_val_' + str(i) , c_vals[i])
        #     initial_v_vec.append(min(getattr(self, 'c_val_' + str(i))))

        # pot_interp.
        # for j in range(num_dots):


        #     _, wavefunc = solve_schrodinger_eq(self.consts, self.gparams, n_sols=1)

        # return wavefuncs



