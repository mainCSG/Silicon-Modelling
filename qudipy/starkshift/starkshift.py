# From module
from ..qutils.math import inner_prod
from ..qutils.solvers import solve_schrodinger_eq

# From external libraries
import numpy as np
import pandas as pd

class StarkShift:
    def __init__(self, gparams, consts):
        self.gparams = gparams
        self.consts = consts

    def delta_g(self, e_interp, c_vals, c_val_names, wavefuncs=None):

        if wavefuncs==None:
            _, wavefuncs = solve_schrodinger_eq(self.consts, self.gparams, n_sols=1)

        c_vals_delta_g = []

        if self.consts.mu_2 is not None:
            mu_2 = self.consts.mu_2
        else:
            raise NotImplementedError("The quadratic Stark shift coefficient in this material \
                                       has not been specified in the constants class")

        for c_val in c_vals:
        # Return the potential interpolated (or calculated) at that particular value
            v_vec = c_val
            new_e = e_interp(v_vec)

            delta_g_list = []
            for wavefunc in wavefuncs:

                #Calcualte the weighted average of the electric field over the wavefunction
                avg_e = self._weighted_average(wavefunc, np.square(new_e))
            
                delta_g = mu_2 * avg_e
                delta_g_list.append(np.real(delta_g))

            
            c_vals_delta_g.append(c_val + delta_g_list)

        # Return the calculated value of delta_g
        delta_gs = ['delta_g_' + str(i+1) for i in range(len(wavefuncs))]
        coulmns = c_val_names + delta_gs
        df = pd.DataFrame(c_vals_delta_g, columns=coulmns)
        return df

    def _weighted_average(self, wavefunc, observable):
        '''
        Calculates the average of an observable, weighted by the probability density of a wavefunction
        Parameters
        ----------
        gparams : GridParameters class
            Contains grid and potential information.
        wavefunc : complex array
            'Bra' wavefunction for the inner product. If grid is 2D, then the 
            array should be in meshgrid format.
        observable : complex array
            Should have the same dimensions  

        Returns
        -------
        exp_val : complex float
            The expectation value calculated 
        '''
        weighted_wavefunc = observable * wavefunc
        exp_val = inner_prod(self.gparams, wavefunc, weighted_wavefunc)
        return exp_val



