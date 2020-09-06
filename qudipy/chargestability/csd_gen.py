'''
File used to generate and plot charge stability diagrams using the constant interaction model.
'''
import math
import sys
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

e = 1.602176634 * 10**-19  # TODO figure out relative imports for common constants


class CSD:
    '''
    Initialize the charge stability diagram class which generates charge stability diagrams based on given capacitance parameters.
    Based off of the analisys in section II.A and Appendix 3 in https://doi.org/10.1103/RevModPhys.75.1.
    This class is intended for testing of the analysis module and comparing extracted input parameter with known input parameters.
    '''
    def __init__(self, c_l, c_r, c_m, c_g1, c_g2):
        '''
         
        Parameters
        ----------
        c_l: Capacitance between the left resevoir and dot 1
        c_r: Capacitance between the right resevoir dot 2
        c_m: Capacitance between dot 1 and 2
        c_g1: Capacitance between gate 1 and dot 1
        c_g2:  Capacitance between gate 2 and dot 2

        Returns
        -------
        None

        '''

        # Capacitances between dots and resevoirs
        self.c_l = c_l
        self.c_r = c_r

        # Capacitances between dots
        self.c_m = c_m

        # Gate-dot capacitances
        self.c_g1 = c_g1
        self.c_g2 = c_g2

        # Calculated constants
        # Total sum of capacitances on dots
        self.c_1 = self.c_l + self.c_g1 + self.c_m
        self.c_2 = self.c_r + self.c_g2 + self.c_m

        # Dot charging energy
        self.e_c1 = e**2 * self.c_1 / (self.c_1 * self.c_2 - self.c_m**2)
        self.e_c2 = e**2 * self.c_2 / (self.c_1 * self.c_2 - self.c_m**2)

        # Electrostatic coupling energy
        self.e_cm = e**2 * self.c_m / (self.c_1 * self.c_2 - self.c_m**2)

    def calculate_energy(self, n_1, n_2, v_g1, v_g2):
        '''Returns energy of dot with occupation n_1, n_2 with applied voltages v_g1, v_g2.
        Dependent on c_l, c_r, c_m, c_g1 and c_g2 defined when object is initialized.

        Parameters
        ----------
        n_1: Occupation on dot 1
        n_2: Occupation on dot 2
        v_g1: voltage on plunger gate 1
        v_g2: voltage on plunger gate 2

        Returns
        -------
        Energy of system in joules
        '''
        # This is formula A12 from Appendix 3 of the paper references at the top of the file
        f = - 1/abs(e) * (
            self.c_g1 * v_g1 * (n_1 * self.e_c1 + n_2 * self.e_cm) + self.c_g2 * v_g2 * (
                n_1 * self.e_cm + n_2 * self.e_c2)) + 1/e**2 * (1/2 * self.c_g1**2 * v_g1**2 * self.e_c1 + 1/2 * self.c_g2**2 *
                                                                v_g2**2 * self.e_c2 + self.c_g1 * v_g1 * self.c_g2 * v_g2 * self.e_cm)

        return 1/2 * n_1**2 * self.e_c1 + 1/2 * n_2**2 * self.e_c2 + n_1 * n_2 * self.e_cm + f

    def _lowest_energy(self, v_g1, v_g2):
        '''Returns occupation (n_1, n_2) with lowest energy for applied gate voltages v_g1, v_g2. 
        Dependent on c_l, c_r, c_m, c_g1 and c_g2 defined when object is initialized.

        Parameters
        ----------
        v_g1: voltage on plunger gate 1
        v_g2: voltage on plunger gate 2

        Returns
        -------
        state: list with occupation in dot 1 and 2

        '''

        # get occupation giving lowest energy assuming a continuous variable function (i.e derivative of 0)
        n_1 = 1/(1 - self.e_cm ** 2/(self.e_c1 * self.e_c2)) * 1/abs(e) * (self.c_g1 * v_g1 * (
             1 - self.e_cm ** 2 / (self.e_c1 * self.e_c2)) + self.c_g2 * v_g2 * (self.e_cm/self.e_c2 - self.e_cm/self.e_c1))
        n_2 = -n_1 * self.e_cm/self.e_c2 + 1 / \
            abs(e) * (self.c_g1 * v_g1 * self.e_cm/self.e_c2 + self.c_g2 * v_g2)

        # goes over 4 closest integer lattice points to find integer solution with lowest energy
        n_trials = [(math.floor(n_1), math.floor(n_2)), (math.floor(n_1) + 1, math.floor(n_2)),
                    (math.floor(n_1), math.floor(n_2) + 1), (math.floor(n_1) + 1, math.floor(n_2) + 1)]
        n_energies = [self.calculate_energy(
            *occupation, v_g1, v_g2) for occupation in n_trials]
        state = n_trials[n_energies.index(min(n_energies))]
        if state[0] >= 0 and state[1] >= 0:
            return state
        if state[0] < 0 and state[1] < 0:
            return [0, 0]
        elif state[0] < 0:
            return [0, state[1]]
        else:
            return [state[0], 0]

    def generate_csd(self, v_g1_max, v_g2_max, v_g1_min=0, v_g2_min=0, c_cs_1=None, c_cs_2=None, num=100, plotting=False, blur=False, blur_sigma=0):
        ''' Generates the charge stability diagram between v_g1(2)_min and v_g1(2)_max with num by num data points in 2D

        Parameters
        ----------
        v_g1_max: maximum voltage on plunger gate 1
        v_g2_max: maximum voltage on plunger gate 2

        Keyword Arguments
        -----------------
        v_g1_max: minimum voltage on plunger gate 1 (default 0)
        v_g2_max: minimum voltage on plunger gate 2 (default 0)
        c_cs_1: coupling between charge sensor and dot 1 (default to None)
        c_cs_2: coupling between charge sensor and dot 2 (default to None)
        num: number of voltage point in 1d, which leads to a num^2 charge stability diagram (default 100)
        plotting: flag indicating whether charge stability diagram should be plotted after completion (default False)
        blur: Flag which dictates whether to do a Gaussian blur on the data (default False)
        blur_sigma: If blur is set to True, use this number as the stanaderd devation in the Gaussian blur (default 0)

        Returns
        -------
        None
        '''
        self.num = num
        self.v_g1_min = v_g1_min
        self.v_g1_max = v_g1_max
        self.v_g2_min = v_g2_min
        self.v_g2_max = v_g2_max
        # Determines how to make the colorbar for the charge stability diagram
        # If capacitances are given, use those as multipliers
        if (c_cs_1 is not None) and (c_cs_2 is not None):
            dot_1_multiplier = c_cs_1
            dot_2_multiplier = c_cs_2
        # Otherwise, create a colormap that gives a uniques color combo to each (n,m) pair
        else:
            max_occupation = self._lowest_energy(v_g1_max, v_g2_max)
            dot_1_multiplier = 1
            dot_2_multiplier = max_occupation[0] + 1

        # Generates all the voltages to be swept
        self.v_1_values = [round(self.v_g1_min + i/num * (self.v_g1_max - self.v_g1_min), 4) for i in range(num)]
        self.v_2_values = [round(self.v_g2_min + j/num * (self.v_g2_max - self.v_g2_min), 4) for j in range(num)]
        # Goes through all the v_1 and v_2 values and generate the csd data

        # Checks if a version after 3.8 is running in order to use the more efficient Walrus operator 
        if sys.sys.version_info >= (3, 8): # check that 
            data = [
                    [v_1, v_2, (p:= self._lowest_energy(v_1, v_2))[0] * dot_1_multiplier + p[1] * dot_2_multiplier
                    ] for v_1 in self.v_1_values for v_2 in self.v_2_values
                ]
        else: # run less efficient version of the code
            data = [
                    [v_1, v_2, self._lowest_energy(v_1, v_2)[0] * dot_1_multiplier + self._lowest_energy(v_1, v_2)[1] * dot_2_multiplier
                    ] for v_1 in self.v_1_values for v_2 in self.v_2_values
               ]

        # Create DataFrame from data and pivot into num by num array
        df = pd.DataFrame(data, columns=['V_g1', 'V_g2', 'Current'])
        self.csd = df.pivot_table(index='V_g1', columns='V_g2', values='Current')

        if blur is True:
            self.csd = pd.DataFrame(gaussian_filter(self.csd, blur_sigma), columns=self.v_1_values, index=self.v_2_values) 

        # Create Dataframe that looks for differences between adjacent pixels, creating a "derivative" of the charge stability diagram
        df_der_row = self.csd.diff(axis=0)
        df_der_col = self.csd.diff(axis=1)
        df_der = pd.concat([df_der_row, df_der_col]).max(level=0)
        self.csd_der = df_der

        # Show plots if flag is set to True
        if plotting is True:

            # Toggles colorbar if charge sensor information is given
            cbar_flag = False
            if (c_cs_1 is not None) and (c_cs_2 is not None):
                cbar_flag = True

            # Plot the chagre stability diagram
            p1 = sb.heatmap(self.csd, cbar=cbar_flag, xticklabels=int(
                num/5), yticklabels=int(num/5), cbar_kws={'label': 'Current (arb.)'})
            p1.axes.invert_yaxis()
            p1.set(xlabel=r'V$_1$', ylabel=r'V$_2$')
            plt.show()

            # Plot the "derivative" of the charge stability diagram
            p2 = sb.heatmap(df_der, cbar=cbar_flag, xticklabels=int(
            num/5), yticklabels=int(num/5))
            p2.axes.invert_yaxis()
            p2.set(xlabel=r'V$_1$', ylabel=r'V$_2$')
            plt.show()
