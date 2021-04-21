'''
File used to generate and plot charge stability diagrams for double dot systems using the constant interaction model.
'''
import numpy as np
import pandas as pd
from math import floor
from ..utils.constants import Constants

consts = Constants("vacuum")

class CIMCSD:
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
        # TODO generatlize this number
        self.n_sites = 2
        
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
        self.e_c1 = consts.e**2 * self.c_1 / (self.c_1 * self.c_2 - self.c_m**2)
        self.e_c2 = consts.e**2 * self.c_2 / (self.c_1 * self.c_2 - self.c_m**2)

        # Electrostatic coupling energy
        self.e_cm = consts.e**2 * self.c_m / (self.c_1 * self.c_2 - self.c_m**2)

    def generate_csd(self, v_g1_max, v_g2_max, v_g1_min=0, v_g2_min=0, num=100):
        ''' Generates the charge stability diagram between v_g1(2)_min and v_g1(2)_max with num by num data points in 2D

        Parameters
        ----------
        v_g1_max: maximum voltage on plunger gate 1
        v_g2_max: maximum voltage on plunger gate 2

        Keyword Arguments
        -----------------
        v_g1_max: minimum voltage on plunger gate 1 (default 0)
        v_g2_max: minimum voltage on plunger gate 2 (default 0)
        num: number of voltage point in 1d, which leads to a num^2 charge stability diagram (default 100)

        Returns
        -------
        None
        '''
        # Save for later
        self.num = num
        self.v_g1_min = v_g1_min
        self.v_g1_max = v_g1_max
        self.v_g2_min = v_g2_min
        self.v_g2_max = v_g2_max

        # Generates all the voltages to be swept
        self.v_1_values = np.around(np.linspace(self.v_g1_min, self.v_g1_max, num), decimals=6)
        self.v_2_values = np.around(np.linspace(self.v_g2_min, self.v_g2_max, num), decimals=6)
        
        # Goes through all the v_1 and v_2 values and generate the csd data
        occupation = [[[self._lowest_energy(v_1, v_2)] for v_1 in self.v_1_values] for v_2 in self.v_2_values]

        # Create a num by num DataFrame from occupation data information as entries
        self.occupation = pd.DataFrame(occupation, index=self.v_1_values, columns=self.v_2_values)

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
        f = - 1/consts.e * (
            self.c_g1 * v_g1 * (n_1 * self.e_c1 + n_2 * self.e_cm) + self.c_g2 * v_g2 * (
                n_1 * self.e_cm + n_2 * self.e_c2)) + 1/consts.e**2 * (1/2 * self.c_g1**2 * v_g1**2 * self.e_c1 + 1/2 * self.c_g2**2 *
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
        n_1 = 1/(1 - self.e_cm ** 2/(self.e_c1 * self.e_c2)) * 1/consts.e * (self.c_g1 * v_g1 * (
             1 - self.e_cm ** 2 / (self.e_c1 * self.e_c2)) + self.c_g2 * v_g2 * (self.e_cm/self.e_c2 - self.e_cm/self.e_c1))
        n_2 = -n_1 * self.e_cm/self.e_c2 + 1 / \
            consts.e * (self.c_g1 * v_g1 * self.e_cm/self.e_c2 + self.c_g2 * v_g2)

        # goes over 4 closest integer lattice points to find integer solution with lowest energy
        trial_occupations = [(floor(n_1), floor(n_2)), (floor(n_1) + 1, floor(n_2)),
                    (floor(n_1), floor(n_2) + 1), (floor(n_1) + 1, floor(n_2) + 1)]
        n_energies = [self.calculate_energy(
            *trial, v_g1, v_g2) for trial in trial_occupations]
        state = trial_occupations[n_energies.index(min(n_energies))]
        if state[0] >= 0 and state[1] >= 0:
            return tuple(state)
        if state[0] < 0 and state[1] < 0:
            return (0, 0)
        elif state[0] < 0:
            return (0, state[1])
        else:
            return (state[0], 0)
