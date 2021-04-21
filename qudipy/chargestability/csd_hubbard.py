'''
File used to generate and plot charge stability diagrams from provided potentials using the Hubbrad model
For more information about the method, see the references https://doi.org/10.1103/PhysRevB.83.235314 and https://doi.org/10.1103/PhysRevB.83.161301
'''

import copy
import numpy as np
import pandas as pd
import itertools
from scipy import linalg as la
from ..utils.constants import Constants

class HubbardCSD:
    '''
    Initialize the charge stability diagram class which generates charge stability diagrams based on given capacitance parameters.
    Based on section III in https://doi.org/10.1103/PhysRevB.83.235314.
    This class is intended for use with NextNano potentials to simulate charge stability diagrams of various designs, but can also be used with analytic potentials.
    '''
    def __init__(self, n_sites, n_e, cap_matrix, h_mu=False, h_t=False, h_u=False, param_dict=dict()):
        '''

        Parameters
        ----------
        n_sites: Number of sites in the system that an electron may occupy.
        n_e: Maximum number of electrons in the system. Must be less than or equal to 2*n_sites
        cap_matrix: Dimensionless matrix which is used to convert from gate voltages to chemical potentials on individual dots

        Keyword Arguments
        -----------------
        h_mu: Whether to include the chemical potential term in Hamiltonian when Hamiltonian is created (default False)
        h_t: Whether to include the tunnel coupling term in Hamiltonian when Hamiltonian is created (default False)
        h_u: Whether to include Coulomb repulsion in Hamiltonian when Hamiltonian is created (default False)
        param_dict: Dictionary of additional parameters in meV. The parameters can be U_ij or t_ij, which correspond to Coulomb repulsion or 
                tunnel coupling between dots i and j respectively. See "Hubbard Charge Stability" tutorial for more information

        Returns
        -------
        None
        '''

        # Save which parts of Hamiltonian to be used for later
        self.h_mu = h_mu
        self.h_t = h_t
        self.h_u = h_u

        # Set all dictonary items to variables of the object
        for key, item in param_dict.items():
            self.__setattr__(key, item)

        if self.h_mu is False:
            raise Exception("Hamiltonian will be independent of gate voltages so no charge stability diagram can be generated")

        # Check that number of electrons doesn't exceed twice the amount of sites
        if n_e > 2 * n_sites:
            raise Exception(f"Number of electrons ({n_e}) exceeds twice the amount of sites ({n_sites}) allowed")
        else:
            self.n_e = n_e
            self.n_sites = n_sites

        cap_matrix = np.array(cap_matrix)
        if cap_matrix.shape != (n_sites, n_sites):
            raise Exception(f"Expected capacitance matrix of shape ({n_sites},{n_sites}), instead got capacitance matrix of shape {cap_matrix.shape}")
        else:
            self.cap_matrix = cap_matrix

        # Generates the basis to be used
        self.__generate_basis()

        # These next steps generate the fixed portion of the Hamiltonian, which is created on initialization since it is independent of voltage

        # First, generate the matrix of the correct size
        self.fixed_hamiltonian = np.zeros((len(self.basis),len(self.basis)))

        # Then add the component to the fixed portion of the Hamiltonian that you want to consider
        if h_t is True:
            self.fixed_hamiltonian += self.__generate_h_t()

        if h_u is True:
            self.fixed_hamiltonian += self.__generate_h_u()

    def generate_csd(self, initial_v, g1, g2, v_g1_max, v_g2_max, num=100):
        '''Generates the charge stability diagram between v_g1(2)_min and v_g1(2)_max with num by num data points in 2D

        Parameters
        ----------
        initial_v: Initial voltage vector on all gates
        g1: Index of first gate to vary in the charge stability diagram
        g2: Index of second gate to vary in the charge stability diagram
        v_g1_max: maximum voltage on g1 in volts
        v_g2_max: maximum voltage on g2 in volts

        Keyword Arguments
        -----------------
        num: number of voltage point in 1d, which leads to a num^2 charge stability diagram (default 100)

        Returns
        -------
        None
        '''

        # Stores parameters for late
        self.num = num
        self.g1 = g1 - 1 #For consistent indexing
        self.g2 = g2 - 1 
        self.initial_v = initial_v
        self.v_g1_min = initial_v[self.g1]
        self.v_g1_max = v_g1_max
        self.v_g2_min = initial_v[self.g2]
        self.v_g2_max = v_g2_max

        # Generate voltage points to sweep over
        self.v_1_values = np.around(np.linspace(self.v_g1_min, self.v_g1_max, num), decimals=6)
        self.v_2_values = np.around(np.linspace(self.v_g2_min, self.v_g2_max, num), decimals=6)

        # Loop over all voltage point combinations in list comprehension
        occupation = [[[self._lowest_energy(self._volt_vect_gen(v_1, v_2))] for v_1 in self.v_1_values] for v_2 in self.v_2_values]

        # Create a num by num DataFrame from occupation data information as entries
        self.occupation = pd.DataFrame(occupation, index=self.v_1_values, columns=self.v_2_values)
    
    def _volt_vect_gen(self, v_1, v_2):
        '''
        Given two voltages v_1 and v_2 applied to gates g1 and g2 respectivey, 
        return the voltage vector with those voltages applied to those gates and everything other
        voltage set to their initial value

        Parameters
        ----------
        v_1: Initial voltage on gate g1 in volts
        v_2: Initial voltage on gate g2 in volts

        Returns
        -------
        Voltage vector of all voltages applied to each gate
        '''
        volt_vect = copy.copy((self.initial_v))
        volt_vect[self.g1] = v_1
        volt_vect[self.g2] = v_2
        return np.array(volt_vect).T

    def _lowest_energy(self, volt_vect):
        '''
        Given a vector of gate voltages, determine the lowest energy occupation

        Parameters
        ----------
        volt_vect: Initial voltage vector on all gates

        Returns
        -------
        Tuple of occupation on each site
        '''
        # Convert from voltages to chemical potentials using the capacitance matrix
        chem_vect = self._volt_to_chem_pot(volt_vect)
        h_mu = np.zeros(self.fixed_hamiltonian.shape)

        # Compute the variable part of the Hamiltonian
        for i in range(self.basis_length):
            h_mu[i][i] = (- chem_vect * self.basis_occupations[i]).sum()

        current_hamiltonian = self.fixed_hamiltonian + h_mu
        eigenvals, eigenvects = la.eigh(current_hamiltonian)
        eigenvects = np.transpose(eigenvects) # To get column eigenvectors not column entries
        lowest_eigenvect = np.squeeze(eigenvects[np.argmin(eigenvals)])
        lowest_eigenvect_prob = np.real(lowest_eigenvect * np.conj(lowest_eigenvect))
        occupation_list = []
        for i in range(self.n_sites):
            occupation_list.append((lowest_eigenvect_prob * getattr(self, 'basis_occupation_' + str(i+1))).sum())
        return tuple(occupation_list)

    def __generate_basis(self):
        '''
        Creates the basis of all possible states given the constraints on the number of sites and number of electrons

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Compute all possible occupations with the cartesian product, and then
        # remove all states that exceed the number of electron specified
        # TODO make this more efficient so we only generate the states we want
        all_combos = list(itertools.product(*[[0,1] for i in range(self.n_sites * 2)])) # * 2 is for spin degeneracy (could add another * 2 for valleys)
        basis = []
        for combo in all_combos:
            if sum(combo) <= self.n_e:
                basis.append(list(combo))

        # Labels each index in the basis state with site number and spin direction (could add valley states in future as well)
        self.sites = [f'site_{n+1}' for n in range(self.n_sites)]
        self.spins = ['spin_up', 'spin_down']
        basis_labels = list(itertools.product(self.sites, self.spins))

        # Count number of electrons in each basis state (useful to determine occupations of states later)
        occupations_list = []
        for i in range(self.n_sites):
            j = 2*i
            setattr(self, 'basis_occupation_' + str(i+1), np.array([sum(x[j:j+2]) for x in basis]))
            occupations_list.append(getattr(self, 'basis_occupation_' + str(i+1)))

        # Set for later use
        self.basis_occupations = np.vstack(occupations_list).transpose()
        self.basis = basis
        self.basis_length = len(basis)
        self.basis_labels = basis_labels

    def __generate_h_t(self):
        '''
        Generates the tunnel coupling term of the Hamiltonian in the Hubbard model

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Create empty matrix to fill
        h_t = np.zeros(self.fixed_hamiltonian.shape)

        # Go over pairs of states that are not the same (since H_t has no diagonal terms)
        for i in range(self.fixed_hamiltonian.shape[0]):
            for j in range(i):
                state_1 = self.basis[i]
                state_2 = self.basis[j]

                if sum(state_1) != sum(state_2):
                    continue # No tunnel coupling between states with different charge occupation
                
                if sum(state_1[::2]) != sum(state_2[::2]):
                    continue # No tunnel coupling between states with different number of spins in each orientation

                # Go over pairs of labels, which correspond to whether particular (location, spin) are occupied
                result = 0
                # count = 0
                for k in range(len(self.basis_labels)):
                    for l in range(k):
                        # Go over pairs of sites
                        for n in range(self.n_sites):
                                for m in range(n):
                                    # Add contribution to result is that tunnel coupling term exists (i.e non-zero)
                                    if hasattr(self, 't_' + str(m+1) + str(n+1)):
                                        result += getattr(self, 't_' + str(m+1) + str(n+1)) * self.__inner_product(state_1, self.__create(self.__annihilate(state_2, k), l))
                                        result += getattr(self, 't_' + str(m+1) + str(n+1)) * self.__inner_product(state_1, self.__create(self.__annihilate(state_2, l), k))

                h_t[i][j] = -result
                
        h_t += h_t.conj().T #Since matrix is symmetric
        return h_t

    def __generate_h_u(self):
        '''
        Generates the Coulomb repulsion term of the Hamiltonian in the Hubbard model

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        # Create empty matrix to fill
        h_u = np.zeros(self.fixed_hamiltonian.shape)

        # Go over all states (but not pairs since H_U is diagonal)
        for i in range(self.fixed_hamiltonian.shape[0]):
                state_1 = self.basis[i]
                # Go over pairs of labels, which correspond to whether particular (location, spin) are occupied
                result = 0
                for k in range(len(self.basis_labels)):
                    for l in range(k):
                        # Then, go over over all site pairs of sites and add their contribution if it is present
                        for j in range(self.n_sites):
                            for m in range(j+1): # j+1 to get same site repulsion
                                if self.basis_labels[k][0] == 'site_' + str(j+1) and self.basis_labels[l][0] == 'site_' + str(m+1):
                                    if hasattr(self, 'U_' + str(m+1) + str(j+1)): # Check if inter-dot coupling is set between these two sites, skipping if is not
                                        result += getattr(self, 'U_' + str(m+1) + str(j+1)) * self.__inner_product(state_1, self.__number(self.__number(state_1, k), l))



                h_u[i][i] = result
        return h_u

    def _volt_to_chem_pot(self, volt_vect):
        '''
        Converts from supplied voltage vector to chemical potential using the capacitance matrix
        Requires the capacitance matrix of the system to already be specified

        Parameters
        ----------
        volt_vect: vector of voltage on each gate

        Keyword Arguments
        -----------------
        None

        Returns
        -------
        Chemical potentials vector, with the same length as volt_vect
        '''
        return self.cap_matrix @ volt_vect

    def __inner_product(self, state_1, state_2):
        '''
        Computes the inner product of two orhtonormal states (lists with 0s and 1s only as entries)
        Is a private function since this is not a general prupose inner product function but is useful for this file 

        Parameters
        ----------
        state_1: First state in the inner product
        state_2: Second state in the inner product

        Returns
        -------
        Either 0 or 1, depending on the inner product
        '''
        if state_1 is None or state_2 is None: # Deals with cases where the coefficient of state is 0, so the inner product is multiplied by 0
            return 0
        elif state_1 == state_2:
            return 1
        else:
            return 0

    def __create(self, state, position):
        '''
        Computes the creation operator acting on a state at a particular position

        Parameters
        ----------
        state: state for the creation operator to be acted on
        position: place where we will try to increase the electron number

        Returns
        -------
        The state incremented by 1 in position, or None
        '''
        state = copy.copy(state) # to avoid overwrites onto object
        if state is None:
            pass # keep state as None
        elif state[position] == 0:
            state[position] = 1
        else:
            state = None
        return state

    def __annihilate(self, state, position):
        '''
        Computes the annihilation operator acting on a state at a particular position

        Parameters
        ----------
        state: state for the annihilation operator to be acted on
        position: place where we will try to decrease the electron number

        Returns
        -------
        The state reduced by 1 in position, or None
        '''
        state = copy.copy(state) # to avoid overwrites onto object
        if state is None:
            pass
        elif state[position] == 1:
            state[position] = 0
        else:
            state = None
        return state

    def __number(self, state, position):
        '''
        Computes the number oparator acting on a state at a particular position

        Parameters
        ----------
        state: state for the number operator to be acted on
        position: place where we will try to count the electron number

        Returns
        -------
        The state (if it has an electron in position) or None
        '''
        return self.__create(self.__annihilate(state, position), position)

    def load_potential(self, file_path):
        # TODO
        pass
