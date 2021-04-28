"""
Hamiltonian matrix functions
Reference: B. Buonacorsi, B. Shaw, J. Baugh. (2020) doi.org/10.1103/PhysRevB.102.125406

@author: aaranyaalex
"""
import numpy as np
from .. import potential as pot
from ..qutils.solvers import build_1DSE_hamiltonian, build_2DSE_hamiltonian
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
from scipy.linalg import block_diag
from types import SimpleNamespace

class Hamiltonian:
    '''
    General class which represents all Hamiltonians and methods that will be common to all Hamiltonians.
    Expected to be a parent class with child classes representing more specific types of Hamiltonians.
    '''
    def __init__(self, fixed_hamiltonian):
        """

        Parameters
        ----------
        fixed_hamiltonian: Square Hermitian matrix that represents the portion of the Hamiltonian that does not change with time

        """
        self.fixed_hamiltonian = fixed_hamiltonian
        self.size = fixed_hamiltonian.shape

    def eigens(self, nsols=1, variable_hamiltonian=None):
        """
        Finds the eigenvalues eigenvectors of the total Hamiltonian of the system, where the variable portion of the Hamiltonian is passed
        as a variable and the fixed portion of the Hamiltonian is 

        Parameters
        ----------
        None

        Keyword Arguments
        -----------------
        nsols: integer number of eigenvalue/eigenvectors to return (default 1)
        variable_hamiltonian: matrix which represents the variable part of the Hamiltonian to be added to the constant part of the Hamiltonian (default None)

        Returns
        -------
        None
        """

        if variable_hamiltonian is not None:
            total_hamiltonian = self.fixed_hamiltonian + variable_hamiltonian
        else:
            total_hamiltonian = self.fixed_hamiltonian

        eigenvals, eigenvects = eigh(total_hamiltonian)
        idx = eigenvals.argsort()
        eigenvals = eigenvals[idx][:nsols]
        eigenvects = np.transpose(eigenvects) # To get column eigenvectors not column entries
        eigenvects = eigenvects[idx][:nsols]
        return eigenvals, eigenvects

    def ground_state(self, variable_hamiltonian=None):
        """
        Wrapper around eigens function which returns the ground state energy and the ground state eigenvector with extra dimensions removed

        Parameters
        ----------
        None

        Keyword Arguments
        -----------------
        variable_hamiltonian: matrix which represents the variable part of the Hamiltonian to be added to the constant part of the Hamiltonian (default None)

        Returns
        -------
        eigval: Number which represents the ground state energy
        eigvect: 1D numpy array which represents the gorund state in th

        """

        # Calculate the eigenvectors and eigenvals using eigens, then only extract the first eigenenergy and 
        eigval, eigvect = self.eigens(variable_hamiltonian=variable_hamiltonian)
        eigval, eigvect = eigval[0], np.squeeze(eigvect)
        return eigval, eigvect

# This function is not for a general Hamiltonian
class HamFunctions:

    def eigens(self, A, params, nsols=1):
        """
        Calculates eigenvalues and eigenvectors for an operator

        Parameters
        ----------
        A : 2D complex array
            Operator of which to find the eigenstates

        params : dict
            Input dictionary containing 'PotInterp' and 'V' keys, corresponding to potential/grid info and control
            pulse vector.

        Keyword Args
        ----------

        nsols : int
            Number of eigenvectors and eigenvalues to solve for.

        Returns
        -------
        e_ens : complex 1D array
            Eigenenergies sorted in ascending order.

        e_vecs : complex 2D array
            Eigenvectors where e_vecs[:, i] corresponds to e_ens[i]

        """
        potential = params['PotInterp'](params['V'])
        e_ens, e_vecs = eigs(A, k=nsols, sigma=potential.min())
        # Sort values
        idx = e_ens.argsort()
        e_ens = e_ens[idx]
        e_vecs = e_vecs[:, idx]

        return e_ens, e_vecs

    def expectation(self, A, wf1, wf2):
        """
        Calculates expectation values of operators with inputted wavefunctions

        Parameters
        ----------
        A : 2D complex array
            Operator of which to take the expectation value
        wf1, wf2 : 1D complex vectors, inputted as kets
            Wavefuntions used to calculate expectation values.

        Returns
        -------
        E: 1D complex float
            Expectation value of Hamiltonian as <wf1| H |wf2>

        """
        # Convert wf1 to bra, check wf2 is ket
        wf1 = wf1.reshape(1, -1).conj()
        wf2 = wf2.reshape(-1, 1)

        E = np.dot(wf1, np.dot(A, wf2))

        return E


class HamGenerator(HamFunctions):

    """
    Child class used to generate shuttling Hamiltonian arrays.

    Class Attributes
    ----------------
    params : SimpleNamespace
        Input parameters for compatible Hamiltonian types

    RealSpace: 2D sparse array
        Real Space Hamiltonian.

    effOrbital: 2D complex array
        Effective Hamiltonian describing orbital dynamics of shuttled electron. Eqn(2) of Reference.

    effSpin: 2D complex array
        Effective Hamiltonian describing orbital, spin and valley dynamics of shuttled electron. Eqn(3) of Reference.

    """

    def __init__(self, i_params):
        """
        Perform input parameter checks, and assign a HamType based on values given. Converts dictionary
        to object with attributes for readability.

        Parameters
        ----------
        i_params : dict
            Input parameters for compatible Hamiltonian types.

            Accepted Key-Values
            -------------------
            'HamType' : str or list of str, optional
                Type of Hamiltonian to be generated. If not in i_params, will be filled according to inputted keys.
                Accepts arguments of "RealSpace", "effOrbital" and "effSpin".

            'TC' : list of floats
                Resonant tunnel coupling points of neighbouring QDs.

            'Eps' : list of floats
                Ground state energies of each QD.

            'OSplit' : list of floats
                Ground to first excited orbital splitting of each QD.

            'VSplit' : list of floats
                Valley splitting of each QD. Includes valley phase.

            'Ez' : float
                Zeeman energy due to static magnetic field.

            'SOC' : list of floats
                Spin-orbit coupling strengths.

            'V' : list of floats
                Control pulse vector at a given time. In form [V1, V2, V3 ...]

            'PotInterp' : PotentialInterpolator object
                Contains grid and potential information.
        """

        # Compatible types and corresponding necessary parameters
        compat = {
            "RealSpace": ['PotInterp', 'V'],
            "effOrbital": ['TC', 'Eps', 'OSplit'],
            "effSpin": ['TC', 'Eps', 'VSplit', 'Ez', 'SOC'],
        }
        params = SimpleNamespace(**i_params)

        if not hasattr(params, 'HamType'):
            # Assign a HamType based on the available attributes
            params.HamType = [key for key in compat if (set(compat[key]).issubset(set(params.__dict__)))]
            # Raise error if no HamType can be formed
            if not params.HamType:
                raise ValueError('Supplied parameters are insufficient to form supported Hamiltonians')
        else:
            try:
                # Find total parameters needed to calculated user-inputted HamType(s)
                if type(params.HamType) is str:
                    total_params = compat[params.HamType]
                else:
                    total_params = sum([compat[item] for item in params.HamType], [])
            except ValueError:
                # Error thrown if user-inputted HamType unsupported
                if type(params.HamType) is list:
                    raise ValueError(
                        f'Invalid Hamiltonian type(s):\n' + f'{set(params.HamType) - set(compat.keys())}.\n'
                                                            f'Compatible types are:\n' + f'{compat.keys()}')
                elif type(params.HamType) is str:
                    raise ValueError(f'Hamiltonian type {params.HamType} is invalid.\nCompatible types are:\n' +
                                     f'{compat.keys()}')

            # Check if inputted parameters contain to total params needed
            if not all(item in list(i_params.keys()) for item in total_params):
                # Show user the improperly formatted / missing keys
                invalid_keys = np.setdiff1d(total_params, list(i_params.keys()))
                raise ValueError(f'Supplied dictionary keys {invalid_keys} are ' +
                                 f'missing or invalid for the {params.HamType} Hamiltonian(s).\nNecessary keys are:\n' +
                                 f'{total_params}')

        # Initialize class attributes
        self.params = params
        self.RealSpace = None
        self.effSpin = None
        self.effOrbital = None

    def generate_ham_array(self):
        """
        Function to generate Hamiltonian arrays

        """
        # Initialize outputs and check inputs
        params = self.params

        if 'RealSpace' in params.HamType:
            # Interpolate potential at given control vector
            potential = params.PotInterp(params.V)
            gparams = pot.GridParameters(params.PotInterp.xcoords, potential=potential)

            # Build appropriate Hamiltonian
            if gparams.grid_type == '1D':
                self.RealSpace = build_1DSE_hamiltonian(params.PotInterp.constants, gparams)
            elif gparams.grid_type == '2D':
                self.RealSpace = build_2DSE_hamiltonian(params.PotInterp.constants, gparams)

        if 'effOrbital' in params.HamType:
            # Use input params to find number of QDs in chain
            n_dots = min(len(params.TC) + 1, len(params.Eps), len(params.OSplit))

            if n_dots < max(len(params.TC) + 1, len(params.Eps), len(params.OSplit)):
                # Cut data if extra data given
                params.TC = params.TC[:n_dots - 1]
                params.Eps = params.Eps[:n_dots]
                params.OSplit = params.OSplit[:n_dots]
                print(f'Supplied data inconsistent, calculating H for a {n_dots}QD-system')

            # Form Hamiltonian using block diagonals
            blocks = np.multiply.outer(params.TC, np.ones((2, 2), int))
            off = np.zeros((0, 2))
            self.effOrbital = block_diag(off.T, *blocks, off) + block_diag(off, *blocks, off.T) + \
                     np.diag(np.insert(np.array(params.OSplit) + np.array(params.Eps), slice(None, None, 1), params.Eps))

        if 'effSpin' in params.HamType:
            # Use input params to find number of QDs in chain
            n_dots = min(len(params.TC) + 1, len(params.Eps), len(params.VSplit))

            if n_dots < max(len(params.TC) + 1, len(params.Eps), len(params.VSplit)):
                # Cut data if extra data given
                params.TC = params.TC[:n_dots - 1]
                params.Eps = params.Eps[:n_dots]
                params.VSplit = params.VSplit[:n_dots]
                print(f'Supplied data inconsistent, calculating H for a {n_dots}QD-system')

            # Initialize two-level operators
            Ax = np.array([[0, 1], [1, 0]], dtype=complex)
            Ay = np.array([[0, -1j], [1j, 0]], dtype=complex)
            Az = np.array([[1, 0], [0, -1]], dtype=complex)
            Ao = np.eye(2)
            # A+ and A-
            A_p = 0.5 * (Ax + 1j * Ay)
            A_m = 0.5 * (Ax - 1j * Ay)

            # Initialize 3D arrays storing k operators for all dots
            k_d = np.zeros((n_dots,)*3, dtype=complex)
            k_x, k_y, k_z = np.zeros((n_dots-1, n_dots, n_dots), dtype=complex),  np.zeros((n_dots-1, n_dots, n_dots), dtype=complex), \
                            np.zeros((n_dots - 1, n_dots, n_dots), dtype=complex)
            k_0 = np.eye(n_dots)

            # Fill diagonals
            k_d[np.diag_indices(n_dots, ndim=3)] = np.ones((n_dots))
            k_x[:, :-1, 1:][np.diag_indices(n_dots-1, ndim=3)] = np.ones((n_dots-1))
            k_x[:, 1:, :-1][np.diag_indices(n_dots-1, ndim=3)] = np.ones((n_dots-1))
            k_y[:, :-1, 1:][np.diag_indices(n_dots - 1, ndim=3)] = np.ones((n_dots - 1)) * -1j
            k_y[:, 1:, :-1][np.diag_indices(n_dots - 1, ndim=3)] = np.ones((n_dots - 1)) * 1j
            k_z[np.diag_indices(n_dots-1, ndim=3)] = np.ones((n_dots-1))
            k_z[:, 1:, 1:][np.diag_indices(n_dots - 1, ndim=3)] = np.ones((n_dots - 1)) * -1

            # Generate H using repeated Kronecker products, 3D arrays summed along 3rd dimension
            k_x = np.einsum("i,ijk->ijk", params.TC, k_x)
            self.effSpin = np.sum(
                np.kron(np.kron(k_d * params.Eps, Ao), Ao) + np.kron(np.kron(k_d * params.VSplit, A_p), Ao) + np.kron(
                    np.kron(k_d * params.VSplit, A_m), Ao), axis=0) + np.sum(
                np.kron(np.kron(k_x, Ao), Ao) + np.kron(np.kron(k_z * params.SOC[0], Ao), Ax) + np.kron(
                    np.kron(k_y * params.SOC[1], Ao), Ax), axis=0) + np.kron(np.kron(k_0 * params.Ez, Ao), Az)
