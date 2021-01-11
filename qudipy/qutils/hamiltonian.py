"""
Hamiltonian matrix functions
Reference: B. Buonacorsi, B. Shaw, J. Baugh. (2020) doi.org/10.1103/PhysRevB.102.125406

@author: aaranyaalex
"""
import numpy as np
import qudipy.potential as pot
from qudipy.qutils.solvers import build_1DSE_hamiltonian, build_2DSE_hamiltonian
from scipy.sparse.linalg import eigs
from scipy.linalg import block_diag
from types import SimpleNamespace


def ham_interp(i_params):
    """
    Function to generate Hamiltonian matrices

    Parameters
    ----------
    i_params : dict
        Input parameters for compatible Hamiltonian types

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

        'Eta' : list of floats
            Spin-orbit strengths.

        'V' : list of floats
            Control pulse vector at a given time. In form [V1, V2, V3 ...]

        'PotInterp' : PotentialInterpolator object
            Contains grid and potential information.


    Returns
    -------
    H_RS: 2D sparse array
        Real Space Hamiltonian.

    effH_O: 2D complex array
        Effective Hamiltonian describing orbital dynamics of shuttled electron. Eqn(2) of Reference.

    effH_S: 2D complex array
        Effective Hamiltonian describing orbital, spin and valley dynamics of shuttled electron. Eqn(3) of Reference.

    """
    # Initialize outputs and check inputs
    params = extract_dict(i_params)
    H_RS = None
    effH_O = None
    effH_S = None

    if 'RealSpace' in params.HamType:
        # Interpolate potential at given control vector
        potential = params.PotInterp(params.V)
        gparams = pot.GridParameters(params.PotInterp.xcoords, potential=potential)

        # Build appropriate Hamiltonian
        if gparams.grid_type == '1D':
            H_RS = build_1DSE_hamiltonian(params.PotInterp.constants, gparams)
        elif gparams.grid_type == '2D':
            H_RS = build_2DSE_hamiltonian(params.PotInterp.constants, gparams)

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
        off = np.empty((0, 2), int)
        effH_O = block_diag(off.T, *blocks, off) + block_diag(off, *blocks, off.T) + \
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

        # 3D arrays storing k operators for all dots
        k_d = np.zeros((n_dots, n_dots, n_dots), dtype=complex)
        k_d[np.diag_indices(n_dots, ndim=3)] = np.ones((n_dots))
        k_x = np.array(
            [np.roll(np.pad(Ax, ((0, n_dots - 2), (0, n_dots - 2))), (n, n), axis=(0, 1)) for n in range(n_dots - 1)])
        k_x = np.einsum("i,ijk->ijk", params.TC, k_x)
        k_y = np.array(
            [np.roll(np.pad(Ay, ((0, n_dots - 2), (0, n_dots - 2))), (n, n), axis=(0, 1)) for n in range(n_dots - 1)])
        k_z = np.array(
            [np.roll(np.pad(Az, ((0, n_dots - 2), (0, n_dots - 2))), (n, n), axis=(0, 1)) for n in range(n_dots - 1)])
        k_0 = np.eye(n_dots)

        # Generate H using repeated Kronecker products, 3D arrays summed along 3rd dimension
        effH_S = np.sum(
            np.kron(np.kron(k_d * params.Eps, Ao), Ao) + np.kron(np.kron(k_d * params.VSplit, A_p), Ao) + np.kron(
                np.kron(k_d * params.VSplit, A_m), Ao), axis=0) + np.sum(
            np.kron(np.kron(k_x, Ao), Ao) + np.kron(np.kron(k_z * params.Eta[0], Ao), Ax) + np.kron(
                np.kron(k_y * params.Eta[1], Ao), Ax), axis=0) + np.kron(np.kron(k_0 * params.Ez, Ao), Az)

    # Returns all Hamiltonians, None if not formed
    return H_RS, effH_O, effH_S


def extract_dict(i_params):
    """
    Helper function to perform input parameter checks, and assign a HamType based on values given. Converts dictionary
    to object with attributes for readability.

    Parameters
    ----------
    i_params : dict
        Input parameters for compatible Hamiltonian types. See ham_interp() docstring for compatible key-value pairs.

    Returns
    -------
    params : SimpleNamespace
        Converted i_params, contains all data necessary for given HamType

    """

    # Compatible types and parameters. Row of compat_params corresponds to index of compat_types.
    compat_types = ['RealSpace', 'effOrbital', 'effSpin']
    compat_params = [['PotInterp', 'V'], ['TC', 'Eps', 'OSplit'], ['TC', 'Eps', 'VSplit', 'Ez', 'Eta']]
    params = SimpleNamespace(**i_params)

    if not hasattr(params, 'HamType'):
        # Assign a HamType based on the available attributes
        params.HamType = [compat_types[compat_params.index(row)] for row in compat_params
                          if (set(row).issubset(set(params.__dict__)))]
        # Raise error if no HamType can be formed
        if not params.HamType:
            raise ValueError('Supplied parameters are insufficient to form supported Hamiltonians')
    else:
        try:
            # Find total parameters needed to calculated user-inputted HamType(s)
            if type(params.HamType) is str:
                total_params = compat_params[compat_types.index(params.HamType)]
            else:
                total_params = sum([compat_params[compat_types.index(item)] for item in params.HamType], [])
        except ValueError:
            # Error thrown if user-inputted HamType unsupported
            raise ValueError(f'Hamiltonian type {params.HamType} is invalid.\nCompatible types are:\n' +
                             f'{compat_types}')

        # Check if inputted parameters contain to total params needed
        if not all(item in list(i_params.keys()) for item in total_params):
            # Show user the improperly formatted / missing keys
            invalid_keys = np.setdiff1d(total_params, list(i_params.keys()))
            raise ValueError(f'Supplied dictionary keys {invalid_keys} are ' +
                             f'missing or invalid for the {params.HamType} Hamiltonian(s).\nNecessary keys are:\n' +
                             f'{total_params}')
    return params


def eigens(ham_matrix, params, nsols=1):
    """
    Calculates eigenvalues and eigenvectors for a Hamiltonian matrix

    Parameters
    ----------
    ham_matrix : 2D complex array
        Hamiltonian matrix

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
    e_ens, e_vecs = eigs(ham_matrix, k=nsols, sigma=potential.min())
    # Sort values
    idx = e_ens.argsort()
    e_ens = e_ens[idx]
    e_vecs = e_vecs[:, idx]

    return e_ens, e_vecs


def expectation(ham_array, wf1, wf2):
    """
    Calculates expectation values of Hamiltonian with input wavefunctions

    Parameters
    ----------
    ham_matrix : 2D complex array
        Hamiltonian matrix
    wf1, wf2 : 1D complex vectors
        Wavefuntions used to calculate expectation values.

    Returns
    -------
    E: 1D complex vector
        Expectation value of Hamiltonian as <wf1| H |wf2>

    """
    # Ensure wf1 is row and wf2 is row
    wf1 = wf1.reshape(1, -1).conj()
    wf2 = wf2.reshape(-1, 1)

    E = np.dot(wf1, np.dot(ham_array, wf2))

    return E
