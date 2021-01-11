"""
Hamiltonian matrix functions

@author: aaranyaalex
"""
import numpy as np
import qudipy.potential as pot
from qudipy.qutils.solvers import build_1DSE_hamiltonian, build_2DSE_hamiltonian
from scipy.sparse.linalg import eigs
from scipy.linalg import block_diag
from types import SimpleNamespace


def ham_interp(i_params):
    params = extract_dict(i_params)
    H_RS = None
    effH_O = None
    effH_S = None

    if 'RealSpace' in params.HamType:
        potential = params.PotInterp(params.V)
        gparams = pot.GridParameters(params.PotInterp.xcoords, potential=potential)

        if gparams.grid_type == '1D':
            H_RS = build_1DSE_hamiltonian(params.PotInterp.constants, gparams)
        elif gparams.grid_type == '2D':
            H_RS = build_2DSE_hamiltonian(params.PotInterp.constants, gparams)

    if 'effOrbital' in params.HamType:
        n_dots = min(len(params.TC)+1, len(params.Eps), len(params.OSplit))

        if n_dots < max(len(params.TC)+1, len(params.Eps), len(params.OSplit)):
            params.TC = params.TC[:n_dots-1]
            params.Eps = params.Eps[:n_dots]
            params.OSplit = params.OSplit[:n_dots]
            print(f'Supplied data inconsistent, calculating H for a {n_dots}QD-system')

        # form Hamiltonian using block diagonals
        blocks = np.multiply.outer(params.TC, np.ones((2, 2), int))
        off = np.empty((0, 2), int)
        effH_O = block_diag(off.T, *blocks, off) + block_diag(off, *blocks, off.T) + \
                 np.diag(np.insert(np.array(params.OSplit)+np.array(params.Eps), slice(None, None, 1), params.Eps))

    if 'effSpin' in params.HamType:
        n_dots = min(len(params.TC) + 1, len(params.Eps), len(params.VSplit))

        if n_dots < max(len(params.TC) + 1, len(params.Eps), len(params.VSplit)):
           params.TC = params.TC[:n_dots - 1]
           params.Eps = params.Eps[:n_dots]
           params.VSplit = params.VSplit[:n_dots]
           params.Eta = params.Eta[:n_dots]
           print(f'Supplied data inconsistent, calculating H for a {n_dots}QD-system')

        # Two level operators
        Ax = np.array([[0, 1], [1, 0]], dtype=complex)
        Ay = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Az = np.array([[1, 0], [0, -1]], dtype=complex)
        Ao = np.eye(2)
        A_p = 0.5*(Ax + 1j * Ay)
        A_m = 0.5*(Ax - 1j * Ay)
        # 3D arrays storing k operators for all dots
        k_d = np.zeros((n_dots, n_dots, n_dots), dtype=complex)
        k_d[np.diag_indices(n_dots, ndim=3)] = np.ones((n_dots))
        k_x = np.array([np.roll(np.pad(Ax, ((0,n_dots-2), (0,n_dots-2))), (n, n), axis=(0, 1)) for n in range(n_dots-1)])
        k_y = np.array([np.roll(np.pad(Ay, ((0,n_dots-2), (0,n_dots-2))), (n, n), axis=(0, 1)) for n in range(n_dots-1)])
        k_z = np.array([np.roll(np.pad(Az, ((0,n_dots-2), (0,n_dots-2))), (n, n), axis=(0, 1)) for n in range(n_dots-1)])
        k_0 = np.eye(n_dots)

        effH_S = np.sum(np.kron(np.kron(k_d*params.Eps, Ao), Ao) + np.kron(np.kron(k_d*params.VSplit, A_p), Ao) + \
                       np.kron(np.kron(k_d*params.Vplit, A_m), Ao), axis=2) + \
                np.sum(np.kron(np.kron(k_x*params.TC, Ao), Ao) + np.kron(np.kron(k_z*params.Eta[0], Ao), Ax) + \
                       np.kron(np.kron(k_y * params.Eta[0], Ao), Ax), axis=2) + \
                np.kron(np.kron(k_0 * params.Ez, Ao), Az)

    return H_RS, effH_O, effH_S


def extract_dict(i_params):
    compat_types = ['RealSpace', 'effOrbital', 'effSpin']
    compat_params = [['PotInterp', 'V'], ['TC', 'Eps', 'OSplit'], ['TC', 'Eps', 'VSplit', 'Ez', 'Eta']]
    params = SimpleNamespace(**i_params)

    if not hasattr(params, 'HamType'):
        # Assign a HamType based on the available parameter attributes
        params.HamType = [compat_types[compat_params.index(row)] for row in compat_params
                          if(set(row).issubset(set(params.__dict__)))]
        # Raise error if no HamType can be formed
        if not params.HamType:
            raise ValueError('Supplied parameters are insufficient to form supported Hamiltonians')
    else:
        try:
            # find total parameters needed to calculated user-inputted HamType(s)
            if type(params.HamType) is str:
                total_params = compat_params[compat_types.index(params.HamType)]
            else:
                total_params = sum([compat_params[compat_types.index(item)] for item in params.HamType], [])
        except ValueError:
            # error if user-inputted HamType unsupported
            raise ValueError(f'Hamiltonian type {params.HamType} is invalid.\nCompatible types are:\n' +
                             f'{compat_types}')

        # check if inputted parameters correspond to total params needed
        if not all(item in list(i_params.keys()) for item in total_params):
            # show user the improperly formatted / missing keys
            invalid_keys = np.setdiff1d(total_params, list(i_params.keys()))
            raise ValueError(f'Supplied dictionary keys {invalid_keys} are ' +
                             f'missing or invalid for the {params.HamType} Hamiltonian(s).\nNecessary keys are:\n' +
                             f'{total_params}')
    return params


def eigens(ham_array, params, nsols=1):
    potential = params['PotInterp'](params['V'])
    e_ens, e_vecs = eigs(ham_array, k=nsols, sigma=potential.min())
    idx = e_ens.argsort()
    e_ens = e_ens[idx]
    e_vecs = e_vecs[:, idx]

    return e_ens, e_vecs


def expectation(ham_array, wf1, wf2):
    wf1 = wf1.reshape(1, -1).conj()
    wf2 = wf2.reshape(-1, 1)

    E = np.dot(wf1, np.dot(ham_array, wf2))

    return E