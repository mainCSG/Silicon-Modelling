"""
Quantum utility solver functions

@author: simba
"""

import numpy as np
import time
import qudipy as qd
import qudipy.exchange as ex
from scipy import sparse
from scipy.sparse.linalg import eigs 
from scipy.linalg import eigh
from qudipy.qutils.qmath import inner_prod

def build_1DSE_hamiltonian(consts, gparams):
    ''' 
    Build a single electron Hamilonian for the 1-dimensional potential 
    specified in the gparams class. The laplacian operator is approximated by
    using a 1D 3-point stencil. The Hamilonian assumes a natural ordering 
    format along the main diagonal.

    Parameters
    ----------
    consts : Constants class
        Contains constants value for material system.
    gparams : GridParameters class
        Contains grid and potential information

    Returns
    -------
    ham_1D : sparse 2D array
        1-dimensional Hamtilonian. The diagonal elements are in natural
        ordering format

    '''
    
    # Build potential energy hamiltonian term
    PE_1D = sparse.diags(gparams.potential)
    
    # Build the kinetic energy hamiltonian term
    
    # Construct dummy block matrix B
    KE_1D = sparse.eye(gparams.nx)*(-2/(gparams.dx**2))
    # Add the +/-1 off diagonal entries for the 1/dx^2 elements
    KE_1D = KE_1D + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),-1)
    KE_1D = KE_1D + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),1)
    
    # Multiply by unit coefficients
    if consts.units == 'Ry':
        KE_1D = -KE_1D
    else:
        KE_1D = -consts.hbar**2/(2*consts.me)*KE_1D    
        
    # Assemble the full Hamiltonian with potential and kinetic terms
    ham_1D = PE_1D + KE_1D
    
    return ham_1D

def build_2DSE_hamiltonian(consts, gparams):
    '''
    Build a single electron Hamilonian for the 2-dimensional potential 
    specified in the gparams class. The laplacian operator is approximated by 
    using a 2D 5-point stencil. The Hamiltonian assumes a natural ordering
    format along the main diagonal.

    Parameters
    ----------
    consts : Constants class
        Contains constants value for material system.
    gparams : GridParameters class
        Contains grid and potential information

    Returns
    -------
    ham_2D : sparse 2D array
        2-dimensional Hamtilonian. The diagonal elements are in natural
        ordering format.

    '''
    
    # Build potential energy hamiltonian term
    PE_2D = sparse.diags(
        np.squeeze(gparams.convert_MG_to_NO(gparams.potential)))
    
    # Build the kinetic energy hamiltonian term
    
    # Construct B matrix
    B = sparse.eye(gparams.nx)*(-2/(gparams.dx**2) - 2/(gparams.dy**2))
    # Add the +/-1 off diagonal entries for the 1/dx^2 elements
    B = B + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),-1)
    B = B + sparse.diags(np.ones(gparams.nx-1)/(gparams.dx**2),1)
    
    # Now create a block diagonal matrix of Bs
    KE_2D = sparse.kron(sparse.eye(gparams.ny), B)
    # Now set the off diagonal entries for the 1/dy^2 elements
    KE_2D = KE_2D + sparse.kron(sparse.diags(np.ones(gparams.ny-1),-1),
                                sparse.eye(gparams.nx)/(gparams.dy**2))
    KE_2D = KE_2D + sparse.kron(sparse.diags(np.ones(gparams.ny-1),1),
                                sparse.eye(gparams.nx)/(gparams.dy**2))
    
    # Multiply by appropriate unit coefficients
    if consts.units == 'Ry':
        KE_2D = -KE_2D
    else:
        KE_2D = -consts.hbar**2/(2*consts.me)*KE_2D    
        
    # Assemble the full Hamiltonian with potential and kinetic terms
    ham_2D = PE_2D + KE_2D
    
    return ham_2D

def solve_schrodinger_eq(consts, gparams, n_sols=1):
    '''
    Solve the time-independent Schrodinger-Equation H|Y> = E|Y> where H is
    the single-electron 1 (or 2)-dimensional Hamiltonian.

    Parameters
    ----------
    consts : Constants class
        Contains constants value for material system.
    gparams : GridParameters class
        Contains grid and potential information.   
        
    Keyword Arguments
    -----------------
    n_sols: int, optional
        Number of eigenvectors and eigenenergies to return. The default is 1.

    Returns
    -------
    eig_ens : complex 1D array
        Lowest eigenenergies sorted in ascending order.
    eig_vecs : complex 2D array
        Corresponding eigenvectors in either natural order (1D) or meshgrid 
        (2D) format. eig_vecs[i] is the ith eigenvector with corresponding
        eigenvalue eig_ens[i].
        

    '''
    
    # Determine if a 1D or 2D grid and build the respective Hamiltonian
    if gparams.grid_type == '1D':
        hamiltonian = build_1DSE_hamiltonian(consts, gparams)
    elif gparams.grid_type == '2D':
        hamiltonian = build_2DSE_hamiltonian(consts, gparams)   
        
    # Solve the schrodinger equation (eigenvalue problem)
    eig_ens, eig_vecs = eigs(hamiltonian.tocsc(), k=n_sols, M=None,
                                           sigma=gparams.potential.min())
    
    # Sort the eigenvalues in ascending order (if not already)
    idx = eig_ens.argsort()   
    eig_ens = eig_ens[idx]
    # Transpose so first index corresponds to the ith e-vector
    eig_vecs = eig_vecs.T
    # Sort eigenvectors to match eigenvalues
    eig_vecs = eig_vecs[idx,:]
    
    # Normalize the wavefunctions and convert to meshgrid format if it's a 2D
    # grid system
    if gparams.grid_type == '2D':
        eig_vecs_mesh = np.zeros((n_sols, gparams.ny, gparams.nx),
                                 dtype=complex)
    for idx in range(n_sols):
        curr_wf = eig_vecs[idx,:]
        
        if gparams.grid_type == '1D':
            norm_val = inner_prod(gparams, curr_wf, curr_wf)
            eig_vecs[idx,:] = curr_wf/np.sqrt(norm_val)
        
        if gparams.grid_type == '2D':
            norm_val = inner_prod(gparams, gparams.convert_NO_to_MG(
                curr_wf), gparams.convert_NO_to_MG(curr_wf))
        
            curr_wf = curr_wf/np.sqrt(norm_val)
            
            eig_vecs_mesh[idx,:,:] = gparams.convert_NO_to_MG(curr_wf)
            
    if gparams.grid_type == "2D":
        eig_vecs = eig_vecs_mesh
    
    return eig_ens, eig_vecs

def solve_many_elec_SE(gparams, n_elec, n_xy_ho, n_se, n_sols=4, 
                       consts=qd.Constants("vacuum"), optimize_omega=True,
                       omega=None, opt_omega_n_se=2, ho_CMEs=None, 
                       path_CME=None, spin_subspace='all'):
    '''
    This function calculates the many electron energy spectra given an
    arbitrary potential landscape. The energy spectra is found using a modified
    LCHO-CI approach where we approximate the single electron orbitals of the
    potential using a basis of harmonic orbitals centered at the coordinate
    system origin. This calculation will load a pre-calculated set of Coulomb
    matrix elements (CMEs) if it is available (if not, it will calculate them
    but this computation time is costly).

    Parameters
    ----------
    gparams : TYPE
        DESCRIPTION.
    n_elec : TYPE
        DESCRIPTION.
    n_xy_ho : TYPE
        DESCRIPTION.
    n_se : TYPE
        DESCRIPTION.
    n_sols : TYPE, optional
        DESCRIPTION. The default is 4.
    consts : TYPE, optiona
        DESCRIPTION. The default is "vacuum".
    optimize_omega : TYPE, optional
        DESCRIPTION. The default is True.
    omega : TYPE, optional
        DESCRIPTION. The default is None.
    opt_omega_n_se : TYPE, optional
        DESCRIPTION, The default is 2.
    ho_CMEs : TYPE, optional
        DESCRIPTION. The default is None.
    path_CME : TYPE, optional
        DESCRIPTION. The default is None.
    spin_subspace : TYPE, optional
        DESCRIPTION. The default is 'all'.

    Returns
    -------
    many_elec_ens : TYPE
        DESCRIPTION.
    many_elec_vecs : TYPE
        DESCRIPTION.

    '''
    
    total_calc_time = time.time()
    
    print('Begining many body energy calculation...\n')
    
    #********************#
    # omega optimization #
    #********************#
    opt_omega_time = time.time()
    if optimize_omega:
        print('Optimizing choice of omega in approximating the single ' +
              'electron orbitals...\n')
        omega_opt, __ = ex.optimize_HO_omega(gparams, 
                                             nx=n_xy_ho[0], ny=n_xy_ho[1],
                                             omega_guess=omega,
                                             n_se_orbs=opt_omega_n_se, 
                                             consts=consts)
        
        print(f'Found an optimal omega of {omega_opt:.2E}.\n')
        opt_omega_time = time.time() - opt_omega_time
        print(f'Elapsed time is {opt_omega_time} seconds.')
        print('Done!\n')
    else:
        omega_opt = omega
        opt_omega_time = time.time() - opt_omega_time
        
    #**********************#
    # Building 2D HO basis #
    #**********************#
    print('Finding 2D harmonic orbitals at origin...\n');
    # Create a new basis of HOs centered at the origin of our dots
    origin_hos = ex.build_HO_basis(gparams, omega=omega_opt, 
                                   nx=n_xy_ho[0], ny=n_xy_ho[1], ecc=1.0,
                                   consts=consts)
    print('Done!\n');
    
    #**********#
    # A matrix #
    #**********#
    print('Finding A matrix...\n')
    
    a_mat_time = time.time()
    __, a_mat, lcho_ens = ex.find_H_unitary_transformation(gparams, origin_hos, 
                                                           consts=consts, 
                                                           unitary=True,
                                                           ortho_basis=True)
    
    # Truncate A in accordance with how many itinerant orbitals we want
    a_mat = a_mat[:n_se, :]
    lcho_ens = lcho_ens[:n_se]
        
    a_mat_time = time.time() - a_mat_time
    print(f'Elapsed time is {a_mat_time} seconds.')
    print('Done!\n')
    
    print(lcho_ens)
    return 0, 0

    #************#
    # CME matrix #
    #************#
    if ho_CMEs:
        print('Harmonic orbital CME matrix supplied as an argument!\n');
    else:
        print('Harmonic orbital CME matrix NOT supplied as an argument!\n'+
              'Will construct the CME matrix now...\n')
        
        cme_time = time.time()
        ho_CMEs = ex.calc_origin_CME_matrix(n_xy_ho[0], n_xy_ho[1], omega=1.0)
        cme_time = time.time() - cme_time
        print(f'Elapsed time is {cme_time} seconds.')
        print('Done!\n')
    
    #***************#
    # Transform CME #
    #***************#
    transform_time = time.time()
    print('Transforming the CME library to single electron basis...\n')
    
    '''
    % Get a subset of the CMEs if the given nOrigins parameter in the
    % simparams file is less than what we've solved for in the library
    % already. Useful for checking convergence wrt number of orbitals. 
    CMEs_lib_sub = getSubsetOfCMEs(sparams, CMEs_lib);
    '''
    
    # Scale the CMEs to match the origin HOs used to construct the
    # transformation matrix
    A = sqrt(consts.hbar / (consts.me * omega_opt))

    # Now do a basis transformation using a_mat
    full_trans_mat = np.kron(a_mat, a_mat)
    
    se_CMEs = full_trans_mat @ (ho_CMEs / A) @ full_trans_mat.conj().T
    
    transform_time = time.time() - transform_time
    print(f'Elapsed time is {transform_time} seconds.')
    print('Done!\n')
    
    #************************************#
    # Build 2nd quantization hamiltonian #
    #************************************#
    
    build2nd_time = time.time()
    print('Building 2nd quantization Hamiltonian and diagonalizing...\n')
    
    # Build the 2nd quantization Hamiltonian and then diagonalize it to
    # obtain the egienvectors and eigenenergies of the many electron system.
    ham_2q = ex.build_second_quant_ham(n_elec, spin_subspace, n_se, lcho_ens,
                                       se_CMEs)
    
    build2nd_time = time.time() - build2nd_time
    print(f'Elapsed time is {build2nd_time} seconds.')
    print('Done!\n')
    
    # Now diagonalize the many electron hamiltonian
    many_elec_ens, many_elec_vecs = eigh(ham_2q, subset_by_index=[0,n_sols-1])
    
    total_calc_time = time.time() - total_calc_time
    
    # Display runtime information/etc.
    print(f'Total simulation time: {total_calc_time:.2f} sec, '+
          f'{total_calc_time/60:.2f} min.')
    
    print(f'Time optimizing omega: {opt_omega_time:.2f} sec, '+
          f'{opt_omega_time/total_calc_time*100:.2f}% of total.')
    
    print(f'Time finding A matrix: {a_mat_time:.2f} sec, '+
          f'{a_mat_time/total_calc_time*100:.2f}% of total.')
    
    print(f'Time transforming CMEs: {transform_time:.2f} sec, '+
          f'{transform_time/total_calc_time*100:.2f}% of total.')
    
    print(f'Time building 2nd quantization H: {build2nd_time:.2f} sec, '+
          f'{build2nd_time/total_calc_time*100:.2f}% of total.')
    
    return many_elec_ens, many_elec_vecs
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        

    