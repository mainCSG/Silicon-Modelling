"""
Functions for calculating the exchange interaction.

@author: simba
"""

import numpy as np
import math
import qudipy as qd
from scipy.linalg import eig
from scipy.optimize import minimize

def optimize_HO_omega(gparams, nx, ny=None, ecc=1.0, omega_guess=1E15, 
                      n_SE_orbs=2, opt_tol=1E-7, consts=qd.Constants("vacuum")):
    '''
    Find an optimal choice of omega used when building a basis of harmonic 
    orbitals centered at the origin which are used to approximate the single 
    electron orbitals of a potential. Minimization is done using the BFGS 
    algorithm. If the algorithm fails, this function will return None.

    Parameters
    ----------
    gparams : GridParamters object
        Contains the grid and potential information.
    nx : int
        Number of modes along the x direction to include in the harmonic 
        orbital basis set.
        
    Keyword Arguments
    -----------------
    ny : int, optional
        Number of modes along the y direction to include in the harmonic 
        orbital basis set. The default is nx.
    ecc : float, optional
        Specify the eccentricity of the harmonic orbitals defined as 
        ecc = omega_y/omega_x. The default is 1.0 (omega_y = omega_x).
    omega_guess : float, optional
        Initial guess for omega in the optimization. If the grid was infinitely
        large and dense, then choice of omega_guess is not important. For finite
        grid sizes, a poor choice of omega can cause the harmonic orbitals 
        themselves to be larger than than the grid itself. This causes
        obvious orthogonality issues. Therefore, it is heavily suggested that
        omega_guess is supplied with a decent guess. The default is 1E15 which
        should be smaller than most reasonable grid spacings in SI units.
    n_SE_orbs : int, optional
        Number of single electron orbitals to compare to when checking how well
        the harmonic orbitals approximate the single electron orbitals for a 
        given choice of omega. The default is 2.
    opt_tol : float, optional
        Optimality tolerance for the BFGS algorithm. The default is 1E-7.
    consts : Constants object, optional
        Specify the material system constants. The default is "vacuum".

    Returns
    -------
    opt_omega : float
        The optimal omega found. If the optimization could not be completed, 
        then None is returned.
    opt_HOs : n-d array
        The basis of optimal harmonic orbitals. The first dimension of the 
        array corresponds to the index of the harmonic orbital.

    '''
    
    # Set default behavior of ny=nx
    if ny is None and gparams.grid_type == '2D':
        ny = nx
    
    # First thing to do is find the first few single electron orbitals from
    # the potential landscape.
    __, se_orbitals = qd.qutils.solvers.solve_schrodinger_eq(consts, gparams, 
                                                             n_sols=n_SE_orbs)

    def find_HO_wf_difference(curr_log_w):
        
        # Undo omega log
        curr_w = 10**curr_log_w
        
        # Find current basis of HOs
        curr_HOs = build_HO_basis(gparams, curr_w, nx, ny, ecc=ecc,
                                  consts=consts)
        
        # Get current overlap matrix between HOs and SE orbitals
        S_matrix = qd.qutils.qmath.find_overlap_matrix(gparams, se_orbitals, 
                                                       curr_HOs)
        
        # S contains all <SE_i|HO_j> inner products. If we have chosen a good 
        # omega, then SUM(|<SE_i|HO_j>|^2) will be close to 1. Therefore, we
        # want to maximize this value.
        min_condition = np.abs(1-np.diag(S_matrix.conj().T @ S_matrix))
        # Average the min condition with respect to the number of SE orbitals
        # so the ideal min condition = 1
        min_condition = np.sum(min_condition)/n_SE_orbs
        
        return min_condition
        
    # Do a search over the log of omega to improve minimization robustness. In
    # particular when changing between different material systems and dot
    # geometries and sizes.
    opt_result = minimize(fun=find_HO_wf_difference,
                             x0=np.log10(omega_guess), method='BFGS',
                             options={'gtol': opt_tol})
            
    # If optimzation was successful, return optimal omega and optimal basis
    if 'success' in opt_result.message:
        print(opt_result.message)
        opt_omega = 10**opt_result.x
        
        opt_HOs = build_HO_basis(gparams, opt_omega, nx, ny, ecc=ecc,
                                  consts=consts)
        
        return opt_omega, opt_HOs
    
    # If optimization failed, return None
    else:
        print(opt_result.message)
        
        return None, None

def build_HO_basis(gparams, omega, nx, ny=0, ecc=1.0,
                   consts=qd.Constants('vacuum')):
    '''
    Build of basis of 1D or 2D harmonic orbitals centered at the origin of the
    coordinate system.

    Parameters
    ----------
    gparams : GridParameters object
        Contains grid and potential information.
    omega : float
        Harmonic frequency of the harmonic orbital basis to build.
    nx : int
        Number of modes along the x direction to include in the basis set.
        
    Keyword Arguments
    -----------------
    ny : int, optional
        Number of modes along the y direction to include in the basis set. 
        Only applicable if gparams is for a '2D' system. The default is 0.
    ecc : float, optional
        Specify the eccentricity of the harmonic orbitals defined as 
        ecc = omega_y/omega_x. The default is 1.0 (omega_y = omega_x).
    consts : Constants object, optional
        Specify the material system constants when building the harmonic
        orbitals. The default assumes 'vacuum' as the material system.

    Returns
    -------
    HOs : array
        The constructed harmonic orbital basis where the first axis of the 
        array corresponds to a different harmonic orbital (HOs[n,:] for 1D and
        HOs[n,:,:] for 2D). If gparams describes a 2D grid then the harmonic
        orbitals are ordered first by y, then by x.

    '''
    
    # Initialize the array for storing all the created harmonic orbitals and
    # get corresponding harmonic confinements along x and y
    omega_x = omega
    # Used for shorter expressions when building harmonic orbitals
    alpha_x = np.sqrt(consts.me*omega_x/consts.hbar)
    if gparams.grid_type == '1D':
        HOs = np.zeros((nx, gparams.nx), dtype=complex)
    elif gparams.grid_type == '2D':
        omega_y = omega_x*ecc
         # Used for shorter expressions when building harmonic orbitals
        alpha_y = np.sqrt(consts.me*omega_y/consts.hbar)
        HOs = np.zeros((nx*ny, gparams.ny, gparams.nx), dtype=complex)
    

    # Construct all of the hermite polynomials we will use to build up the
    # full set of HOs. We will store each nth hermite polynomial to make this
    # more efficient while using the recursion formula to find higher order
    # polynomials.
    def _get_hermite_n(n, hermite_sub1, hermite_sub2, x_arg):
        '''
        Helper function for finding the n_th hermite polynomial IF the previous
        two nth polynomials are known.

        Parameters
        ----------
        n : int
            Specify which nth hermite polynomial currently being calculated.
        hermite_sub1 : array
            The H_{n-1} hermite polynomial (if applicable).
        hermite_sub2 : array
            The H_{n-2} hermite polynomial (if applicable).
        x_arg : array
            x-coordinates for the current hermite polynomial.

        Returns
        -------
        array
            The H_n hermite polynomial.

        '''
        
        # Base case 0
        if n == 0:
            return np.ones(x_arg.size)
        # Base case 1
        elif n == 1:
            return 2*x_arg
        # All other cases
        else:
            return 2*x_arg*hermite_sub1 - 2*(n-1)*hermite_sub2
    
    # Construct all the hermite polynomials which we will use to build up the
    # full set of HOs.   
    # x first
    x_hermites = np.zeros((nx, gparams.nx), dtype=complex)    
    for idx in range(nx):
        if idx == 0:
            x_hermites[idx,:] = _get_hermite_n(idx, [], [], alpha_x*gparams.x)
        elif idx == 1:
            x_hermites[idx,:] = _get_hermite_n(idx, [], [], alpha_x*gparams.x)
        else:
            x_hermites[idx,:] = _get_hermite_n(idx, x_hermites[idx-1,:],
                                              x_hermites[idx-2,:],
                                              alpha_x*gparams.x)
    # y now (if applicable)
    if gparams.grid_type == '2D':
        y_hermites = np.zeros((ny, gparams.ny), dtype=complex)  
        for idx in range(ny):
            if idx == 0:
                y_hermites[idx,:] = _get_hermite_n(idx, [], [], alpha_y*gparams.y)
            elif idx == 1:
                y_hermites[idx,:] = _get_hermite_n(idx, [], [], alpha_y*gparams.y)
            else:
                y_hermites[idx,:] = _get_hermite_n(idx, y_hermites[idx-1,:],
                                                  y_hermites[idx-2,:],
                                                  alpha_y*gparams.y)

    # Now that the hermite polynomials are built, construct the 1D harmonic
    # orbitals
    # x first
    x_HOs = np.zeros((nx, gparams.nx), dtype=complex)
    for idx in range(nx):
        # Build harmonic orbital
        coeff = 1/np.sqrt(2**idx*math.factorial(idx))*(alpha_x**2/math.pi)**(1/4)
        x_HOs[idx,:] = coeff*np.exp(-alpha_x**2*gparams.x**2/2)*x_hermites[idx,:]
        
    # y now (if applicable)
    if gparams.grid_type == '2D':
        y_HOs = np.zeros((ny, gparams.ny), dtype=complex)
        for idx in range(ny):
            # Build harmonic orbital
            coeff = 1/np.sqrt(2**idx*math.factorial(idx))*(alpha_y**2/math.pi)**(1/4)
            y_HOs[idx,:] = coeff*np.exp(-alpha_y**2*gparams.y**2/2)*y_hermites[idx,:]

    # If building for a 2D grid, build the 2D harmonic orbital states
    if gparams.grid_type == '1D':
        HOs = x_HOs
    elif gparams.grid_type == '2D':
        idx_cnt = 0 # Used for saving harmonic orbitals to HOs array
        for x_idx in range(nx):
            # Get current x harmonic orbital and convert to meshgrid format
            curr_x_HO = x_HOs[x_idx,:]
            curr_x_HO, _ = np.meshgrid(curr_x_HO,np.ones(gparams.ny))
            for y_idx in range(ny):
                # Get current y harmonic orbital and convert to meshgrid format
                curr_y_HO = y_HOs[y_idx,:]
                _, curr_y_HO = np.meshgrid(np.ones(gparams.nx),curr_y_HO)
                
                # Make 2D harmonic orbital
                HOs[idx_cnt,:,:] = curr_x_HO*curr_y_HO                
                idx_cnt += 1
                
    return HOs
        
# HELP WITH COMING UP WITH A SHORTER NAME 
# Also, this is a good example of a code that should be "generalized" to 
# handle a Hamiltonian class
def find_H_unitary_transformation(gparams, new_basis, 
                              consts=qd.Constants('vacuum'), unitary=True,
                              ortho_basis=False):
    '''
    This function takes an inputted Hamiltonian (gparams) and does a
    transformation into a new basis (new_basis).

    Parameters
    ----------
    gparams : GridParameters object
        Contains grid and potential information.
    new_basis : n-d array
        A multi-dimensional array corresponding to the basis we will transform
        the Hamiltonian into. The first dimension should correspond to the 
        index of each basis state.
        
    Keyword Arguments
    -----------------
    consts : Constants object, optional
        Specify the material system constants. The default is "vacuum".
    unitary : bool
        Specify if you want the unitary transformation that performs the basis
        transformation to be returned. Requires evaluation of the eigenvalue 
        probem. The default is True.
    ortho_basis : bool
        Specify if the new_basis is orthogonal or not. When False, we will 
        calculate the overlap matrix during the transformation.  When True, no
        overlap matrix is calculated. If you know ahead of time that your basis
        is orthogonal, specifying True can save some computation overhead by 
        not calculating the overlap matrix. The default is False.

    Returns
    -------
    ham_new : 2d array
        The Hamiltonian written in the new basis.
    U : 2d array
        The unitary transformation that yields U*H*U^-1 = H' where H is the
        original hamiltonian and H' is the hamiltonian in the new basis. This
        is only returned when unitary=True.

    '''
    
    # First thing is to build the Hamiltonian
    if gparams.grid_type == '1D':
        ham = qd.qutils.solvers.build_1DSE_hamiltonian(consts, gparams)
    elif gparams.grid_type == '2D':
        ham = qd.qutils.solvers.build_2DSE_hamiltonian(consts, gparams)
        
    # Intialize hamiltonian for the basis transformation
    n_basis_states = new_basis.shape[0]
    ham_new = np.zeros((n_basis_states,n_basis_states), dtype=complex)
    
    # Now rewrite Hamiltonian in new basis by evaluating inner products <i|H|j>
    # First upper triangular elements
    for i in range(n_basis_states):
        # Ket state
        state_R = new_basis[i,:]
        # Convert to NO if a 2D state
        if gparams.grid_type == '2D':
            state_R = gparams.convert_MG_to_NO(state_R)
            
        # Evaluate H|j>
        state_R = ham @ state_R
        
        # Convert back to MG if a 2D state
        if gparams.grid_type == '2D':
            state_R = gparams.convert_NO_to_MG(state_R)
            
        for j in range(i+1,n_basis_states):
            # Bra state
            state_L = new_basis[j,:]

            # Evaluate <i|H|j>
            ham_new[i,j] = qd.qutils.qmath.inner_prod(gparams, state_L, state_R)

    # Now lower triangular elements
    ham_new += ham_new.conj().T
    
    # Now diagonal elements
    for i in range(n_basis_states):
        # Ket and bra states
        state_R = new_basis[i,:]
        state_L = state_R
        
        # Convert ket to NO if a 2D state
        if gparams.grid_type == '2D':
            state_R = gparams.convert_MG_to_NO(state_R)
            
        # Evaluate H|i>
        state_R = ham @ state_R
        
        # Convert back to MG if a 2D state
        if gparams.grid_type == '2D':
            state_R = gparams.convert_NO_to_MG(state_R)

        # Evaluate <i|H|i>
        ham_new[i,i] = qd.qutils.qmath.inner_prod(gparams, state_L, state_R)
            
    # Correct any numerical issues and force Hamiltonian to be hermitian
    ham_new = (ham_new + ham_new.conj().T)/2;
    
    # Now calculate the unitary transformation to convert H -> H' (if desired)
    # U*H*U^-1 = H'
    if unitary is False:
        return ham_new
    else:
        # If basis is declared to be orthogonal, assuming overlap matrix
        # is identity, otherwise calculate it.
        if ortho_basis is True:
            eig_ens, U = eig(ham_new)
        else:
            S_matrix = qd.qutils.qmath.find_overlap_matrix(gparams, new_basis,
                                                           new_basis)
            eig_ens, U = eig(ham_new, S_matrix)

    # Sort unitary by eigenvalue
    U = U[:,eig_ens.argsort()]

    return ham_new, U
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        