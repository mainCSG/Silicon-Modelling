__all__ = [
    'inner_prod','partial_trace', 'project_up', 'project_down', 'matrix_sqrt', 
        'partial_trace_general' # , 'fidelity', 'purity'
        ]

import numpy as np
import math
from numpy import linalg as la

def inner_prod(gparams, wf1, wf2):
    '''
    Evaluates the inner product between two complex wavefunctions.

    Parameters
    ----------
    gparams : GridParameters class
        Contains grid and potential information.
    wf1 : complex array
        'Bra' wavefunction for the inner product. If grid is 2D, then the 
        array should be in meshgrid format.
    wf2 : complex array
        'Ket' wavefunction for the inner product. If grid is 2D, then the 
        array should be in meshgrid format.
        
    Keyword Arguments
    ----------------
    None.

    Returns
    -------
    inn_prod : complex float
        The inner product between the two inputted wavefunctions.

    '''
    
    if gparams.grid_type == '1D':
        inn_prod = np.trapz(np.multiply(wf1.conj(),wf2), x=gparams.x)
    elif gparams.grid_type == '2D':
        inn_prod = np.trapz(np.trapz(np.multiply(wf1.conj(), wf2), 
                                     x=gparams.x, axis=1), gparams.y)
            
    return inn_prod

   
def project_up(rho, elem):
    """
    Projects the system density matrix onto the spin-up state of 
    the specified electron(s) (without renormalizing it)
    
    Parameters
    ----------
    rho : 2D complex array
        Matrix of the dimensions 2**N x 2**N (density matrix in our case).
    elem : int / iterable of ints
        Number(s) of the electron(s) whose state(s) is to project the system 
        density matrix on.
        
    Keyword Arguments
    ----------------
    None.
    
    Returns
    -------
    projected_rho : complex 2D array
        new density matrix (non-renormalized) of the dimensions 
        2**m x 2**m, where m = N - dim(elem) 

    """
    dim = rho.shape[0]    #dimension of the matrix
    N = int(math.log2(rho.shape[0]))   
            #number of qubits encoded by density matrix
    
    if isinstance(elem, int):
        
        projected_rho=[]
            #the idea behind bitwise shifts is that every matrix in tensor product
            #doubles the dimension of the resulting matrix, i. e. adds one binary 
            #digit to the number. It follows from the definition of Kronecker
            #product that the most significant bit defines the element of the
            #leftmost matrix in Kronecker product, and vice versa
        for i in range(0, dim):
            if (i >> (N - elem)) % 2 == 0:
                var = []
                for j in range(0, dim):
                    if (j >> (N - elem)) % 2 == 0:
                        var.append(rho[i][j])
                projected_rho.append(var)
            else:
                continue
    
        return  np.array(projected_rho)
    
    elif isinstance(elem, (tuple, list, set) ):
        projected_rho = rho.copy()
        elems_sorted = sorted(elem, reverse=True)
            #iterable sorted in the reversed order; necessary for the correct 
            # consecutive application of the project_up operations 
        for el in elems_sorted:
            projected_rho = project_up(projected_rho, el)
        return projected_rho
    
    else:
        print("Qubits that are traced out should be defined by a single int \
              number or an iterable of ints. Try again")
            

def project_down(rho, elem):
    """
    Projects the system density matrix onto the spin-down state 
    of the specified electron(s) (without renormalizing it).
    
    Parameters
    ----------
    rho : 2D array
        Matrix of the dimensions 2**N x 2**N (density matrix in our case).
    elem : int / iterable of ints
        Number(s) of the electron(s) whose state(s) is to project the system 
        density matrix on.
        
    Keyword Arguments
    ----------------
    None.
    
    Returns
    -------
    projected_rho : complex 2D array
        new density matrix (non-renormalized) of the dimensions 
        2**m x 2**m, where m = N - dim(elem) 
    """

    dim = rho.shape[0]    #dimension of the matrix
    N = int(math.log2(rho.shape[0]))   
            #number of qubits encoded by density matrix
    
    if isinstance(elem, int):
        
        projected_rho = []
            #the idea behind bitwise shifts is that every matrix in tensor product
            #doubles the dimension of the resulting matrix, i. e. adds one binary 
            #digit to the number. It follows from the definition of Kronecker
            #product that the most significant bit defines the element of the
            #leftmost matrix in Kronecker product, and vice versa
        for i in range(0, dim):
            if (i >> (N - elem))%2 == 1:
                var = []
                for j in range(0, dim):
                    if (j >> (N - elem)) % 2 == 1:
                        var.append(rho[i][j])
                projected_rho.append(var)
            else:
                continue
    
        return np.array(projected_rho)
    
    elif isinstance(elem, (tuple,list,set)):
        projected_rho = rho.copy()
        elems_sorted = sorted(elem, reverse=True)
            #iterable sorted in the reversed order; necessary for the correct 
            #consecutive application of the project_down operations 
        for el in elems_sorted:
            projected_rho = project_down(projected_rho, el)
        return projected_rho
    
    else:
        print("Qubits that are traced out should be defined by a single int \
              number or a tuple of ints. Try again")


def partial_trace(rho, elem):
    """
    Finds partial trace with respect to the specified qubit(s)
    
    Parameters
    ----------
    rho : 2D array
        matrix of the dimensions 2**N x 2**N (density matrix in our case)
    elem : int / iterable of ints
        number(s) of the electron(s) whose state(s) is/are averaged out

    Returns
    -------
    traced_rho: complex 2D array
        new density matrix of the dimensions 2**m x 2**m, m = N - dim(elem) 

    """
    if isinstance(elem, int):
        return project_up(rho, elem) + project_down(rho, elem)
    
    elif isinstance(elem, (tuple, set, list)):
        traced_rho = rho.copy()
        elems_sorted = sorted(elem, reverse=True)
            #iterable sorted in the reversed order; necessary for the correct 
            # consecutive application of the project_up operations 
        for el in elems_sorted:
            traced_rho = partial_trace(traced_rho, el)
        return traced_rho
    else:
        #error with the input 
        raise ValueError("Qubits that are traced out should be defined by a  \
                         single int number or an iterable of ints. Try again")

def matrix_sqrt(A):
    """
    Calculates a square root of the specified matrix.

    Parameters
    ----------
    A : numpy array
        matrix to calculate the square root of

    Returns
    -------
    sqrt_A: complex 2D array
        square root of the matrix

    """
    w,v = la.eig(A)
    sqrt_A = v @ np.diag(np.sqrt((1 + 0j)*w)) @ la.inv(v)
    return sqrt_A
    


def partial_trace_general(rho, dim, traced_subsystem):
    '''
    This code takes the partial trace of a density matrix.  It is adapted from
    the open-source TrX file written by Toby Cubitt for MATLAB.
     
    Parameters
    ----------
    rho : complex 2D array
        A 2D array describing a density matrix.
    dim : 1D integer array
        An array of the dimensions of each subsystem for the whole system for
        psi. [2,4,2] corresponds to a system of size 2x4x2 with 3 subsystems.
    traced_subsystem : 1D array
        An array of the subsystems to trace out. For dim=[2,4,2], sys=[1,3] 
        would trace out the first and third subsystems leaving only a subspace
        with size 4.
        
    Keyword Arguments
    -----------------
    None.

    Returns
    -------
    traced_rho : complex 2D array
        The resulting rho after taking the partial trace.
        
    '''
    # Convert inputs to numpy arrays if not already inputted as such.
    rho = np.array(rho)
    traced_subsystem = np.array(traced_subsystem) - 1  #-1 is added in order 
        # to accommodate conventional numbering rule
        # (starting with 1) on the user's side
    dim = np.array(dim)
    
    # Check inputs
    if any(traced_subsystem > len(dim)-1) or any(traced_subsystem < 0):
        print(traced_subsystem > len(dim))
        print(any(traced_subsystem < 0))
        # Error with sys variable
        raise ValueError("Invalid subsytem in traced_qubits.")
    
    #if ((len(dim) == 1 and np.mod(len(rho)/dim,1) != 0) 
    #    and len(rho) != np.prod(dim)):
        
    if (rho.shape[0] != np.prod(dim) or rho.shape[1] != np.prod(dim)):
        # Error with dim or psi variables
        raise ValueError("Size of rho inconsistent with dim.")
    
    # Get rid of any singleton dimensions in dim
    traced_subsystem = np.setdiff1d(traced_subsystem, np.argwhere(dim == 1))
    dim = np.concatenate([dim[idx] for idx in np.argwhere(dim != 1)])
    
    # Number of subsystems
    n = len(dim)
    dim_reversed = dim[::-1]
    subsystems_keep = np.array([idx for idx in np.linspace(0, n - 1, n) 
                                if idx not in traced_subsystem])
    # Dimension of psi to trace out
    dim_trace = dim[traced_subsystem].prod()
    # Dimension of psi leftover after partial trace
    dim_keep = len(rho)/dim_trace
    
    # Reshape density matrix into tensor with one row and one column index for
    # each subsystem, permute traced subsytem indices to the end, reshape
    # again so that first two indices are row and column multi-indices for
    # kept subsystems and third index is a flatted index for traced subsytems,
    # then sum third index over "diagonal" entries.
    perm = n-1 - np.concatenate([subsystems_keep[::-1], 
                                    subsystems_keep[::-1]-n,
                                       traced_subsystem, traced_subsystem - n])
    rho1 = np.reshape(rho, np.concatenate([dim_reversed, dim_reversed]),
                      order='F')
    rho2 = np.transpose(rho1, perm.astype(int))
    rho = np.reshape(rho2, np.array([dim_keep, dim_keep, dim_trace ** 2],
                                    dtype=np.uint), order='F')
    traced_rho = np.sum(rho[:,:,range(0, dim_trace ** 2,dim_trace + 1)], axis=2)
            
    return traced_rho



def fidelity(rho, rho_reference):
    """
    Calculates fidelity of the system density matrix with respect to 
    the reference one

    Parameters
    ----------
    rho : 2D complex array
        The system density matrix
    rho_reference : 2D complex array
        The density matrix (initial, anticipated final, etc.) to compare 
        with the system density matrix

    Keyword Arguments
    ------------------
    None.
    
    Returns
    -------
    fid : float
        Value of fidelity defined as in the write-up, see "Spin simulator"
        chapter: https://www.overleaf.com/3252553442tbqcmxntqvtk.

    """  
    fid = np.real(np.trace(matrix_sqrt(
                             matrix_sqrt(rho_reference) @ rho
                                 @ matrix_sqrt(rho_reference))) ** 2)
    return fid


def purity(rho):
    """
    Calculates purity of the density matrix tr(rho^2).

    Parameters
    ----------
    rho : 2D complex array
        The system density matrix
    
    Keyword Arguments
    ------------------
    None.
    
    Returns
    -------
    pur : float
        value of purity defined as tr(rho^2).

    """
    pur = np.real(np.trace(rho @ rho))
    return pur




