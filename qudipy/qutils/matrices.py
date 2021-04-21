"""
Constant matrices used to define quantum gates

@author: hromecB
"""
import numpy as np

#constant Pauli 2x2 matrices

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, complex(-0.0, -1.0)], [complex(0.0, 1.0), 0]], 
                               dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)

def x(N, k):
    """
    Creates matrix X_k of dimensions 2**N x 2**N 
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k : int
        Position of the Pauli X matrix in the tensor product.
    
    Keyword Arguments
    -----------------
    None.

    Returns
    -------
    x_k: 2D complex array
        Matrix X_k of dimensions 2**N x 2**N 

    """
    if k == 1:
        return np.kron(PAULI_X, np.eye(2 ** (N - 1)))
    x_k = PAULI_I
    for m in range(2, N + 1):
        if k is m:
            x_k = np.kron(x_k, PAULI_X)
        else:
            x_k = np.kron(x_k, PAULI_I)
    else:
        return x_k


def y(N, k):
    """
    Creates matrix Y_k of dimensions 2**N x 2**N 
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k : int
        Position of the Pauli Y matrix in the tensor product.
        
    Keyword Arguments
    -----------------
    None.

    Returns
    -------
    y_k: 2D complex array
        Matrix Y_k of dimensions 2**N x 2**N 

    """
    if k == 1:
        return np.kron(PAULI_Y, np.eye(2 ** (N - 1)))
    y_k = PAULI_I
    for m in range(2, N + 1):
        if k is m:
            y_k = np.kron(y_k, PAULI_Y)
        else:
            y_k = np.kron(y_k, PAULI_I)
    else:
        return y_k


def z(N, k):
    """
    Creates matrix Z_k of dimensions 2**N x 2**N 
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k : int
        Position of the Pauli Z matrix in the tensor product.
   
    Keyword Arguments
    -----------------
    None.

    Returns
    -------
    z_k: 2D complex array
        Matrix Z_k of dimensions 2**N x 2**N 

    """
    if k == 1:
        return np.kron(PAULI_Z, np.eye(2 ** (N - 1)))
    z_k = PAULI_I
    for m in range(2, N + 1):
        if k is m:
            z_k = np.kron(z_k, PAULI_Z)
        else:
            z_k = np.kron(z_k, PAULI_I)
    else:
        return z_k

#ladder operators X_k ± i Y_k

def sigma_plus(N, k):
    """
    Defines a raising operator of the k-th qubit

    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k : int
        Position of the raising operator in the tensor product.

    Returns
    -------
    : complex 2D array
        The raising operator X_k + i Y_k
    """
    return x(N, k) + complex(0.0, 1.0) * y(N, k)


def sigma_minus(N, k):
    """
    Defines a lowering operator of the k-th qubit

    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k : int
        Position of the raising operator in the tensor product.

    Returns
    -------
    : complex 2D array
        The lowering operator X_k - i Y_k
    """
    return x(N, k) - complex(0.0, 1.0) * y(N, k)
    
def e_up(N, k):
    """
    Defines matrix that projects k-th qubit on the state |0〉
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k : int
        Position of the projection matrix in the tensor product.
        
    Keyword Arguments
    -----------------
    None.
    
    Returns
    -------
    : 2D complex array
        Matrix |0〉〈0|_k of dimensions 2**N x 2**N 

    """
    return 0.5 * (unit(N) + z(N, k))


def e_down(N, k):
    """
    Defines matrix that projects k-th qubit on the state |1〉
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k : int
        Position of the projection matrix in the tensor product.
        
    Keyword Arguments
    -----------------
    None.
    
    Returns
    -------
    : 2D complex array
        Matrix |1〉〈1|_k of dimensions 2**N x 2**N 
    """
    return 0.5 * (unit(N) - z(N, k))


def unit(N):
    """
    Defines unit matrix of dimensions 2**N x 2**N
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
        
    Keyword Arguments
    -----------------
    None.
    
    Returns
    -------
    : 2D complex array
       Unit matrix of dimensions 2**N x 2**N
    """
    
    return np.eye((2 ** N), (2 ** N), dtype=complex)


def cnot(N, ctrl, trgt):
    """
    Defines a matrix for CNOT gate.
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    ctrl: int
        Index of the control qubit.
    trgt: int
        Index of the target qubit.
        
    Keyword Arguments
    -----------------
    None.

    Returns
    -------
    : 2D complex array
        Matrix for CNOT gate

    """
    return e_up(N, ctrl) + e_down(N, ctrl) @ x(N, trgt)


def swap(N, k1, k2):
    """
    Defines SWAP gate matrix for the qubits with the indices k1, k2.
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k1, k2: int
        Indices of the qubits.

    Keyword Arguments
    -----------------
    None.

    Returns
    -------
    : 2D complex array
        Matrix for SWAP gate 

    """
    return cnot(N, k1, k2) @ cnot(N, k2, k1) @ cnot(N, k1, k2)


def sigma_product(N, k1, k2):
    """
    Defines the dot product of two Pauli vectors.
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k1, k2: int
        Indices of the qubits.

    Keyword Arguments
    -----------------
    None.

    Returns
    -------
    : 2D complex array
        The inner product \vec{sigma_k1} \cdot \vec{sigma_k2}

    """
    return x(N, k1) @ x(N, k2) + y(N, k1) @ y(N, k2) + z(N, k1) @ z(N, k2)


def rswap(N, k1, k2):
    """
    Defines sqrt(SWAP) gate matrix for the qubits with the indices k1, k2.
    
    Parameters
    ----------
    N : int
        Number of 1-qubit degrees of freedom in the operator.
    k1, k2: int
        Indices of the qubits.

    Keyword Arguments
    -----------------
    None.

    Returns
    -------
    : 2D complex array
        Matrix for SWAP gate 

    """
    return (complex(0.25, -0.25) * sigma_product(N, k1, k2) 
                                                + complex(1.5, 0.5) * unit(N))