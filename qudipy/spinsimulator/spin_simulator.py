# -*- coding: utf-8 -*-
"""

Script for calculating spin evolution.

I strongly recommend that you read the "Spin simulator" chapter of the 
write-up **before** familiarizing yourself with this code:
    https://www.overleaf.com/project/5f4fcbdd5566fb0001f3b6aa

@author: bkhromet
"""

import numpy as np

from math import pi, log2, exp, sin, cos, inf

import warnings
import tqdm  #for progress bar

import qudipy.qutils.matrices as matr
import qudipy.qutils.math as qmath

from qudipy.circuit import ControlPulse
from qudipy.utils.constants import Constants 


#material system is chosen to be vacuum by default because such parameters as 
#effective mass or dielectric constant do not matter for spin simulations;

consts = Constants("vacuum")       

#helper functions

def p(B_0, T):
    """
    Calculates the population of the spin-up state in the system with strong
    constant magnetic field.
    
    Parameters
    ----------
    B_0 : float
        Zeeman field [T].
    T : float
        Temperature [K].
      
    Keyword Arguments
    -----------------
    None.
    
    Returns
    -------
    p_ : float
        Population of the spin-up state.

    """
    p_ = 0
    if T > 1e-2:
        p_ = 1 / (exp (2 * consts.muB * B_0 / (consts.kB * T)) + 1)
    return p_

    

def J_sigma_product(N, k1, k2):
    """
    Matrix coupled to the exchange parameter J_{k_1, k_2}. Takes into account
    that there is no exchange with itself.
    
    Parameters
    ----------
    N : int
        Number of electrons in the system.
    k1, k2 : int
        Indexes of the sigma matrices in the tensor product
        (numbered from 1 to N)
        
    Keyword Arguments
    -----------------
    None.
    
    Returns
    -------
    j_sigma_product_ : 2D numpy array
        The dot product of two Pauli vectors divided by 4\hbar .
    """ 
    
    j_sigma_product_ = np.zeros((2 ** N, 2 ** N))
    if k1 != k2:
         j_sigma_product_ = matr.sigma_product(N, k1, k2) / (4 * consts.hbar)  
    return j_sigma_product_ 
    
def x_sum(N):
    """
    Sum highligted with green in equation 1.2.1 of the write-up. Used to 
    reduce the total number of multiplications during the time evolution 
    
    Parameters
    ----------
    N : int
        Number of electrons in the system.
        
    Keyword Arguments
    -----------------
    None.
    
    Returns
    -------
    : 2D complex array
        Sum of X_k-matrices for all k in [1,N] weighted by
        consts.muB/consts.hbar .

    """
    return consts.muB / consts.hbar * sum(matr.x(N, k) for k in range(1, N+1))


def y_sum(N):
    """
    Sum highligted with cyan in equation 1.2.1 of the write-up. Used to 
    reduce the total number of multiplications during the time evolution 
    
    Parameters
    ----------
    N : int
        Number of electrons in the system.
        
    Keyword Arguments
    -----------------
    None.
    
    Returns
    -------
    : 2D complex array
        Sum of Y_k-matrices for all k in[1,N] weighted 
        by consts.muB / consts.hbar.

    """
    return consts.muB / consts.hbar * sum(matr.y(N, k) for k in range(1, N+1))


def z_sum_omega(N, B_0, f_rf):
    """
    Sum highligted with brown in equation 1.2.1 of the write-up. Used to 
    reduce the total number of multiplications during the time evolution 
    simulation.
    
    Parameters
    ----------
    N : int
        Number of electrons in the system.
    B_0 : float
        Zeeman field [T].
    f_rf : float
        Frequency of the ESR field [Hz].

    Keyword Arguments
    -----------------
    None.
    
    Returns
    -------
    : 2D complex array
        Sum of Z_k-matrices for all k in [1,N] 
        weighted by i(omega - omega_rf)/2.

    """
    return ((consts.muB * B_0 / consts.hbar - pi * f_rf)
                * sum(matr.z(N, k) for k in range(1, N+1)))


def z_sum_p(N, B_0, T, T_1):
    """
    Sum highligted with red in equation 1.2.1 of the write-up. Used to 
    reduce the total number of multiplications during the time evolution 
    simulation.
    
    Parameters
    ----------
    N : int
        number of electrons in the system
    B_0 : float
        Zeeman field [T].
    T : float
        Temperature [K].
    T_1 : float
        Spin relaxation time.
    
    Keyword Arguments
    -----------------    
    None.
    
    Returns
    -------
    : 2D complex array
        Sum of Z_k-matrices for all k in [1,N] weighted by (2*p(B_0,T)-1) / T_1


    """
    return (2 * p(B_0, T) - 1) / T_1 * sum(matr.z(N, k) for k in range(1, N+1))
    

#list of dictionaries of constant matrices


def const_dict(N_0, T, B_0, f_rf, T_1):
    """
    List of dictionaries of all constant matrices.
    
    Parameters
    ----------
    N_0 : int
        The maximal number of quantum dots in the system that host electrons.
    B_0 : float
        Zeeman field [T].
    f_rf : float
        Frequency of the ESR field [Hz].
    T : float
        Temperature [K].
    T_1 : float
        Spin relaxation time.
    
    Keyword Arguments
    -----------------
    None.
    

    Returns 
    -------
    const_dict_: list of dicts
        The list of all constant matrices needed for an efficient computation.
        Entries of each dictionary:
        - "Xs" - list of X_k
        - "Ys" - list of Y_k
        - "Zs" - list of Z_k
        - "sigma_pluses" - array of matr.sigma_plus_k_l
        - "sigma_minuses" - list of matr.sigma_minus_k
        - "J_sigma_products" - (symmetric) matrix of 
            (\frac{1}{4 * \hbar} \vec{sigma_k1} \cdot \vec{sigma_k2})
        - "x_sum" (multiplied by consts.muB/consts.hbar)
        - "y_sum" (multiplied by consts.muB/consts.hbar)
        - "z_sum_omega"
        - "Z_sum_p"
        The functions that give entries of the dictionaries are defined above. 
        Please see the write-up for the explanation of them (they are 
        highlighted with different colors in the write-up).        
        Each dictionary corresponds to a different dimensionality of the system 
        (from 1 to N_0)
    """
    const_dict_ = []
    for N in range(1,N_0+1):
        
        Xs = [matr.x(N, k) for k in range(1, N+1)]
        Ys = [matr.y(N, k) for k in range(1, N+1)]
        Zs = [matr.z(N, k) for k in range(1, N+1)]
        
        sigma_pluses = [matr.sigma_plus(N, k) for k in range(1, N+1)]
        sigma_minuses = [matr.sigma_minus(N, k) for k in range(1, N+1)]
        
        J_sigma_products = [[J_sigma_product(N, k1, k2) for k2 in 
                                     range(1, N+1)] for k1 in range(1, N+1)] 
                       
        const_dict_.append({"Xs":Xs, "Ys":Ys, "Zs":Zs,  
                                "sigma_pluses":sigma_pluses, 
                                "sigma_minuses":sigma_minuses, 
                                "J_sigma_products":J_sigma_products,
                                "x_sum":x_sum(N), "y_sum":y_sum(N), 
                                "z_sum_omega":z_sum_omega(N, B_0, f_rf), 
                                "z_sum_p":z_sum_p(N, B_0, T, T_1)})
    
    return const_dict_


class SpinSys:
    """
    The class describes the spin system. Attributes comprise the system
    density matrix and constant system parameters. Methods construct 
    Hamiltonian  and time-evolve the system with pulses.
    
    Attributes
    ----------
    rho : 2D complex array
        Density matrix. Initialized with the value before pulse
            
    N_0 : int
        Maximal possible number of electrons in the system (could be bigger
            than the dimensions of rho; useful in case we load new electrons. 
    time: float
        Point in time [s] at which the system is defined.
    T_1: float
        Spin relaxation time [s].
    T_2: float
        Spin dephasing time [s].
    B_0 : float
        Zeeman field [T].
    T : float
        Temperature [K].
    f_rf: float
        RF field frequency [Hz] (optionally passed in sys_param_dict)
    cdict: dictionary of 2D complex arrays
        Collection of costant matrices used to speed up the simulation. Have
        all the dimensinons from 2 x 2 up to 2**N_0 x 2**N_0.
        
    Methods
    ----------
    hamiltonian(cdict_N, pulse_params=None):
        Creates system Hamiltonian at a particular point of time
    lindbladian(rho_mod, const_dict_N, pulse_params=None):
        Calculates the right-hand side of the Lindblad equation
        at a particular point of time
    evolve(pulse, rho_reference=None, eval_fidelity=False, eval_purity=False,
               track_qubits=None, eval_Bloch_vectors=False, 
                   track_points_per_pulse=100):
        Performs system evolution under a given pulse. Outputs a dictionary of
        the specified tracking parameters
    track_subsystem(self, track_qubits, eval_Bloch_vectors=False):
        Gives the specified system submatrices and Bloch vectors (if tracked)
  
    """
        
    def __init__(self, state, N_0=None, time=0, T_1=inf, T_2=inf, 
                     B_0=0, T=0, f_rf=None):
        """
        Initializes the SpinSys class object.
        
        Parameters
        ----------
        state : complex array or a tuple/list of such objects
            Wavefunction, density matrix, or list of them corresponding to 
            different qubits defining a system. The density matrix of
            the whole system is created from this.
        
        Keyword Arguments
        -----------------    
        N_0 : int, optional
            Maximal possible number of electrons in the system (could be bigger
            than the dimensions of rho). By default, N_0 is assumed to be the
            dimension of the specified rho: dim(rho) = 2**N_0 x 2**N_0.
        time: float, optional
            Point in time [s] at which the system is defined. The default is 0.
        T_1: float, optional
            Spin relaxation time [s]. The default is infinity.
        T_2: float, optional
            Spin dephasing time [s]. The default is infinity.
        B_0 : float, optional
            Zeeman field [T]. The default is zero.
        T : float, optional
            Temperature [K]. The default is zero.
        f_rf: float, optional
            The RF field frequency [Hz]. By default, it is equated to
            the Larmor frequency calculated from B_0.
   
        Returns
        -------
        None.

        """
        def __is_power_2(num):
            """
            Hidden function that checks if the integer number is an integer
            power of 2 

            Parameters
            ----------
            num : int
                Integer to be checked.

            Returns
            -------
            ispower2 : bool
                True if num is a power of 2, False otherwise

            """
            while (num % 2 == 0):  
                num = num / 2
            return num == 1
        
        # checking the correctness of input
        err_msg = "The state is defined incorrectly. Please try again"
        if isinstance(state, list):
            stat = state.copy()
        elif isinstance(state, tuple):
            stat = list(state)
        elif isinstance(state, np.ndarray): 
            stat = [np.copy(state),]
        else: 
            raise ValueError(err_msg)
            
        if (isinstance(stat, (tuple, list))
                    and all(isinstance(entry, np.ndarray) for entry in stat)):
            for i in range(len(stat)):    
                # making all entries 2D arrays
                if stat[i].ndim == 1:
                    stat[i] = np.array([stat[i]])
                elif stat[i].ndim != 2:
                    raise ValueError(err_msg)
                    
                #transforming all wavefunctions into density matrices
                if np.shape(stat[i])[0] == 1:
                    stat[i] = np.kron(stat[i].T, stat[i].conj())
                elif np.shape(stat[i])[1] == 1:
                    stat[i] = np.kron(stat[i], stat[i].T.conj())
                if not (np.shape(stat[i])[0] == np.shape(stat[i])[1] and 
                           __is_power_2(np.shape(stat[i])[0])):
                    raise ValueError(err_msg)
        else:       
            raise ValueError(err_msg)
            
        #creating the density matrix of the whole system
        rho = np.array([[1]])
        for entry in stat:
            rho = np.kron(rho, entry)

        self.rho = rho
        self.time = time
        
        if N_0 is None: 
            #assuming the current dimensionality of rho is maximum possible
            #throughout the system evolution
            self.N_0 = int(log2(self.rho.shape[0]))
        else:
            self.N_0 = N_0
        
        self.T_1 = T_1
        self.T_2 = T_2
        self.B_0 = B_0
        self.T = T
        
        if f_rf is None:
            self.f_rf = 2 * consts.muB * B_0 / consts.h
        else: 
            self.f_rf = f_rf
        
        self.cdict = const_dict(self.N_0, self.T, self.B_0, self.f_rf, 
                                                                    self.T_1)
                
        #methods
        
    def hamiltonian(self, const_dict_N, pulse_params=None):
        """
        Builds the system Hamiltonian at a particular point of time.

        Parameters
        ----------
        const_dict_N: dictionary
            Dictionary of constant matrices 2**N x 2**N
        
        Keyword Arguments
        -----------------
        pulse_params : dictionary, optional
             Dictionary of values of delta_g[i], 
             J[i], B_rf taken at a particular point of time. 
             The default is None interpreted as 'no pulse'

        Returns
        -------
        ham: 2D complex array
            Array that represents the Hamiltonian.

        """
        ham = const_dict_N["z_sum_omega"].copy()
                #copy is in order not to modify the const_dict_N entries; 
                #the term z_sum_omega is always present in the Hamiltonian
            
        if pulse_params is not None:
            #deducing N from const_dict_N
            N = int(log2(const_dict_N["x_sum"].shape[0])) 
            
            # building a list of delta_g_i values, i=1...N           
            delta_g=[0] * N 
            delta_g_dict = {int(k[8:]):v for (k,v) in pulse_params.items() 
                     if k[:8]=="delta_g_"}      

            for i in range(1, N+1):
                delta_g[i-1] = delta_g_dict.get(i, 0)
            
            # building a list of J_{i,i+1} values, i=1...N-1. For now, we  
            # assume that the qubits are numbered sequentially so that all 
            # nearest-neighbor exchange parameters could be indexed with
            # a single index i
            
                J = []
                J_dict = {int(k[2:]):v for (k,v) in pulse_params.items() 
                         if k[:2]=="J_"}
                for i in range(N-1):
                    J.append(J_dict.get(i+1, 0))
            
            #Incorporating values of B_rf
                    
            B_rf = pulse_params.get("B_rf", 0)
            phi = pulse_params.get("phi", 0)
            
            if delta_g != [0] * N:
                for k in range(N):
                    ham += (consts.muB * self.B_0 * delta_g[k]
                                * const_dict_N["Zs"][k]) / (2 * consts.hbar)
                
            if J != [0]*(N-1):
                for k in range(N-1):
                    ham += J[k] * const_dict_N["J_sigma_products"][k][k+1]
            
            if B_rf != 0:
                ham += const_dict_N["x_sum"]* B_rf * cos(pi * phi / 180)
                ham += const_dict_N["y_sum"]* B_rf * sin(pi * phi / 180)
              
        return ham
    

    def lindbladian(self, rho_mod, const_dict_N, pulse_params=None):
        """
        Creates right-hand side of the Lindblad equation at a particular
        point of time

        Parameters
        ----------
        rho_mod: 2D complex array 
            The intermediate density matrix that the
            Runge-Kutta method is used for.
        const_dict_N: dictionary
            Dictionary of constant matrices 2**N x 2**N.

        Keyword Arguments
        -----------------
        pulse_params : dictionary, optional
             Dictionary of values of delta_g[i], J[i], B_rf taken at a 
             particular point of time. 
             The default is None interpreted as 'no pulse'.

        Returns
        -------
        lin : 2D complex array
            the right-hand side of the Lindblad equation

        """
        N = int(log2(const_dict_N["x_sum"].shape[0])) 
            #automatically calculating the size
        
        ham = np.zeros((2**N, 2**N), dtype=complex)
        if pulse_params is not None:
            ham  += self.hamiltonian(const_dict_N, pulse_params)
        
        lin = ( 1j *( rho_mod @ ham - ham @ rho_mod) + const_dict_N["z_sum_p"] 
               - (2/self.T_1 + 1/(2*self.T_2)) * N * rho_mod )

        for k in range(N):
            lin  += (  p(self.B_0, self.T) / self.T_1 * 
                    const_dict_N["sigma_pluses"][k] @ 
                    rho_mod.dot(const_dict_N["sigma_minuses"][k]) +  
                    (1 - p(self.B_0, self.T)) / self.T_1 * 
                    const_dict_N["sigma_minuses"][k] @
                    rho_mod.dot(const_dict_N["sigma_pluses"][k]))
 
        return lin
        
    
    def evolve(self, pulses, 
               rho_reference=None, eval_fidelity=False, eval_purity=False,
               track_qubits=None, eval_Bloch_vectors=False,
                track_points_per_pulse=100
               ):
        """
        Function that performs spin system evolution under the external pulse,
        i.e. updates the system density matrix.

        Parameters
        ----------
        pulses : ControlPulse object, or tuple/list of such objects
            Contains pulse parameters at different points of time. The tuple/
            list of pulses could be specified as an irregular array like
            [pulse1, pulse2, [pulse3, (pulse4, pulse5)], pulse6], which is 
            convenient for dealing with compound pulses. 
                 
        Keyword Arguments
        -----------------
        rho_reference: 2D complex array, optional
            The density matrix (initial, anticipated final, etc.) to compare 
            with the density matrix during the simulation. If specified, the
            fidelity of system density matrix with respect to it can be 
            calculated. The default is None.
        eval_fidelity : bool, optional
            Indicates whether to track the fidelity of the density matrix with 
            respect to the specified rho_reference (if it is specified). 
            The default is False.
        eval_purity : bool, optional
            Indicates whether to track the purity
            of the density matrix. 
            The default is False.   
        track_qubits : int / iterable of ints, optional
            Index (indices) of qubit(s) to track during the pulse(s) 
            (i. e. evaluate 1-qubit density submatrices for them).
            The default is None.            
        eval_Bloch_vectors: bool, optional
            Indicates whether to calculate Bloch vectors for the tracked 
            qubits along with their density submatrices.
            The default is False.            
        track_points_per_pulse: int, optional
            Number of points of time during the simulation to save and
            the tracked parameters specified above and then output them.
            The default is 100. 
            
        Returns
        -------
        ret_dict: dictionary of numpy arrays
            Includes some of the following pairs key-value (the length of each 
            array is given by track_points_per_pulse variable):
            - "time": 1D float array 
                  Points of time that correspond to the values contained in 
                  the other arrays defined below.
            - "purity": 1D float array
                  Purity of the system density matrix (if eval_purity is True).
            - "fidelity": 1D float array
                  Fidelity of the system density matrix with respect to 
                  rho_reference (if it is defined, and if eval_fidelity is True).
            - "submatrix_{i}": 1D array of complex 2D arrays
                  Density submatrix of the i-th electron, if i is in 
                  track_qubits iterable.
            - "sigma_x_{i}" : 1D float array
              "sigma_y_{i}" : 1D float array
              "sigma_z_{i}" : 1D float array
                  Bloch vector components of the i-th electron, if i
                  is in track_qubits iterable, and eval_Bloch_vectors is True.

        """
        ret_dict = {}    #dictionary to be returned 
              
        #calculating the values at the beginning of the evolution
        if eval_fidelity:
            if rho_reference is not None:
                rho_ref = rho_reference.copy()
                    #copied just in case to keep the 
                    #passed reference matrix unchanged
                ret_dict["fidelity"]=[qmath.fidelity(self.rho, rho_ref)]
            else:
                print("The reference matrix is not specified. "
                      "Fidelity has not been calculated")
                    
        if eval_purity:
            ret_dict["purity"] = [qmath.purity(self.rho)]
            
        if track_qubits is not None:
            track_dict = self.track_subsystem(
                track_qubits, eval_Bloch_vectors)
            track_keys = track_dict.keys()
            ret_dict.update({k:[track_dict[k]] for k in track_keys})
        elif eval_Bloch_vectors:
            warnings.warn("Qubits are not specified; "
                          "Bloch vectors are not tracked")
            
        ret_dict["time"] = [self.time]
                     
        N = int(log2(self.rho.shape[0]))
        cdict_N = self.cdict[N-1]
      
        # nested function that deals with the evolution of a single pulse,
        # or recursively calls each single pulse one after another
        def __nested_evolve(pulse, retdict):
            #evolution when a single pulse is specified
            
            if isinstance(pulse, ControlPulse):                
                pulse_length = pulse.length
                
                #number of values of delta_g, B_rf, etc contained in the pulse.
                #To calculate it, a control parameter (B_rf, etc)
                #is chosen arbitrarily 
                
                num_values = np.shape(pulse.ctrl_pulses
                                      [list(pulse.ctrl_names)[0]])[0]
                
                if pulse_length==0:       
                    delta_t = self.T_2 / 100   #TODO change to an appropriate 
                                            #number; 100 is chosen arbitrarily
                else:
                    delta_t = 1.0* pulse_length /  (num_values-1)
                    
                #checking if dimensions of arrays in pulse are consistent
                dim_correct=True
                for name in pulse.ctrl_names:
                    dim_correct &= (pulse_length / 
                                     (np.shape(pulse.ctrl_pulses[name])[0] - 1) 
                                        == delta_t)
                
                if not dim_correct:
                    raise ValueError("Inconsistent dimensions of the pulse "
                                     "entry. Please try again")
        
                else:
                    tran = tqdm.tqdm(range(num_values))
                    #range starts with 0 for a proper display of progress bar
                    for n in tran:
                        #progress bar
                        tran.set_description(pulse.name + ", step %i" % (n+1))
                        if n != 0:   
                            #Runge-Kutta method
                            prev_pulse = {k:v[n-1] for (k,v) 
                                           in pulse.ctrl_pulses.items()}
                            cur_pulse={k:v[n] for (k,v) 
                                           in pulse.ctrl_pulses.items()}
                            avg_pulse={k:(0.5 * (v[n-1]+v[n])) for (k,v) 
                                           in pulse.ctrl_pulses.items()}
                            
                            K1 = self.lindbladian(self.rho, cdict_N, prev_pulse)
                            K2 = self.lindbladian(self.rho + 0.5 * delta_t * K1, 
                                                      cdict_N, avg_pulse )
                            K3 = self.lindbladian(self.rho + 0.5 * delta_t * K2, 
                                                      cdict_N, avg_pulse )
                            K4 = self.lindbladian(self.rho + delta_t * K3, 
                                                      cdict_N, cur_pulse)
                            #updating density matrix
                            
                            self.rho = self.rho + delta_t/6 * (K1 + 2 * K2 + 
                                                                  2 * K3 + K4)
                #______________________________________________________________
                            # defining when to write values of fidelity, etc.
                            
                            # the following formulas are correct for the case
                            # track_points_per_pulse > 3
                            if (num_values - 1) % (track_points_per_pulse - 1) == 0:
                                track_step = max(int((num_values - 1) /  
                                                  (track_points_per_pulse - 1)), 1)
                            else:
                                track_step = max(int((num_values-1) /  
                                                  (track_points_per_pulse - 2)), 1)
                            
                            #writing values in the output dictionary
                            if n % track_step == 0 or n == num_values-1:
                                #update time point
                                retdict["time"].append(self.time + n * delta_t)
              
                                #implementing tracking:
                                if track_qubits is not None:
                                    for key in track_keys:
                                        retdict[key].append(self.track_subsystem( 
                                            track_qubits, eval_Bloch_vectors)[key])  
                                if eval_fidelity and rho_reference is not None:    
                                    retdict["fidelity"].append(
                                        qmath.fidelity(self.rho, rho_ref))
                                if eval_purity:
                                    retdict["purity"].append(qmath.purity(self.rho))
            
                #updating the time attribute after the pulse
                self.time += pulse_length
                
            #recursive call of pulses embedded in the "pulse" iterable
            elif (isinstance(pulse, (list,tuple))):
                for p in pulse:
                    __nested_evolve(p, retdict)
            
            else:
                raise ValueError("The pulse input format is incorrect."
                                 " Please try again")
        
        __nested_evolve(pulses, ret_dict)
 
        #making all lists 1D arrays
        ret_dict = {key:np.array(val) for key, val in ret_dict.items()}

        return ret_dict


###### called when optional parameters of 'evolve' are specified as True ######
   
    def track_subsystem(self, track_qubits=None, eval_Bloch_vectors=False):
        """
        Gives the specified system submatrices and Bloch vectors (if tracked)

        Parameters
        ----------
        None.
        
        Keyword Arguments
        -----------------
        track_qubits : int/iterable of ints, optional
            Defines the qubit(s) whose density submatrices are tracked
        eval_Bloch_vectors: bool, optional
            Indicates whether to calculate Bloch vector component(s)
            
        Returns
        -------
        ret_dict: dictionary of numpy arrays
            Dictionary containing subsystem density matrices, and their 
            Bloch vectors if specified.

        """
        ret_dict = {}
        
        N = int(log2(self.rho.shape[0]))  
            #automatically calculating the size
        
        Nset = set(range(1, N+1))
        trqub = set()   # a set of tracked qubits; using set automatically 
                        # discard possible repetitions
        
        ifint = isinstance(track_qubits, int)    #track a single qubit
        ifiterable = (isinstance(track_qubits, (tuple, list, set))
              and all(isinstance(val, int) for val in track_qubits))
                #track multiple qubits
        
        if not (ifint or ifiterable):
            raise ValueError("The tracked qubits should be properly specified"  
                             "by an int or an iterable of ints. None of the"  
                                 "qubits has been tracked")
        else:
            if ifint:
                trqub = {track_qubits}
            if ifiterable :
                trqub = set(track_qubits) 

            for qub in trqub:
                submatrix = qmath.partial_trace(self.rho, (Nset-{qub}))
                subm = "submatrix_{}".format(qub)
                ret_dict[subm] = submatrix
                if eval_Bloch_vectors:
                    ret_dict["sigma_x_{}".format(qub)] = np.trace(submatrix 
                                                                @ matr.PAULI_X)
                    ret_dict["sigma_y_{}".format(qub)] = np.trace(submatrix 
                                                                @ matr.PAULI_Y)
                    ret_dict["sigma_z_{}".format(qub)] = np.trace(submatrix 
                                                                @ matr.PAULI_Z)
                
        return ret_dict
            
            
        