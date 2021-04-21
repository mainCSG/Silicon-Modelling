# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:20:47 2020

The script contains functions that generate simple constant pulses 
in spin space.

@author: bkhromet
"""
#For data manipulation
import numpy as np

import math

# Circuit module containing control pulses and ideal circuits
from qudipy.circuit import ControlPulse
# Constants class
from qudipy.utils.constants import Constants

#material system is chosen to be vacuum by default because such parameters as 
#effective mass or dielectric constant do not matter for spin simulations;
consts = Constants("vacuum")

def rot_square(qubits, N, axis, theta, B_0, B_rf, f_rf=None,
               delta_g=0., num_val=300):
    """
    Function that creates a constant (square) ROTX, ROTY or ROTZ pulse 
    for a given RF field.
    
    Parameters
    ----------
    qubits: int / iterable of ints
        The qubit(s) to be exposed to the rotation pulse.
    N: int
        Number of qubits in the system.
    axis: string
        Specifies the axis of rotation: "X", "Y" or "Z".
    theta: float
        Specifies the angle of rotation (in degrees).
    B_0: float
        Zeeman field.
    B_rf: float
        constant ESR field magnitude during the pulse
    
        
    Keyword Arguments
    -----------------
    f_rf: float, optional
        The frequency of the ESR field. By default, it is equated to
            the Larmor frequency calculated from B_0.
    delta_g: float, optional
        Approximate value of deviation g-factor used to cancel the effect 
        of global RF field on the unspecified qubits.
    num_val: int, optional
        number of data points
            
    Returns
    -------
    rotpulse: ControlPulse object / tuple of such objects 
        Pulse / tuple of pulses corresponding to the requested rotation 
        on the Bloch sphere  
    """
    if f_rf is None:
        f_rf = 2* consts.muB * B_0 / consts.hbar 
    if axis == "X" or axis == "Y":
        phis = np.full(num_val, 0.)
        if axis == "Y":
            phis = np.full(num_val, 90)
        if theta < 0:
            phis = phis + 180
        bs = np.full(num_val, B_rf)
        
        pulse_length = abs((theta * math.pi / 180) * 
                    consts.hbar / (2 * consts.muB * B_rf))
        
        rotpulse = ControlPulse("ROT{}_{}".format(axis, theta), 
                    "effective", pulse_length= pulse_length) 
                        #to comply with Brandon's code which uses picoseconds
                        
        rotpulse.add_control_variable("phi", phis)
        rotpulse.add_control_variable("B_rf", bs)
        
        # adding deviation g-factors to make unused qubits 
        # effectively "idle" under the pulse
        
        ifint = isinstance(qubits, int)    #track a single qubit
        ifiterable = (isinstance(qubits, (tuple,list, set))
              and all(isinstance(val, int) for val in qubits))
               
        if ifiterable: 
            set_qubits = set(qubits) 
        elif ifint:
            set_qubits = {qubits}
        else:
            raise ValueError("The tracked qubits should be specified"  
                             "by an int or an iterable of ints. None of the"  
                                 "qubits have been detuned to idle")
        
        # tuning the target qubit(s) on resonance, the variables are named in 
        # compliance with the write-up
        omega = 2 * consts.muB * B_0 / consts.hbar
        omega_capital = 2 * consts.muB * B_rf / consts.hbar
        
        if B_0 != 0:
            dg0 = 2.0 * (2 * math.pi * f_rf / omega - 1)
            
            for qub in set_qubits:
                rotpulse.add_control_variable("delta_g_{}".format(qub), 
                                                  np.full(num_val, dg0))        
    
        idle_qubs = set(range(1, N + 1)) - set_qubits
        # calculating the exact value of delta_g, see the write-up for the 
        # derivation

        #number of full rotations on the Bloch sphere for the idling qubit
        nrot = int(math.sqrt((omega * (1 + 0.5 * delta_g) - 2 * math.pi * 
                              f_rf)**2 + omega_capital ** 2) * 
                               pulse_length / (2 * math.pi) ) + 1
        
        dg1 = (math.sqrt((2 * math.pi * nrot/pulse_length)**2 
               - omega_capital**2) - omega +2 * math.pi *f_rf ) * 2 / omega
        
        dg2 = (-math.sqrt((2 * math.pi * nrot/pulse_length)**2
               - omega_capital**2) - omega +2 * math.pi *f_rf ) * 2 / omega
        
        #choosing the closest value
        exact_delta_g = dg1 if abs(dg1-delta_g) < abs(dg2-delta_g) else dg2
            
        for qub in idle_qubs:
            rotpulse.add_control_variable("delta_g_{}".format(qub), 
                                         np.full(num_val, exact_delta_g))
                
        return rotpulse
        #del rotpulse
    elif axis=="Z":
        return [rot_square(qubits, N, "X", -90, B_0, B_rf, 
                                                   f_rf, delta_g, num_val), 
                rot_square(qubits, N, "Y", theta, B_0, B_rf,
                                                   f_rf, delta_g, num_val), 
                rot_square(qubits, N, "X", 90, B_0, B_rf, 
                                                   f_rf, delta_g, num_val)]
    else:
        raise ValueError("Incorrect input of axis, please try again")
        return 0
    
def swap(qubits, N, J, B_0=0, f_rf=None, num_val=300):
    """
    A function that builds a simple SWAP gate.
    
    Parameters
    ----------
    qubits: tuple of 2 ints
        Pair of indices of the two interacting qubits. Temporarily, only the 
        case of two neighboring qubits is supported. 
    N: int
        Number of qubits in the system.
    J: float
        Exchange coupling between the specified qubits
        
    Keyword Arguments
    -----------------
    B_0: float
        Zeeman field. Default is zero.
    f_rf: float, optional
        The frequency of the ESR field specified for the system. By default, 
        it is equated to the Larmor frequency calculated from B_0.
    num_val: int, optional
        number of data points 
            
    Returns
    -------
    swappulse: ControlPulse object
        Pulse corresponding to the SWAP between the two specified neighboring 
        qubits
    """
    if len(qubits) != 2:
        raise ValueError("The indices of the interacting qubits are specified " 
                         "incorrectly. There should be a list/tuple with two "
                         "integers")
    if abs(qubits[0] - qubits[1]) != 1:
        raise ValueError("Temporarily, only the exchange between neighboring"
                         "qubits is supported")
    qubit = min(qubits)    
    Js = np.full(num_val, J)
    swappulse = ControlPulse("SWAP_{}_{}".format(qubit, qubit + 1), 
                                "effective", pulse_length = consts.h / (2 * J)) 
    swappulse.add_control_variable("J_{}".format(qubit), Js)
    
    #tuning all qubits on resonance
    if B_0 != 0:
        omega = 2 * consts.muB * B_0 / consts.hbar
        dg0 = 2.0 * (2 * math.pi * f_rf / omega - 1)
        for qub in range(1, N + 1):
            swappulse.add_control_variable("delta_g_{}".format(qub), 
                                         np.full(num_val, dg0))
    
    return swappulse

def rswap(qubits, N, J, B_0=0, f_rf=None, num_val=300):
    """
    A function that builds a simple √SWAP gate.
    
    Parameters
    ----------
    qubits: tuple of 2 ints
        Pair of indices of the two interacting qubits. Temporarily, only the 
        case of two neighboring qubits is supported. 
    N: int
        Number of qubits in the system.
    J: float
        Exchange coupling between the specified qubits.   
        
    Keyword Arguments
    -----------------
    B_0: float
        Zeeman field. Default is zero.
    f_rf: float, optional
        The frequency of the ESR field specified for the system.By default, 
        it is equated to the Larmor frequency calculated from B_0.
    num_val: int, optional
        number of data points 
            
    Returns
    -------
    rswappulse: ControlPulse object
        Pulse corresponding to the SWAP between the two specified neighboring 
        qubits
    """
    if len(qubits) != 2:
        raise ValueError("The indices of the interacting qubits are specified " 
                         "incorrectly. There should be a list/tuple with two "
                         "integers")
    if abs(qubits[0] - qubits[1]) != 1:
        raise ValueError("Temporarily, only the exchange between neighboring"
                         "qubits is supported")
    qubit = min(qubits)    
    Js = np.full(num_val, J)
    rswappulse = ControlPulse("RSWAP_{}_{}".format(qubit, qubit + 1), 
                                "effective", pulse_length = consts.h / (4 * J)) 
    rswappulse.add_control_variable("J_{}".format(qubit), Js)
    
    #tuning all qubits on resonance
    if B_0 != 0:
        omega = 2 * consts.muB * B_0 / consts.hbar
        dg0 = 2.0 * (2 * math.pi * f_rf / omega - 1)
        for qub in range(1, N + 1):
            rswappulse.add_control_variable("delta_g_{}".format(qub), 
                                         np.full(num_val, dg0))

    return rswappulse

