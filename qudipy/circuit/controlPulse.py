"""
Class for a control pulse

@author: simba
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
from copy import deepcopy

class ControlPulse:
    
    def __init__(self, pulse_name, pulse_type, pulse_length=-1, ideal_gate=None):
        '''
        Initialize a ControlPulse object.

        Parameters
        ----------
        pulse_name : string
            Name of pulse.
        pulse_type : string
            Specify whether pulse is described with "experimental" or 
            "effective" control variables.
            
        Keyword Arguments
        -----------------
        pulse_length : int, optional
            Total length of pulse in [s]. The default is -1.
            This is an optional keyword because 
            sometimes you may wish to vary the pulse length across different
            simulations for a given pulse. Pulse length can be set later using
            the set_pulse_length() method.
        ideal_gate : string, optional
            The ideal gate keyword for this control pulse. The default is None.

        Returns
        -------
        None.

        '''
        
        self.pulse_type = pulse_type
        
        self.name = pulse_name
        self.length = pulse_length # Units are s
        
        self.ctrl_pulses = {
            }
        self.ctrl_names = []
        self.n_ctrls = 0
        
        self.ideal_gate = ideal_gate
        
        self.ctrl_time = None
        
        self.map_exists = False
        
    def __call__(self, time_pts, call_inner=True):
                
        '''
        Call method for class.

        Parameters
        ----------
        time_pts : 1D float array
            Time points where we want to obtain the interpolated pulse values.
            
        Keyword Arguments
        -----------------
        call_inner : bool, optional
            When True, this will use the mappings between the outer and inner 
            control variable layers if the mappings exists. When False, this 
            will only use the outer layer for the __call__ method. The default
            is True.

        Returns
        -------
        interp_pulse : 2D float array
            The interpolated pulse values at the inputted time points.

        '''
        
        # Check that we actually have a valid pulse length set
        if self.length == -1:
            print('Cannot call control pulse object.\nPulse length has not '+
                  'been specified.\nPlease set using the .set_pulse_length()'+
                  'method.')
            return
        
        # Check if the interpolators have been constructed. If not, then 
        # make them.
        if not hasattr(self,'ctrl_interps'):
            self.__generate_ctrl_interpolators()
        
        # Loop through each control variable and get the interpolated pulse
        # for each time point.
        interp_pulse = np.zeros((len(time_pts),len(self.ctrl_names)))
        for ctrl_idx, ctrl in enumerate(self.ctrl_names):
            interp_pulse[:,ctrl_idx] = self.ctrl_interps[ctrl](time_pts)
            
        # If there is a mapping between the outer control variables and an
        # inner control variable layer, then we need to do an additional
        # interpolation to get those values.
        if self.map_exists and call_inner:
            # Intialize inner layer of interpolated pulse
            interp_pulse_inner = np.zeros((len(time_pts), 
                                           len(self.ctrl_names_inner)))
            
            # Go ahead and do the outer-inner layer interpolation
            for ctrl_idx, ctrl_inner in enumerate(self.ctrl_names_inner):
                interp_pulse_inner[:,ctrl_idx] = \
                    self.ctrl_interps_inner[ctrl_inner](interp_pulse)
            
            return interp_pulse_inner
        else:
            return interp_pulse
    
    def copy(self):
        '''
        This method will do a deep copy of the current class object.

        Returns
        -------
        ControlPulse
            Deep copy of current ControlPulse object.

        '''
        return deepcopy(self)
    
    def __get_base_pulse(self, get_inner=False):
        '''
        Will return the base control pulse as a 2D array where the first index
        corresponds to the pulse values for each variable and the second index
        corresponds to a different control variable. The second index is ordered
        according to the .ctrl_names attribute.

        Keyword Arguments
        -----------------
        get_inner : bool, optional
            Will return the base pulse in terms of the inner control pulse
            layer (if it exists).

        Returns
        -------
        base_pulse : 2D array
            A 2D array composed of each variable control pulses.

        '''
        
        if not hasattr(self,'ctrl_names_inner') and get_inner:
            raise ValueError("Cannot return the base pulse in terms of the"+ 
                             "inner layer because a control mapptin has not"+ 
                             "been defined.")
        elif get_inner:
            ctrl_dict = self.ctrl_pulses_inner
            names = self.ctrl_names_inner
        else:
            ctrl_dict = self.ctrl_pulses
            names = self.ctrl_names
        
        # Now populate the inner dictionary of inner control pulses
        base_pulse = np.zeros((len(list(ctrl_dict.values())[0]),
                               len(names)))
        for idx, ctrl_base in enumerate(names):
            base_pulse[:,idx] = ctrl_dict[ctrl_base]
            
        return base_pulse
    
    def plot(self, plot_ctrls='all', time_int='full', n='base', show_inner=False,
             show_base=False):
        '''
        Plot the control pulse. Can plot a subset of the control variables 
        and within some time interval.

        Keyword Arguments
        -----------------
        plot_ctrls : list of strings, optional
            Specify the name of each control variable pulse you wish to plot
            or plot 'all'. The default is 'all'.
        time_int : list of floats
            Specify the time interval over which to plot the pulse. Cannot be
            less than 0 or greater than the current pulse length. The default 
            is the full time interval.
        n : int, optional
            Number of points to use in the pulse when plotting. The default is
            the number of points in the base control pulse.
        show_inner : bool, optional
            Will show the pulse in terms of the inner control variable layer
            if it exists. The default is False which plots the outer control
            variable layer.
        show_base : bool, optional
            Will show the pulse points without any interpolation (i.e. 
            step-like interpolation). Ignores time_int and n keyword arguments.
            The default is False.

        Returns
        -------
        None.

        '''
        
        # Get the time interval and time points to plot
        if time_int == 'full':
            min_time = 0
            max_time = self.length
        else:
            min_time = time_int[0]
            max_time = time_int[1]
                    
        # Get the actual pulse and corresponding time points
        if show_base or n == 'base':
            # If ctrl_time is not defined, then we need to initialize the
            # control interpolators
            if self.ctrl_time is None:
                self.__generate_ctrl_interpolators()
                
            t_pts = self.ctrl_time
        else:
            if not isinstance(n, int):
                raise ValueError('n must be an integer or ''base''.')
            t_pts = np.linspace(min_time,max_time,n)
        
        if show_base:
            pulse = self.__get_base_pulse(show_inner)
            # Check if the interpolators have been constructed. If not, then 
            # make them.
            if not hasattr(self,'ctrl_interps'):
                self.__generate_ctrl_interpolators()
        else:
            pulse = self(t_pts, show_inner)
        
        # Check the plot_ctrls input
        if not isinstance(plot_ctrls,(list,tuple,set)):
            if plot_ctrls.lower() == 'all':
                if show_inner and self.map_exists:
                    plot_ctrls = self.ctrl_names_inner
                else:
                    plot_ctrls = self.ctrl_names
            # A single control variable was inputted but wasn't wrapped in a
            # list, tuple, or set, so we will wrap it in a list
            elif plot_ctrls in self.ctrl_names:
                plot_ctrls = [plot_ctrls]
            else:
                raise ValueError('Unrecognized input for control variables '+
                                 f'to plot {plot_ctrls}.\nAllowed inputs are '+
                                 'either ''all'' or a list of allowed names:\n'+
                                 f' {self.ctrl_names}')
        # If a list of names was specified, check that all are valid ctrl 
        # variable names
        elif not (set(plot_ctrls).issubset(self.ctrl_names) or 
                  set(plot_ctrls).issubset(self.ctrl_names_inner)):
            raise ValueError('Unrecognized input for control variables '+
                             f'to plot {plot_ctrls}.\nAllowed inputs are '+
                             'either ''all'' or a list of allowed names.\n'+
                             f'Outer layer variables: {self.ctrl_names}\n'+
                             f'Inner layer variables: {self.ctrl_names_inner}')
           
        # For each pulse specified by ctrl_vars, plot those pulses
        ctrl_idxs = []
        for ctrl in plot_ctrls:
            if show_inner:
                ctrl_idxs.append(self.ctrl_names_inner.index(ctrl))
            else:
                ctrl_idxs.append(self.ctrl_names.index(ctrl))
            
        # Figure out scale to apply (if any) on time axis to make more
        # readable as default length units are ps
        if 1E-15 < max(t_pts) <= 1E-12:
            scale = 1E15
            units = '[fs]'
        elif 1E-12 < max(t_pts) <= 1E-9:
            scale = 1E12
            units = '[ps]'
        elif 1E-9 < max(t_pts) <= 1E-6:
            scale = 1E9
            units = '[ns]'
        elif 1E-6 < max(t_pts) <= 1E-3:
            scale = 1E6
            units = '[us]'
        elif 1E-3 < max(t_pts):
            scale = 1E3
            units = '[ms]'
            
        # Generate figure
        plt.figure()
        if show_base:
            plt.plot(t_pts*scale, pulse[:,ctrl_idxs], marker='o', 
                     linestyle='None')    
        else:
            plt.plot(t_pts*scale, pulse[:,ctrl_idxs])   
        if show_inner:
            lgd_names = [self.ctrl_names_inner[idx] for idx in ctrl_idxs]
        else:
            lgd_names = [self.ctrl_names[idx] for idx in ctrl_idxs]
        plt.legend(lgd_names,loc='best')
        plt.xlabel('Time ' + units)
        plt.show()
        
    def set_pulse_length(self, pulse_length):
        '''
        Change the pulse length by scaling the time axis of the pulse with 
        respect to the previously specified pulse.

        Parameters
        ----------
        pulse_length : int
            Pulse legnth for control pulse in [s].

        Returns
        -------
        None.

        '''
                
        # Keep track of old length
        old_length = self.length
        self.length = pulse_length
        
        # If previous length is -1, then pulse length has not been specified
        # yet so we can just set the attribute as we did above and then return
        if old_length == -1:
            return
            
        # If self.ctrl_time is defined, then we need to just scale all the
        # current time points
        if self.ctrl_time is not None:
            self.ctrl_time *= pulse_length/old_length
            
        # Check if the interpolators have been constructed. If they have, 
        # then we need to update them with the new ctrl_time attribute
        if hasattr(self,'ctrl_interps'):
            self.__generate_ctrl_interpolators()
        
    def add_control_variable(self, var_name, var_pulse):
        '''
        Adds a control variable to the pulse. If the variable name is time,
        then it will store the time points in the ctrl_time variable.
        
        Parameters
        ----------
        var_name : string
            Name of new control variable being added.
        var_pulse : 1D array
            The corresponding control pulse for the added control variable.

        Returns
        -------
        None.

        '''
        
        # Check that the new control variable pulse has the same number of 
        # points as all the other control variables (if this is first one, 
        # then save the number of points)
        if hasattr(self,'n_pts'):
            if len(var_pulse) != self.n_pts:
                raise ValueError('Number of pulse points is not equal to the '+
                                 'current\nnumber of pulse points in previously '+
                                 'loaded control variable pulses.\n'+
                                 f'Expected {self.n_pts}, got {len(var_pulse)}.')
        else:
            self.n_pts = len(var_pulse)
        
        if var_name.lower() == 'time':
            self.ctrl_time = np.array(var_pulse)
            # Double check that the time points start at 0.
            if not np.isclose(self.ctrl_time[0], 0):
                raise ValueError(f'Cannot load time variable for {self.name}'+
                                 ' because time points do not start at 0.')

            self.set_pulse_length(var_pulse[-1])
        else:
            self.ctrl_pulses[var_name] = np.array(var_pulse)
            # self.ctrl_pulses.keys() should be in the order items are added 
            # to the dict if using python >3.7. But to keep things backwards
            # compatible, we will manually append our own list.
            self.ctrl_names.append(var_name)
            self.n_ctrls = len(self.ctrl_names)
            
    def __generate_ctrl_interpolators(self):
        '''
        Loop through every control variable and make a 1D time interpolation 
        object.

        Returns
        -------
        None.

        '''
        
        # Initialize an empty dictionary to store all the interpolators
        self.ctrl_interps = {}
        
        # Check if the time array is set, if not, then assume each point in
        # the pulse is linearly spaced in time
        if self.ctrl_time is None:
            # Build a linear time array
            self.ctrl_time = np.linspace(0, self.length, self.n_pts)
        
        # For each pulse, find the time axis points and then make the 1D
        # interpolator. All ctrl pulses will be linearly interpolated.
        for ctrl in self.ctrl_names:
            curr_pulse = self.ctrl_pulses[ctrl]
                
            self.ctrl_interps[ctrl] = interp1d(self.ctrl_time, curr_pulse)
            
    def define_ctrl_mapping(self, map_grid, map_data, ctrl_name, 
                            overwrite=False):
        '''
        This method creates an "inner" layer for the control pulse to translate
        between the control variables used to original define the object and 
        the control variables used in the actual Hamiltonian. If the object
        was originally created using the actual Hamiltonian control variables,
        then this method is not needed. The original control variables are
        referred to as the "outer" layer.
        
        This method will only create a mapping between the outer control
        variables and a single inner control variable. To add more inner
        control variables, this function must be called repeatedly.

        Parameters
        ----------
        map_grid : TYPE
            DESCRIPTION.
        map_data : TYPE
            DESCRIPTION.
        ctrl_name : TYPE
            DESCRIPTION.
        overwrite : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        # Check if there already exists a mapping for this variable and
        # warn user about overwriting it.
        if hasattr(self, 'ctrl_names_inner'):
            if ctrl_name in self.ctrl_names_inner and not overwrite:
                raise Warning("There already exists a mapping for the control "+
                              f"variable {ctrl_name}. \n If you wish to overwrite "+
                              "this mapping, set the keyword arg overwrite=True.")

                return
        # Otherwise, this is our first time making a mapping and need to 
        # initialize the dictionary for all the inner layer interpolants
        else:
            self.ctrl_interps_inner = {}
        
        # Make the regular grid interpolator object to convert from the 
        # original control variables to this new control variable
        if not isinstance(map_grid, tuple):
            map_grid = tuple(map_grid)
        interp_obj = RegularGridInterpolator(map_grid, map_data)
        
        self.ctrl_interps_inner[ctrl_name] = interp_obj
        
        # Now that the interpolator has been made, let's update the names 
        # of this inner control pulse layer
        if not hasattr(self,'ctrl_names_inner'):
            # Initialize the inner control variable names list
            self.ctrl_names_inner = []
            # Initialize a new dictionary of inner control pulses
            self.ctrl_pulses_inner = {
                }
            
            # We have created a mapping, so change the attribute flag
            self.map_exists = True
            
        if not ctrl_name in self.ctrl_names_inner:
            self.ctrl_names_inner.append(ctrl_name)
            self.n_ctrls_inner = len(self.ctrl_names_inner)
           
        # Get out control pulse
        outer_pulse = self.__get_base_pulse()

        self.ctrl_pulses_inner[ctrl_name] = interp_obj(outer_pulse)
        
    
            
        
        
        
        
        
        
        
        
        
        