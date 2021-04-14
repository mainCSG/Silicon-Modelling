"""
Functions for loading data, either from files or analytically defining them.

@author: simba
"""

import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.interpolate import interp2d
from itertools import product

import qudipy as qd
from qudipy.potential import GridParameters
from qudipy.potential.potentialinterpolator import PotentialInterpolator
        
def build_interpolator(load_data_dict, constants=qd.Constants(), 
                       y_slice=None):
    '''
    This function constructs an interpolator object for either a group of 
    potential or electric field files.
    Parameters
    ----------
    all_data_sep : dict
        Dictionary containing the x and y coordinates for the loaded files,
        the potential data for each loaded file, and the corresponding votlage
        vector for each file.
        Fields = ['coords', 'potentials', 'ctrl_vals']
        
    Keyword Arguments
    ----------
    constants : Constants object, optional
        Constants object containing material parameter details. The default is
        a Constants object assuming air as the material system.
    y_slice : float, optional
        Used to create a interpolator of only 1D poetentials. Specify a slice 
        along the y-axis at which to take the 1D potential when constructing
        the interpolator. Units should be specified in [m]. 
        The default is None.
    
    Returns
    -------
    interp_obj : Mod_RegularGridInterpolator class
        Interpolant object for the data inputted into the function.
    '''
    
    # Get first set of x and y coordinates
    x_coords = load_data_dict['coords'][0]
    y_coords = load_data_dict['coords'][1]
    
    # Extract all the control values
    all_ctrls = np.asarray(load_data_dict['ctrl_vals'])
    
    # Get total number of ctrls (including singleton dims)
    n_ctrls = len(load_data_dict['ctrl_vals'][0])
        
    # Find which gate voltages have singleton dimension. We need to keep track
    # because the interpolator cannot handle singleton dimensions
    single_dims = []
    n_dims = []
    ctrl_values = []
    for idx in range(n_ctrls):
        n_unique = len(set(all_ctrls[:,idx]))
        if n_unique == 1:
            single_dims.append(idx)
        else:
            n_dims.append(n_unique)
            ctrl_values.append(sorted(list(set(all_ctrls[:,idx]))))
        
    # Now assemble the data to be interpolated
    temp_n_dims = [range(n) for n in n_dims]
    
    # Add the y and x coordinate lengths so we know the expected dimensions of 
    # the total nd array of data to interpolate
    if y_slice is None:
        all_data_stacked = np.zeros((np.prod(n_dims),len(y_coords),len(x_coords)))
        n_dims.extend([len(y_coords),len(x_coords)])  
    else:
        all_data_stacked = np.zeros((np.prod(n_dims),len(x_coords)))
        n_dims.extend([len(x_coords)])  

    # Go and stack the potential data together and then reshape it into
    # correct format
    if y_slice is not None:
        y_idx = qd.utils.find_nearest(y_coords, y_slice)[0]
    for idx, curr_gate_idx in enumerate(product(*temp_n_dims)):
        if y_slice is None:
            all_data_stacked[idx,:,:] = load_data_dict['potentials'][idx]
        else:
            all_data_stacked[idx,:] = np.squeeze(
                load_data_dict['potentials'][idx][y_idx,:])
    
    all_data_stacked = np.reshape(all_data_stacked,(n_dims))
    
    # Construct the interpolator
    if y_slice is None:
        ctrl_values.extend([y_coords,x_coords])
    else:
        ctrl_values.extend([x_coords])
    interp_obj = PotentialInterpolator(ctrl_values, load_data_dict['ctrl_names'],
                                        all_data_stacked, single_dims, constants,
                                        y_slice)
    
    return interp_obj

def load_potentials(ctrl_vals, ctrl_names, f_type='pot', f_dir=None, 
                    f_pot_units='J', f_dis_units='m', trim_x=None, trim_y=None):
    '''
    This function loads many potential files specified by all combinations of 
    the control values given in ctrl_vals and the control names given in
    ctrl_names. The potential files MUST be 2D potential slices (if you have
    3D nextnano simulations, you must preprocess them first). Potential files
    are assumed to follow the syntax: 
    'TYPE_C1NAME_C1VAL_C2NAME_C2VAL_..._CNNAME_CNVAL.txt'
    where TYPE = 'Uxy' or 'Ez'. 
    Refer to tutorial for a more explicit example.
    
    Parameters
    ----------
    ctrl_vals : list of list of floats
        List of relevant control values for the files to load.  The first list
        index corresponds to the ith control variable and the second list
        index correspond to the ith value for that control variable.
    ctrl_names : list of strings
        List of each ctrl variable name. Must be the same length as ctrl_vals 
        first dimension.
        
    Keyword Arguments
    -----------------
    f_type : string, optional
        Type of file to load (either potential or electric field). Acceptable 
        arguments include ['pot','potential','Uxy','electric','field','Ez'].
        Default is potential. The default is 'pot'
    f_dir : string, optional
        Path to find files specified in f_list. The default is is the current
        working directory.
    f_pot_units : string, optional
        Units of the potential in the files to load. Units from file will be
        converted to J.
        Supported inputs are 'J' and 'eV'.
    f_dis_units : string, optional
        Units of the x and y coordinates in the files to load. Units from file
        will be converted to m. 
        Supported inputs are 'm' and 'nm'. 
    trim_x : list of floats, optional
        Specify min and max bounds of x-axis in [m] to save when loading the 
        files. Data points outside this window will be trimmed and not saved 
        in the loaded files. The default is None.
    trim_y : list of floats, optional
        Specify min and max bounds of y-axis in [m] to save when loading the 
        files. Data points outside this window will be trimmed and not saved 
        in the loaded files. The default is None.
    
    Returns
    -------
    all_files : dict
        Dictionary containing the x and y coordinates for the loaded files,
        the potential data for each loaded file, and the corresponding votlage
        vector for each file.
        Fields = ['coords', 'potentials', 'ctrl_vals']
    '''

    # Check inputs
    if len(ctrl_vals) != len(ctrl_names):
        raise ValueError('Incorrect number of control names given, must be ' +
                         'equal to first dimension of ctrl_vals ' +
                         f' {len(ctrl_vals)}.')
    
    # Check if dir was given
    if f_dir is None:
        f_dir = ''

    all_files = {}
    cval_array = []
    pots_array = []
    # Loop through all combinations of gate voltages to load the files
    for idx, curr_cvals in enumerate(product(*ctrl_vals)):
        # Now build up the current file name
        # First figure out type of file to load (electric or potential)
        if f_type in ['pot', 'potential', 'Uxy']:
            f_name = 'Uxy'
        elif f_type in ['field', 'electric', 'Ez']:
            f_name = 'Ez'
            
        for name, val in zip(ctrl_names, curr_cvals):
            f_name = f_name+ '_' + name + '_' + "{:.3f}".format(val)
        
        f_name += '.txt'
        
        # After file name is constructed, load the data from file into a larger
        # list containing information about all the loaded files.

        # Load file
        data = pd.read_csv(f_dir+f_name, header=None).to_numpy()
        
        # Extract items
        x = data[0,1:]
        y = data[1:,0]
        pot = data[1:,1:]        

        # Convert units if needed
        if f_pot_units == 'eV':
            # Just need to get electron charge
            constants = qd.Constants('vacuum')
            pot *= constants.e
            
        if f_dis_units == 'nm':
            x *= 1E-9
            y *= 1E-9
        
        if idx == 0:
            # If trim wasn't specified then we want to store the whole x axis
            if trim_x is None:
                trim_x = [np.min(x), np.max(x)]
            # If trim wasn't specified then we want to store the whole y axis
            if trim_y is None:
                trim_y = [np.min(y), np.max(y)]
                
            # Get bool mask and trim coordinates
            x_idx_mask = np.logical_and(x >= trim_x[0], x <= trim_x[1])
            y_idx_mask = np.logical_and(y >= trim_y[0], y <= trim_y[1])
            
            new_x = x[x_idx_mask]
            new_y = y[y_idx_mask]
            
            # Get new coordinate points by rounding number of coordinates 
            # points to a power of 2 for both x and y (for faster ffts).
            new_x_len = 1 if len(new_x) == 0 else 2**(len(new_x) - 1).bit_length()
            new_y_len = 1 if len(new_y) == 0 else 2**(len(new_y) - 1).bit_length()
            new_x = np.linspace(new_x.min(), new_x.max(), new_x_len)
            new_y = np.linspace(new_y.min(), new_y.max(), new_y_len)
            
            # Have coordinates be a namedtuple
            Coordinates = namedtuple('Coordinates',['x','y'])
            all_files['coords'] = Coordinates(new_x,new_y)
            
        cval_array.append(list(curr_cvals))

        # Do a spline interpolation to find potential at the 'trimmed' and 
        # power of 2 coordiante points.
        f = interp2d(x, y, pot, kind='cubic')
        new_pot = f(new_x, new_y)
        pots_array.append(new_pot)
        
    all_files['ctrl_vals'] = cval_array
    all_files['ctrl_names'] = ctrl_names
    all_files['potentials'] = pots_array
    
    
    return all_files

def analytical_potential(ctrl_vals, ctrl_names, function, x_range, y_range):
    '''
    Allows one to create an analytical potential in the xy-plane from a function
    Parameters
    ----------
    ctrl_vals : list of list of floats
        List of relevant control values used to define the potential. The first list
        index corresponds to the ith control variable and the second list
        index correspond to the ith value for that control variable.
    ctrl_names : list of strings
        List of each ctrl variable name. Must be the same length as ctrl_vals 
        first dimension.
    function : callable
        Function which defines the 2D potential you wish to map out. function
        must take the list ctrl_vals as the first argument and a GridParameters 
        object as the second argument. Refer to tutorial for an explicit example
    x_range : 1D List or Numpy array
        grid points along x where you want to potential to be calculated
    y_range : 1D List or Numpy array
        grid points along y where you want to potential to be calculated

    Returns
    -------
    analytical_potential : dictionary
        Dictionary containing the x and y coordinates, the potential at each
        coordinate point, and the control variable vector and control variable
        name.
    '''

    # From the x and y ranges provided, create the GridParameters
    gparams = GridParameters(x_range, y_range)

    # Go over all products of control values and calculate potential
    cval_array = []
    pots_array = []
    for curr_cvals in product(*ctrl_vals):
        pots_array.append(function(curr_cvals, gparams))
        cval_array.append(list(curr_cvals))

    # Create named tuple for coordinates
    Coordinates = namedtuple('Coordinates',['x','y'])

    # Load everything into a dictionary that is returned
    analytic_potential = {}
    analytic_potential['coords'] = Coordinates(x_range, y_range)
    analytic_potential['ctrl_vals'] = cval_array
    analytic_potential['ctrl_names'] = ctrl_names
    analytic_potential['potentials'] = pots_array

    return analytic_potential
    
