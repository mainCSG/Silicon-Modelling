"""
GridParameters class

@author: simba
"""
import numpy as np

class GridParameters:
    '''
    
    Initialize the grid parameter class. Handles all things related to the 
    grid settings for the simulations.
        
    '''
    
    def __init__(self, x, y=None, potential=np.array([])):
        '''
        
        Parameters
        ----------
        x : array
            Grid coordinates along x with uniform spacing.
            
        Keyword Arguments
        -----------------
        y : array
            Grid coordinates along y with uniform spacing.
        potential : array, optional
            Potential values along x-y coordinates where 2DEG is formed. Must
            be in meshgrid format (y,x). The default is an empty numpy array.

        Returns
        -------
        None.

        '''
        self.potential = np.array(potential)
        
        self.x = np.array(x)
        self.dx = x[1] - x[0]
        self.nx = len(x)  
                
        # Check if y coordinates were inputted as an argument
        if y is None:
            self.grid_type = '1D'
            
            # Check that coordinate data matches potential but ignore if the 
            # potential is not defined
            if potential.shape[0] != 0:
                if self.nx != self.potential.shape[0]:
                    raise ValueError("x coordinate grid points do not match"\
                                    " number of potential x-coordinates.")
                        
        # y coordinates were inputted
        else:         
            self.grid_type = '2D'
            self.y = np.array(y)
            self.dy = y[1] - y[0]
            self.x_mesh, self.y_mesh = np.meshgrid(x, y, sparse=False, indexing='xy')
            self.ny, self.nx = self.x_mesh.shape
            
            # Check that coordinate data matches potential but ignore if the 
            # potential is not defined
            if potential.shape[0] != 0:
                # Check that coordinate data matches potential
                if self.nx != self.potential.shape[1]:
                    raise ValueError("x coordinate grid points do not match"\
                                    " number of potential x-coordinates.")
                if self.nx != self.potential.shape[1]:
                    raise ValueError("y coordinate grid points do not match"\
                                    " number of potential y-coordinates.")
        
        
    def convert_MG_to_NO(self, data_MG):
        '''
        
        Converts data from meshgrid to natural order format       

        Parameters
        ----------
        dataMG : 2D array
            Data to convert in meshgrid format.

        Returns
        -------
        data_NO : 1D array
            Converted data in natural order format.

        '''
        
        data_MG = np.transpose(data_MG)
        data_NO = np.reshape(data_MG, (self.nx*self.ny, 1), order='F');

        return data_NO
        
        
    def convert_NO_to_MG(self, data_NO):
        '''
        
        Converts data from natural order to meshgrid format

        Parameters
        ----------
        data_NO : 1D array
            Data to convert in natural order format

        Returns
        -------
        data_MG : 2D array
            Converted data in meshgrid order format.

        '''
        
        data_MG = np.reshape(data_NO, (self.nx,self.ny), order='F');
        data_MG = np.transpose(data_MG)
        
        return data_MG
    
    def update_potential(self, new_pot):
        '''
        
        Assign a new potential to the class

        Parameters
        ----------
        new_pot : 2D array in meshgrid format
            New potential to assign.

        Returns
        -------
        None.

        '''
        
        self.potential = new_pot
        
        
        
        
        
        
        
        
        
        