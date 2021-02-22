"""
Constants class

@author: simba
"""

class Constants:
    
    def __init__(self, material_system="vacuum"):
        '''
        
        Keyword Arguments
        -----------------
        material_system : string, optional
            String specifying which material system the constant class is for.
            Currently allowed systems are: ["vacuum","Si/SiO2", "Si/SiGe", "GaAs"].
            Default is vacuum.

        Returns
        -------
        None.

        '''
        # Default units are SI [International system]
        self.units = "SI"
        
        # Mathematical constants
        self.pi = 3.141592653589793         # pi
        
        # Physical constants
        self.h = 6.62607015E-34              # Planck's constant [J*s]
        self.hbar = self.h/(2*self.pi)      # Reduced Planck's constant [J*s]
        self.e = 1.602176634E-19          # electron charge [C]
        self.m0 = 9.10938356E-31            # free electron mass [kg]
        self.c = 2.99792458E8               # speed of light [m/s]
        self.muB = 9.274009994E-24          # Bohr magneton [J/T]
        self.eps0 = 8.85418782E-12          # Vacuum permitivity [F/m]
        self.kB = 1.380649E-23              # Boltzmann constant [J/K]
                  
        # Material system constants
        # Supported material systems include [Si/SiO2, Si/SiGe, GaAs, air]
        self.material_system = material_system
        
        if material_system == "Si/SiO2":
            self.epsR = 7.8                 # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0*0.191         # Effective mass [kg]
            self.mu_2 = 2.2 * (1e-9)**2     # Quadratic Stark shift coefficient [m^2/V^2]
        elif material_system == "Si/SiGe":
            self.epsR = 12.375              # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0*0.191         # Effective mass [kg]
            self.mu_2 = None                # Quadratic Stark shift coefficient [m^2/V^2]
        elif material_system == "GaAs":
            self.epsR = 12.4                # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0*0.067         # Effective mass [kg]
            self.mu_2 = None                # Quadratic Stark shift coefficient [m^2/V^2]
        elif material_system.lower() == "vacuum" or "air":
            self.epsR = 1                   # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0               # Effective mass [kg]
            self.mu_2 = None                # Quadratic Stark shift coefficient [m^2/V^2]
        else:
            # If no or unrecognized material system specified assume "air"
            self.epsR = 1                   # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0               # Effective mass [kg]
            self.mu_2 = None                # Quadratic Stark shift coefficient [m^2/V^2]
            print("WARNING: Material system either not recognized or defined.\n"+
                  "Assuming ""vacuum"" as the material system.\n"+
                  "Allowed material systems are: [""Si/SiO2"", ""Si/SiGe"","+
                                                 " ""GaAs"", ""vacuum""]")   
            
        
        
