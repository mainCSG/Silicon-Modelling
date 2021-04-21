# QuDiPy Development Guidelines

The below guidelines are relatively obvious. The aim of this document is to standardize the way we write code for the qudipy module. Adopting a consistent framework will ensure that PRs are handled more easily and quickly.

1. Brandon (@theycallmesimba), Stephen (@bleutooth65), and Bohdan (@hromecB) are the current maintainers of the repository. All pull requests must be reviewed and approved by at least two of them before merging into master.

2. You should never work directly on the master branch. Always start your own development branch and submit a PR to merge with master.

3. Before merging you need to run all tutorials and unittests to ensure you have not broken anything.

4. External repositories are allowed as long as they are well maintained (for example numpy, scikit-learn, pandas, tqdm). This is subjective, so use your judgement.

5. Every function needs to have a docstring where there Parameters, Keyword Arguments, and Returns variables are clarified. An example of a good docstring is shown in qudipy/potential/potentialInterpolator.py. If you are using an external source as a reference (such as an academic paper or blog post), make sure to reference it in the docstring description. The general format for a docstring is

```
Parameters
----------
var_1 : type
    Description.
var_2 : type
    Description.
   
Keyword Arguments
-----------------
var_3 : type, optional
    Description.
        
Returns
-------
var_ret: type
     Description. 
```

6. If you have added code you believe accomplishes a mathematically non-trivial task, please add a short write up of the procedure to the QuDiPy math write up document: https://www.overleaf.com/3252553442tbqcmxntqvtk. You can then reference your write up in the appropriate function's docstring.

7. The qudipy module and submodules should always be loaded as:
	* `import qudipy as qd`
	* `import qudipy.potential as pot`
	* `import qudipy.chargestability as csd`
	* `import qudipy.qutils.math as qmath`
	* `import qudipy.qutils.matrices as matr`
	* `import qudipy.spinsimulator.spin_simulator as sps`

8. Common qudipy classes should be implemented as (although in somecases this may not always be appropriate):
	* `gparams = qd.potential.GridParameters()` or `gparams = pot.GridParameters()`
	* `consts = qd.Constants()`

9. Common external repositories should be loaded as:
	* `import numpy as np`
	* `import pandas as pd`
	* `import scipy as sp`
	* `from scipy import linalg as la`
	* `import matplotlib.pyplot as plt`
	* `import seaborn as sns`

8. The lowest python version supported is python 3.6.

9. Please report bugs as an issue on github.

10. Follow PEP 8 style guidelines.

11. Code should always prioritize readability first and optimality second.

12. Comment, comment, comment.

