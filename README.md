# QuDiPy (Quantum Dots in Python)

This respository aims to be a general purpose tool for simulating electron dynamics in quantum dots with an emphasis on quantum information processing. We adopt an effective mass formalism which enables simulations across a variety of material systems and heterostructures. The code base is under construction by the Coherent Spintronics Group (CSG) at the University of Waterloo. The three codebase maintainers are Brandon Buonacorsi (@theycallmesimba), Stephen Harrigen (@bleutooth65), and Bohdan Khromets (@hromecB). Reach out to any of us if you have questions regarding this project.

Version 1.0 is currently under development. When completed, we aim to have the following capabilities:
- Simulation of charge stability diagrams using either a constant interaction model or a Hubbard Hamiltonian model. In addition, this module can fit charge stability data to extract capacitances for the constant interaciton model.
- Real space time evolution simulations useful for simulating orbital dynamics of a single electron in a potential landscape.
- General purpose spin simulator capable of simulating n-spin systems using an effective Hamiltonian that incorporates ESR and the exchange interaction.
- Estimation of the stark-shift in a silicon heterostructure.
- Calculation of the many-electron spectra of a general quantum dot network using a modified LCHO-CI approach. These energy spectra can be mapped onto a Heisenberg Hamiltonian to determine the pairwise exchange interactions of electrons in the dot network.
- Tools for finding optimal control pulses such as GRAPE and constant-adiabatic pulse engineering.

You can see the general progress in the module by going to the tutorials folder and following along the Jupyter notebooks.  Currently, the data for the tutorials are hosted elsewhere.  If you would like these available to you, please email brandonbuonacorsi@gmail.com (subject line: QuDiPy).  We emphasize that this is work in progress.

For development guidelines, see https://github.com/mainCSG/QuDiPy/blob/master/Development%20Guidelines.md.
