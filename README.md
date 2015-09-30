# StellarEOS

The purpose of this project is to provide Python interfaces to various microphysics
codes commonly used in astrophysical simulations. At the moment, the planned scope
includes the following:
1. Equations of State
2. Opacity
Others may be added if there is sufficient interest.

The interfaces developed should:
1. Be minimal and usable.
2. Be well documented
3. Take inputs and give outputs in C.G.S.K. (centimeter, gram, second, kelvin) units. The only
exception to this is for mean particle masses, which are taken and given in units of the proton mass.
4. Have units noted both near the relevant functions in the code and in the broader documentation.
5. Depend only on Python, NumPy, and SciPy. Compilers are okay (but see 7 below).
6. Rely on as simple a build system as possible, and be platform-independent.
7. Require minimal modifications to the underlying tables or the routines which come with them.
Where required, these modifications must be documented clearly, and the unmodified files must be
distributed as well.
8. Be distributed with the relevant tables/routines.
9. Accept inputs in the form of both scalars and NumPy arrays, and provide output with the same
dimensionality as the input, with exceptions made for routines which return a vector of outputs
corresponding to distinct physical quantities.
10. Come with test cases and examples.

Helper scripts for merging outputs from multiple different codes are also welcome and encouraged,
though this should result in a strictly one-way dependency (i.e. the individual code/table
interfaces should not depend in any way on the helper scripts).

The tables currently contained in this project are:

OPAL2005EOS - http://opalopacity.llnl.gov/EOS_2005/
SCVH EOS - http://adsabs.harvard.edu/abs/1995ApJS...99..713S

Finally, this is all very much a work in progress, so if any of the above doesn't currently hold
true, please consider helping make it so!

 - Adam S. Jermyn