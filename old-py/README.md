# old-py/

> **Note**  
> This Python+1D attempted implementation of "MCDN" (old name) for Stokes geometry,
> using Ed's 1D non-Firedrake meshes, is deprecated.  See the FASCD-based implementation in `py/`.

Program `steady.py` is intended to solve the steady ice geometry problem (SIGP) on an interval using Stokes dynamics.  It uses Firedrake for the Stokes velocity/pressure solve, so one needs to activate the venv.  For detailed information do

        $ ./steady.py -h | less

