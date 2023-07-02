# multilevel-stokes-geometry

This is a project to apply the FASCD (full approximation scheme constraint decomposition) method, using Stokes dynamics, to compute the geometry of glaciers.

## paper/

This is a draft of

  * Bueler, E. and Mitchell, L. (in preparation). Multilevel computation of glacier
    geometry from Stokes dynamics.

## py/

Python/Firedrake codes in this directory support the above paper.  A copy of repository https://bitbucket.org/pefarrell/fascd/src/master/ is required; the ideas of FASCD are documented by the paper in https://github.com/bueler/mcd-extended:

  * E. Bueler & P. Farrell (in preparation), A full approximation scheme
    multilevel method for nonlinear variational inequalities

## talk/

These are the slides for [my talk at SIAM GS21](https://meetings.siam.org/sess/dsp_programsess.cfm?sessioncode=70836), which was held virtually.  The PDF of the slides are [here](http://pism.github.io/uaf-iceflow/bueler-siamgs2021.pdf).

## history of this project

Earlier development occurred in the [mg-glaciers](https://github.com/bueler/mg-glaciers) repository, but this has been split out.  Before the completion of Bueler & Farrell (in preparation), there was a 1D-only attempt in `old-py/` which did not get to a working multilevel method.
