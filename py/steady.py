#!/usr/bin/env python3
'''Solve steady-geometry Stokes obstacle problem by a multilevel constraint decomposition method.'''

# TODO:
#   1. widen the default sheet?; consider -domainlength 60.0e3 -domeL 25.0e3
#   2. replace meshlevel.py with use of Firedrake interval mesh (requires
#      new implementation of monotone restriction); this would allow
#      parallelization
#   3. copy mg-glaciers/py/mcdn.py and build it out

# RUNS:
#   actually converges
#      ./steady.py -sweepsonly -J JJ -smoother jacobislow -Hmin HH
#      with JJ=1,2,3,4,5 and HH=0,10,100;  HH=0 seems as good as any
#      also sort of works with JJ=6 when HH=100 and -mz 8
#      also works with -smoother gsslow
#   uses jacobicolor (faster than jacobislow) and works:
#      for JJ in 3 4 5 6; do ./steady.py -sweepsonly -monitor -omega 0.7 -Hmin 20.0 -mz 8 -irtol 1.0e-2 -J $JJ; done
#      but fails to converge for JJ=7

import sys
import argparse
import numpy as np
from firedrake import *

from meshlevel import MeshLevel1D
from problem import IceProblem
from smoother import SmootherStokes
from mcdn import mcdnvcycle
from visualize import showiteratecmb

parser = argparse.ArgumentParser(description='''
Solve the 1D obstacle problem for the steady geometry of a glacier using
Stokes dynamics:  For given climate a(x) and bed elevation b(x),
find s(x) in the closed, convex subset
    K = {r | r >= b}
so that the variational inequality (VI) holds,
    F(s)[r-s] >= 0   for all r in K.
Note s solves the surface kinematical equation, as an interior PDE, in the
inactive set {x | s(x) > b(x)}.

Solution is by the nonlinear (FAS) extension of the multilevel constraint
decomposition (MCD) method of Tai (2003), or by sweeps of the smoother.

Initial implementation generates a dome geometry on a flat bed as initial state
and then tries to converge from there.

References:
  * Bueler, E. and Mitchell, L. (2022). Multilevel computation of glacier
    geometry from Stokes dynamics.  In preparation.
  * Tai, X.-C. (2003). Rate of convergence for some constraint
    decomposition methods for nonlinear variational inequalities.
    Numer. Math. 93 (4), 755--786.
''',
    formatter_class=argparse.RawTextHelpFormatter,
    allow_abbrev=False,  # bug in python 3.8 causes this to be ignored
    add_help=False)
adda = parser.add_argument
adda('-coarsesweeps', type=int, default=1, metavar='N',
     help='smoother sweeps on coarsest grid (default=%(default)s)')
adda('-cyclemax', type=int, default=100, metavar='N',
     help='maximum number of (multilevel) cycles (default=%(default)s)')
adda('-domainlength', type=float, default=30.0e3, metavar='L',
     help='solve on [0,L] (default=%(default)s m)')
adda('-domeH0', type=float, default=1000.0, metavar='L',
     help='center height of dome formula ice sheet (default=%(default)s m)')
adda('-domeL', type=float, default=10.0e3, metavar='L',
     help='half-width of dome formula ice sheet (default=%(default)s m)')
adda('-downsweeps', type=int, default=0, metavar='N',
     help='smoother sweeps before coarse-mesh correction (default=%(default)s)')
adda('-eps', type=float, metavar='X', default=1.0e-2,  # FIXME sensitive
    help='regularization used in viscosity (default=%(default)s)')
adda('-Hmin', type=float, metavar='X', default=0.0,
    help='minimum ice thickness (default=%(default)s)')
adda('-irtol', type=float, default=1.0e-3, metavar='X',
     help='reduce norm of inactive residual (default=%(default)s)')
adda('-jcoarse', type=int, default=0, metavar='J',
     help='coarse mesh is jth level (default jcoarse=0 gives 1 node)')
adda('-J', type=int, default=3, metavar='J',
     help='fine mesh is Jth level (default J=%(default)s)')
adda('-monitor', action='store_true', default=False,
     help='show CP residual norm at each step')
adda('-mz', type=int, default=4, metavar='MZ',
     help='number of (x,z) extruded mesh levels (default=%(default)s)')
adda('-o', metavar='FILEROOT', type=str, default='',
     help='save .pvd and final image in .png to FILEROOT.*')
adda('-oimage', metavar='FILE', type=str, default='',
     help='save final image, e.g. .pdf or .png')
adda('-omega', type=float, metavar='X', default=1.0,  # FIXME sensitive
    help='scale by this factor in smoother iteration (default=%(default)s)')
adda('-padding', action='store_true', default=False,
     help='put Hmin thickness of ice in ice-free locations')
adda('-printwarnings', action='store_true', default=False,
     help='print pointwise feasibility warnings')
adda('-smoother', choices=['richardson', 'gsslow', 'jacobislow', 'jacobicolor'],
     metavar='X', default='jacobicolor',
     help='smoother (default=%(default)s)')
adda('-steadyhelp', action='store_true', default=False,
     help='print help for steady.py and end (vs -help for PETSc options)')
adda('-sweepsonly', action='store_true', default=False,
     help='do smoother sweeps as cycles, instead of multilevel')
adda('-upsweeps', type=int, default=2, metavar='N',
     help='smoother sweeps after coarse-mesh correction (default=%(default)s)')
adda('-viewperturb', type=int, default=None, nargs='+', metavar='N ...',
     help='view u,p perturbations at these nodes to .pvd file; use with -o')
args, unknown = parser.parse_known_args()
if args.steadyhelp:
    parser.print_help()
    sys.exit(0)

if args.padding:
    assert args.Hmin > 0.0, 'padding requires minimum positive thickness'
if args.viewperturb is not None:
    assert len(args.o) > 0, 'use -view perturb with -o'

# problem and mesh hierarchy
problem = IceProblem(args)
if args.sweepsonly:
    # build fine-level mesh only
    finemesh = MeshLevel1D(j=args.J, xmax=args.domainlength)
    finemesh.b = problem.bed(finemesh.xx())
else:
    # build mesh hierarchy: list of MeshLevel1D with indices [0,..,J]
    # (some coarse levels will be unused if jcoarse > 0)
    assert args.J >= args.jcoarse >= 0
    hierarchy = [None] * (args.J + 1)             # list [None,...,None]
    for j in range(args.J + 1):
        hierarchy[j] = MeshLevel1D(j=j, xmax=args.domainlength)
        hierarchy[j].b = problem.bed(hierarchy[j].xx())
    finemesh = hierarchy[args.J]

# fine-level data
ella = finemesh.ellf(problem.source(finemesh.xx()))  # source ell[v] = <a,v>
s = problem.initial(finemesh.xx())

# set-up smoother
smooth = SmootherStokes(args)

# generate .pvd file with initial state (see first residual())
if args.o:
    smooth.savestatenextresidual(args.o + '_0.pvd')

# solve
if args.sweepsonly:
    print('SWEEPSONLY with smoother "%s" ...' % args.smoother)
else:
    print('MCDN with smoother "%s" ...' % args.smoother)
r = smooth.residual(finemesh, s, ella)  # generates file if -o above
normF0 = smooth.inactiveresidualnorm(finemesh, s, r, finemesh.b)
if args.monitor:
    print('   0: %.4e' % normF0)
for j in range(args.cyclemax):
    if args.sweepsonly:
        r = smooth.smoothersweep(finemesh, s, ella, currentr=r)
    else:
        s += mcdnvcycle(args, smooth, hierarchy, s, ella)
        r = smooth.residual(finemesh, s, ella)  # needed? return it from MCDN?
    normF = smooth.inactiveresidualnorm(finemesh, s, r, finemesh.b)
    if args.monitor:
        print('%4d: %.4e' % (j+1, normF))
    if normF < args.irtol * normF0:
        print('iteration CONVERGED at step %d by F<(%.2e)F0' \
              % (j+1, args.irtol))
        break
    elif normF > 100.0 * normF0:
        print('iteration DIVERGED by F>100F0 at step %d' % (j+1))
        break
    elif j + 1 == args.cyclemax:
        print('iteration REACHED CYCLEMAX at step %d' % (j+1))

# output to .pvd
if args.o:
    smooth.savestatenextresidual(args.o + '_%d.pvd' % (j+1))
    smooth.residual(finemesh, s, ella)  # extra residual call generates file
    if args.viewperturb is not None:
        smooth.savename = args.o + '_perturb.pvd'
        smooth.viewperturb(s, args.viewperturb)

# output an image
if args.oimage:
    showiteratecmb(finemesh, s, problem.source(finemesh.xx()),
                   filename=args.oimage)
