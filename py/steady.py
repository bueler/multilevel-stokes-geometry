#!/usr/bin/env python3
'''Solve steady-geometry Stokes obstacle problem by a multilevel constraint decomposition method.'''

# TODO:
#   2. widen the default sheet; consider -domainlength 60.0e3 -domeL 25.0e3
#   3. implement -smoother jacobicolor with default coloring mode being 3 ice thicknesses
#   4. copy mg-glaciers/py/mcdn.py and build it out

# RUNS:
#   actually converges
#      ./steady.py -sweepsonly -J JJ -smoother jacobislow -Hmin HH
#      with JJ=1,2,3,4,5 and HH=0,10,100;  HH=0 seems as good as any
#      also sort of works with JJ=6 when HH=100 and -mz 8
#      also works with -smoother gsslow

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from firedrake import *

from meshlevel import MeshLevel1D
from problem import IceProblem, secpera
from smoother import SmootherStokes
#from mcdn import mcdnsolver

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

Initial implementation generates Bueler profile geometry as initial state
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
adda('-coarse', type=int, default=1, metavar='N',
     help='smoother sweeps on coarsest grid (default=%(default)s)')
adda('-cyclemax', type=int, default=100, metavar='N',
     help='maximum number of (multilevel) cycles (default=%(default)s)')
adda('-domainlength', type=float, default=30.0e3, metavar='L',
     help='solve on [0,L] (default=%(default)s m)')
adda('-domeH0', type=float, default=1000.0, metavar='L',
     help='center height of dome formula ice sheet (default=%(default)s m)')
adda('-domeL', type=float, default=10.0e3, metavar='L',
     help='half-width of dome formula ice sheet (default=%(default)s m)')
adda('-down', type=int, default=0, metavar='N',
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
adda('-smoother', choices=['richardson', 'jacobislow', 'gsslow'],
     metavar='X', default='richardson',
     help='smoother (default=%(default)s)')
adda('-steadyhelp', action='store_true', default=False,
     help='print help for steady.py and end (vs -help for PETSc options)')
adda('-sweepsonly', action='store_true', default=False,
     help='do smoother sweeps as cycles, instead of multilevel')
adda('-up', type=int, default=2, metavar='N',
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

# mesh hierarchy: a list of MeshLevel1D with indices [0,..,levels-1]
assert args.J >= args.jcoarse >= 0
levels = args.J - args.jcoarse + 1
hierarchy = [None] * (levels)             # list [None,...,None]
for j in range(levels):
    hierarchy[j] = MeshLevel1D(j=j+args.jcoarse, xmax=args.domainlength)

# fine-level problem data
problem = IceProblem(args)
mesh = hierarchy[-1]
b = problem.bed(mesh.xx())
ellf = mesh.ellf(problem.source(mesh.xx()))  # source functional ell[v] = <f,v>
s = problem.initial(mesh.xx())

# set-up smoother
smooth = SmootherStokes(args, b)

def output(filename, description):
    '''Either save result to an image file or use show().  Supply '' as filename
    to use show().'''
    if filename is None:
        plt.show()
    else:
        print('saving %s to %s ...' % (description, filename))
        plt.savefig(filename, bbox_inches='tight')

def final(mesh, s, cmb, filename=''):
    '''Generate graphic showing final iterate and CMB function.'''
    mesh.checklen(s)
    xx = mesh.xx()
    xx /= 1000.0
    plt.figure(figsize=(15.0, 8.0))
    plt.subplot(2,1,1)
    plt.plot(xx, s, 'k', linewidth=4.0)
    plt.xlabel('x (km)')
    plt.ylabel('surface elevation (m)')
    plt.subplot(2,1,2)
    plt.plot(xx, cmb * secpera, 'r')
    plt.grid()
    plt.ylabel('CMB (m/a)')
    plt.xlabel('x (km)')
    output(filename, 'image of final iterate')

if args.sweepsonly:
    print('sweepsonly with smoother "%s" ...' % args.smoother)
    if args.o:
        smooth.savestatenextresidual(args.o + '_0.pvd')
    r = smooth.residual(mesh, s, ellf)
    normF0 = smooth.inactiveresidualnorm(mesh, s, r, b)
    if args.monitor:
        print('   0: %.4e' % normF0)
    for j in range(args.cyclemax):
        r = smooth.smoothersweep(mesh, s, ellf, b, currentr=r)
        normF = smooth.inactiveresidualnorm(mesh, s, r, b)
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
    if args.o:
        smooth.savestatenextresidual(args.o + '_%d.pvd' % (j+1))
    smooth.residual(mesh, s, ellf)  # extra residual call
    if args.viewperturb is not None:
        smooth.savename = args.o + '_perturb.pvd'
        smooth.viewperturb(s, args.viewperturb)
else:
    raise NotImplementedError('MCDN not implemented')

if args.oimage:
    final(mesh, s, smooth.source(mesh.xx()), filename=args.oimage)
