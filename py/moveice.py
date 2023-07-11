from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="""
Solve 2D steady ice geometry problem, i.e. surface kinematical equation
VI problem, where ice surface velocity is fake.  Solves
   < F(s), q - s > >= 0   for all q
where F is diffusion-regularized surface kinematical operator
   F(s) = - eps grad^2 s - u_s . n_s - a
and eps>0 is small.  This is a (regularized) advection VI.  Compare to
F in fascd/examples/sia.py, which models flow by SIA.
""",
    formatter_class=RawTextHelpFormatter)
parser.add_argument('-bumps', action='store_true', default=False,
                    help='generate bumpy, but smooth, bed topography')
parser.add_argument('-eps', type=float, default=0.001, metavar='EPS',
                    help='diffusivity regularization in weak form [default 0.001]')
parser.add_argument('-levs', type=int, default=4, metavar='L',
                    help='number of mesh levels [default 4]')
parser.add_argument('-mcoarse', type=int, default=10, metavar='MC',
                    help='number of cells in each direction in coarse mesh [default=10]')
parser.add_argument('-nofas', action='store_true', default=False,
                    help='use a non-scalable Newton solver instead of FASCD')
parser.add_argument('-o', metavar='FILE', type=str, default='',
                    help='output file name for Paraview format (.pvd)')
parser.add_argument('-quad', action='store_true', default=False,
                    help='Q1 instead of P1')
parser.add_argument('-uscale', type=float, default=30.0, metavar='X',
                    help='scale of velocity, in m/a [default 30.0]')
parser.add_argument('-velocity', type=str, default='constant', metavar='X',
                    choices = ['constant','radial','rotate'],
                    help='choose velocity field')
args, unknown = parser.parse_known_args()

assert args.levs >= 1, 'at least one level required'
assert args.mcoarse >= 1, 'at least one cell in coarse mesh'

from firedrake import *
L = 1800.0e3        # domain is [0,L]^2, with fields centered at (xc,xc)
xc = L/2
secpera = 31556926.0

# following constants are used only in accumulation() and dome()
n = Constant(3.0)
p = n + 1
A = Constant(1.0e-16) / secpera
g = Constant(9.81)
rho = Constant(910.0)
Gamma = 2*A*(rho * g)**n / (n+2)
domeL = 750.0e3
domeH0 = 3600.0

def accumulation(x):
    # https://github.com/bueler/sia-fve/blob/master/petsc/base/exactsia.c#L51
    R = sqrt(dot(x - as_vector([xc, xc]), x - as_vector([xc, xc])))
    r = conditional(lt(R, 0.01), 0.01, R)
    r = conditional(gt(r, domeL - 0.01), domeL - 0.01, r)
    s = r / domeL
    C = domeH0**(2*n + 2) * Gamma / (2 * domeL * (1 - 1/n) )**n
    pp = 1/n
    tmp1 = s**pp + (1-s)**pp - 1
    tmp2 = 2*s**pp + (1-s)**(pp-1) * (1-2*s) - 1
    a = (C / r) * tmp1**(n-1) * tmp2
    return a

def dome(x):
    # https://github.com/bueler/sia-fve/blob/master/petsc/base/exactsia.c#L83
    R = sqrt(dot(x - as_vector([xc, xc]), x - as_vector([xc, xc])))
    mm = 1 + 1/n
    qq = n / (2*n + 2)
    CC = domeH0 / (1-1/n)**qq
    z = R / domeL
    tmp = mm * z - 1/n + (1-z)**mm - z**mm
    expr = CC * tmp**qq
    sexact = conditional(lt(R, domeL), expr, 0)
    return sexact

def bumpybed(x):
    xx, yy = x[0] / L, x[1] / L
    b = + 5.0 * sin(pi*xx) * sin(pi*yy) \
        + sin(pi*xx) * sin(3*pi*yy) - sin(2*pi*xx) * sin(pi*yy) \
        + sin(3*pi*xx) * sin(3*pi*yy) + sin(3*pi*xx) * sin(5*pi*yy) \
        + sin(4*pi*xx) * sin(4*pi*yy) - 0.5 * sin(4*pi*xx) * sin(5*pi*yy) \
        - sin(5*pi*xx) * sin(2*pi*yy) - 0.5 * sin(10*pi*xx) * sin(10*pi*yy) \
        + 0.5 * sin(19*pi*xx) * sin(11*pi*yy) + 0.5 * sin(12*pi*xx) * sin(17*pi*yy)
    return b

def fakevelocity(x):
    '''Generate a fake 3D velocity field at surface of ice, with zero z component.'''
    U0 = args.uscale / secpera
    if args.velocity == 'constant':
        return as_vector([Constant(U0), Constant(U0), 0.0])
    xx = x - as_vector([xc, xc])
    if args.velocity == 'radial':  # the bad case!
        return as_vector([U0 * xx[0] / xc, U0 * xx[1] / xc, 0.0])
    elif args.velocity == 'rotate':
        return as_vector([- U0 * xx[1] / xc, U0 * xx[0] / xc, 0.0])
    else:
        raise NotImplementedError

base = RectangleMesh(args.mcoarse, args.mcoarse, L, L, quadrilateral=args.quad)
mh = MeshHierarchy(base, args.levs-1)
mesh = mh[-1]
x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 1)
s = Function(V, name="s = surface elevation")  # initialize to zero

a = interpolate(accumulation(x), V)
a.rename("a = accumulation")
if args.bumps:
    b = bumpybed(x)
    B0 = 200.0  # (m); amplitude of bumps
    lb = interpolate(B0 * b, V)
else:
    lb = interpolate(Constant(0), V)
lb.rename("lb = bedrock topography")

Wvelocity = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
Us = Function(Wvelocity).interpolate(fakevelocity(x))
ns = as_vector([-s.dx(0), -s.dx(1), 1.0])
v = TestFunction(V)
F = args.eps * inner(grad(s), grad(v)) * dx + (-inner(Us, ns) - a) * v * dx

bcs = DirichletBC(V, 0, "on_boundary")
problem = NonlinearVariationalProblem(F, s, bcs)

s.assign(lb)  # initialize to zero thickness (s-lb=0), thus admissible

if args.nofas:
    sp = {"snes_type": "vinewtonrsls",
        "snes_rtol": 1.0e-4,
        "snes_atol": 1.0e-6,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_linesearch_type": "basic",
        #"snes_linesearch_type": "bt",
        #"snes_linesearch_order": "2",  # order 1 broken in older petsc, but fixed?
        "snes_max_it": 1000,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"}
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, options_prefix="")
    solver.solve(bounds=(lb, interpolate(Constant(50000.0),V)))
else:
    sp = {#"fascd_monitor": None,
        "fascd_converged_reason": None,
        "fascd_rtol": 1.0e-4,
        "fascd_atol": 1.0e-6,
        "fascd_max_it": 20,
        "fascd_cycle_type": "full",  # or "nested"
        "fascd_full_iterations": 2,
        #"fascd_levels_snes_type": "vinewtonssls",
        "fascd_levels_snes_max_it": 4, # IMPORTANT PARAMETER on hi-res meshes
        "fascd_levels_snes_linesearch_type": "bt",
        "fascd_levels_snes_linesearch_order": "2",  # order 1 broken in older petsc, but fixed?
        #"fascd_levels_snes_monitor": None,
        #"fascd_levels_snes_converged_reason": None,
        "fascd_levels_ksp_type": "gmres",
        "fascd_levels_ksp_max_it": 3,
        "fascd_levels_ksp_converged_maxits": None,
        "fascd_levels_pc_type": "asm",
        "fascd_levels_sub_pc_type": "ilu",
        #"fascd_levels_ksp_type": "preonly",
        #"fascd_levels_pc_type": "lu",
        #"fascd_levels_pc_factor_mat_solver_type": "mumps",
        "fascd_coarse_snes_linesearch_type": "bt",
        "fascd_coarse_snes_linesearch_order": "2",
        "fascd_coarse_snes_linesearch_damping": 0.7,
        #"fascd_coarse_snes_converged_reason": None,
        #"fascd_coarse_snes_monitor": None,
        "fascd_coarse_ksp_type": "preonly",
        #"fascd_coarse_pc_type": "lu",
        #"fascd_coarse_pc_factor_mat_solver_type": "mumps",
        "fascd_coarse_pc_type": "redundant",
        "fascd_coarse_redundant_pc_type": "lu",
        "fascd_coarse_snes_rtol": 1.0e-4,
        "fascd_coarse_snes_atol": 1.0e-6}
    import sys
    sys.path.append('fascd/')  # needed so FASCDSolver can find operators, nlvsrhs, etc.
    from fascd import FASCDSolver
    #from fascd.fascd import FASCDSolver
    solver = FASCDSolver(problem, solver_parameters=sp, options_prefix="", bounds=(lb, None),
                         admiss_eps=1.0,  # 1.0 meter error gets truncated in Vcycle
                         debug_figures1d=False, warnings_convergence=True)
    solver.solve()

from firedrake.petsc import PETSc
icevol = assemble((s - lb) * dx)
with s.dat.vec_ro as vs:
    _, smax = abs(vs).max()
mfine = args.mcoarse * 2**(args.levs-1)
PETSc.Sys.Print('done (%s, %s velocity):' \
                   % ('bumps' if args.bumps else 'dome', args.velocity))
PETSc.Sys.Print('    mcoarse=%d, %d levels, %d x %d mesh, h=%.3f km' \
                % (args.mcoarse,args.levs,mfine,mfine,L/(1000*mfine)))
PETSc.Sys.Print('    ice volume = %.4e km^3,  max s = %.2f m' \
                % (icevol / 1.0e9, smax))

if args.o:
    Us *= secpera
    Us.rename("U_s = surface velocity (m/a)")
    sdome = Function(V, name="sdome (for reference)").interpolate(dome(x))
    PETSc.Sys.Print('writing to %s ...' % args.o)
    File(args.o).write(a,s,sdome,Us,lb)
