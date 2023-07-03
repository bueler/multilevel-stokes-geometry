from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="""
Solve 2D steady ice geometry problem, i.e. surface kinematical equation
VI problem, where ice surface velocity is fake.  This is an advection
VI.  Compare pefarrell/fascd/examples/sia.py
""",
    formatter_class=RawTextHelpFormatter)
parser.add_argument('-bumps', action='store_true', default=False,
                    help='generate bumpy, but smooth, bed topography')
parser.add_argument('-levs', type=int, default=5, metavar='L',
                    help='number of mesh levels [default 5]')
parser.add_argument('-mcoarse', type=int, default=10, metavar='MC',
                    help='number of cells in each direction in coarse mesh [default=10]')
parser.add_argument('-nofas', action='store_true', default=False,
                    help='use a non-scalable Newton solver instead of FASCD')
parser.add_argument('-o', metavar='FILE', type=str, default='',
                    help='output file name for Paraview format (.pvd)')
parser.add_argument('-quad', action='store_true', default=False,
                    help='Q1 instead of P1')
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

def fakevelocity(H, x):
    '''Generate a fake 3D velocity field at surface of ice.
    Has zero z component.  Note 0 <= thresh(H) <= 1.'''
    U0 = 200.0 / secpera            # velocity scale is U0 = 200 m/a
    H0 = 1000.0                     # velocity is smoothly chopped around H0 = 1000
    c = 1.0 - exp(-3.0)
    k = H0 / (1.0 + ln(c))
    thresh = 1.0 - c * exp(-H / k)  # thresh=0.05 at H=0, thresh=0.63 at H=H0
    xx = x - as_vector([xc, xc])
    U = U0 * thresh * xx / xc
    return as_vector([0.6 * U[0], U[1], 0.0])

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

W = VectorFunctionSpace(mesh, 'CG', 1, dim=3)
Us = Function(W).interpolate(fakevelocity(dome(x), x))

# FIXME I need to think about advection-dominated VI problems!
# with these settings i can get
#   python3 moveice.py -levs 2 -nofas -o foo.pvd -snes_max_it 50
# to work

Deps = 0.01
v = TestFunction(V)
# F is weak form of
#    - u_s . n_s - a = 0
# compare following to F in fascd/examples/sia.py
ns = as_vector([-s.dx(0), -s.dx(1), 1.0])
F = Deps * inner(grad(s), grad(v)) * dx + inner(- inner(Us, ns) - a, v) * dx

# FIXME note performance is the same independent of bcs.  not really a hint

#bcs = DirichletBC(V, 0, "on_boundary")
#problem = NonlinearVariationalProblem(F, s, bcs)
problem = NonlinearVariationalProblem(F, s, None)

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
    solver.solve(bounds=(lb, interpolate(Constant(5000.0),V)))
else:
    sp = {#"fascd_monitor": None,
        "fascd_converged_reason": None,
        "fascd_rtol": 1.0e-3,  # can be tightened to 1.0e-4 without changing discretization error
        "fascd_atol": 1.0e-6,
        "fascd_max_it": 50,
        "fascd_cycle_type": "full",  # or "nested"
        "fascd_full_iterations": 2,
        #"fascd_levels_snes_type": "vinewtonssls",
        "fascd_levels_snes_max_it": 4, # IMPORTANT PARAMETER on hi-res meshes
        "fascd_levels_snes_linesearch_type": "bt",
        "fascd_levels_snes_linesearch_order": "2",  # order 1 broken in older petsc, but fixed?
        #"fascd_levels_snes_monitor": None,
        "fascd_levels_snes_converged_reason": None,
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
        "fascd_coarse_snes_converged_reason": None,
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
    from fascd.fascd import FASCDSolver
    solver = FASCDSolver(problem, solver_parameters=sp, options_prefix="", bounds=(lb, None),
                         admiss_eps=1.0,  # 1.0 meter error gets truncated in Vcycle
                         debug_figures1d=False, warnings_convergence=True)
    solver.solve()

from firedrake.petsc import PETSc
mfine = args.mcoarse * 2**(args.levs-1)
PETSc.Sys.Print('done (%s): mcoarse=%d, %d levels, %d x %d mesh, h=%.3f km' \
                % ('bumps' if args.bumps else 'dome',
                   args.mcoarse,args.levs,mfine,mfine,L/(1000*mfine)))

if args.o:
    Us *= secpera
    Us.rename("U_s = surface velocity (m/a)")
    PETSc.Sys.Print('writing to %s ...' % args.o)
    File(args.o).write(a,s,Us,lb)
