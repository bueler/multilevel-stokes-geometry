'''Module for SmootherStokes class derived from SmootherObstacleProblem.'''

import numpy as np
import firedrake as fd
from basesmoother import SmootherObstacleProblem

secpera = 31556926.0        # seconds per year

def extend(mesh, f):
    '''On an extruded mesh extend a function f(x,z), already defined on the
    base mesh, to the mesh using the 'R' constant-in-the-vertical space.'''
    Q1R = fd.FunctionSpace(mesh, 'P', 1, vfamily='R', vdegree=0)
    fextend = fd.Function(Q1R)
    fextend.dat.data[:] = f.dat.data_ro[:]
    return fextend

def D(w):
    return 0.5 * (fd.grad(w) + fd.grad(w).T)

class SmootherStokes(SmootherObstacleProblem):
    '''Smoother for solving the steady-geometry Stokes problem.  Generates an
    extruded mesh for each residual evaluation.  Implements projected
    nonlinear versions of the Richardson and Jacobi smoothers.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        # smoother name
        self.name = 'SmootherStokes'
        # physical parameters
        self.g = 9.81                    # m s-2
        self.rhoi = 910.0                # kg m-3
        self.nglen = 3.0
        self.A3 = 1.0e-16 / secpera      # Pa-3 s-1;  EISMINT I ice softness
        self.B3 = self.A3**(-1.0/3.0)    # Pa s(1/3);  ice hardness
        # used in Stokes solver
        self.Dtyp = 1.0 / secpera        # s-1
        self.sc = 1.0e-7                 # velocity scale for symmetric scaling
        # parameters for initial condition, a Bueler profile; see van der Veen
        #   (2013) section 5.3
        self.buelerL = 10.0e3       # half-width of sheet
        self.buelerH0 = 1000.0      # center thickness
        self.Gamma = 2.0 * self.A3 * (self.rhoi * self.g)**self.nglen
        self.Gamma /= (self.nglen + 2.0)
        # we store the basemesh info and the bed elevation
        self.basemesh = None
        self.mx = None
        self.b = None
        self.saveflag = False
        self.savename = None

    def savestatenextresidual(self, name):
        '''On next call to residual(), save the state.'''
        self.saveflag = True
        self.savename = name

    def _regDu2(self, u):
        reg = self.args.eps * self.Dtyp**2
        return 0.5 * fd.inner(D(u), D(u)) + reg

    def stresses(self, mesh, u):
        ''' Generate effective viscosity and tensor-valued deviatoric stress
        from the velocity solution.'''
        Q1 = fd.FunctionSpace(mesh,'Q',1)
        Du2 = self._regDu2(u)
        r = 1.0 / self.nglen - 1.0
        assert self.nglen == 3.0
        nu = fd.Function(Q1).interpolate(0.5 * self.B3 * Du2**(r/2.0))
        nu.rename('effective viscosity (Pa s)')
        TQ1 = fd.TensorFunctionSpace(mesh, 'Q', 1)
        tau = fd.Function(TQ1).interpolate(2.0 * nu * D(u))
        tau /= 1.0e5
        tau.rename('tau (bar)')
        return nu, tau

    def savestate(self, mesh, u, p, kres):
        ''' Save state and diagnostics into .pvd file.'''
        assert self.saveflag == True
        assert self.savename is not None
        assert len(self.savename) > 0
        nu, tau = self.stresses(mesh, u)
        u *= secpera
        u.rename('velocity (m a-1)')
        p /= 1.0e5
        p.rename('pressure (bar)')
        kres.rename('kinetic residual (a=0)')
        print('saving u,p,nu,tau,kres to %s' % self.savename)
        fd.File(self.savename).write(u,p,nu,tau,kres)
        self.saveflag = False
        self.savename = None

    def solvestokes(self, mesh, printsizes=False):
        '''Solve the Glen-Stokes problem on the input extruded mesh.
        Returns the separate velocity and pressure solutions.'''

        # set up mixed method for Stokes dynamics problem
        V = fd.VectorFunctionSpace(mesh, 'Lagrange', 2)
        W = fd.FunctionSpace(mesh, 'Lagrange', 1)
        if printsizes:
            print('      dimensions n_u = %d, n_p = %d' % (V.dim(), W.dim()))
        Z = V * W
        up = fd.Function(Z)
        scu, p = fd.split(up)       # scaled velocity, unscaled pressure
        v, q = fd.TestFunctions(Z)

        # symmetrically-scaled Glen-Stokes weak form
        fbody = fd.Constant((0.0, - self.rhoi * self.g))
        sc = self.sc
        Du2 = self._regDu2(scu * sc)
        assert self.nglen == 3.0
        nu = 0.5 * self.B3 * Du2**((1.0 / self.nglen - 1.0)/2.0)
        F = ( sc*sc * fd.inner(2.0 * nu * D(scu), D(v)) \
              - sc * p * fd.div(v) - sc * q * fd.div(scu) \
              - sc * fd.inner(fbody, v) ) * fd.dx

        # zero Dirichlet on base (and stress-free on top and cliffs)
        bcs = [ fd.DirichletBC(Z.sub(0), fd.Constant((0.0, 0.0)), 'bottom')]

        # Newton-LU solve Stokes, split, descale, and return
        par = {'snes_linesearch_type': 'bt',
               'snes_max_it': 200,
               'snes_rtol': 1.0e-4,    # not as tight as default 1.0e-8
               'snes_stol': 0.0,       # expect CONVERGED_FNORM_RELATIVE
               'ksp_type': 'preonly',
               'pc_type': 'lu',
               'pc_factor_shift_type': 'inblocks'}
        fd.solve(F == 0, up, bcs=bcs, options_prefix='s', solver_parameters=par)
        u, p = up.split()
        u *= sc
        return u, p

    def createbasemesh(self, mesh1d):
        '''Create a Firedrake interval base mesh matching mesh1d.  Also store
        the bed elevation.'''
        self.mx = mesh1d.m + 1
        self.basemesh = fd.IntervalMesh(self.mx, length_or_left=0.0,
                                        right=mesh1d.xmax)
        self.b = self.phi(mesh1d.xx())

    def extrudetogeometry(self, s, report=False):
        '''Generate extruded mesh over self.basemesh, to height s.  The icy
        columns get their height from s, with minimum height args.Hmin.  By
        default the extruded mesh has empty (0-element) columns if ice-free
        according to s.  If args.padding==True then the whole extruded mesh has
        the same layer count.  Optional reporting of mesh stats.'''
        assert self.basemesh is not None
        if report:
            print('mesh: base of %d elements (intervals)' \
                  % self.mx)
        # extrude to temporary total height 1.0
        mz = self.args.mz
        if self.args.padding:
            assert self.args.Hmin > 0.0, \
                'padding requires minimum positive thickness'
            mesh = fd.ExtrudedMesh(self.basemesh, mz, layer_height=1.0 / mz)
            if report:
                print('      extruded is padded, has %d x %d elements' \
                      % (self.mx, mz))
        else:
            layermap = np.zeros((self.mx, 2), dtype=int)  # [[0,0], ..., [0,0]]
            thk = s - self.b
            thkelement = ( (thk[:-1]) + (thk[1:]) ) / 2.0
            icyelement = (thkelement > self.args.Hmin + 1.0e-3)
            layermap[:,1] = mz * np.array(icyelement, dtype=int)
            # FIXME: in parallel we must provide local, haloed layermap
            mesh = fd.ExtrudedMesh(self.basemesh, layers=layermap,
                                   layer_height=1.0 / mz)
            if report:
                icycount = sum(icyelement)
                print('      extruded has %d x %d icy elements and %d ice-free base elements' \
                      % (icycount, mz, self.mx - icycount))
        # put s(x) into a Firedrake function on the base mesh
        P1base = fd.FunctionSpace(self.basemesh, 'Lagrange', 1)
        sbase = fd.Function(P1base)
        sbase.dat.data[:] = np.maximum(s, self.args.Hmin)
        # change mesh height to s(x)
        x, z = fd.SpatialCoordinate(mesh)
        xxzz = fd.as_vector([x, extend(mesh, sbase) * z])
        coords = fd.Function(mesh.coordinates.function_space())
        mesh.coordinates.assign(coords.interpolate(xxzz))
        return mesh

    def extracttop(self, mesh1d, mesh, field):
        '''On an extruded mesh with some ice-free (i.e. empty) columns, loop
        over the base mesh finding top cells where ice is present, then top
        nodes, and evaluate the field there.  Only works for Q1 fields.
        (Thanks Lawrence Mitchell.)'''
        assert self.basemesh is not None
        # get the cells from basemesh and mesh
        bmP1 = fd.FunctionSpace(self.basemesh, 'Lagrange', 1)
        bmcnm = bmP1.cell_node_map().values
        Q1 = fd.FunctionSpace(mesh, 'Lagrange', 1)
        cnm = Q1.cell_node_map().values
        coff = Q1.cell_node_map().offset  # node offset in column
        # get the cell-wise indexing scheme
        section, iset, facets = Q1.cell_boundary_masks
        # facets ordered with sides first, then bottom, then top
        off = section.getOffset(facets[-1])
        dof = section.getDof(facets[-1])
        topind = iset[off:off+dof]  # nodes on top of a cell
        assert len(topind) == 2
        # loop over base mesh cells computing top-node field value
        f = mesh1d.zeros()
        for cell in range(self.basemesh.cell_set.size):
            start, extent = mesh.cell_set.layers_array[cell]
            ncell = extent - start - 1
            if ncell == 0:
                continue  # leave r unchanged for these base mesh nodes
            topcellnodes = cnm[cell, ...] + coff * ncell - 1
            f_all = field.dat.data_ro[topcellnodes] # at ALL nodes in top cell
            f[bmcnm[cell,...]] = f_all[topind]
        return f

    def residual(self, mesh1d, s, ella):
        '''Compute the residual functional, namely the surface kinematical
        residual for the entire domain, for a given iterate s.  In symbols
        matching the paper, returns  r = F(s)[.] - ella(.)  where . ranges over
        hat functions on mesh1d.  Note mesh1D is a MeshLevel1D instance and
        ella(.) = <a(x),.> in V^j' is the CMB.  This residual evaluation
        calls extrudemesh() to set up an (x,z) Firedrake mesh by extrusion of a
        (stored) base mesh.  If saveupname is a string then the Stokes
        solution (u,p) is saved to that file.  The returned residual array is
        defined on mesh1d and is in the dual space V^j'.'''
        firstcall = (self.basemesh == None)
        if firstcall: # if needed, generate self.basemesh from mesh1d
            self.createbasemesh(mesh1d)
        # solve the Glen-Stokes problem on the extruded mesh
        mesh = self.extrudetogeometry(s, report=firstcall)
        u, p = self.solvestokes(mesh, printsizes=firstcall)
        # evaluate kinematic part of residual  (note - u|_s . n_s = u ds/dx - w)
        _, z = fd.SpatialCoordinate(mesh)
        kres_ufl = u[0] * z.dx(0) - u[1]
        Q1 = fd.FunctionSpace(mesh, 'Lagrange', 1)
        kres = fd.Function(Q1).interpolate(kres_ufl)
        if self.saveflag:
            self.savestate(mesh, u, p, kres)
        # get kinematic residual r = - u|_s . n_s on z = s(x)
        if self.args.padding:
            # in this case the 'top' BC nodes are all we need
            topbc = fd.DirichletBC(Q1, 1.0, 'top')
            r = kres.dat.data_ro[topbc.nodes]
        else:
            # if some columns are ice-free then nontrivial extraction needed
            r = self.extracttop(mesh1d, mesh, kres)
        # include the climatic mass balance: - u|_s . n_s - a
        return mesh1d.ellf(r) - ella

    def smoothersweep(self, mesh1d, s, ella, phi, currentr=None):
        '''Do in-place smoothing on s(x).  On input, set currentr to a vector
        to avoid re-computing the residual.  Computes and returns the residual
        after the sweep.'''
        mesh1d.checklen(s)
        mesh1d.checklen(ella)
        mesh1d.checklen(phi)
        if currentr is None:
            currentr = self.residual(mesh1d, s, ella)
        if self.args.smoother == 'richardson':
            negd = self.richardsonsweep(s, phi, currentr)
        elif self.args.smoother == 'jacobislow':
            negd = self.jacobislowsweep(mesh1d, s, ella, phi, currentr)
        elif self.args.smoother == 'gsslow':
            negd = self.gsslowsweep(mesh1d, s, ella, phi, currentr)
        mesh1d.WU += 1
        if self.args.monitor and len(negd) > 0:
            print('      negative diagonal entries: ', end='')
            print(negd)
        return self.residual(mesh1d, s, ella)

    def richardsonsweep(self, s, phi, r):
        '''Do in-place projected nonlinear Richardson smoothing on s(x):
            s <- max(s - omega * r, phi)
        User must adjust omega to reasonable level.  (Do this with SIA-type
        stability criterion argument.)'''
        np.maximum(s - self.args.omega * r, phi, s)
        return []

    def jacobislowsweep(self, mesh1d, s, ella, phi, r,
                        eps=1.0, dump=False):
        '''Do in-place projected nonlinear Jacobi smoothing on s(x)
        where the diagonal entry d_i = F'(s)[psi_i,psi_i] is computed
        by VERY SLOW finite differencing of expensive residual calculations.
        If d_i > 0 then
            snew_i <- max(s_i - omega * r_i / d_i, phi_i)
        but otherwise
            snew_i <- phi_i.
        After snew is completed we do s <- snew.'''
        snew = s.copy()
        negd = []
        for j in range(1, len(s)-1): # loop over interior points
            sperturb = s.copy()
            sperturb[j] += eps
            if dump:
                self.savestatenextresidual(self.args.o + '_jacobi_%d.pvd' % j)
            rperturb = self.residual(mesh1d, sperturb, ella)
            d = (rperturb[j] - r[j]) / eps
            if d > 0.0:
                snew[j] = max(s[j] - self.args.omega * r[j] / d, phi[j])
            else:
                snew[j] = phi[j]
                negd.append(j)
        s[:] = snew[:] # in-place copy
        return negd

    def gsslowsweep(self, mesh1d, s, ella, phi, r,
                    eps=1.0, dump=False):
        '''Do in-place projected nonlinear Gauss-Seidel smoothing on s(x)
        where the diagonal entry d_i = F'(s)[psi_i,psi_i] is computed
        by VERY SLOW finite differencing of expensive residual calculations.
        If d_i > 0 then
            s_i <- max(s_i - omega * r_i / d_i, phi_i)
        but otherwise
            s_i <- phi_i.'''
        negd = []
        for j in range(1, len(s)-1): # loop over interior points
            sperturb = s.copy()
            sperturb[j] += eps
            if dump:
                self.savestatenextresidual(self.args.o + '_gs_%d.pvd' % j)
            rperturb = self.residual(mesh1d, sperturb, ella)
            d = (rperturb[j] - r[j]) / eps
            if d > 0.0:
                s[j] = max(s[j] - self.args.omega * r[j] / d, phi[j])
            else:
                s[j] = phi[j]
                negd.append(j)
            # must recompute residual for s (nonlocal!)
            r = self.residual(mesh1d, s, ella)
        return negd

    def phi(self, x):
        '''For now we have a flat bed.'''
        return np.zeros(np.shape(x))

    def source(self, x):
        '''Continuous source term, i.e. mass balance, for Bueler profile.
        See van der Veen (2013) equations (5.49) and (5.51).  Assumes x
        is a numpy array.'''
        n = self.nglen
        invn = 1.0 / n
        r1 = 2.0 * n + 2.0                   # e.g. 8
        s1 = (1.0 - n) / n                   #     -2/3
        C = self.buelerH0**r1 * self.Gamma   # A_0 in van der Veen is Gamma here
        C /= ( 2.0 * self.buelerL * (1.0 - 1.0 / n) )**n
        xc = self.args.domainlength / 2.0
        X = (x - xc) / self.buelerL          # rescaled coord
        m = np.zeros(np.shape(x))
        # usual formula for 0 < |X| < 1
        zzz = (abs(X) > 0.0) * (abs(X) < 1.0)
        if any(zzz):
            Xin = abs(X[zzz])
            Yin = 1.0 - Xin
            m[zzz] = (C / self.buelerL) \
                     * ( Xin**invn + Yin**invn - 1.0 )**(n-1.0) \
                     * ( Xin**s1 - Yin**s1 )
        # fill singular origin with near value
        if any(X == 0.0):
            Xnear = 1.0e-8
            Ynear = 1.0 - Xnear
            m[X == 0.0] = (C / self.buelerL) \
                          * ( Xnear**invn + Ynear**invn - 1.0 )**(n-1.0) \
                          * ( Xnear**s1 - Ynear**s1 )
        # extend by ablation
        if any(abs(X) >= 1.0):
            m[abs(X) >= 1.0] = min(m)
        return m

    def initial(self, x):
        '''Default initial shape is the Bueler profile.  See van der Veen (2013)
        equation (5.50).  Assumes x is a numpy array.'''
        n = self.nglen
        p1 = n / (2.0 * n + 2.0)                  # e.g. 3/8
        q1 = 1.0 + 1.0 / n                        #      4/3
        Z = self.buelerH0 / (n - 1.0)**p1         # outer constant
        xc = self.args.domainlength / 2.0
        X = (x - xc) / self.buelerL               # rescaled coord
        Xin = abs(X[abs(X) < 1.0])                # rescaled distance from
                                                  #   center, in ice
        Yin = 1.0 - Xin
        s = np.zeros(np.shape(x))                 # correct outside ice
        s[abs(X) < 1.0] = Z * ( (n + 1.0) * Xin - 1.0 \
                                + n * Yin**q1 - n * Xin**q1 )**p1
        return s
