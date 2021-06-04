'''Module for SmootherStokes class derived from SmootherObstacleProblem.'''

import numpy as np
import firedrake as fd
from basesmoother import SmootherObstacleProblem
from problem import secpera, g, rhoi, nglen, B3

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

    def __init__(self, args, b, admissibleeps=1.0e-10):
        super().__init__(args, admissibleeps=admissibleeps)
        self.b = b
        # smoother name
        self.name = 'SmootherStokes'
        # used in Stokes solver
        self.Dtyp = 1.0 / secpera        # s-1
        self.sc = 1.0e-7                 # velocity scale for symmetric scaling
        # we store the basemesh info and the bed elevation
        self.basemesh = None
        self.mx = None
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
        r = 1.0 / nglen - 1.0
        assert nglen == 3.0
        nu = fd.Function(Q1).interpolate(0.5 * B3 * Du2**(r/2.0))
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
        kres.rename('kinematic residual (a=0)')
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
        fbody = fd.Constant((0.0, - rhoi * g))
        sc = self.sc
        Du2 = self._regDu2(scu * sc)
        assert nglen == 3.0
        nu = 0.5 * B3 * Du2**((1.0 / nglen - 1.0)/2.0)
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
        mesh1d.checklen(self.b)
        self.basemesh = fd.IntervalMesh(self.mx, length_or_left=0.0,
                                        right=mesh1d.xmax)

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

    def kinematical(self, mesh, u):
        ''' Evaluate kinematic part of residual from given velocity u, namely
        as a field defined on the whole extruded mesh:
            kres = u ds/dx - w.
        Note n_s = <-s_x, 1> so this is <u,w> . n_s.'''
        _, z = fd.SpatialCoordinate(mesh)
        kres_ufl = u[0] * z.dx(0) - u[1]
        Q1 = fd.FunctionSpace(mesh, 'Lagrange', 1)
        kres = fd.Function(Q1).interpolate(kres_ufl)
        return kres

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
        # get kinematical part of residual
        kres = self.kinematical(mesh, u)
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

    def viewperturb(self, s, klist, eps=1.0):
        '''For given s(x), compute solution perturbations from s[k] + eps,
        i.e. lifting surface by eps, at an each icy interior node in klist.
        Saves du,dp,dres to file self.savename, a .pvd file.'''
        assert self.basemesh is not None
        assert self.savename is not None
        assert len(self.savename) > 0
        # solve the Glen-Stokes problem on the unperturbed extruded mesh
        meshs = self.extrudetogeometry(s)
        us, ps = self.solvestokes(meshs)
        kress = self.kinematical(meshs, us)
        # solve on the PERTURBED extruded mesh
        sP = s.copy()
        for k in klist:
            if k < 1 or k > len(s)-2:
                print('WARNING viewperturb(): skipping non-interior node k=%d' \
                      % k)
            elif s[k] > self.b[k] + 0.001:
                sP[k] += eps
            else:
                print('WARNING viewperturb(): skipping bare-ground node k=%d' \
                      % k)
        meshP = self.extrudetogeometry(sP)
        uP, pP = self.solvestokes(meshP)
        kresP = self.kinematical(meshP, uP)
        # compute difference as a function on the unperturbed mesh
        V = fd.VectorFunctionSpace(meshs, 'Lagrange', 2)
        W = fd.FunctionSpace(meshs, 'Lagrange', 1)
        du = fd.Function(V)
        du.dat.data[:] = uP.dat.data_ro[:] - us.dat.data_ro[:]
        du *= secpera
        du.rename('du (m a-1)')
        dp = fd.Function(W)
        dp.dat.data[:] = pP.dat.data_ro[:] - ps.dat.data_ro[:]
        dp /= 1.0e5
        dp.rename('dp (bar)')
        # dres is difference of full residual cause a(x) cancels
        dres = fd.Function(W)
        dres.dat.data[:] = kresP.dat.data_ro[:] - kress.dat.data_ro[:]
        dres.rename('dres')
        print('saving perturbations du,dp,dres to %s' % self.savename)
        fd.File(self.savename).write(du,dp,dres)

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
