'''Module for ObstacleSmoother.'''

import numpy as np

# one smoother type sets up a PETSc Mat and KSP system
import sys
import petsc4py
petsc4py.init(sys.argv)  # must come before "import PETSc"
from petsc4py import PETSc

class ObstacleSmoother:
    '''A smoother on an obstacle problem.  Works on a mesh of class MeshLevel1D and calls a solver of class GlenStokes.  Note the mesh holds the bed
    elevation (obstacle).

    Implements:
        * nonlinear Richardson (pointwise)
        * nonlinear Gauss-Seidel (pointwise)
        * nonlinear Jacobi (pointwise, coloring)
        * Newton reduced-space (coloring)
    In all cases the Jacobian entries are computed by finite-differencing,
    but only the diagonal is used in the pointwise smoothers.  The coloring
    smoothers get multiple rows of the Jacobian matrix by assigning different
    colors to points which are more than a certain number of ice thicknesses
    apart.

    The public interface implements residual evaluation and application of the
    in-place smoother:
        smooth = ObstacleSmoother(args, solver)
        r = smooth.residual(mesh1d, s, ella)
        smooth.smoothersweep(mesh1d, s, ella)
    Note smoothersweep() calls a smoother from the dictionary self.smoothers.

    There is also evaluation of the CP norm,
        irnorm = smooth.cpresidualnorm(mesh1d, s, r)
    and a routine to trigger output:
        smooth.savestatenextresidual(filename)
    '''

    def __init__(self, args, solver):
        self.args = args
        self.solver = solver
        self.created = False
        self.saveflag = False
        self.smoothers = {'richardson':  self._richardsonsweep,
                          'gsslow':      self._gsslowsweep,
                          'jacobicolor': self._jacobicolorsweep,
                          'newtonrs':    self._newtonrs}

    def _checkadmissible(self, mesh1d, w, phi):
        '''Check admissibility and stop if not.'''
        for p in range(1, mesh1d.m+1):
            if w[p] < phi[p]:
                print('ERROR: inadmissible w[%d]=%e < phi[%d]=%e (m=%d)' \
                      % (p, w[p], p, phi[p], mesh1d.m))
                sys.exit(0)

    def _sweepindices(self, mesh1d, forward=True):
        '''Generate indices for sweep.'''
        if forward:
            ind = range(1, mesh1d.m+1)    # 1,...,m
        else:
            ind = range(mesh1d.m, 0, -1)  # m,...,1
        return ind

    def cpresidualnorm(self, mesh1d, s, r):
        '''Compute the norm of the residual values at nodes where the constraint
        is NOT active.  Where the constraint is active the residual r=F(s) in
        the complementarity problem is allowed to have any positive value;
        only the residual at inactive nodes is relevant to convergence.'''
        F = r.copy()
        F[s <= mesh1d.b] = np.minimum(F[s <= mesh1d.b], 0.0)
        return mesh1d.l2norm(F)

    def savestatenextresidual(self, name):
        '''On next call to residual(), save the state.'''
        self.saveflag = True
        self.savename = name

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
        # set up base mesh (if needed) and extruded mesh
        if not self.created:
            self.solver.createbasemesh(mx=mesh1d.m+1, xmax=mesh1d.xmax)
        mesh = self.solver.extrudetogeometry(s, mesh1d.b,
                                             report=not self.created)
        # solve the Glen-Stokes problem on the extruded mesh
        u, p, kres = self.solver.solve(mesh, printsizes=not self.created)
        if not self.created:
            self.created = True
        if self.saveflag:
            self.solver.savestate(mesh, u, p, kres, savename=self.savename)
            self.saveflag = False
        # get kinematic residual r = - u|_s . n_s on z = s(x)
        r = self.solver.extracttop(mesh, kres)
        # include the climatic mass balance: - u|_s . n_s - a
        return mesh1d.ellf(r) - ella

    def smoothersweep(self, mesh1d, s, ella, currentr=None):
        '''Do in-place smoothing on s(x).  On input, set currentr to a vector
        to avoid re-computing the residual.  Computes and returns the residual
        after the sweep.'''
        mesh1d.checklen(s)
        mesh1d.checklen(ella)
        mesh1d.checklen(mesh1d.b)
        if currentr is None:
            currentr = self.residual(mesh1d, s, ella)
        negd = self.smoothers[self.args.smoother](mesh1d, s, ella, currentr)
        mesh1d.WU += 1
        if self.args.monitor and len(negd) > 0:
            print('      negative diagonal entries: ', end='')
            print(negd)
        return self.residual(mesh1d, s, ella)

    def _richardsonsweep(self, mesh1d, s, ella, r):
        '''Do in-place projected nonlinear Richardson smoothing on s(x):
            s <- max(s - omega * r, b)
        User must adjust omega to reasonable level.  (Do this with SIA-type
        stability criterion argument.)'''
        np.maximum(s - self.args.omega * r, mesh1d.b, s)
        return []

    def _gsslowsweep(self, mesh1d, s, ella, r,
                     eps=1.0, dump=False):
        '''Do in-place projected nonlinear Gauss-Seidel smoothing on s(x)
        where the diagonal entry d_i = F'(s)[psi_i,psi_i] is computed
        by VERY SLOW finite differencing of expensive residual calculations
        at every point.
        For each i, compute r = F(s).  If d_i > 0 then
            s_i <- max(s_i - omega * r_i / d_i, b_i)
        but otherwise
            s_i <- phi_i.
        Note s_i is updated immediately.'''
        negd = []
        for j in range(1, len(s)-1): # loop over interior points
            sperturb = s.copy()
            sperturb[j] += eps
            if dump:
                self.savestatenextresidual(self.args.o + '_gs_%d.pvd' % j)
            rperturb = self.residual(mesh1d, sperturb, ella)
            d = (rperturb[j] - r[j]) / eps
            if d > 0.0:
                s[j] = max(s[j] - self.args.omega * r[j] / d, mesh1d.b[j])
            else:
                s[j] = mesh1d.b[j]
                negd.append(j)
            # must recompute residual for s (nonlocal!)
            r = self.residual(mesh1d, s, ella)
        return negd

    def _colors(self, mesh1d, s):
        '''Compute c which is the number of colors and the gap between nodes
        with the same color.  Nodes of the same color are separated by
        cperthickness times the maximum ice thickness.  Set -cperthickness X
        where X is greater than ice sheet width to turn off coloroing.  (Thus
        VERY SLOW finite differencing without coloring.)'''
        thkmax = max(s - mesh1d.b)
        if thkmax == 0.0:
            c = mesh1d.m + 1
        else:
            c = int(np.ceil(self.args.cperthickness * thkmax / mesh1d.h))
            c = min([c, mesh1d.m + 1])
        if c >= mesh1d.m + 1:
            print('      [coloring off]')
        else:
            print('      c = %d colors' % c)
        return c

    def _jacobicolorsweep(self, mesh1d, s, ella, r,
                          eps=1.0, dump=False):
        '''Do in-place projected nonlinear Jacobi smoothing on s(x)
        where the diagonal entry d_i = F'(s)[psi_i,psi_i] is computed
        by SLOW finite differencing of expensive residual calculations, but
        using coloring.
        First r = F(s).  Then for each i, if d_i > 0 then
            snew_i <- max(s_i - omega * r_i / d_i, b_i)
        but otherwise
            snew_i <- b_i.
        After snew is completed we do s <- snew.'''
        c = self._colors(mesh1d, s)
        snew = mesh1d.b.copy() - 1.0  # snew NOT admissible; note check below
        snew[0], snew[mesh1d.m+1] = mesh1d.b[0], mesh1d.b[mesh1d.m+1]
        negd = []
        for k in range(c):
            # note jlist = [k+1,] (singleton) if k+1+c >= mesh1d.m+1
            jlist = np.arange(k+1, mesh1d.m+1, c, dtype=int)
            sperturb = s.copy()
            sperturb[jlist] += eps
            if dump:
                self.savestatenextresidual(self.args.o + '_jacobi_%d.pvd' % j)
            rperturb = self.residual(mesh1d, sperturb, ella)
            for j in jlist:
                d = (rperturb[j] - r[j]) / eps
                if d > 0.0:
                    snew[j] = max(s[j] - self.args.omega * r[j] / d,
                                  mesh1d.b[j])
                else:
                    snew[j] = mesh1d.b[j]
                    negd.append(j)
        # check on whether coloring scheme hit each node
        self._checkadmissible(mesh1d, snew, mesh1d.b)
        s[:] = snew[:] # in-place copy
        return negd

    def _fdjacobianband(self, mesh1d, s, ella,
                        currentr=None, eps=20.0, dump=False, band=1):
        '''Compute a banded approximation A of the Jacobian, using coloring,
        by evaluating the residual function.  A is band-limited to [-band,band]
        around the diagonal; band = 1 gives A tridiagonal.'''
        if currentr is None:
            r = self.residual(mesh1d, s, ella)
        else:
            r = currentr
        A = PETSc.Mat()
        A.create(PETSc.COMM_WORLD)
        A.setSizes((mesh1d.m, mesh1d.m))
        A.setFromOptions()
        A.setUp()  # FIXME need pre-allocation and parallel
        c = self._colors(mesh1d, s)
        # FIXME assert band + 2 <= c
        negd = []
        for k in range(c):
            # nodes of the color k
            nodelist = np.arange(k+1, mesh1d.m+1, c, dtype=int) # 1-based
            #    (note jlist = [k+1,] (singleton) if k+1+c >= mesh1d.m+1)
            sperturb = s.copy()
            sperturb[nodelist] += eps
            if dump:
                self.savestatenextresidual(self.args.o \
                                           + '_newtonrs_color%d.pvd' % k)
            rperturb = self.residual(mesh1d, sperturb, ella)
            # fill k-colored rows by finite differences
            for jnode in nodelist:
                row = jnode - 1
                # columns around diagonal
                col = list(range(max(0, row-band), min(mesh1d.m, row+band+1)))
                val = []
                for l in col:
                    lnode = l + 1
                    ajl = (rperturb[lnode] - r[lnode]) / eps
                    val.append(ajl)
                    if lnode == jnode and ajl < 0.0:
                        negd.append(jnode)
                A.setValues([row,], col, val)
        A.assemblyBegin()
        A.assemblyEnd()
        return A, negd

    def _changableset(self, mesh1d, s, r):
        '''Return index set (PETSc IS) for changable nodes, i.e. those with
        either s > b or r <= 0.'''
        #     FIXME include marginal active nodes?
        II = []
        for j in range(1, mesh1d.m+1):  # j is 1-based
            if s[j] > mesh1d.b[j] or r[j] <= 0.0:
                II.append(j - 1)
        ii = PETSc.IS()
        ii.createGeneral(II)
        return ii

    def _projectedlinesearch(self, mesh1d, s, ds, ella, r):
        '''Do projected line search as in Bueler 2021, Chapter 12, equations
        (12.18) and (12.19).  Starts from s and tries direction ds.
        Returns s which satisfies an Armijo criterion using the CP norm.'''
        sigma = 1.0e-4
        F0 = self.cpresidualnorm(mesh1d, s, r)
        beta = 1.0
        lsmax = 12
        for l in range(lsmax):
            sls = s.copy() + beta * ds                 # apply direction
            np.maximum(sls, mesh1d.b, out=sls)         # project
            rls = self.residual(mesh1d, sls, ella)     # new residual
            Fls = self.cpresidualnorm(mesh1d, sls, rls)
            if Fls <= (1.0 - sigma * beta) * F0:       # Armijo
                print('      line search: %d reductions' % l)
                return sls
            if l == lsmax - 1:
                print('WARNING: line search failure')
                return sls
            beta *= 0.5                                # next beta

    def _newtonrs(self, mesh1d, s, ella, r):
        '''Do one in-place reduced-space Newton step on s(x).  PETSc KSP solves
        the reduced linear step equations
           Jac_{I,I}(s) ds_I = - r_I(s)
        Also ds_A = 0 to form search direction ds.  Then we do projected line
        search to update s.'''
        # get Jacobian approximation (PETSc Mat)
        Jac, negd = self._fdjacobianband(mesh1d, s, ella, currentr=r)
        # get indices of changable (inactive) nodes (PETSc IS)
        ii = self._changableset(mesh1d, s, r)
        #ii.view()
        iione = np.array(ii, dtype=int) + 1  # 1-based for indexing nodes
        mA = ii.getSize()
        # extract changable submatrix from Jacobian
        JacII = Jac.createSubMatrix(ii, iscol=ii)
        #JacII.view()
        mII, nII = JacII.getSize()
        print('      size(J_{I,I}) = %d x %d' % (mII,nII))
        assert mA == mII == nII >= 1
        # right-hand side of Newton system
        rhs = PETSc.Vec()
        rhs.create(PETSc.COMM_WORLD)
        rhs.setSizes(mA)
        rhs.setFromOptions()
        rhs.setValues(range(mA), -r[iione])
        rhs.assemblyBegin()
        rhs.assemblyEnd()
        dsII = rhs.duplicate()
        # solve Newton system for step ds
        #   (note solver feedback works, for example: -newt_ksp_view
        #    -newt_ksp_converged_reason -newt_ksp_view_mat -newt_ksp_view_rhs
        #    -newt_pc_type svd -newt_pc_svd_monitor)
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setOptionsPrefix('newt_')
        ksp.setOperators(A=JacII, P=JacII)
        ksp.setType('preonly')
        ksp.getPC().setType('lu')
        ksp.setFromOptions()
        ksp.solve(rhs, dsII)
        #dsII.view()
        # search direction has ds=0 where not changable
        ds = mesh1d.zeros()
        ds[iione] = dsII[:]
        # get updated s from line search
        s[:] = self._projectedlinesearch(mesh1d, s, ds, ella, r)
        self._checkadmissible(mesh1d, s, mesh1d.b)
        return negd
