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

    Implements nonlinear Richardson plus nonlinear GS and Jacobi methods based
    on the computation of the Jacobian diagonal entries by finite-differencing.
    (These methods do not assume the residual has an easily-accessible
    Jacobian diagonal.)

    The public interface implements residual evaluation and application of the
    in-place smoother:
        smooth = ObstacleSmoother(args, solver)
        r = smooth.residual(mesh1d, s, ella)
        smooth.smoothersweep(mesh1d, s, ella)
    There is also evaluation of the CP norm,
        irnorm = smooth.inactiveresidualnorm(mesh1d, s, r)
    and a routine to trigger output:
        smooth.savestatenextresidual(filename)
    '''

    def __init__(self, args, solver):
        self.args = args
        self.solver = solver
        self.ieps = 0.001  # if s[j] > b[j] + ieps then node j is inactive
        self.created = False
        self.saveflag = False

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

    def inactiveresidualnorm(self, mesh1d, s, r):
        '''Compute the norm of the residual values at nodes where the constraint
        is NOT active.  Where the constraint is active the residual r=F(s) in
        the complementarity problem is allowed to have any positive value;
        only the residual at inactive nodes is relevant to convergence.'''
        F = r.copy()
        F[s < mesh1d.b + self.ieps] = np.minimum(F[s < mesh1d.b + self.ieps],
                                                 0.0)
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
        if self.args.smoother == 'richardson':
            negd = self._richardsonsweep(mesh1d, s, ella, currentr)
        elif self.args.smoother == 'gsslow':
            negd = self._gsslowsweep(mesh1d, s, ella, currentr)
        elif self.args.smoother == 'jacobicolor':
            negd = self._jacobicolorsweep(mesh1d, s, ella, currentr)
        elif self.args.smoother == 'newtonrslu':
            negd = self._newtonrslu(mesh1d, s, ella, currentr)
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
        '''Compute c which is the number of colors (and the gap between nodes
        with the same color).
        Nodes of the same color are separated by cperthickness
        times the maximum ice thickness.  Set e.g. -cperthickness 1.0e10
        to use VERY SLOW finite differencing without coloring.'''
        thkmax = max(s - mesh1d.b)
        c = int(np.ceil(self.args.cperthickness * thkmax / mesh1d.h))
        return min([c, mesh1d.m+1])

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
        print('      c = %d colors' % c)
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

    def _newtonrslu(self, mesh1d, s, ella, r,
                    eps=1.0, dump=False, band=1):
        '''Do one in-place projected, reduced-space Newton step on s(x).
        First computes an approximated Jacobian using coloring.  It is
        band-limited to [-band,band] relative to the diagonal; band = 1 gives
        tridiagonal.  We require band + 2 <= c where c is the gap generated by
        coloring.  We apply LU from PETSc to solve the linear step equations,
        and do projected line search.'''
        # do coloring
        c = self._colors(mesh1d, s)
        assert band + 2 <= c
        # find inactive indices; FIXME include marginal active nodes?
        II = []
        for j in range(1, mesh1d.m+1):
            if s[j] > mesh1d.b[j] + self.ieps:
                II.append(j)
        mA = len(II)
        print('      c = %d colors and mA = %d inactive nodes' % (c,mA))
        assert mA >= 1
        # generate Jacobian approximation by finite differencing
        Jac = PETSc.Mat()
        Jac.create(PETSc.COMM_WORLD)
        Jac.setSizes((mA,mA))
        Jac.setFromOptions()
        Jac.setUp()  # FIXME consider pre-allocation
        negd = []
        for k in range(c):
            # note jlist = [k+1,] (singleton) if k+1+c >= mesh1d.m+1
            jlist = np.arange(k+1, mesh1d.m+1, c, dtype=int)
            #print(jlist)
            sperturb = s.copy()
            sperturb[jlist] += eps
            if dump:
                self.savestatenextresidual(self.args.o + '_jacobi_%d.pvd' % j)
            rperturb = self.residual(mesh1d, sperturb, ella)
            for j in list(set(II) & set(jlist)):  # colored inactive nodes
                row = II.index(j)
                col, val = [], []
                for l in range(j-band,j+band+1):
                    if l in II:
                        col.append(II.index(l))
                        ajl = (rperturb[l] - r[l]) / eps
                        val.append(ajl)
                        if l == j and ajl < 0.0:
                            negd.append(j)
                Jac.setValues([row,],col,val)
        Jac.assemblyBegin()
        Jac.assemblyEnd()
        #Jac.view()
        # right-hand side of Newton system:  Jac(s) ds = - r(s)
        mr = PETSc.Vec()
        mr.create(PETSc.COMM_WORLD)
        mr.setSizes(mA)
        mr.setFromOptions()
        mr.setValues(range(mA), -r[II])
        mr.assemblyBegin()
        mr.assemblyEnd()
        #mr.view()
        # solve for step in s
        # solver feedback works, for example:
        #   -newt_ksp_view
        #   -newt_ksp_converged_reason
        #   -newt_ksp_view_mat
        #   -newt_ksp_view_rhs
        #   -newt_pc_type svd -newt_pc_svd_monitor
        ksp = PETSc.KSP()
        ksp.create(PETSc.COMM_WORLD)
        ksp.setOptionsPrefix('newt_')
        ksp.setOperators(A=Jac, P=Jac)
        ksp.setType('preonly')
        ksp.getPC().setType('lu')
        ksp.setFromOptions()
        # solve Newton step
        ds = mr.duplicate()
        ksp.solve(mr, ds)
        #ds.view()
        # FIXME: do projected line search
        s[II] += ds
        return negd
