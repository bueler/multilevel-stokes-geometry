'''Module for SmootherStokes class derived from SmootherObstacleProblem.'''

import numpy as np

class SmootherObstacleProblem:
    '''A smoother on an obstacle problem.  Works on a mesh of class MeshLevel1D and calls a solver of class GlenStokes.  Note the mesh holds the bed
    elevation (obstacle).

    Implements Richardson plus FD Jacobian GS and Jacobi methods.  These
    methods do not assume the residual functional has an easily-accessible
    Jacobian or Jacobian diagonal.

    The public interface implements residual evaluation and application of the
    in-place smoother:
        smooth = SmootherObstacleProblem(args, solver)
        r = smooth.residual(mesh1d, s, ella)
        smooth.smoothersweep(mesh1d, s, ella)
    There is also evaluation of the CP norm,
        irnorm = smooth.inactiveresidualnorm(mesh1d, s, r)
    and a routine to trigger output:
        smooth.savestatenextresidual(filename)
    '''

    def __init__(self, args, solver, admissibleeps=1.0e-10):
        self.args = args
        self.solver = solver
        self.admissibleeps = admissibleeps
        self.created = False
        self.saveflag = False

    def _checkadmissible(self, mesh1d, w, phi):
        '''Check admissibility and stop if not.'''
        for p in range(1, mesh1d.m+1):
            if w[p] < phi[p] - self.admissibleeps:
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

    def inactiveresidualnorm(self, mesh1d, s, r, ireps=0.001):
        '''Compute the norm of the residual values at nodes where the constraint
        is NOT active.  Where the constraint is active the residual r=F(s) in
        the complementarity problem is allowed to have any positive value;
        only the residual at inactive nodes is relevant to convergence.'''
        F = r.copy()
        F[s < mesh1d.b + ireps] = np.minimum(F[s < mesh1d.b + ireps], 0.0)
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
        elif self.args.smoother == 'jacobislow':
            negd = self._jacobislowsweep(mesh1d, s, ella, currentr)
        elif self.args.smoother == 'jacobicolor':
            negd = self._jacobicolorsweep(mesh1d, s, ella, currentr)
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
        If d_i > 0 then
            s_i <- max(s_i - omega * r_i / d_i, b_i)
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
                s[j] = max(s[j] - self.args.omega * r[j] / d, mesh1d.b[j])
            else:
                s[j] = mesh1d.b[j]
                negd.append(j)
            # must recompute residual for s (nonlocal!)
            r = self.residual(mesh1d, s, ella)
        return negd

    def _jacobicolorsweep(self, mesh1d, s, ella, r,
                          eps=1.0, dump=False):
        '''Do in-place projected nonlinear Jacobi smoothing on s(x)
        where the diagonal entry d_i = F'(s)[psi_i,psi_i] is computed
        by SLOW finite differencing of expensive residual calculations, but
        using coloring.  Nodes of the same color are separated by cperthickness
        times the maximum ice thickness.  Set -cperthickness 1.0e10 (for
        example) to use VERY SLOW finite differencing without coloring.
        If d_i > 0 then
            snew_i <- max(s_i - omega * r_i / d_i, b_i)
        but otherwise
            snew_i <- b_i.
        After snew is completed we do s <- snew.'''
        thkmax = max(s - mesh1d.b)
        c = int(np.ceil(self.args.cperthickness * thkmax / mesh1d.h))
        c = min([c,mesh1d.m+1])
        print('      c = %d' % c)
        snew = mesh1d.b.copy() - 1.0  # snew NOT admissible; note check below
        snew[0], snew[mesh1d.m+1] = mesh1d.b[0], mesh1d.b[mesh1d.m+1]
        negd = []
        for k in range(c):
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
