'''Module for abstract base class SmootherObstacleProblem.'''

__all__ = ['SmootherObstacleProblem']

import sys
from abc import ABC, abstractmethod
import numpy as np

class SmootherObstacleProblem(ABC):
    '''Abstact base class for a smoother on an obstacle problem.  Works on
    any mesh of class MeshLevel1D.'''

    def __init__(self, args, admissibleeps=1.0e-10):
        self.args = args
        self.admissibleeps = admissibleeps
        self.name = 'base'

    def _checkadmissible(self, mesh, w, phi):
        '''Check admissibility and stop if not.'''
        for p in range(1, mesh.m+1):
            if w[p] < phi[p] - self.admissibleeps:
                print('ERROR: inadmissible w[%d]=%e < phi[%d]=%e (m=%d)' \
                      % (p, w[p], p, phi[p], mesh.m))
                sys.exit(0)

    def _sweepindices(self, mesh, forward=True):
        '''Generate indices for sweep.'''
        if forward:
            ind = range(1, mesh.m+1)    # 1,...,m
        else:
            ind = range(mesh.m, 0, -1)  # m,...,1
        return ind

    def shownonzeros(self, z):
        '''Print a string indicating locations where array z is zero.'''
        Jstr = ''
        for k in range(len(z)):
            Jstr += '_' if z[k] == 0.0 else '*'
        print('  %d nonzeros: ' % sum(z > 0.0) + Jstr)

    def inactiveresidualnorm(self, mesh, w, r, phi, ireps=0.001):
        '''Compute the norm of the residual values at nodes where the constraint
        is NOT active.  Where the constraint is active the residual F(w) in the
        complementarity problem is allowed to have any positive value; only the
        residual at inactive nodes is relevant to convergence.'''
        F = r.copy()
        F[w < phi + ireps] = np.minimum(F[w < phi + ireps], 0.0)
        return mesh.l2norm(F)

    def smoother(self, iters, mesh, w, ell, phi):
        '''Apply iters sweeps of obstacle-problem smoother on mesh to modify w in place.  Alternate directions.'''
        forward = True
        for _ in range(iters):
            self.smoothersweep(mesh, w, ell, phi, forward=forward)
            forward = not forward

    @abstractmethod
    def residual(self, mesh, w, ell):
        '''Compute the residual functional for given iterate w.  Note
        ell is a source term in V^j'.'''

    @abstractmethod
    def smoothersweep(self, mesh, w, ell, phi, forward=True):
        '''Apply one sweep of obstacle-problem smoother on mesh to modify w in
        place.'''
