'''Module implementing a V-cycle of the nonlinear full approximation storage
(FAS) extension of the multilevel constraint decomposition (MCD) method.'''

__all__ = ['mcdnvcycle']

import numpy as np

def _levelreport(indent, j, m, sweeps):
    indentprint(indent - j, 'level %d: %d sweeps over m=%d nodes' \
                             % (j, sweeps, m))

def _coarsereport(indent, m, sweeps):
    indentprint(indent, 'coarsest: %d sweeps over m=%d nodes' \
                         % (sweeps, m))

def mcdnvcycle(args, smoother, hierarchy, s, ella, b, levels=None):
    '''Apply one V-cycle of the MCDN method.  Input args is an options
    dictionary with parameters.  Input smoother is of type
    SmootherObstacleProblem.  Note hierarchy[j] is of type MeshLevel1D,
    for j=0,...,args.J, but only args.jcoarse,...,args.J are actually used.
    This method generates all the defect constraints hierarchy[j].chi for
    j <= J, but hierarchy[j].b must
    be defined for all mesh levels.  The input iterate w is in V^j and
    linear functional ell is in V^J'.  The coarse solver is the same as
    the smoother.'''

    raise NotImplementedError('MCDN not implemented')
    assert args.down >= 0 and args.up >= 0 and args.coarse >= 0

    # set up on finest level
    hierarchy[J].checklen(w)
    hierarchy[J].checklen(ell)
    hierarchy[J].ell = ell
    hierarchy[J].g = w

    # downward
    for k in range(J, 0, -1):
        # compute next defect constraint using monotone restriction
        hierarchy[k-1].chi = hierarchy[k].mR(hierarchy[k].chi)
        # define down-obstacle
        phi = hierarchy[k].chi - hierarchy[k].cP(hierarchy[k-1].chi)
        # smooth the correction y
        if args.mgview:
            _levelreport(levels-1, k, hierarchy[k].m, args.down)
        hierarchy[k].y = hierarchy[k].zeros()
        obsprob.smoother(args.down, hierarchy[k], hierarchy[k].y,
                         hierarchy[k].ell, phi)
        # update residual
        wk = hierarchy[k].g + hierarchy[k].y
        F = obsprob.residual(hierarchy[k], wk, hierarchy[k].ell)
        # g^{k-1} is current solution restricted onto next-coarsest level
        hierarchy[k-1].g = hierarchy[k].iR(wk)
        # source on next-coarsest level
        hierarchy[k-1].ell = \
            obsprob.applyoperator(hierarchy[k-1], hierarchy[k-1].g) \
            - hierarchy[k].cR(F)

    # coarse mesh solver = smoother sweeps on correction y
    if args.mgview:
        _coarsereport(levels-1, hierarchy[0].m, args.coarse)
    hierarchy[0].y = hierarchy[0].zeros()
    obsprob.smoother(args.coarse, hierarchy[0], hierarchy[0].y,
                     hierarchy[0].ell, hierarchy[0].chi)

    # upward
    hierarchy[0].z = hierarchy[0].y
    for k in range(1, J+1):
        # accumulate corrections
        hierarchy[k].z = hierarchy[k].cP(hierarchy[k-1].z) + hierarchy[k].y
        if args.up > 0:
            # smooth the correction y;  up-obstacle is chi[k] not phi (see paper)
            if args.mgview:
                _levelreport(levels-1, k, hierarchy[k].m, args.up)
            obsprob.smoother(args.up, hierarchy[k], hierarchy[k].z,
                             hierarchy[k].ell, hierarchy[k].chi)
    return hierarchy[J].z
