'''Dome ice problem.'''

from abc import ABC
import numpy as np

# public physical parameters
secpera = 31556926.0        # seconds per year
g = 9.81                    # m s-2
rhoi = 910.0                # kg m-3
nglen = 3.0
A3 = 1.0e-16 / secpera      # Pa-3 s-1;  EISMINT I ice softness
B3 = A3**(-1.0/3.0)         # Pa s(1/3);  ice hardness

class IceProblem(ABC):
    '''Class for a particular ice problem.'''

    def __init__(self, args):
        self.args = args
        # parameters for initial condition, a dome; see van der Veen
        #   (2013) section 5.3
        self.domeL = 10.0e3       # half-width of sheet
        self.domeH0 = 1000.0      # center thickness
        # derived constant
        self.Gamma = 2.0 * A3 * (rhoi * g)**nglen
        self.Gamma /= (nglen + 2.0)

    def bed(self, x):
        '''For now we have a flat bed.'''
        return np.zeros(np.shape(x))

    def source(self, x):
        '''Continuous source term, the climatic mass balance (CMB).
        See van der Veen (2013) equations (5.49) and (5.51).  Assumes x
        is a numpy array.'''
        n = nglen
        invn = 1.0 / n
        r1 = 2.0 * n + 2.0                   # e.g. 8
        s1 = (1.0 - n) / n                   #     -2/3
        C = self.domeH0**r1 * self.Gamma     # A_0 in van der Veen is Gamma here
        C /= ( 2.0 * self.domeL * (1.0 - 1.0 / n) )**n
        xc = self.args.domainlength / 2.0
        X = (x - xc) / self.domeL          # rescaled coord
        m = np.zeros(np.shape(x))
        # usual formula for 0 < |X| < 1
        zzz = (abs(X) > 0.0) * (abs(X) < 1.0)
        if any(zzz):
            Xin = abs(X[zzz])
            Yin = 1.0 - Xin
            m[zzz] = (C / self.domeL) \
                     * ( Xin**invn + Yin**invn - 1.0 )**(n-1.0) \
                     * ( Xin**s1 - Yin**s1 )
        # fill singular origin with near value
        if any(X == 0.0):
            Xnear = 1.0e-8
            Ynear = 1.0 - Xnear
            m[X == 0.0] = (C / self.domeL) \
                          * ( Xnear**invn + Ynear**invn - 1.0 )**(n-1.0) \
                          * ( Xnear**s1 - Ynear**s1 )
        # extend by ablation
        if any(abs(X) >= 1.0):
            m[abs(X) >= 1.0] = min(m)
        return m

    def initial(self, x):
        '''Default initial shape is the dome profile.  See van der Veen (2013)
        equation (5.50).  Assumes x is a numpy array.'''
        n = nglen
        p1 = n / (2.0 * n + 2.0)                  # e.g. 3/8
        q1 = 1.0 + 1.0 / n                        #      4/3
        Z = self.domeH0 / (n - 1.0)**p1         # outer constant
        xc = self.args.domainlength / 2.0
        X = (x - xc) / self.domeL               # rescaled coord
        Xin = abs(X[abs(X) < 1.0])                # rescaled distance from
                                                  #   center, in ice
        Yin = 1.0 - Xin
        s = np.zeros(np.shape(x))                 # correct outside ice
        s[abs(X) < 1.0] = Z * ( (n + 1.0) * Xin - 1.0 \
                                + n * Yin**q1 - n * Xin**q1 )**p1
        return s
