#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

x = np.linspace(0.0,10.0,11)
# following data was generated by stokesdomain.py with print(b[::100]) etc.
b = np.array([0.5300000000000001, 0.36185948536513635, -0.18136049906158566, -0.1558830996397852, 0.16787164932467638, 0.07119577782212608, 0.4226854163999131, 1.218121471138974, 1.592419336666987, 2.269802550645665, 3.5125890501455257])
s = np.array([0.5300000000000001, 0.36185948536513635, -0.18136049906158566, 1.4801848778600042, 2.3962987740708668, 2.4711957778221265, 2.6511125411461034, 2.5, 1.592419336666987, 2.269802550645665, 3.5125890501455257])

def isgap(a, b):
    return abs(a - b) > 1.0e-4

def plotextruded(x, b, s, mz=5, Hmin=0.0):
    '''Plot the extruded mesh.  mz is the number of element layers.  Hmin > 0
    produces a cliff at the margin.  Make the top of the top element strong.'''
    assert all(s >= b)
    plt.plot(x, b, 'k-')
    dz = np.maximum(s - b, Hmin) / mz # everywhere at least Hmin/mz
    # plot verticals
    for j in range(len(x)):
        if isgap(b[j],s[j]) or (j > 0 and isgap(b[j-1],s[j-1])) \
                            or (j < len(x) - 1 and isgap(b[j+1],s[j+1])):
            snew = b[j] + mz * dz[j]
            plt.plot([x[j], x[j]], [b[j], snew], 'k-')
    # plot element tops
    for j in range(len(x)-1):
        if isgap(b[j],s[j]) or isgap(b[j+1],s[j+1]):
            for k in range(mz):
                zl = b[j] + (k+1) * dz[j]
                zr = b[j+1] + (k+1) * dz[j+1]
                if k + 1 == mz:  # show strong top
                    plt.plot([x[j], x[j+1]], [zl, zr], 'k-', lw=3.0)
                else:
                    plt.plot([x[j], x[j+1]], [zl, zr], 'k-')
            # show original s near cliffs
            if Hmin > 0.0 and \
                    (not isgap(b[j],s[j]) or not isgap(b[j+1],s[j+1])):
                plt.plot([x[j], x[j+1]], [s[j], s[j+1]], 'k:', lw=3.0)
        else:
            plt.plot([x[j], x[j+1]], [s[j], s[j+1]], 'k-', lw=3.0)
    plt.axis('off')

# extruded FE domain figure
plt.figure(figsize=(16,4))
plotextruded(x, b, s)
plotextruded(x + 9.5, b, s, Hmin = 0.5)
writeout('extruded.pdf')
