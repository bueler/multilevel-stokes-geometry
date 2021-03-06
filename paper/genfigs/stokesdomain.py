#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from writeout import writeout

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"

fsize=18.0
bigfsize=24.0

def genbasicfig(xshift=0.0):
    x = np.linspace(0.0,10.0,1001)
    # bed elevation
    b = 0.07*(x-3.0)**2 + 0.2*np.sin(2.0*x) - 0.1
    plt.plot(x + xshift, b, 'k', lw=2.5)
    # current thickness for Omega^{n-1}
    h0 = 3.0
    L = 3.0
    firstshape = h0*(-0.2 + np.sqrt(np.maximum(0.0,1.0 - (x-5)**2/L**2)))
    thk = np.maximum(0.0, firstshape)
    # surface
    s = b + thk
    offset = 0.1
    plt.plot(x + xshift, s + offset, 'k--', lw=3.0)
    return x + xshift, s, b

def drawclimate(x,s):
    plt.text(x[10], s[10]+2.7, r'$a$', fontsize=bigfsize, color='k')
    for j in range(10):
        xarr = x[50+100*j]
        if j>0:
            magarr = 0.6*np.sin(np.pi/2 + 0.6*xarr)
        else:
            magarr = 0.05
        plt.arrow(xarr, s.max()+0.3, 0.0, magarr,
                  lw=1.5, head_width=0.1, color='k')

# domain notation figure
plt.figure(figsize=(10,5.5))
x, s, b = genbasicfig()
plt.text(x[600] - 1.0, b[600] + 0.4 * s[600], r'$\Lambda_s$',
         fontsize=bigfsize, color='k')
drawclimate(x,s)
# mark top surface
plt.text(x[250], s[250] + 0.7, r'$s$', fontsize=bigfsize, color='k')
# mark bottom surface
plt.text(x[650], b[650] - 0.5, r'$b$', fontsize=bigfsize, color='k')
# show \Omega
yR = min(b) - 0.5
plt.plot([min(x),max(x)],[yR,yR],color='k',lw=1.0)
plt.text(x[875],yR+0.2,r'$\Omega$',fontsize=fsize)
plt.axis([0.0,10.0,yR-0.8,4.5])
plt.axis('off')
writeout('stokesdomain.pdf')

# IDO (ice dynamics operator) figure
plt.figure(figsize=(16,4))
x, s, _ = genbasicfig(xshift=0.0)
plt.text(x[300], s[300] + 1.2, r'$s$', fontsize=bigfsize, color='k')
plt.text(x[500], s[500] - 1.2, r'$\Lambda_s$', fontsize=bigfsize, color='k')
plt.arrow(10.0, 1.5, 1.0, 0.0,
          lw=1.5, head_width=0.1, color='k')
plt.text(10.4, 1.6, r'$\Phi$', fontsize=bigfsize, color='k')
x, s, b = genbasicfig(xshift=11.0)
offset = 0.1
xl, xr = min(x[s>b]), max(x[s>b])
dx = xr - xl
xsc = (x[s>b] - xl) / dx
y = s[s>b] - offset + 1.2 * np.sin(5.5*np.pi*xsc) * np.cos(np.pi*xsc)**4
plt.plot(x[s>b], y, color='k', lw=1.0)
plt.plot(x[x < xl], s[x < xl] + offset, color='k', lw=1.0)
plt.plot(x[x > xr], s[x > xr] + offset, color='k', lw=1.0)
plt.text(x[760] + 0.2, s[760] + 0.7, r'$\Phi(s)$',
         fontsize=bigfsize, color='k')
plt.axis([0.0,22.0,-0.5,4.0])
plt.axis('off')
writeout('idoaction.pdf')
