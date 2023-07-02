'''Module for matplotlib visualization of glaciers.'''

from problem import secpera
import matplotlib.pyplot as plt

def output(filename, description):
    '''Either save result to an image file or use show().  Supply '' as filename
    to use show().'''
    if filename is None:
        plt.show()
    else:
        print('saving %s to %s ...' % (description, filename))
        plt.savefig(filename, bbox_inches='tight')

def showiteratecmb(mesh, s, cmb, filename=''):
    '''Generate graphic showing final iterate and CMB function.  mesh is of
    type MeshLevel1D and s, cmb are vectors on the mesh.'''
    mesh.checklen(s)
    mesh.checklen(cmb)
    xx = mesh.xx()
    xx /= 1000.0
    plt.figure(figsize=(15.0, 8.0))
    plt.subplot(2,1,1)
    plt.plot(xx, s, 'k', linewidth=4.0)
    plt.xlabel('x (km)')
    plt.ylabel('surface elevation (m)')
    plt.subplot(2,1,2)
    plt.plot(xx, cmb * secpera, 'r')
    plt.grid()
    plt.ylabel('CMB (m/a)')
    plt.xlabel('x (km)')
    output(filename, 'image of iterate and CMB')
