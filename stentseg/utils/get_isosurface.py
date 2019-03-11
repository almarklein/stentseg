""" Functionality to get isosurface using marching cubes and store as stentgraph
with seedpoints

Author: Maaike Koenrades. Created 2019

Based on https://www.programcreek.com/python/example/88834/skimage.measure.marching_cubes
Other option https://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/
"""
import skimage 
import matplotlib.pyplot as plt
import numpy as np
from stentseg.stentdirect import stentgraph
import copy
from stentseg.utils import PointSet, _utils_GUI
from stentseg.utils.visualization import show_ctvolume
import visvis as vv
from stentseg.utils.picker import pick3d

#todo: error library conflict with Qt!! versions? Pyside conflict? Visvis or QT libraries??

def save_isosurface_stl(vol, threshold=600, name=None):
    """To create and save and stl based on This will produce a 3d printable stl based on self.volume_data. 
    
    """
    #todo: wip, does not work
    # from stl import mesh # ??
    from skimage import measure
    
    if name is None:
        name = input('What should the filename be?') + '.stl'
    
    verts, faces = measure.marching_cubes(vol, threshold)   #Marching Cubes algorithm
    
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0, 0, 1]
    mesh.set_facecolor(face_color)
    #todo: mesh.Mesh error. has no object Mesh; version? of mesh should be imported?
    solid = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            solid.vectors[i][j] = verts[f[j],:]

    solid.save(name)


def get_isosurface3d(vol, threshold=300, showsurface=False):
    """
    Generate isosurface to return the vertices (surface_points), based on HU threshold
    Vertices are scaled using vol.sampling and vol.origin
    return: verts, faces, pp_verts, mesh
    """
    from skimage import measure, morphology
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    verts, faces = measure.marching_cubes(vol,threshold) # verts in z,y,x
    #verts, faces, normals, values = measure.marching_cubes_lewiner(vol,level=threshold) 
    #version after 0.12 returns verts, faces, normals, values
    
    vertpoints = copy.deepcopy(verts)
    # set scale and origin
    vertpoints[:,0] *= vol.sampling[0] # sampling z,y,x
    vertpoints[:,1] *= vol.sampling[1]
    vertpoints[:,2] *= vol.sampling[2]
    vertpoints[:,0] += vol.origin[0]
    vertpoints[:,1] += vol.origin[1]
    vertpoints[:,2] += vol.origin[2]
    
    pp = vertpoints.copy() # pp are the surfacepoints
    pp[:,0] = vertpoints[:,-1] # to x y x
    pp[:,-1] = vertpoints[:,0] # to x y z
    
    verts = copy.deepcopy(pp) # vertice points xyz scaled and set to vol origin
    pp_verts = PointSet(pp)
    
    if showsurface:
        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.1)
        face_color = [0, 0, 1]
        mesh.set_facecolor(face_color)
        
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.add_collection3d(mesh)
        
        ax.set_xlim(0, vol.shape[0])
        ax.set_ylim(0, vol.shape[1])
        ax.set_zlim(0, vol.shape[2])
        
        plt.show()
    
    else:
        mesh = None
    
    return verts, faces, pp_verts, mesh


def show_surface_and_vol(vol, pp_isosurface, showVol='MIP', clim =(-200,1000), isoTh=300, climEditor=True):
    """ Show the generated isosurface in original volume
    """
    f = vv.figure()
    ax = vv.gca()
    ax.daspect = 1,1,-1
    ax.axis.axisColor = 1,1,1
    ax.bgcolor = 0,0,0
    # show vol and iso vertices
    show_ctvolume(vol, showVol=showVol, isoTh=isoTh, clim=clim, climEditor=climEditor)
    label = pick3d(vv.gca(), vol)
    vv.plot(pp_isosurface, ms='.', ls='', mc= 'r', alpha=0.2, mw = 4)
    a = vv.gca()
    f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a]) )
    print('------------------------')
    print('Use keys 1, 2, 3, 4 and 5 for preset anatomic views')
    print('Use v for a default zoomed view')
    print('Use x to show and hide axis')
    print('------------------------')
    
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    
    return label
    

def isosurface_to_graph(pp):
    """ Store isosurface points to graph as seedpoints
    pp is PointSet of isosurface vertex points
    """
    
    modelnodes = stentgraph.StentGraph()
    # add pp as nodes    
    for i, p in enumerate(pp):
        p_as_tuple = tuple(p.flat)
        modelnodes.add_node(p_as_tuple)
    
    return modelnodes


## TESTING

if __name__ == "__main__":

    import os
    import numpy as np
    import visvis as vv
    from visvis import ssdf
    from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
    from stentseg.utils.picker import pick3d
    
    cropname = 'prox'
    what = 'avgreg'
    ptcode = 'chevas_01'
    ctcode = '12months' 
    basedir = select_dir(r'F:\Nellix_chevas\CT_SSDF\SSDF_automated')
    
    # Load volumes
    s = loadvol(basedir, ptcode, ctcode, cropname, what)
    
    # sampling was not properly stored after registration for all cases: reset sampling
    vol_org = copy.deepcopy(s.vol)
    s.vol.sampling = [vol_org.sampling[1], vol_org.sampling[1], vol_org.sampling[2]] # z,y,x
    s.sampling = s.vol.sampling
    vol = s.vol

    ## Get isosurface
    verts, faces, pp_vertices, mesh = get_isosurface3d(vol, threshold=300, showsurface=False)
    
    ## Show isosurface and original volume
    show_surface_and_vol(vol, pp_vertices, showVol='MIP')
    
    ## Centerline test
    from stentseg.utils.centerline import (find_centerline, 
    points_from_nodes_in_graph, points_from_mesh, smooth_centerline)
    
    start1 = [(113.6, 89.5, 34.3)] # pt 1 chim vessel
    ends =   [(128.5, 94.1, 31.6)]
    
    j=0
    centerline1 = find_centerline(pp2, start1[j], ends[j], step= 1, 
        ndist=10, regfactor=0.5, regsteps=1, verbose=False) # centerline1 is a PointSet
    
    
    ## Save to .stl test
    # filename = r'D:\Profiles\koenradesma\Desktop\isotest'
    # pp = save_isosurface_stl(vol, name=filename)



