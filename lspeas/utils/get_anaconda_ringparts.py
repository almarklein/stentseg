""" Module to obtain the different parts and locations of the Anaconda dual ring

Hooks, struts, 2nd ring, top ring
""" 

from stentseg.stentdirect import stentgraph
from sklearn.cluster import KMeans
import numpy as np
from stentseg.utils import PointSet
import visvis as vv
from stentseg.utils.visualization import show_ctvolume
from stentseg.utils.picker import pick3d
from stentseg.utils.utils_graphs_pointsets import point_in_pointcloud_closest_to_p
 

def add_nodes_edge_to_newmodel(modelnew, model,n,neighbour):
    """ Get edge and nodes with attributes from model and add to newmodel
    """
    c = model.edge[n][neighbour]['cost']
    ct = model.edge[n][neighbour]['ctvalue']
    p = model.edge[n][neighbour]['path']
    pdeforms = model.edge[n][neighbour]['pathdeforms']
    modelnew.add_node(n, deforms = model.node[n]['deforms'])
    modelnew.add_node(neighbour, deforms = model.node[neighbour]['deforms'])
    modelnew.add_edge(n, neighbour, cost = c, ctvalue = ct, path = p, pathdeforms = pdeforms)
    return


def _get_model_hooks(model):
    """Get model hooks
    Return model without hooks and model with hooks only
    """
    import numpy as np
    from stentseg.stentdirect import stentgraph
    
    # initialize 
    model_noHooks = model.copy()
    model_hooks = stentgraph.StentGraph() # graph for hooks
    hooknodes = list() # remember nodes that belong to hooks 
    for n in sorted(model.nodes() ):
        if model.degree(n) == 1:
            neighbour = list(model.edge[n].keys())
            neighbour = neighbour[0]
            add_nodes_edge_to_newmodel(model_hooks,model,n,neighbour)
            hooknodes.append(neighbour)
            model_noHooks.remove_node(n) # this also removes the connecting edge
    
    return model_hooks, model_noHooks
    

def get_model_struts(model, nstruts=8):
    """Get struts between R1 and R2
    Detects them based on z-orientation and length
    Runs _get_model_hooks 
    """
    from stentseg.stentdirect.stentgraph import _edge_length
    from stentseg.stentdirect import stentgraph
    import numpy as np
    
    # remove hooks if still there
    models = _get_model_hooks(model)
    model_hooks, model_noHooks = models[0], models[1]
    # initialize
    model_h_s = model_hooks.copy() # struts added to hook model
    model_struts = stentgraph.StentGraph()
    directions = []
    for n1, n2 in model_noHooks.edges():
        e_length = _edge_length(model, n1, n2)
        if (3.5 < e_length < 12): # struts OLB21 are 4.5-5.5mm OLB34 9-9.5
            vector = np.subtract(n1,n2) # nodes, paths in x,y,z
            vlength = np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
            direction = abs(vector / vlength)
            directions.append([direction, n1, n2]) # direction and nodes
#             print(direction)
    d = np.asarray(directions) # n x 3 (direction,n1,n2) x 3 (xyz)
    ds = sorted(d[:,0,2], reverse = True) # highest z direction first
    for i in range(nstruts):
        indice = np.where(d[:,0,2]==ds[i])[0][0] # [0][0] to get int in array in tuple
        n1 = tuple(d[indice,1,:])
        n2 = tuple(d[indice,2,:])
        add_nodes_edge_to_newmodel(model_struts,model,n1,n2)  
        add_nodes_edge_to_newmodel(model_h_s,model,n1,n2)  
    
    model_R1R2 = model_noHooks.copy()
    model_R1R2.remove_edges_from(model_struts.edges())
#     print('************')
    
    return model_struts, model_hooks, model_R1R2, model_h_s, model_noHooks


def get_model_rings(model_R1R2):
    """Get top ring and 2nd ring from model containing two sepatate rings.
    First run _get_model_struts
    """
    import networkx as nx
    import numpy as np

    model_R2 = model_R1R2.copy()
    model_R1 = model_R1R2.copy()
    # struts must be removed
    clusters = list(nx.connected_components(model_R1R2))
    assert len(clusters) == 2  # 2 rings
    c1, c2 = np.asarray(clusters[0]), np.asarray(clusters[1])
    c1_z, c2_z = c1[:,2], c2[:,2]  # x,y,z
    if c1_z.mean() < c2_z.mean(): # then c1/cluster[0] is topring; daspect = 1, 1,-1
        model_R2.remove_nodes_from(clusters[0])
        model_R1.remove_nodes_from(clusters[1])
    else:
        model_R2.remove_nodes_from(clusters[1])
        model_R1.remove_nodes_from(clusters[0])
        
    return model_R1, model_R2    


def get_midpoints_peaksvalleys(model):
    """ Get midpoints near the peaks and valleys
    """
    # remove hooks, pop nodes
    model_hooks1, model_noHooks1 = _get_model_hooks(model)
    stentgraph.pop_nodes(model_noHooks1)
    midpoints_peaks_valleys = PointSet(3)
    for n1, n2 in sorted(model_noHooks1.edges()):
        if stentgraph._edge_length(model_noHooks1, n1, n2) < 10: # in mm
            # get midpoint for edges near struts
            mid = (n1[0]+n2[0])/2, (n1[1]+n2[1])/2, (n1[2]+n2[2])/2
            path = model_noHooks1.edge[n1][n2]['path']
            mid_and_pathpoint = point_in_pointcloud_closest_to_p(path, mid)
            midpoints_peaks_valleys.append(mid_and_pathpoint[0]) # append pp_point
    return midpoints_peaks_valleys


def identify_peaks_valleys(midpoints_peaks_valleys, model, vol, vis=True):
    """ Given a cloud of points containing 2 peak and 2 valley points for R1
    and R2, identify and return these locations. Uses clustering and x,y,z
    """
    # detect clusters to further label locations
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(midpoints_peaks_valleys)
    centroids = kmeans.cluster_centers_ # x,y,z
    labels = kmeans.labels_
    
    # identify left, right, ant, post centroid (assumes origin is prox right anterior)
    left = list(centroids[:,0]).index(max(centroids[:,0])) # max x value
    right = list(centroids[:,0]).index(min(centroids[:,0])) 
    anterior = list(centroids[:,1]).index(min(centroids[:,1])) # min y value
    posterior = list(centroids[:,1]).index(max(centroids[:,1]))
    # get points into grouped arrays
    cLeft, cRight, cAnterior , cPosterior = [], [], [], []
    for i, p in enumerate(midpoints_peaks_valleys):
        if labels[i] == left:
            cLeft.append(tuple(p.flat))
        elif labels[i] == right:
            cRight.append(tuple(p.flat))
        elif labels[i] == anterior:
            cAnterior.append(tuple(p.flat))
        elif labels[i] == posterior:
            cPosterior.append(tuple(p.flat))
    
    cLeft = np.asarray(cLeft)
    cRight = np.asarray(cRight)
    cAnterior = np.asarray(cAnterior)
    cPosterior = np.asarray(cPosterior)
    # divide into R1 R2
    R1_left = cLeft[list(cLeft[:,2]).index(min(cLeft[:,2]))] # min z for R1; valley
    R2_left = cLeft[list(cLeft[:,2]).index(max(cLeft[:,2]))] # valley
    R1_right = cRight[list(cRight[:,2]).index(min(cRight[:,2]))] # min z for R1; valley
    R2_right = cRight[list(cRight[:,2]).index(max(cRight[:,2]))] # valley
    R1_ant = cAnterior[list(cAnterior[:,2]).index(min(cAnterior[:,2]))] # min z for R1; peak
    R2_ant = cAnterior[list(cAnterior[:,2]).index(max(cAnterior[:,2]))] # peak
    R1_post = cPosterior[list(cPosterior[:,2]).index(min(cPosterior[:,2]))] # min z for R1; peak
    R2_post = cPosterior[list(cPosterior[:,2]).index(max(cPosterior[:,2]))] # peak
    
    if vis==True:
        # visualize identified locations
        f = vv.figure(1); vv.clf()
        f.position = 968.00, 30.00,  944.00, 1002.00
        a = vv.gca()
        colors = ['r', 'g', 'b', 'm', 'c', 'y', 'w', 'k']
        for i, p in enumerate([R1_left,R2_left,R1_right,R2_right,R1_ant,R2_ant,R1_post,R2_post]):
            vv.plot(p[0], p[1], p[2], ms='.', ls='', mc=colors[i], mw=14 )
        vv.legend('R1 left','R2 left','R1 right','R2 right','R1 ant','R2 ant','R1 post','R2 post')
        show_ctvolume(vol, model, showVol='MIP', clim=(0,2500))
        pick3d(vv.gca(), vol)
        model.Draw(mc='b', mw = 10, lc='g')
        for i in range(len(midpoints_peaks_valleys)):
            vv.plot(midpoints_peaks_valleys[i], ms='.', ls='', mc=colors[labels[i]], mw=6)
        a.axis.axisColor= 1,1,1
        a.bgcolor= 0,0,0
        a.daspect= 1, 1, -1  # z-axis flipped
        a.axis.visible = True
    
    return R1_left,R2_left,R1_right,R2_right,R1_ant,R2_ant,R1_post,R2_post

def save_model_in_ssdf(s, newmodel, basedir, ptcode, ctcode, cropname, modelname, what='avgreg'):
    """ save model by replacing the model in the provided ssdf
    e.g. modelname = 'model_R1'
    e.g. save_model_in_ssdf(s, model_R1, basedir, ptcode, ctcode, cropname, 'modelR1')
    """
    # replace model
    s.model = newmodel.pack()
    # Save
    filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, modelname+what)
    ssdf.save(os.path.join(basedir, ptcode, filename), s)
    print('saved to disk in {} as {}.'.format(basedir, filename) )

if __name__ == '__main__':
    import os
    import visvis as vv
    from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
    import numpy as np
    from stentseg.utils import PointSet
    from stentseg.stentdirect import stentgraph
    from stentseg.stentdirect.stentgraph import create_mesh
    from stentseg.utils.visualization import show_ctvolume
    from visvis import ssdf
    
    # Select the ssdf basedir
    basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                        r'D:\LSPEAS\LSPEAS_ssdf', r'F:\LSPEAS_ssdf_backup')
    
    # Select dataset to register
    ptcode = 'LSPEAS_002'
    ctcode = 'discharge'
    cropname = 'ring'
    modelname = 'modelavgreg'
    
    # Load the stent model and mesh
    s = loadmodel(basedir, ptcode, ctcode, cropname, modelname)
    model = s.model
    model2 = model.copy()
    
    # Load static CT image to add as reference
    s2 = loadvol(basedir, ptcode, ctcode, cropname, 'avgreg')
    vol = s2.vol
    
    f = vv.figure(1); vv.clf()
    a = vv.subplot(1,2,1)
    a.axis.axisColor = 1,1,1
    a.axis.visible = False
    a.bgcolor = 0,0,0
    a.daspect = 1, 1, -1
    t = vv.volshow(vol, clim=(0, 2500), renderStyle='mip')
    # model.Draw(mc='b', mw = 10, lc='g')
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
    
    # get hooks and struts
    model_struts, model_hooks, model_R1R2, model_h_s, model_noHooks = get_model_struts(model, nstruts=8)
    # model_hooks.Draw(mc='r', mw = 10, lc='r')
    model_noHooks.Draw(mc='b', mw = 10, lc='b')
    # model_struts.Draw(mc='m', mw = 10, lc='m')
    
    a2 = vv.subplot(1,2,2)
    a2.axis.axisColor = 1,1,1
    a2.axis.visible = False
    a2.bgcolor = 0,0,0
    a2.daspect = 1, 1, -1
    t2 = vv.volshow(vol, clim=(0, 2500), renderStyle='mip')
    vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
    vv.title('Model for LSPEAS %s  -  %s' % (ptcode[7:], ctcode))
    
    # gets rings separate
    model_R1, model_R2 = get_model_rings(model_R1R2)
    model_R1.Draw(mc='y', mw = 10, lc='y')
    model_R2.Draw(mc='c', mw = 10, lc='c')
    
    
    