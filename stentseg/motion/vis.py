""" Module that provides functionality for visualization moving stent grafts.
"""

def create_mesh_with_values(graph, valueskey='path_curvature_change', radius=1.0):
    """ Create a Mesh object from the graph. The values of the mesh
    encode the *given values* at the points on the *paths* in the graph.
    E.g. curvature change at each point
    """ 
    from visvis.processing import lineToMesh, combineMeshes
    from visvis import Pointset  # lineToMesh does not like the new PointSet class
    from stentseg.utils import PointSet
    import numpy as np
    
    # Init list of meshes
    meshes = []
    
    for n1,n2 in graph.edges():
        # obtain path and deforms of edge
        path = graph.edge[n1][n2]['path']
        # obtain values for points on path
        pathValues = graph.edge[n1][n2][valueskey]
        # create mesh for path
        values = np.vstack(pathValues)
        path, values = Pointset(path), values   
        meshes.append( lineToMesh(path, radius, 8, values) ) # 8 refers to number of vertices on tube

    # Combine meshes and return
    if meshes:
        return combineMeshes(meshes)
    else:
        return None

# todo: wip
def make_mesh_dynamic_with_abs_displacement(mesh,deforms_f,origin,dim ='z',motion='amplitude',radius=1.0,**kwargs):
    """ Create Mesh object with color values from mesh, e.g. of vessel
    Input:  origin from volume
            deforms forward??
            invertZ, True of False. Inverts vertices value for z
    Output: 
    """
    import numpy as np
    from stentseg.motion.displacement import _calculateAmplitude, _calculateSumMotion
    from stentseg.motion.dynamic import get_mesh_deforms
    from visvis.processing import lineToMesh
    from visvis import Pointset  # lineToMesh does not like the new PointSet class
    
    #todo: mesh deforms_f use correct this way?
    pp, pp_mesh_deforms = get_mesh_deforms(mesh, deforms_f, origin, **kwargs) # removes vertice duplicates
    pp_mesh_displacements = []
    for n_point in range(len(pp_mesh_deforms[0])): # n vertices
            pointDeforms = [] # vertice deforms
            for i in range(len(pp_mesh_deforms)): # n (10) phases
                pointDeform = pp_mesh_deforms[i][n_point]
                pointDeforms.append(np.asarray(pointDeform)[0])
            if motion == 'amplitude': # max distance between two pointpositions
                dmax = _calculateAmplitude(pointDeforms, dim=dim)[0]
                pp_mesh_displacements.append(dmax)
            elif motion == 'sum':
                dsum = _calculateSumMotion(pointDeforms, dim=dim)
                pp_mesh_displacements.append(dsum)

    # create mesh
    # mesh._values
    mesh = []
    values = np.vstack(pp_mesh_displacements)
    points, values = Pointset(pp), np.asarray(values)   
    mesh.append( lineToMesh(points, radius, 8, values) )
    
    return mesh
    
    
def create_mesh_with_abs_displacement(graph, radius = 1.0, dim = 'xyz', motion = 'amplitude'):
    """ Create a Mesh object from the graph. The values of the mesh
    encode the *absolute displacement* at the points on the *paths* in the graph.
    Displacement can be the absolute displacement of a point in xyz, xy, z, y, or x
    direction over the cardiac cycle.
    """ 
    from visvis.processing import lineToMesh, combineMeshes
    from visvis import Pointset  # lineToMesh does not like the new PointSet class
    from stentseg.utils import PointSet
    import numpy as np
    from stentseg.motion.displacement import _calculateAmplitude, _calculateSumMotion
    
    # Init list of meshes
    meshes = []
    
    for n1,n2 in graph.edges():
        # obtain path and deforms of edge
        path = graph.edge[n1][n2]['path']
        pathDeforms = graph.edge[n1][n2]['pathdeforms']
        # get displacement during cardiac cycle for each point on path
        pathDisplacements = []
        for pointDeforms in pathDeforms:
            if motion == 'amplitude': # max distance between two pointpositions
                dmax = _calculateAmplitude(pointDeforms, dim=dim)[0]
                pathDisplacements.append(dmax)
            elif motion == 'sum':
                dsum = _calculateSumMotion(pointDeforms, dim=dim)
                pathDisplacements.append(dsum)            
        # create mesh for path
        values = np.vstack(pathDisplacements)
        path, values = Pointset(path), np.asarray(values)   
        meshes.append( lineToMesh(path, radius, 8, values) )

    # Combine meshes and return
    if meshes:
        return combineMeshes(meshes)
    else:
        return None


def create_mesh_with_deforms(graph, deforms, origin, radius=1.0, fullPaths=True): 
    """ Create a Mesh object from the graph. The values of the mesh
    encode the *deformation* at the points on the *paths* in the graph.
    
    The values array can be used to set the value to any kind
    of information derived from the nodes.
    
    """
    from visvis.processing import lineToMesh, combineMeshes
    from visvis import Pointset  # lineToMesh does not like the new PointSet class
    from stentseg.utils import PointSet
    import numpy as np
    
    # Init list of meshes
    meshes = []
    
    for n1,n2 in graph.edges():
        # Obtain path of edge and make mesh
        if fullPaths:
            path = graph.edge[n1][n2]['path']
            path = PointSet(path)  # Make a new class PointSet

            # Get deformation for all points in path
            # todo: perhaps pirt *should* be aware of the origin!
            # it isn't now, so we need to take it into account here
            g_deforms = []
            samplePoints = path - PointSet([o for o in reversed(origin)], dtype='float32')
            for deform in deforms:
                delta_z = deform.get_field_in_points(samplePoints, 0).reshape(-1, 1)
                delta_y = deform.get_field_in_points(samplePoints, 1).reshape(-1, 1)
                delta_x = deform.get_field_in_points(samplePoints, 2).reshape(-1, 1)
                delta = PointSet( np.concatenate((delta_x, delta_y, delta_z), axis=1) )
                g_deforms.append(delta)
            
            # Attach deformation to each point.
            # Needed because the points do not know their own index
            values = PointSet(1)
            for i, point in enumerate(path):
                mov = PointSet(3)
                for j in range(len(g_deforms)):
                    mov.append(g_deforms[j][i])
                d = ( (mov[:,0]**2 + mov[:,1]**2 + mov[:,2]**2)**0.5 ).reshape(-1,1)  # magnitude in mm
                dtot = d.sum()  # a measure for deformation of a point
                values.append(np.asarray(dtot))
        else:
            # Straight path
            path, values = PointSet(3), PointSet(1)
            path.append(n1); path.append(n2)
            mov1 = graph.node[n1]['deforms']
            mov2 = graph.node[n2]['deforms']
            d1 = ( (mov1[:,0]**2 + mov1[:,1]**2 + mov1[:,2]**2)**0.5 ).reshape(-1,1)  # magnitude in mm
            dtot1 = d1.sum()  # a measure for deformation of a point
            d2 = ( (mov2[:,0]**2 + mov2[:,1]**2 + mov2[:,2]**2)**0.5 ).reshape(-1,1)  
            dtot2 = d2.sum()
            values.append(np.asarray(dtot1)); values.append(np.asarray(dtot2))
            #values = [n1._angleChange, n2._angleChange]
        
        path, values = Pointset(path), np.asarray(values)    
        meshes.append( lineToMesh(path, radius, 8, values) )
    
    # Combine meshes and return
    if meshes:
        return combineMeshes(meshes)
    else:
        return None


def convert_mesh_values_to_angle_change(m, g, i=None):
    """ Convert the values of the mesh to a single value representing
    the angular change at motionIndex i, or to the maximum angular
    change if i is None.
    
    """
    # todo: this still uses the old mesh model class
    # Get list of angle changes
    pp = vv.Pointset(1)
    if i is None:
        for node in g:
            pp.append( node._angleChange )
    else:
        for node in g:
            pp.append( node._angleChanges[i] )
    angleChanges = pp.data.ravel().copy()
    
    # Get components of values
    I1 = m._values[:,0].astype('int32')
    I2 = m._values[:,1].astype('int32')
    F2 = m._values[:,2]
    F1 = 1.0 - F2
    
    # Get values and update
    return F1 * angleChanges[I1] + F2 * angleChanges[I2]

    