""" Module that provides functionality for visualization moving stent grafts.
"""


def create_mesh_with_values(g, radius=1.0, simplified=True): 
    """ Create a Mesh object from the graph. The values of the mesh
    encode triplets (node1, node2, weight) where node1 and node2 are
    indices to nodes in the graph and weight is the relative proximity
    to these nodes.
    
    The values array can then be used to set the value to any kind
    of information derived from the nodes.
    
    """
    from visvis.processing import lineToMesh, combineMeshes
    # todo: this still uses the old mesh model class
    
    # Init list of meshes
    meshes = []
    
    for e in g.GetEdges():
        # Obtain path of edge and make mesh
        if simplified:
            # Straight path
            path, values = vv.Pointset(3), vv.Pointset(3)
            path.append(e.end1._data); values.append(e._i1, e._i2, 0)
            path.append(e.end2._data); values.append(e._i1, e._i2, 1)
            #values = [e.end1._angleChange, e.end2._angleChange]
        else:
            path = vv.Pointset(e.props[2].data)
            #values = values.reshape(-1, 1)
            values = vv.Pointset(3)
            for i in np.linspace(0.0, 1.0, len(path)):
                values.append(e._i1, e._i2, i)
        meshes.append( lineToMesh(path, radius, 8, values) )
    
    # Combine meshes and return
    if meshes:
        return combineMeshes(meshes)
    else:
        return None


def create_mesh_with_abs_displacement(graph, radius = 1.0, dimensions = 'xyz', motion = 'sum'):
    """ Create a Mesh object from the graph. The values of the mesh
    encode the *absolute displacement* at the points on the *paths* in the graph.
    Displacement can be the absolute displacement of a point in xyz, xy or z
    direction over the cardiac cycle.
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
        pathDeforms = graph.edge[n1][n2]['pathdeforms']
        # get displacement during cardiac cycle for each point on path
        pathDisplacements = []
        for pointDeforms in pathDeforms:
            if motion == 'amplitude': # max amplitude between pointpositions
                for point in path:
                    pointIndices = point + pointDeforms # 10 positions for each point
                    amplitude = []
                    for i, indice in enumerate(pointIndices):
                        vectors = indice - pointIndices
                        if dimensions == 'xyz':
                            d = (vectors[:,0]**2 + vectors[:,1]**2 + vectors[:,2]**2)**0.5  # 3Dvector length in mm
                        elif dimensions == 'xy':
                            d = (vectors[:,0]**2 + vectors[:,1]**2 )**0.5  # 2Dvector length in mm
                        elif dimensions == 'z':
                            d = abs(vectors[:,2])  # 1Dvector length in mm
                        amplitude.append(d.max())
                    maxamplitude = max(amplitude) # for point
                pathDisplacements.append(maxamplitude)
            elif motion == 'sum':
                vectors = []
                npositions = len(pointDeforms)
                for j in range(npositions):
                    if j == npositions-1:  # -1 as range starts at 0
                        # vector from point at 90% RR to 0%% RR
                        vectors.append(pointDeforms[j]-pointDeforms[0])
                    else:
                        vectors.append(pointDeforms[j]-pointDeforms[j+1])
                vectors = np.vstack(vectors)
                if dimensions == 'xyz':
                    d = (vectors[:,0]**2 + vectors[:,1]**2 + vectors[:,2]**2)**0.5  # 3Dvector length in mm
                elif dimensions == 'xy':
                    d = (vectors[:,0]**2 + vectors[:,1]**2 )**0.5  # 2Dvector length in mm
                elif dimensions == 'z':
                    d = abs(vectors[:,2])  # 1Dvector length in mm
                displacement = d.sum() # displacement of a point
                pathDisplacements.append(displacement)            
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


def remove_stent_from_volume(vol, graph, stripSize=5):
    """ Give the high intensity voxels that belong to the stent a
    lower value, so that the stent appears to be "removed". This is for
    visualization purposes only. Makes use of known paths in graph model.
    """
    from visvis import Pointset

    vol2 = vol.copy()
    for n1,n2 in graph.edges():
        path = graph.edge[n1][n2]['path']
        path = Pointset(path)  # Make a visvis pointset
        stripSize2 = 0.5 * stripSize
        for point in path:
            z,y,x = vol2.point_to_index(point)
            vol2[z-stripSize:z+stripSize2+1, y-stripSize:y+stripSize+1, x-stripSize:x+stripSize+1] = 0
            # remove less in distal direction -> stripSize2
    return vol2