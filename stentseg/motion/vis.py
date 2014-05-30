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


def remove_stent_from_volume(vol, g, stripSize=5):
    """ Make the high intensity voxels that belong to the stent of a
    lower value, so that the stent appears to be "removed". This is for
    visualization purposes only.
    """
    # todo: this still uses the old mesh model class
    vol2 = vol.copy()
    for e in g.GetEdges():
        for node in e.GetPath(e.end1):
    #for node in g:
            z,y,x = vol2.point_to_index(node)
            stripSize2 = 2 * stripSize
            vol2[z-stripSize:z+stripSize+1, y-stripSize:y+stripSize+1, x-stripSize:x+stripSize+1] = 0
    return vol2

