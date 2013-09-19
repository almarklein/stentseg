""" The graph implementation to segment stent grafts.
"""

import numpy as np
import visvis as vv
from visvis.pypoints import Point, Pointset
from pyzolib import ssdf
import time

from visvis.utils import graph
from stentseg.utils import gaussfun


class StentNode(graph.Node):
    """ StentNode(x,y,z,...)
    A Node object represents a point in 2D or 3D space.
    It has a list of references to edge objects. Via these
    edges, the connected Node objects can be obtained.
    This class inherits the Node object.
    """
    
    def __init__(self, *args, **kwargs):
        graph.Node.__init__(self, *args, **kwargs)
        self._isBendNode = False # todo: remove this
        self._inserted = ''
    
    
    def RemoveEdges(self, enc, th):
        """ RemoveEdges(enc, th) 
        This will let the node examine its edges. If it thinks
        any of these should be disconnected, it will ask the node
        on the other end if it agrees (using RequestToRemoveEdge).
        If so, the edge is disconnected.
        
        Edges that are not expected are removed, unles they
        are very strong (>th).
        
        See StentGraph.Prune_weak() for more information.
        """
        
        # Indices, walk through them in reversed order
        ii = range(len(self._edges))
        
        for i in reversed(ii):
            c = self._edges[i]            
            if i < enc:                
                # Keep edge, 
                # the worst edges have already been removed
                pass
            else:
                # The edge must be very strong to maintain it
                if c.props[1] < th:
                    n = c.GetOtherEnd(self)
                    if n.RequestToRemoveEdge(c, enc):
                        c.Disconnect()
    
    
    def RequestToRemoveEdge(self, c, enc): # Method of node
        """ RequestToRemoveEdge(c, enc)
        Another node may ask this node to remove the edge between the
        two. If this node agrees (returns True), the calling node will 
        disconnect it.
        """
        
        # Get which edge
        i = self._edges.index(c)        
        
        # What to do...
        if i >= enc:
            # Yes, we have plenty stronger edges
            return True 
        
        # Default
        return False
    
    
    def Trim(self, maxLength, removedEdges=None):
        """ Trim(nNodes, maxLength)
        Trim loose ends. If this node has only one edge, the 
        edge is removed. Then the node previously at the other
        side of that edge is trimmed.
        
        If the string of edges thus removed exceeds maxLength, the
        edges are restored.
        """
        
        # First call in the recursion?
        if removedEdges is None:
            removedEdges = []        
        elif len(removedEdges) > maxLength:
            # Restore edges; this tail is too long to remove
            for c in removedEdges:
                c.Connect()
        
        if len(self._edges) == 1:
            
            # Get edge and next node
            c = self._edges[0]
            node = c.GetOtherEnd(self)
            
            # Disconnect edge
            c.Disconnect()
            removedEdges.append(c)
            
            # Proceed
            node.Trim(maxLength, removedEdges)

    
class PotentialEdge(graph.Edge):
    """ PotentialEdge(node1, node2, cost, ii1, ii2)
    A potential edge stores the indices of the paths in the array
    starting from where the two nodes meet. At a later stage, 
    the single full path is created and the minimum CT-value obtained.
    """
    
    def __init__(self, p1, p2, cost, ii1, ii2):
        graph.Edge.__init__(self, p1, p2, cost, ii1, ii2)
    
    def Update(self, cost, ii1, ii2):
        """ Update(cost, i1, i2)
        Update the cost and the indices where the fronts meet. 
        """
        self.props = [cost, ii1, ii2]


class PathEdge(graph.Edge):
    """ PathEdge(node1, node2, cost, minCT, path)
    A PathEdge object represents an edge between two
    Node objects. It stores the two ends, the costs, the path,
    and optional extra information. 
    """
    
    def __init__(self, p1, p2, *props):
        graph.Edge.__init__(self, p1, p2, *props)
    
    @property
    def path(self):
        """ Short ref to the path. """
        return self.props[2]
    
    
    def GetPath(self, end):
        """ GetPath(end)
        Returns the path from the given end to the other end. This is 
        always a copy of the attribute self.props[2].
        """        
        if not self.IsEnd(end):
            raise ValueError("Given end is not a valid end for this edge.")
        if self.end1 == end:
            return self.props[2].copy()
        else:
            return Pointset( np.flipud(self.props[2].data) )
    
    
    def SmoothPath(self, ntimes=1):
        """ SmoothPath()
        Smooth the path by using a running average of width 3.
        """
        path = self.props[2]
        for iter in range(ntimes):
            tmp = path[1:-1] + path[0:-2] + path[2:]
            path[1:-1] = tmp / 3.0

    

class SplineEdge:
    pass


class StentGraph(graph.Graph):
    """ nodes = StentGraph()
    An implementation of the ConnectedNodes class that is more 
    specific for stent segmentation. It for example has several 
    methods for graph processing. 
    """
    
    def AppendNode(self, p):
        """ AppendNode(p)
        Append a node (if a point or non-stentNode is given, it is converted
        to a StentNode). The appended instance is returned.
        """
        if isinstance(p, StentNode):
            pass
        elif isinstance(p, (Point, graph.Node)):
            p = StentNode(p)            
        else:
            raise ValueError("Only StentNode objects should be appended.")
        # Append and return
        self.append(p)
        return p
    
    
    def CreateEdge(self, p1, p2, *props):
        """ CreateEdge(self, p1, p2, cost, minCT, path)
        Creates an edge instance using class PathEdge.
        The edge replaces an existing edge only if the
        cost of the new edge is lower, and is discarted
        otherwise. 
        """
        
        # Check nodes
        p1, p2 = self._CheckNodes(p1, p2)
        
        # get new and old edge
        cnew = PathEdge(p1, p2, *props)
        cold = p1.GetEdge(p2) # can be None
        
        # set new egde if there was none, or if cost is lower
        if cold is None:
            cnew.Connect()
        elif cold.props[0] > cnew.props[0]:
            cold.Disconnect()
            cnew.Connect()
        
        # return edge
        return cnew
    
    
    def CreatePotentialEdge(self, i1, i2, cost, ii1, ii2):
        """ CreatePotentialEdge(i1, i2, cost, ii1, ii2)
        Adds a potential edge to the tree. 
        i1 and i2 represent the nodes to connect (as indices in this list).
        ii1 and ii2 are the indices of the voxels where the two paths meet.
        cost is the cost of the edge. The edge is updated
        only if the cost is lower then the current edge between these
        points.
        To make this method as fast as possible, very little checks 
        are performed.
        """
        
        # get point objects
        p1 = self[i1]
        p2 = self[i2]
        
        # get old edge        
        cold = p1.GetEdge(p2) # can be None
        
        # set new edge if there was none, or update if cost is lower
        if cold is None:
            cnew = PotentialEdge(p1, p2, cost, ii1, ii2)
            cnew.Connect()
        elif cold.props[0] > cost:
            cold.Update(cost, ii1, ii2)
    

    def ConvertPotentialEdges(self, m, costToCtValue):
        """ ConvertPotentialEdges(self, mcpObject)
        Convert all potential edges to path edges, using
        the mcpObject.
        """

        def getMinCTvalueOnPath(ii):
            # find minimum cumCost
            pathCosts = [m.costs_f[i] for i in ii]
            ctValues = [costToCtValue(cost) for cost in pathCosts]
            return min(ctValues)
        
        # create function to convert edge
        def convert(c):
            # Get the two paths and swap paths if we must. 
            # This can be necessary because  PotentialEdge.Update() 
            # can be used with the order reversed.
            ii1, ii2 = c.props[1], c.props[2]
            if m.MakeIntPos(c.end1) != ii1[-1]:
                ii1, ii2 = ii2, ii1
            # calc min costs
            minCT = min(getMinCTvalueOnPath(ii1), getMinCTvalueOnPath(ii2))
            # create single path represented as points
            pp = Pointset(m.costs.ndim)
            for i in reversed(ii1):
                pp.append( m.MakePointPos(i) )
            for i in ii2:
                pp.append( m.MakePointPos(i) )
            # create new edge and replace
            cnew = PathEdge(c.end1, c.end2, c.props[0], minCT, pp)
            c.Disconnect()
            cnew.Connect()
        
        # We'll simply go by each node, and convert all edges
        # that are PotentialEdge instances.
        for node in self:
            cc = [c for c in node._edges] # make shallow copy
            for c in cc:
                if isinstance(c, PotentialEdge):
                    convert(c)
    
    
    def Draw(self, mc='g', lc='y', mw=7, lw=0.6, alpha=0.5, axes=None, simple=False):
        """ Draw(self, mc='g', lc='y', mw=7, lw=0.6, 
                alpha=0.5, axes=None, simple=False)
        Draw nodes and edges. 
        """ 
        
        # we can only draw if we have any nodes
        if not len(self):
            return
        
        if simple:
            # Use as a normal graph
            graph.Graph.Draw(self, mc, lc, mw, lw, alpha, axes)
        else:
            # Draw nodes only
            graph.Graph.Draw(self, mc, '', mw, 0, alpha, axes)
            
            # Draw edges ourselves
            if lc and lw:
                cc = self.GetEdges()
                pp = Pointset(self[0].ndim)
                for c in cc:                
                    pp.append(c.path[0])
                    for p in c.path[1:-1]:
                        pp.append(p)
                        pp.append(p)
                    pp.append(c.path[-1])                    
                tmp = vv.plot(pp, ms='', ls='+', lc=lc, lw=lw, 
                        axesAdjust=0, axes=axes, alpha=alpha)
                self._lines[1] = tmp
    
    
    def CreateMesh(self, radius=1.0, fullPaths=True):
        """ CreateMesh(radius=1.0)
        Create a polygonal model from the stent and return it as
        a visvis.BaseMesh object. To draw the mesh, instantiate
        a normal mesh using vv.Mesh(vv.gca(), thisMesh).
        """
        from visvis.processing import lineToMesh, combineMeshes
        
        # Init list of meshes
        meshes = []
        
        for edge in self.GetEdges():
            # Obtain path of edge and make mesh
            if fullPaths:                
                path = edge.GetPath(edge.end1)
            else:
                path = Pointset(3)
                path.append(edge.end1); path.append(edge.end2)
            meshes.append( lineToMesh(path, radius, 8) )
        
        # Combine meshes and return
        if meshes:
            return combineMeshes(meshes)
        else:
            return None
    
    def SortEdges(self):
        """ SortEdges()
        Sort edge lists of all nodes by their cost. 
        Some prune methods need this.
        """
        
        # Label all edges
        for i in range(len(self)):
            p = self[i]
            for c in p._edges:
                if c.end1 is p:
                    c._i1 = i
                else:
                    c._i2 = i
        
        # Define sorter function
        def edgehash(x):    
            return str(x.props[0]) + '_' + str(x._i1) + '_' + str(x._i2)
        
        # Sort the edges of each node
        for node in self:            
            node._edges.sort( key=graph.edgeHash )
    
    
    def SmoothPaths(self, ntimes=1):
        """ SmoothPaths()
        Smooth all paths.
        """
        for c in self.GetEdges():
            c.SmoothPath(ntimes)
    
    
    def SplitEdge(self, c, end, i, noCheck=False):
        """ SplitEdge(c, end, i)
        Split the given edge in two by inserting a node in position i
        of the path, measured from the given end.
        
        If noCheck is True, it is not checked whether the edge's ends
        are valid nodes (which is a relatively expensive test).
        
        The newly added node is returned.
        """
        
        # Perform checks
        if not c._connected or not isinstance(c, PathEdge):
            raise RuntimeError("Can only split connected PathEdge objects.")
        if not noCheck:
            # This check is very slow, so can be disabled with a flag
            if not (c.end1 in self and c.end2 in self):
                raise RuntimeError("Can not split because endpoints do not belong to this node list.")
        
        # Get path and split in two parts with the new node's pos overlapping
        path = c.GetPath(end)
        path1 = path[:i+1]
        path2 = path[i:]
        
        # Get the two refNodes
        refNode1 = end
        refNode2 = c.GetOtherEnd(end)
        
        # Create new node to put in between
        newNode = StentNode(path[i])
        self.append(newNode)
        
        # Create new edges
        c1 = PathEdge(refNode1, newNode, c.props[0], c.props[1], path1)
        c2 = PathEdge(newNode, refNode2, c.props[0], c.props[1], path2)
        
        # Switch edge (Note that the order is important for BendTracer)
        c.Disconnect()
        c1.Connect()
        c2.Connect()
        
        # Return new node
        return newNode
    
    
    def PopNode(self, node):
        """ PopNode(node)
        
        Pop the node from the graph, connecting it's two neighbours with 
        each-other with the combined path.
        Should only be called for nodes with exactly two neighbours.
        
        Returns the combined edge.
        """
        
        # Check
        if node.degree != 2:
            tmp = "Can only pop nodes with two edges, this one has %i."
            raise RuntimeError(tmp%node.degree)
        
        # get paths and the nodes at the other end
        c1, c2 = node._edges[0], node._edges[1]
        node1, node2 = c1.GetOtherEnd(node), c2.GetOtherEnd(node)  
        
        # Get the paths and build new path
        path1, path2 = c1.GetPath(node1), c2.GetPath(node) # not a typo
        path1.extend(path2[1:]) # do not take the node's position twice
        
        # Calculate min-ct and cost. Ctvalue, should clearly be the new 
        # minimum. The cost should be the maximum, I guess...        
        minCT = min(c1.props[1], c2.props[1])
        # cost = c1.props[0]+c2.props[0]
        cost = max(c1.props[0], c2.props[0])
        
        # Create new edge and replace
        cnew = PathEdge(node1, node2, cost, minCT, path1)
        c1.Disconnect()
        c2.Disconnect()
        cnew.Connect()
        
        # Remove node
        self.remove(node)
        
        # Done
        return cnew
    
    
    def Prune_veryWeak(self, th):
        """ Prune_veryWeak(th)
        
        All edges are evaluated and their cost is compared to the
        given threshold. If it is lower, the edge is very bad 
        and removed.
        """
        
        # Go
        for c in self.GetEdges():
            if c.props[1] < th:
                c.Disconnect()
    
    
    def Prune_weak(self, enc, th):
        """ Prune_weak(enc, th)
        
        Each node is asked to evaluate its edges. If it thinks 
        an edge should be removed, it asks the node at the other
        end of that edge if it agrees. Only if they agree, the 
        edge is removed. The decision is based on the smallest 
        CT value on the path, but the paths are sorted by strength 
        (mcp-cost).
        
        Note: requires SortEdges() to be run first.
        
        Rationale: Edges with a path that contains a very low CT
        value can always be discarted. Edges in excess of the expected
        number of edges can only be maintained if their lowest CT 
        value is so high it proves the existance of a wire between
        the nodes.
        
        """
        
        # Let the nodes do the work
        for node in self:
            node.RemoveEdges(enc, th)
    
    
    def Prune_redundant(self):
        """ Prune_redundant()
        
        Remove redundant edges. 
        
        A connection is redundant if both ends of a connection are 
        the same node.
        
        A connection is redundant if the two ends are also connected by
        a connection that is stronger.
        
        A connection is redundant if a weak connection (high mcp cost) 
        connects two nodes which are already connected via two other nodes
        wich are each stronger.
        
        This is applied by analysing the edges one by one, starting
        at the weakest connection.
        
        This process is relatively cheap and can be performed at multiple
        stages to clean up the graph.
        """
        
        # Get set of edges and reverse order (weak first)
        cc = self.GetEdges()
        cc.reverse()
        
        # Analyse each edge
        for c in cc:
            # Get neighbours and costs for end1
            nn1, cc1 = c.end1.GetNeighbours(), c.end1.GetProps(0)
            # Get neighbours of end2
            nn2 = c.end2.GetNeighbours()
            
            # Go on testing until we disconnected 
            disconnected = False
            
            if not disconnected:
                
                # Remove if both ends are the same node
                if c.end1 is c.end2:
                    c.Disconnect()
                    disconnected = True
            
            if not disconnected:
                
                # Remove if connects two nodes by two edges.
                # Note that we only have to iterate from one end
                for c2 in c.end1._edges:
                    if c2 is not c and c2.GetOtherEnd(c.end1) is c.end2:
                        if c.props[0] <= c2.props[0]:
                            c.Disconnect()            
                            disconnected = True
                            break
            
            if not disconnected:
                
                # If we see a 2-path-connection connecting the same
                # nodes, disconnect ourselves if the cost of the 
                # two paths are lower.
                for i in range(len(nn1)):
                    if nn1[i] in nn2:
                        c2 = c.end2.GetEdge(nn1[i])                    
                        if (    (c2 is not None) and 
                                cc1[i] <= c.props[0] and 
                                c2.props[0] <= c.props[0] ):
                            c.Disconnect()
                            disconnected = True
                            break
    
    
    def Prune_smallGroups(self, th=8):
        """ Prune_smallGroups()
        Small groups that consist of less then "th" inter-connected 
        nodes are removed. 
        """
        groups = self.CollectGroups()
        
        for group in groups:
            if len(group) < th:
                for node in group:
                    self.Remove(node)
        return groups
    
    
    def Prune_unconnectedNodes(self):
        """ Prune_unconnectedNodes()
        Remove all nodes that are not connected to any other nodes. 
        """
        
        for node in [node for node in self]:
            if not node._edges:
                self.remove(node)
    
    
    def Prune_trim(self, maxLength):
        """ Prune_trim(maxLength)
        Trim loose ends that are smaller than the specified length.
        """
        for node in self:
            node.Trim(maxLength)

    def Prune_pop(self):
        """ Prune_pop()
        Pop all nodes with degree 2 (having exactly two edges).
        A series of connected nodes can thus be reduced by a single node
        with an edge that connects to itself.
        After this, the cornerpoints need to be detected.
        """        
        # Note that this can pop corners that were inserted as crossings,
        # but not as corners.
        for node in [node for node in self]:
            if (node.degree == 2) and (node._inserted != 'corner'):
                if node not in node.GetNeighbours():
                    self.PopNode(node)
    
    
    def Prune_addCornerNodes(self):
        """ Prune_addCornerNodes()
        Detects positions on each edge where it bends and places
        a new node at these positions.
        """
        
        #edges = self.GetEdges()
        #edges.sort(key=lambda x:x.props[0])
        for edge in self.GetEdges():
            self._addCornerNodesInEdge(edge)
    
    
    def _addCornerNodesInEdge(self, c):
        """ _addCornerNodesInEdge(edge)
        Helper method to be called recusively.
        """
        
        # Get path and calculate indices to place new nodes
        path = c.props[2] # = c.GetPath(c.end1)
        I = detectCorners(path)
        
        if I:
            # Split path in multiple sections
            paths = []
            i_prev=0
            for i in I:
                paths.append( path[i_prev:i+1] ) # note the overlap
                i_prev = i
            paths.append( path[i_prev:] )
            
            # Create new nodes (and insert)
            node2s = []
            for i in I:  
                tmp = StentNode(path[i])
                tmp._inserted = 'corner'
                node2s.append(tmp)
                self.append(tmp)
            node2s.append(c.end2)
            
            # Create new edges (and connect)
            node1 = c.end1
            for i in range(len(paths)):
                node2 = node2s[i]
                cnew = PathEdge(node1, node2, 
                    c.props[0], c.props[1], paths[i])
                cnew.Connect()
                node1 = node2
            
            # Check whether we should still process a bit...
            popThisNode = None
            if c.end1 is c.end2 and len(I)>1:
                popThisNode = c.end1
            
            # Disconnect old edge
            c.Disconnect()
            
            # Pop node and process new bit if we have to
            # We use recursion, but can only recurse once 
            if popThisNode and popThisNode.degree<=2:                
                cnew = self.PopNode(popThisNode)
                self._addCornerNodesInEdge(cnew)
    
    
    def Prune_repositionCrossings(self):
        """ Prune_repositionCrossings()
        Repositions crossings by shifting them along a path that is common
        for two edges.
        """
        
        # Define function to obtain edge pairs
        def getEdgePairs(node):
            d = node.degree
            for i1 in range(d):
                for i2 in range(i1+1,d):
                    yield node._edges[i1], node._edges[i2]
        
        nodeList = [node for node in self]
        iter = 0
        while iter < len(nodeList):            
            node = nodeList[iter]
            iter+=1
            
            # Do not process nodes we inserted ourselves
            if node._inserted:
                continue 
            
            for e1, e2 in getEdgePairs(node):
                
                # Get the two paths to compare and walk untill they diverge
                path1, path2 = e1.GetPath(node), e2.GetPath(node)                
                i = 0
                for i in range(1,min(len(path1), len(path2))):
                    if path1[i] != path2[i]:
                        break
                
                # Was there a bit of the path that was the same?
                i -= 1 
                if i:
                    
                    # Get nodes at the end of both edges
                    node1 = e1.GetOtherEnd(node)
                    node2 = e2.GetOtherEnd(node)
                    
                    # Create new node right before where they split
                    newNode = StentNode(path1[i])
                    self.append(newNode)
                    newNode._inserted = 'crossing'
                    
                    # Create new connections
                    enew0 = PathEdge(node, newNode, # just use props of edge 1
                        e1.props[0], e1.props[1], path1[:i+1]) 
                    enew1 = PathEdge(newNode, node1,
                        e1.props[0], e1.props[1], path1[i:])
                    enew2 = PathEdge(newNode, node2,
                        e2.props[0], e2.props[1], path2[i:])                    
                    
                    # Disconnect old connections
                    e1.Disconnect()
                    e2.Disconnect()
                    
                    # Connect
                    for enew in [enew0, enew1, enew2]:
                        enew.Connect()
                    
                    # If the node has two edges left, pop it.
                    # Otherwise revisit it 
                    if node.degree == 2:
                        self.PopNode(node)
                        #print('crossing repositioned')
                    else:
                        nodeList.append(node)
                        #print('crossing inserted')
                    
                    # Don't look further now
                    break



def detectCorners(path, th=3, smoothFactor=2, angTh=45):
    """ detectCorners(path, th=5, smoothFactor=3, angTh=45)
    Return the indices on the given path were corners are detected.
    """
    
    # Initialize angles
    angles = np.zeros((len(path),), dtype=np.float64)
    
    # angle to radians
    angTh = angTh * np.pi / 180
    
    for i in range(len(path)):
        
        # Are we far enough down the wire that we can calculate 
        # the vectors?
        if i > th*2:
            
            # Get current point, point between vector, and point at end
            p1 = path[i]                
            p2 = path[i-th]                
            p3 = path[i-2*th]
            
            # Calculate vectors          
            vec1 = p1-p2
            vec2 = p2-p3
            
            # Calculate and store angle between them, (abs for 2D)
            angles[i-th] =  abs(vec1.angle(vec2))
    
    
    # Smooth angles and take derivative
    angles2 = angles
    angles = gaussfun.gfilter(angles, smoothFactor)# * smoothFactor**0.5
    
    # Detect local maximuma
    tmp = angles[th:-th]
    localmax = (    (tmp > angles[th-1:-th-1])
                &   (tmp > angles[th+1:-th+1]) 
                &   (tmp > angTh)
                )
    
    # Return indices
    I, = np.where(localmax)
    return [i+th for i in I]
