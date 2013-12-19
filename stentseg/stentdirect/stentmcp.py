""" A version of MCP_Connect from skimage that is tweaked to work
with our stent segmentation algorithm.
"""

import numpy as np
import visvis as vv
import skimage.graph

from . import stentGraph


class MCP_StentDirect(skimage.graph.MCP_Connect):
    """ MCP_StentDirect(costs, nodes, th, sampling)
    Subclass of MCP_Connect to find the connections between seed points
    and turn these into an undirected graph.
    """
    
    def __init__(self, costs, nodes, th, sampling):
        skimage.graph._mcp.MCP_Connect.__init__(self, costs, sampling=sampling)
        
        # Store the inputs
        self._nodes = nodes
        self._costs = costs
        self._th = th
        self._sampling = sampling
        
        # Init connections
        self._connections = {}
        self._connectioncount = 0
    
    
    def create_connection(self, id1, id2, pos1, pos2, cost1, cost2):
        """ Hook: this is called by the MCP algorithm when a connection 
        is found.
        """
        # Get hash and pack value
        hash = min(id1, id2), max(id1, id2)
        val = id1, id2, pos1, pos2, max(cost1, cost2)
        # Set, overwrite or leave connection as is?
        if hash in self._connections:
            currentcost = self._connections[hash][-1]
            if val[-1] < currentcost:
                self._connections[hash] = val
        else:
            self._connections[hash] = val
    
    
    def traceback_to_Pointset(self, path):
        pp = vv.Pointset(len(path[0]))
        for p in reversed(path):
            pp.append(tuple(reversed(p)))
        sampling_factor = np.array(list(reversed(self._sampling)))
        return pp * sampling_factor
    
    
    def finalize_connections(self, nodes, costToCtValue):
        """ Turn the connections into a proper edge in the graph.
        """
        for (id1, id2, pos1, pos2, cost) in self._connections.values():
            
            # Get the two parts of the traceback
            tb1 = self.traceback(pos1)
            tb2 = self.traceback(pos2)
            
            # Turn into pointset and glue the two parts together
            pp1 = self.traceback_to_Pointset( list(reversed(tb1)) )
            pp2 = self.traceback_to_Pointset( tb2 )
            pp1.extend(pp2)
            
            # Get CT values along the path
            pathCosts = [self._costs[(p.z,p.y,p.x)] for p in pp1]
            ctValues = [costToCtValue(cost) for cost in pathCosts] 
            
            # Create a connection object (i.e. an edge), and connect it
            node1 = nodes[id1]
            node2 = nodes[id2]
            cnew = stentGraph.PathEdge(node1, node2, cost, min(ctValues), pp1)
            cnew.Connect()
    
    
    def goal_reached(self, index, cumcost):
        """ Hook: we can stop the algorithm if the cumulative cost is
        sufficiently high. 
        """
        if cumcost > self._th:
            return 2
        else:
            return 0

