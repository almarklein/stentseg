""" A version of MCP_Connect from skimage that is tweaked to work
with our stent segmentation algorithm.
"""

import numpy as np
import visvis as vv
from . import stentGraph

import skimage.graph
class MCP_StentDirect(skimage.graph.MCP_Connect):
    
    def __init__(self, costs, nodes, th):
        skimage.graph._mcp.MCP_Connect.__init__(self, costs, sampling=costs.sampling)
        self._sampling = costs.sampling
        self._th = th
        self._nodes = nodes
        self._costs = costs
        self._shape = costs.shape
        self._connections = {}
        self._connectioncount = 0
    
    
    def create_connection(self, id1, id2, path1, path2, cost1, cost2):
        hash = min(id1, id2), max(id1, id2)
        val = id1, id2, path1, path2, max(cost1, cost2)
        if hash in self._connections:
            currentcost = self._connections[hash][-1]
            # Sometimes it is more, sometimes less ...
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
        for (id1, id2, path1, path2, cost) in self._connections.values():
            #path1, path2 =_unravel_index_fortran([path1, path2], self._costs.shape)
            
            tb1 = self.traceback(path1)
            tb2 = self.traceback(path2)
            
#             tb1 = self.unravel_traceback(path1)
#             tb2 = self.unravel_traceback(path2)
            
            pp1 = self.traceback_to_Pointset( list(reversed(tb1)) )
            pp2 = self.traceback_to_Pointset( tb2 )
            pp1.extend(pp2)
            
            pathCosts = [self._costs[(p.z,p.y,p.x)] for p in pp1]
            ctValues = [costToCtValue(cost) for cost in pathCosts] 
            node1 = nodes[id1]
            node2 = nodes[id2]
            cnew = stentGraph.PathEdge(node1, node2, cost, min(ctValues), pp1)
            cnew.Connect()
    
    def goal_reached(self, index, cumcost):
        if cumcost > self._th:
            return 2
        else:
            return 0
