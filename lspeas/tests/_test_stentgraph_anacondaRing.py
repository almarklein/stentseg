import numpy as np
import networkx
import visvis as vv
from visvis import ssdf

from stentseg.stentdirect import StentDirect, StentDirect_old, getDefaultParams, stentgraph, stentgraph_anacondaRing
from stentseg.stentdirect.stentgraph import _sorted_neighbours

## Test prune_weak_exceptions for proximal ring

def test_weak4_exception():
    """    5 --- 4 --- 3 ---7
                /     /
        6 --- 1 --- 2 --- 8
    
    """
    
    # Test that edge 1-2 is NOT removed
    graph = stentgraph.StentGraph()
    graph.add_edge(1, 2, cost=4, ctvalue=20)
    graph.add_edge(1, 4, cost=2, ctvalue=50)  
    graph.add_edge(1, 6, cost=1, ctvalue=50)  
    #
    graph.add_edge(2, 3, cost=2, ctvalue=35) # low CT
    graph.add_edge(2, 8, cost=2, ctvalue=50)
    #
    graph.add_edge(3, 4, cost=2, ctvalue=50)
    graph.add_edge(3, 7, cost=2, ctvalue=50)
    graph.add_edge(4, 5, cost=1, ctvalue=50)
    
    #prune_weak(graph, 2, 80)
    
    # Check graph
    assert graph.number_of_edges() == 8
    
    # return the graph
    return graph

graph = test_weak4_exception()

## Test prune_weak_exceptions for triangle cases

def test_weak5_exception():
    """   6 ------ 2 --- 7 ---
                  / \    (\) 
           4 --- 1---3 --- 5 ---
    
    Result: Edge 2-3 does not cause problems when strong or weak and high or low CTvalue.
            When 1-2 and 2-3 are weak and of low CT value:
                 2-3 not removed when 5-7 is of high CTvalue -> strong tetragon
                 2-3 is removed when 5-7 is of low CTvalue -> weak tetragon or when 5-7 is not existent
                 
    """
    
    # Test that edge 1-2 is removed
    graph = stentgraph.StentGraph()
    graph.add_edge(1, 2, cost=4, ctvalue=20)
    graph.add_edge(1, 4, cost=1, ctvalue=50)  
    graph.add_edge(1, 3, cost=2, ctvalue=50)  
    #
    graph.add_edge(2, 3, cost=3, ctvalue=30)
    graph.add_edge(2, 6, cost=1, ctvalue=50)
    graph.add_edge(2, 7, cost=2, ctvalue=50)
    #
    #graph.add_edge(5, 7, cost=1, ctvalue=50)
    graph.add_edge(5, 3, cost=2, ctvalue=40)
    
    #prune_weak(graph, 2, 80)
    
    # Check graph
    assert graph.number_of_edges() == 7  #or 8
    
    # return the graph
    return graph

graph = test_weak5_exception()


## Test prune_weak_exceptions for triangle within tetragon

def test_weak6_exception():
    """   6 ------ 2 --- 7 ---
                  / \  /  
           4 --- 1---3 --- 5 ---
    
    Rationale: this situation can occur when there is a cross connection within 
    a tetragon. We need to make sure that 1-2 is recognized as part of tetragon,
    independent of 2-3 values, so edge 7-3 must always be considered.
    (solved with *continue* statements in function prune_weak_exception)
    """
    
    # Test that edge 1-2 is NOT removed
    graph = stentgraph.StentGraph()
    graph.add_edge(1, 2, cost=4, ctvalue=20)
    graph.add_edge(1, 4, cost=1, ctvalue=50)  
    graph.add_edge(1, 3, cost=2, ctvalue=50)  
    #
    graph.add_edge(2, 3, cost=1, ctvalue=50)
    graph.add_edge(2, 6, cost=1, ctvalue=50)
    graph.add_edge(2, 7, cost=2, ctvalue=50)
    #
    graph.add_edge(3, 7, cost=1, ctvalue=50)
    graph.add_edge(5, 3, cost=2, ctvalue=40)
    
    #prune_weak(graph, 2, 80)
    
    # Check graph
    assert graph.number_of_edges() == 8
    
    # return the graph
    return graph

graph = test_weak6_exception()


## Test prune_weak_exceptions for multiple tetragons for n1-n2

def test_weak7_exception():
    """    6 --- 4 --- 3 ---7
                /     /
        5 --- 1 --- 2 --- 8
         \     \    /
          ------ 9
    
    Rationale: when multiple tetragons exist for n1-n2, the edge n1-n2 can be 
    removed twice, which causes an error, or be removed wrongly in a first
    instance --> code changed
    """
    
    # Test that edge 1-2 is NOT removed
    graph = stentgraph.StentGraph()
    graph.add_edge(1, 2, cost=4, ctvalue=20) # weak
    graph.add_edge(1, 4, cost=2, ctvalue=50)  
    graph.add_edge(1, 5, cost=1, ctvalue=50)
    graph.add_edge(1, 9, cost=5, ctvalue=30) # very weak  
    #
    graph.add_edge(2, 3, cost=2, ctvalue=50)
    graph.add_edge(2, 8, cost=2, ctvalue=50)
    graph.add_edge(2, 9, cost=3, ctvalue=50) # moderate weak
    #
    graph.add_edge(3, 4, cost=2, ctvalue=50)
    graph.add_edge(3, 7, cost=2, ctvalue=50)
    graph.add_edge(4, 6, cost=1, ctvalue=50)
    graph.add_edge(9, 5, cost=3, ctvalue=30) # moderate weak
    
    # Check graph
    assert graph.number_of_edges() == 11
    
    # Prune edges
    #prune_weak_exception(graph, 2, 40)
    
    # Return the graph
    return graph

graph = test_weak7_exception()


## Modified prune_weak - v2           old version

#def prune_weak_exception(graph, enc, ctvalue):

enc = 2
ctvalue = 40

# First, get a sorted list of edges
edges = graph.edges()
cnt = 0    
for (n1, n2) in edges:
    c = graph[n1][n2]
    if c['ctvalue'] < ctvalue:
        # We *might* remove this edge
        
        # Check for each node what their "extra edges" are
        extra_edges1 = _sorted_neighbours(graph, n1)
        extra_edges2 = _sorted_neighbours(graph, n2)
        
        # If this edge is "extra" as seen from both ends, we *might* remove it
        if n1 in extra_edges2[enc:] and n2 in extra_edges1[enc:]:
            print('Eligable edge for removal: ',n1,'-',n2 ,graph.edge[n1][n2])
            tetragon = 0
            # Test whether edge n1,n2 is part of *strong* tetragon
            for node1 in extra_edges1:
                if node1 == n2:
                    continue  # go to next iteration, do not consider n1-n2 but 
                              # do consider edges with neighbours with higher cost than n1-n2 
                for node2 in extra_edges2:
                    if node2 == n1:
                        continue
                    if graph.has_edge(node1,node2):
                        t1 = graph[node1][node2]
                        t2 = graph[n1][node1]
                        t3 = graph[n2][node2]
                        print('Tetragon found with connecting edge: ',node1,'-',node2,t1)
                        if t1['ctvalue'] >= ctvalue:# and t2['ctvalue'] >= ctvalue and t3['ctvalue'] >= ctvalue:
                            tetragon = 1            # strong
                        else:                       
                            tetragon = 2            # weak
                            break
            if tetragon == 0:
                print('Eligable edge',n1,'-',n2,'not part of tetragon so removed')
                graph.remove_edge(n1, n2)
                print('******************')
            if tetragon == 1:
                print('Eligable edge',n1,'-',n2,'part of *strong* tetragon so not removed')
                print('******************')
                cnt +=1
            if tetragon == 2:
                print('Eligable edge',n1,'-',n2,'part of *weak* tetragon so removed')
                graph.remove_edge(n1, n2)
                print('******************')

print('In total %i eligable edges for removal were part of *strong* tetragon so not removed' % cnt)
print('******************')                
print('Original  edges: ',edges)
print('Resulting edges: ',graph.edges())



##
##



## Modified prune_redundant                    # old version

def prune_redundant(graph, ctvalue):
    """
    Remove redundant edges. 
    
    A connection is redundant if a weak connection (high mcp cost) 
    connects two nodes which are already connected via two other nodes
    wich are each stronger. Exception should be when this is one of the possibly two 
    triangles in the proximal fixation rings between the hooks
    
    In other words, this function tests each triangle of egdes and removes
    the weakest one (except if its above the given ctvalue ***, was seen as an 
    exception in prune_weak? or is an edge between the proximal fixation hooks).
    
    """
    
    # First, get a sorted list of edges (weakest first)
    SORTBY = 'cost'
    edges = graph.edges()
    edges.sort(key=lambda x: graph.edge[x[0]][x[1]][SORTBY])
    if SORTBY == 'cost':
        edges.reverse()
        
    for (n1, n2) in edges:
        _prune_redundant_edge(graph, n1, n2, ctvalue)


def _prune_redundant_edge(graph, n1, n2, min_ctvalue):  # ***** Modified *****
    """ Check if we should prune a given single redundant edge. 
    """
    
    # Do not allow connections to same node. Note: such edges should
    # not be produced by the MCP alg, but we check them to be sure
    if n1 == n2:
        print('Warning: detected node that was connected to itself.')
        graph.remove_edge(n1, n2)
        return
    
    # Get neightbours for n1 and n2
    nn1 = set(graph.edge[n1].keys())
    nn1.discard(n2)
    nn2 = set(graph.edge[n2].keys())
    nn2.discard(n1)
    # Get cost and ctvalue for this edge
    cost = graph.edge[n1][n2]['cost']
    ctvalue = graph.edge[n1][n2]['ctvalue']
    
    # If ct value is high enough, leave edge intact
    if ctvalue >= min_ctvalue:
        return
    
    # Note: the graph type that we use prohibits having more than one
    # edge between any two nodes.
    
    # Check if there are two edges that are stronger than this edge
    for node1 in nn1:
        for node2 in graph.edge[node1].keys():
            if node2 == n2:
                print('Edge',n1,'-',n2,'part of triangle')
                if ((graph.edge[n1][node1]['cost'] < cost) and
                    (graph.edge[node1][node2]['cost'] < cost)):
                        weak = 1
                        if ((graph.edge[n1][node1]['ctvalue'] > min_ctvalue*1.1) and
                        (graph.edge[node1][node2]['ctvalue'] > min_ctvalue*1.1)):
                            weak = 0
                            print('Eligable edge',n1,'-',n2,'part of *strong* triangle so not removed')
                            print('******************')                           
                        if weak == 1:
                            graph.remove_edge(n1, n2)
                            print('Eligable edge',n1,'-',n2,'NOT part of *strong* triangle so removed')
                            print('******************')  
                            return
    
#     for node1 in nn2:
#         for node2 in graph.edge[node1].keys():
#             if node2 == n1:
#                 if ((graph.edge[n2][node1]['cost'] < cost) and
#                     (graph.edge[node1][node2]['cost'] < cost) and
#                     (graph.edge[n2][node1]['ctvalue'] > min_ctvalue) and
#                     (graph.edge[node1][node2]['ctvalue'] > min_ctvalue)):
#                     
#                     graph.remove_edge(n1, n2)
#                     return

## Test prune_redundant_exceptions for triangle within tetragon

def test_redundant1_exception():
    """   6 ------ 2 --- 7 ---
                  / \  /  
           4 --- 1---3 --- 5 ---
    
    Rationale: hooks also form a triangle and sometimes there is also a redundant egde (3-7) part 
    of another triangle connected to the 'hooks' triangle
    """
    
    # Test that edge 1-3 is NOT removed and 3-7 is removed
    graph = stentgraph.StentGraph()
    graph.add_edge(1, 3, cost=4, ctvalue=20)
    graph.add_edge(1, 4, cost=2, ctvalue=50)  
    graph.add_edge(1, 2, cost=1, ctvalue=50)  
    #
    graph.add_edge(2, 3, cost=1, ctvalue=50)
    graph.add_edge(2, 6, cost=2, ctvalue=50)
    graph.add_edge(2, 7, cost=2, ctvalue=40)
    #
    graph.add_edge(3, 7, cost=3, ctvalue=30)
    graph.add_edge(5, 3, cost=2, ctvalue=50)
    
    #prune_weak(graph, 2, 80)
    
    # Check graph
    assert graph.number_of_edges() == 8
    
    # return the graph
    return graph

graph = test_redundant1_exception()

print('Original  edges: ',graph.edges())

prune_redundant(graph,40)

print('Resulting edges: ',graph.edges())





