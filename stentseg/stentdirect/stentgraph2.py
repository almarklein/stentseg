"""
Next version for stentseg-specific graph class and functionality.
Use networkx instead of vispy.util.graph.
In progress. Step 1 and step2 of stentdirect work now.
"""

import networkx as nx
from stentseg.utils.new_pointset import PointSet
import visvis as vv
from visvis import ssdf



class StentGraph(nx.Graph):
    """ A graph class specific to the stentseg package.
    It has functionality for drawing in visvis and exporting to ssdf.
    """
    
    def draw(self, mc='g', lc='y', mw=7, lw=0.6, alpha=0.5, axes=None, simple=False):
        """ Draw in visvis.
        """
        
        # we can only draw if we have any nodes
        if not self.number_of_nodes():
            return
        
        # Convert nodes to Poinset
        ppn = PointSet(3)
        for n in self.nodes_iter():
            ppn.append(*n)
        
        # Convert edges to Pointset
        ppe = PointSet(3)
        for n1, n2 in self.edges_iter():
            ppe.append(*n1)
            ppe.append(*n2)
        
        # Plot markers (i.e. nodes)
        vv.plot(ppn, ms='o', mc=mc, mw=mw, ls='', alpha=alpha, 
                axes=axes, axesAdjust=False)
        
        # Plot lines
        vv.plot(ppe, ls='+', lc=lc, lw=lw, ms='', alpha=alpha, 
                axes=axes, axesAdjust=False)
    
    
    def pack(self):
        """ Pack the graph to an ssdf struct.
        This method is not stent-specific and is a generic graph export to ssdf.
        """
        # Prepare
        s = ssdf.new()
        s.nodes = []
        s.edges = []
        # Serialize
        s.graph = ssdf.loads(ssdf.saves(self.graph))
        for node in self.nodes_iter():
            #node_info = {'node': node, 'attr': self.node[node]}
            node_info = node, self.node[node]
            s.nodes.append(node_info)
        for edge in self.edges_iter():
            #edge_info = {'node1': edge[0], 'node2':edge[1], 'attr': self.edge[edge[0]][edge[1]]}
            edge_info = edge[0], edge[1], self.edge[edge[0]][edge[1]]
            s.edges.append(edge_info)
        return s
    
    
    def unpack(self, s):
        """ Populate the graph with the given ssdf struct, which should
        be a grapgh export via the pack method.
        """
        self.clear()
        self.graph.update(s.graph.__dict__)
        for node_info in s.nodes:
            #node, attr = node_info['node'], node_info['attr'].__dict__
            node, attr = node_info
            node = node if not isinstance(node, list) else tuple(node)
            attr = attr if isinstance(attr, dict) else attr.__dict__
            self.add_node(node, attr)
        for edge_info in s.edges:
            #node1, node2, attr = edge_info['node1'], edge_info['node2'], node_info['attr'].__dict__
            node1, node2, attr = edge_info
            node1 = node1 if not isinstance(node1, list) else tuple(node1)
            node2 = node2 if not isinstance(node2, list) else tuple(node2)
            attr = attr if isinstance(attr, dict) else attr.__dict__
            self.add_edge(node1, node2, attr)




if __name__ == '__main__':
    
    # Custom stent
    g = StentGraph(summary='dit is een stent!', lala=3)
    g.add_node((10,20), foo=3)
    g.add_node((30,40), foo=5)
    g.add_edge((1,1), (2,2), bar=10)
    g.add_edge((10,20),(1,1), bar=20)
    
    # Auto generate
    import random
    n = 500
    p=dict((i,(random.gauss(0,2),random.gauss(0,2))) for i in range(n))
    g_ = nx.random_geometric_graph(n, 0.1, dim=3, pos=p)
    print('ok')
    g = StentGraph(summary='dit is een stent!', lala=3)
    g.add_nodes_from(g_.nodes_iter())
    g.add_edges_from(g_.edges_iter())
    
    
    fname = '/home/almar/test.ssdf'
    ssdf.save(fname, g.pack())
    
    g2 = StentGraph()
    g2.unpack(ssdf.load(fname))
    
    print(nx.is_isomorphic(g, g2))
    
    
    
    
