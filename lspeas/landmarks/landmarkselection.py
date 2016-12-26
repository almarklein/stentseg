""" Class to perform landmark selection in one or multiple volumes

Copyright Maaike Koenrades
"""

from stentseg.utils.visualization import DrawModelAxes
from stentseg.utils.picker import label2worldcoordinates
from stentseg.stentdirect import stentgraph
from visvis import ssdf


class LandmarkSelector:
    """ LandmarkSelector. Create MIP by default
    """
    def __init__(self, ptcode, vol, axes=None, **kwargs):
        
        self.fig = vv.figure(1); vv.clf()
        self.fig.position = 8.00, 30.00,  944.00, 1500.00
        self.phase = 0
        self.vol = vol
        self.graph = stentgraph.StentGraph()
        self.s = vv.ssdf.new()
        self.points = [] # selected points
        self.nodepoints = []
        self.pointindex = 0 # for selected points
        self.defaultzoom = 0.025 # check current zoom with foo.ax.GetView()
        
        if axes is None:
            self.ax = vv.gca()
        else:
            self.ax = axes
        
        # a2 = vv.subplot(322)
        # a3 = vv.subplot(323)
        
        self.label = DrawModelAxes(self.vol, ax=self.ax) # label of clicked point
        
        vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)')
        vv.title('CT Volume %i%% for LSPEAS %s ' % (phase, ptcode[7:]))
        
        # create button
        a_select = vv.Wibject(self.fig)
        a_select.position = 0.5, 0.7, 0.6, 0.5 # x, y, w, h
        
        # Create text objects
        self._labelcurrentIndexT = vv.Label(a_select) # for text title
        self._labelcurrentIndexT.position = 125,180
        self._labelcurrentIndexT.text = ' Total selected ='
        self._labelcurrentIndex = vv.Label(a_select)
        self._labelcurrentIndex.position = 225,180
        
        # Create Select button
        self._select = False
        self._butselect = vv.PushButton(a_select)
        self._butselect.position = 10,150
        self._butselect.text = 'Select'
        
        # Create Back button
        self._back = False
        self._butback = vv.PushButton(a_select)
        self._butback.position = 125,150
        self._butback.text = 'Undo'
        
        # Create Close button
        self._finished = False
        self._butclose = vv.PushButton(a_select)
        self._butclose.position = 10,210
        self._butclose.text = 'Finish'
        
        # Create Reset View button
        self._resetview = False
        self._butresetview = vv.PushButton(a_select)
        self._butresetview.position = 10,180
        self._butresetview.text = 'Reset Zoom' # back to default zoom
        
        # bind event handlers
        self.fig.eventClose.Bind(self._onFinish)
        self._butclose.eventPress.Bind(self._onFinish)
        self._butselect.eventPress.Bind(self._onSelect)
        self._butback.eventPress.Bind(self._onBack)
        self._butresetview.eventPress.Bind(self._onView)
        
        self._updateTextIndex()
        
    def _updateTextIndex(self):
        # show number last selected point
        l = self._labelcurrentIndex
        i = self.pointindex
        # set text
        if i == 0:
            l.text = '' # empty, none selected yet
        else:
            l.text = '   '+str(i)
        
    def _onSelect(self, event):
        """
        """
        coordinates = np.asarray(label2worldcoordinates(self.label), 
                      dtype=np.float32) # x,y,z
        n = tuple(coordinates.flat)
        self.points.append(n)
        scale = 0.2
        alpha = 1
        # create object sphere for point
        view = self.ax.GetView()
        #todo: plot instead of solidShere? better vis?
        node_point = vv.solidSphere(translation = (n), scaling = (scale,scale,scale))
        node_point.faceColor = (0,1,0,alpha) # 'g' but with alpha
        node_point.visible = True
        node_point.node = n
        node_point.nr = self.pointindex
        self.ax.SetView(view)
        # store
        self.graph.add_node(n)
        self.nodepoints.append(node_point)
        # update index of total selected points
        self.pointindex += 1
        self._updateTextIndex()
        # self.updateVisPoints()
    
    def _onFinish(self, event):
        self._finished = True
        print(self.points) 
        print('Finish was pressed')
        return self.points
        #todo: go to next vol phase
        
    def _onBack(self, event):
        # remove last selected point
        if not (self.pointindex <0): # index always 0 for first point
            self.points.pop(self.pointindex-1) # pop last point from list
            self.pointindex += -1
            self._updateTextIndex()
            removednode = self.nodepoints.pop(-1)
            removednode.Destroy()
        print(self.points)    
        print('Undo was pressed')
        
    def _onView(self, event):
        view = self.ax.GetView()
        view['zoom'] = self.defaultzoom
        self.ax.SetView(view) 
    
    def Run(self):
        vv.processEvents()


 

# self.a1 = vv.subplot(131) 


ls = LandmarkSelector(ptcode, vol, clim=clim, showVol=showVol, axVis=True)

# model = ls.graph # model.nodes() return selected points