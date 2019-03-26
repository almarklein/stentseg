""" Class to perform landmark selection in one or multiple volumes

Copyright Maaike Koenrades
"""

from stentseg.utils.visualization import DrawModelAxes
from stentseg.utils.picker import label2worldcoordinates
from stentseg.stentdirect import stentgraph
from visvis import ssdf
import visvis as vv
import numpy as np
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel


class LandmarkSelector:
    """ LandmarkSelector. Create MIP by default
    """
    def __init__(self, dirsave,ptcode,ctcode,cropname, s, what='phases', axes=None, **kwargs):
        """ s is struct from loadvol
        """
        
        self.fig = vv.figure(1); vv.clf()
        self.fig.position = 0.00, 29.00,  1680.00, 973.00
        self.defaultzoom = 0.025 # check current zoom with foo.ax.GetView()
        self.what = what
        self.ptcode = ptcode
        self.dirsave = dirsave
        self.ctcode = ctcode
        self.cropname = cropname
        if self.what == 'phases':
            self.phase = 0
        else:
            self.phase = self.what # avgreg
        # self.vol = s.vol0
        self.s = s # s with vol(s)
        self.s_landmarks = vv.ssdf.new()
        self.graph = stentgraph.StentGraph()
        self.points = [] # selected points
        self.nodepoints = []
        self.pointindex = 0 # for selected points
        try:
            self.vol = s.vol0 # when phases
        except AttributeError:
            self.vol = s.vol # when avgreg
        
        self.ax = vv.subplot(121)
        self.axref = vv.subplot(122)
        
        self.label = DrawModelAxes(self.vol, ax=self.ax) # label of clicked point
        self.axref.bgcolor = 0,0,0
        self.axref.visible = False
       
        # create axis for buttons
        a_select = vv.Wibject(self.ax) # on self.ax or fig?
        a_select.position = 0.55, 0.7, 0.6, 0.5 # x, y, w, h
        
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
        
        # Create Next/Save button
        self._finished = False
        self._butclose = vv.PushButton(a_select)
        self._butclose.position = 10,230
        self._butclose.text = 'Next/Save'
        
        # # Create Save landmarks button
        # self._save = False
        # self._butsave = vv.PushButton(a_select)
        # self._butsave.position = 125,230
        # self._butsave.text = 'Save|Finished'
        
        # Create Reset-View button
        self._resetview = False
        self._butresetview = vv.PushButton(a_select)
        self._butresetview.position = 10,180
        self._butresetview.text = 'Default Zoom' # back to default zoom
        
        # bind event handlers
        self.fig.eventClose.Bind(self._onFinish)
        self._butclose.eventPress.Bind(self._onFinish)
        self._butselect.eventPress.Bind(self._onSelect)
        self._butback.eventPress.Bind(self._onBack)
        self._butresetview.eventPress.Bind(self._onView)
        # self._butsave.eventPress.Bind(self._onSave)
        
        self._updateTextIndex()
        self._updateTitle()
        
    def _updateTitle(self):
        if self.what == 'phases':
            vv.title('CT Volume {}% for LSPEAS {} '.format(self.phase, self.ptcode[7:]))
        else:
            vv.title('CT Volume {} for LSPEAS {} '.format(self.phase, self.ptcode[7:]))
    
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
        scale = 0.25
        alpha = 1
        # create object sphere for point
        view = self.ax.GetView()
        #todo: plot instead of solidShere? better vis?
        # graph.Draw(mc='r', mw = 10, lc='y')
        # point = vv.plot(point[0], point[1], point[2], 
        #                 mc = 'm', ms = 'o', mw = 8, alpha=0.5)
        node_point = vv.solidSphere(translation = (n), scaling = (scale,scale,scale))
        node_point.faceColor = (0,1,0,alpha) # 'g' but with alpha
        node_point.visible = True
        node_point.node = n
        node_point.nr = self.pointindex
        self.ax.SetView(view)
        # store
        self.graph.add_node(n, number=self.pointindex)
        self.nodepoints.append(node_point)
        # update index of total selected points
        self.pointindex += 1
        self._updateTextIndex()
    
    def _onFinish(self, event):
        self._finished = True
        print(self.points)
        phase = self.phase
        # store model and pack
        storegraph = self.graph
        self.s_landmarks['landmarks{}'.format(phase)] = storegraph.pack() # s.vol0 etc
        if self.what == 'phases':
            # go to next phase 
            self.phase+= 10 # next phase
            self.points = [] # empty for new selected points
            self.nodepoints = []
            self.pointindex = 0 
            self.vol = self.s['vol{}'.format(self.phase)] # set new vol
            self.graph = stentgraph.StentGraph() # new empty graph
            self._updateTitle()
            self._updateTextIndex()
            self.ax.Clear() # clear the axes. Removing all wobjects
            self.label = DrawModelAxes(self.vol, ax=self.ax)
            
            # draw vol and graph of 0% in axref
            model = stentgraph.StentGraph()
            model.unpack(self.s_landmarks.landmarks0 )
            self.axref.Clear()
            DrawModelAxes(self.s.vol0, model, ax=self.axref)
            self.axref.visible = True
            vv.title('CT Volume 0% for LSPEAS {} with selected landmarks'.format(
                self.ptcode[7:]))
            
            self.ax.camera = self.axref.camera
        
        # === Store landmarks graph ssdf ===
        dirsave = self.dirsave
        ptcode = self.ptcode
        ctcode = self.ctcode
        cropname = self.cropname
        what = self.what
        saveLandmarkModel(self, dirsave, ptcode, ctcode, cropname, what)
        
        print('Next/Finish was pressed - Landmarks stored')
        return
        
    # def _onSave(self, event):
    #     """ save landmarks graph
    #     """
    #     dirsave = self.dirsave
    #     ptcode = self.ptcode
    #     ctcode = self.ctcode
    #     cropname = self.cropname
    #     what = self.what
    #     saveLandmarkModel(self, dirsave, ptcode, ctcode, cropname, what)
    
    def _onBack(self, event):
        # remove last selected point
        if not (self.pointindex <0): # index always 0 for first point
            self.points.pop(self.pointindex-1) # pop last point from list
            self.pointindex += -1
            self._updateTextIndex()
            removednode = self.nodepoints.pop(-1)
            self.graph.remove_node(removednode.node)
            removednode.Destroy()
        print(self.points)    
        print('Undo was pressed')
        
    def _onView(self, event):
        view = self.ax.GetView()
        view['zoom'] = self.defaultzoom
        self.ax.SetView(view) 
    
    def Run(self):
        vv.processEvents()


def saveLandmarkModel(ls, dirsave, ptcode, ctcode, cropname, what):
    """ Save graph with landmarks; do not store volumes again
    Use within or outside class LandmarkSelector
    """
    import os
    
    s = ls.s
    s2 = ls.s_landmarks
    s2.sampling = s.sampling
    s2.origin = s.origin
    s2.stenttype = s.stenttype
    s2.croprange = s.croprange # keep for reference
    for key in dir(s):
            if key.startswith('meta'):
                suffix = key[4:]
                s2['meta'+suffix] = s['meta'+suffix]
    s2.what = what
    s2.params = 'LandmarkSelector'
    s2.stentType = s.stenttype
    
    # Save
    filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'landmarks'+what)
    ssdf.save(os.path.join(dirsave, filename), s2)
    print('')
    print('saved to disk to {}.'.format(os.path.join(dirsave, filename)) )


def makeLandmarkModelDynamic(basedir, ptcode, ctcode, cropname, what='landmarksavgreg',
                     savedir=None):
    """ Make model dynamic with deforms from registration 
        (and store/overwrite to disk)
    """
    #todo: change in default and merge with branch landmarks?
    import pirt
    from stentseg.motion.dynamic import (incorporate_motion_nodes, 
                                         incorporate_motion_edges)
    from visvis import ssdf
    import os
    
    if savedir is None:
        savedir = basedir
    # Load deforms
    s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
    deformkeys = []
    for key in dir(s):
        if key.startswith('deform'):
            deformkeys.append(key)
    deforms = [s[key] for key in deformkeys]
    deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
    paramsreg = s.params
    
    # Load model where landmarks were stored
    # s2 = loadmodel(savedir, ptcode, ctcode, cropname, what)
    fname = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, what)
    s2 = ssdf.load(os.path.join(savedir, fname))
    # Turn into graph model
    model = stentgraph.StentGraph()
    model.unpack(s2[what])
    
    # Combine ...
    incorporate_motion_nodes(model, deforms, s2.origin)
    incorporate_motion_edges(model, deforms, s2.origin)
    
    # Save back
    filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, what)
    s2.model = model.pack()
    s2.paramsreg = paramsreg
    ssdf.save(os.path.join(savedir, filename), s2)
    print('saved to disk to {}.'.format(os.path.join(savedir, filename)) )


if __name__ == '__main__':
    
    ls = LandmarkSelector(ptcode, s, what=what, clim=clim, showVol=showVol, axVis=True)
    
    # s_landmarks = ls.s_landmarks # ssdf struct with graphs for landmark models 
    # model0 = s_landmarks.landmarks0 # model0.nodes returns selected points
    # model10 = s_landmarks.landmarks10
    # model20 = s_landmarks.landmarks20
    # etc.