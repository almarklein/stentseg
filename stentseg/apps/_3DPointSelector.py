""" Module 3D Point Selector

Provides functionality view slices and to select points in multiplanar reconstructions.
"""
import os, time, sys
import numpy as np
import visvis as vv
from visvis.utils.pypoints import Point, Pointset, Aarray

import OpenGL.GL as gl
import OpenGL.GLU as glu

class VolViewer:
    """ VolViewer. View (CT) volume while scrolling through slices x,y or z depending on the direction chosen
    """
    def __init__(self, vol, direction, axes=None):
        self.direction = direction
        # Store vol and init
        if self.direction == 0:
            self.vol = vol
        elif self.direction == 1:
            self.vol = np.transpose(vol,(1,0,2))
            self.vol.origin = (vol.origin[1],vol.origin[0],vol.origin[2])
            self.vol.sampling = (vol.sampling[1],vol.sampling[0],vol.sampling[2])
        elif self.direction == 2:
            self.vol = np.transpose(vol,(2,0,1))
            self.vol.origin = (vol.origin[2],vol.origin[0],vol.origin[1])
            self.vol.sampling = (vol.sampling[2],vol.sampling[0],vol.sampling[1])
        else:
            S('No valid input for direction, only 1,2 or 3 is possible')
        self.slice = 0
    
        # Prepare figure and axex
        if axes is None:
            self.a = vv.gca()
        else:
            self.a = axes
            
        self.f = vv.gcf()
            
        # Create slice in 2D texture
        # self.t = vv.imshow(self.vol[self.slice,:,:],clim = [self.vol.min(), self.vol.max()],axes=self.a)
        self.t = vv.imshow(self.vol[self.slice,:,:],axes=self.a)
        
        # Bind
        self.a.eventScroll.Bind(self.on_scroll)
        self.eventPositionUpdate = vv.events.BaseEvent(self)
        
        axes.eventMouseDown.Bind(self.on_click)
                
        # Fig properties
        self.a.bgcolor = [0,0,0]
        self.a.axis.visible = False
        self.a.showAxis = False
    
    def on_scroll(self, event):
        self.slice += int(event.verticalSteps)
        if self.slice > (self.vol.shape[0]-1):
            self.slice = (self.vol.shape[0]-1)
        if self.slice < 0:
            self.slice = 0
        self.show()
        return True
        
    def on_click(self, event):        
        # get current mouse position
        self._refpos = [round(event.x2d,1), round(event.y2d,1)]
        #print(self._refpos)
        self.eventPositionUpdate.Fire()
        return self._refpos
    
    def show(self):
        self.t.SetData(self.vol[self.slice,:,:])
        
    def GetCurrentSlice(self):
        ctslice = self.slice
        CurrentSlice = round(self.vol.origin[0] + ctslice * self.vol.sampling[0],1)
        return CurrentSlice
        
    def SetCurrentSlice(self, slicenr):
        ctslice = (slicenr - self.vol.origin[0])/ self.vol.sampling[0]
        self.slice = ctslice
        self.show()
    

class PointSelect3D:
    """ A helper class for 3d point select. Use the select3dpoint function to
    perform manual point selection.
    """    
    def __init__(self, vol, a_transversal, a_coronal, a_sagittal, a_MIP, a_text, nr_of_stents):
        self.nr_of_stents = nr_of_stents
        self.f = vv.gcf()
        self.vol = vol
        
        # Create empty list of endpoints
        self.endpoints = []
        self.endpoints = ['xx,yy,zz'] * nr_of_stents * 2
        self.endpointsindex = 0
        
        # Create text objects
        self._labelcurrent = vv.Label(a_text)
        self._labelx = vv.Label(a_text)   
        self._labelxslice = vv.Label(a_text)     
        self._labely = vv.Label(a_text)
        self._labelyslice = vv.Label(a_text)        
        self._labelz = vv.Label(a_text)
        self._labelzslice = vv.Label(a_text)
        self._labelcurrent.position = -250,10
        self._labelx.position = -250,35
        self._labelxslice.position = -200,35
        self._labely.position = -250,55
        self._labelyslice.position = -200,55
        self._labelz.position = -250,75
        self._labelzslice.position = -200,75
        
        self._labelendpointstext =[]
        self._labelendpointstext.append(vv.Label(a_text))
        self._labelendpointstext[0].position = 100,-5
        self._labelendpointstext.append(vv.Label(a_text))
        self._labelendpointstext[1].position = 230,-5
        for i in range(2,self.nr_of_stents+2):
            self._labelendpointstext.append(vv.Label(a_text))
            self._labelendpointstext[i].position = 40,15+(20*(i-2))
        
        self._labelendpoints = []
        for i in range(0,self.nr_of_stents * 2,2):
            self._labelendpoints.append(vv.Label(a_text))
            self._labelendpoints[i].position = 100,15+(20*(i/2)),50,20
            self._labelendpoints.append(vv.Label(a_text))
            self._labelendpoints[i+1].position = 230,15+(20*(i/2)),50,20
        
        # Create Select button
        self._select = False
        self._butselect = vv.PushButton(a_text)
        self._butselect.position = -110,150
        self._butselect.text = 'Select'
        
        # Create Back button
        self._back = False
        self._butback = vv.PushButton(a_text)
        self._butback.position = 10,150
        self._butback.text = 'Back'
        
        # Create Close button
        self._finished = False
        self._butclose = vv.PushButton(a_text)
        self._butclose.position = -50,180
        self._butclose.text = 'Finish'
        
        # Get short name for sampling
        if isinstance(vol, Aarray):
            self._sam = sam = vol.sampling
        else:
            self._sam = None
            sam = (1,1,1)
        
        # Display the slices and 3D MIP
        self.b1 = VolViewer(vol, 0, axes=a_transversal)
        self.b2 = VolViewer(vol, 1, axes=a_coronal)
        self.b3 = VolViewer(vol, 2, axes=a_sagittal)
        
        renderstyle = 'mip'
        a_MIP.daspect = 1,1,-1
        self.b4 = vv.volshow(vol, clim=(0,2500), renderStyle = renderstyle, axes=a_MIP)
        
        # set axis settings
        for a in [a_transversal, a_coronal, a_sagittal, a_MIP]:
            a.bgcolor = [0,0,0]
            a.axis.visible = False
            a.showAxis = True 
                
        # get current slice number        
        Zslice = self.b1.GetCurrentSlice()        
        Yslice = self.b2.GetCurrentSlice()
        Xslice = self.b3.GetCurrentSlice()
        size = vol.shape
        
        # create lines for position of x,y and z slices
        origin = vol.origin
        Zrange = (origin[0], (size[0]*sam[0])+origin[0])
        Xrange = (origin[1], (size[1]*sam[1])+origin[1])
        Yrange = (origin[2], (size[2]*sam[2])+origin[2])
        
        self.l11 = vv.Line(a_transversal,[(Yslice,Xrange[0]),(Yslice,Xrange[1])])
        self.l12 = vv.Line(a_transversal,[(Yrange[0],Xslice),(Yrange[1],Xslice)])

        self.l21 = vv.Line(a_coronal,[(Zslice,Zrange[0]),(Zslice,Zrange[1])])
        self.l22 = vv.Line(a_coronal,[(Yrange[0],Xslice),(Yrange[1],Xslice)])

        self.l31 = vv.Line(a_sagittal, [(Zslice,Zrange[0]),(Zslice,Zrange[1])])
        self.l32 = vv.Line(a_sagittal, [(Xrange[0],Yslice),(Xrange[1],Yslice)])
        
        # change color of the lines
        for i in [self.l11,self.l12,self.l21,self.l22,self.l31,self.l32]:
            i.lc = 'g'
            
        # create a point in the MIP figure for the current position
        self.mippoint = vv.Line(a_MIP, [(Zslice,Xslice,Yslice)])
        self.mippoint.ms = 'o'
        self.mippoint.mw = 5
        self.mippoint.mc = 'g'
        self.mippoint.alpha = 0.9
        
        # Get list of all range wobjects
        self._volviewers = [self.b1, self.b2, self.b3]
            
        # Bind events
        fig = a_text.GetFigure()
        fig.eventClose.Bind(self._OnFinish)
        self._butclose.eventPress.Bind(self._OnFinish)
        self._butselect.eventPress.Bind(self._OnSelect)
        self._butback.eventPress.Bind(self._OnBack)
        
        for r in self._volviewers:
            r.eventPositionUpdate.Bind(self._OnMouseClickAxis)
        for s in range(len(self._labelendpoints)):
            self._labelendpoints[s].eventMouseDown.Bind(self._OnMouseClickEndpoint)
        
        # Almost done
        self._SetTexts()
        self.updatePosition()
    
    def _OnMouseClickEndpoint(self,event):
        index = self._labelendpoints.index(event.owner)
        self.endpointsindex = index
        self.updateText()
        
    def _OnMouseClickAxis(self,event):
        # Get ranges of wobject that fired the event
        rangex, rangey = event.owner._refpos[0], event.owner._refpos[1]
                
        # Update slices in onther wobjects
        if event.owner is self.b1:
            self.b2.SetCurrentSlice(rangey)
            self.b3.SetCurrentSlice(rangex)
        elif event.owner is self.b2:
            self.b1.SetCurrentSlice(rangey)
            self.b3.SetCurrentSlice(rangex)
        elif event.owner is self.b3:
            self.b1.SetCurrentSlice(rangey)
            self.b2.SetCurrentSlice(rangex)
        else:
            print('unknown owner! %s' % repr(event.owner))
                                
    def _SetTexts(self):
        # Get short names for labels
        lx, ly, lz = self._labelx, self._labely, self._labelz
        
        # Apply texts
        self._labelcurrent.text = 'Current Position:'
        lx.text =  'X: '
        ly.text =  'Y: '
        lz.text =  'Z: '
        
        self._labelendpointstext[0].text = 'StartPoints'
        self._labelendpointstext[1].text = 'EndPoints'
        for i in range(2,(self.nr_of_stents)+2):
            self._labelendpointstext[i].text = 'Stent %1d:' % int(i-1)
            
        for i in range(self.nr_of_stents*2):
            self._labelendpoints[i].text = self.endpoints[i]
            
        for i in range(self.nr_of_stents*2):
            if i == self.endpointsindex:
                self._labelendpoints[i].textColor = 'b'
            else:
                self._labelendpoints[i].textColor = 'k'
                
            
    def _OnSelect(self, event):
        Position = self.updatePosition()
        if self.endpointsindex <= len(self.endpoints)-1:
            self.endpoints[self.endpointsindex] = Position
            self.endpointsindex += 1
            self.updateText()
        #print(self.endpoints)
        #print('Current position = ' + str(Position))
        
    def _OnBack(self, event):
        if not(self.endpointsindex <0):
            self.endpoints[self.endpointsindex] = 'xx,yy,zz'
            self.updateText()
        print(self.endpoints)    
        print('Back Pressed')
        
    def _OnFinish(self, event):
        self._finished = True
        return self.endpoints
        print('Finish Pressed')
        
    def updatePosition(self):
        # get current slice numbers
        Zslice = self.b1.GetCurrentSlice()
        Yslice = self.b2.GetCurrentSlice()
        Xslice = self.b3.GetCurrentSlice()
        
        # update lines
        self.l11.SetXdata([Xslice,Xslice])
        self.l12.SetYdata([Yslice,Yslice])
        self.l21.SetXdata([Xslice,Xslice])
        self.l22.SetYdata([Zslice,Zslice])
        self.l31.SetXdata([Yslice,Yslice])
        self.l32.SetYdata([Zslice,Zslice]) 
        
        # update Point
        self.mippoint.SetXdata([Xslice])
        self.mippoint.SetYdata([Yslice])
        self.mippoint.SetZdata([Zslice])     
        
        # update current slice text
        self._labelxslice.text = str(Xslice)
        self._labelyslice.text = str(Yslice)
        self._labelzslice.text = str(Zslice)
        
        # return Position
        Position = (Xslice, Yslice, Zslice)
        return Position
        
    def updateText(self):
        for i in range(self.nr_of_stents*2):
            self._labelendpoints[i].text = str(self.endpoints[i])
            if i == self.endpointsindex:
                self._labelendpoints[i].textColor = 'b'
            else:
                self._labelendpoints[i].textColor = 'k'
    
    def Run(self):
        vv.processEvents()
        self.updatePosition()

def select3dpoints(vol, nr_of_stents, fig=None):
    """ Manually select 3d points in a volume. In the given figure (or a new 
    figure if None), three axes are created that display the transversal, 
    sagittal and coronal slices of the volume. The user can then use the mouse 
    to scroll to the correct slice and select the current position as an 
    endpoint of a stent. 
    Input: Number of stents to select start- and endpoints for.
    """
        
    # Create figure 
    if fig is None:
        fig = vv.figure()    
        figCleanup = True
    else:
        fig.Clear()
        figCleanup = False
    
    # Create four axes and a wibject to attach text labels to  
    fig.position = 0, 22, 750, 700 
    fig.title = '3D Point Selector' 
    a1 = vv.subplot(321)
    a2 = vv.subplot(322)
    a3 = vv.subplot(323)
    a4 = vv.subplot(324)
    a5 = vv.Wibject(fig)
    
    # x-richting, y-richting, x-breedte?, y-breedte?
    a5.position = 0.5, 0.7, 0.5, 0.5
    
    # Set settings
    for a in [a1, a2, a3, a4]:
        a.showAxis = False
    
    # Create PointSelect instance
    pointselect3d = PointSelect3D(vol, a1, a3, a2, a4, a5, nr_of_stents)
    
    # Enter a mainloop
    while not pointselect3d._finished:
        pointselect3d.Run()
        time.sleep(0.01)
    
    # Clean up figure (close if we opened it)
    fig.Clear()
    fig.DrawNow()
    if figCleanup:    
        fig.Destroy()
    
    # Done (return points)
    Startpoints = []
    Endpoints = []
    for i in range(nr_of_stents):
        if isinstance(pointselect3d.endpoints[i*2],tuple):
            Startpoints.append(pointselect3d.endpoints[i*2])
        if isinstance(pointselect3d.endpoints[(i*2)+1],tuple):
            Endpoints.append(pointselect3d.endpoints[(i*2)+1])
    
    return Startpoints, Endpoints
