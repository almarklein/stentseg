# Copyright 2017 MA Koenrades. Created april 2017

""" When showing a deformable mesh show a slider that moves with the index of 
motion, i.e. the phases or the steps of the RR interval in the ECG-gated CT data
"""

import visvis as vv
import numpy as np
import os, time, sys

class ecgSlider:
    """ ECG slider. Create slider and run with _show_model
    functionality on w, q, s and escape
    """
    def __init__(self, dm, fig, ax, motionPlay=(10,0.5), **kwargs):
        """ dm is deformable mesh
        """
        
        self.container = vv.Wibject(ax)
        self.slider = vv.Slider(self.container, fullRange=(0,90), value=0 )
        self.slider.showTicks = True
        self.slider.edgeColor = 0,0,0
        self.container.bgcolor = 1,0,0
        yslider = fig.position[3]-fig.position[1]-35
        self.container.position = 0.15, -20, 0.7, 20
        self.slider.position = 0, 0, 1, 15
        self.fig = fig
        self.maxRange = self.slider.fullRange.max
        self.dm = dm
        self.motionPlay = motionPlay 
        self._finished = False
        
        # Bind
        fig.eventClose.Bind(self._OnClose)
        fig.eventKeyDown.Bind(self._OnKey)
        fig.eventPosition.Bind(self._OnPosition)
        
        self._updateSlider()
        
        
    def _updateSlider(self):
        """ update slider value with motionIndex of deformable mesh
        """
        motionIndex = self.dm.motionIndex
        ecgFrac = motionIndex/self.dm.motionCount
        self.slider.value = ecgFrac * self.maxRange
    
    def _OnClose(self, event):
        self._finished = True
    
    def _OnPosition(self, event):
        # update position
        yslider = self.fig.position[3]+self.fig.position[1]-35
        self.slider.position = 0.15, yslider, 0.7, 20
        
    def _OnKey(self, event):
        if event.text == 'w':
            self.dm.MotionStop()
        if event.text == 'q':
            self.dm.MotionPlay(self.motionPlay[0], self.motionPlay[1])
        if event.text == 'e':
            # show hide slider
            showSlider = self.slider.visible
            if showSlider == False:
                self.slider.visible = True
            else:
                self.slider.visible = False
        if event.key == vv.KEY_ESCAPE:
            # destroy slider
            self._finished = True
            # self.slider.Destroy()
    
    def Run(self):
        vv.processEvents()
        self._updateSlider()
    

def runEcgSlider(dm, fig, ax, motionPlay=(10,0.5), **kwargs):
    """
    """
    # init class ecgSlider
    ecg = ecgSlider(dm, fig, ax, motionPlay=motionPlay)
    print("To control the ECG-slider use w (wait), q (play) and e (show/hide)")
    # Enter a mainloop
    while not ecg._finished:
        ecg.Run()
        time.sleep(0.01)
    
    return ecg