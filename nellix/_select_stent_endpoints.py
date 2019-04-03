'''
Created on 28 jun. 2016

@author: TomLoonen
'''

class _Select_Stent_Endpoints:
    def __init__(self,ptcode,ctcode,basedir, clim=None):
        """ select start and endpoints to be used for centerline generation
        """
        from stentseg.apps._3DPointSelector import select3dpoints
        # from stentseg.apps.ui_dialog import MyDialog
        # from PyQt4 import QtGui
        
        import imageio
        import os
        from stentseg.utils.datahandling import select_dir, loadvol
        import copy
        
        
        # Select dataset to register
        cropname = 'prox'
        what = 'avgreg'
        
        # Load volumes
        s = loadvol(basedir, ptcode, ctcode, cropname, what)
        vol_org = copy.deepcopy(s.vol)
        s.vol.sampling = [vol_org.sampling[1], vol_org.sampling[1], vol_org.sampling[2]]
        s.sampling = s.vol.sampling
        
        # # Select nr of stents
        # app = QtGui.QApplication([])
        # m = MyDialog()
        # m.show()
        # m.exec_()
        # nr_of_stents = int(m.combo.currentText())
        
        # Endpoint selection
        points = select3dpoints(s.vol,nr_of_stents = 6, clim=clim)
        self.StartPoints = points[0]
        self.EndPoints = points[1]
        
        print('StartPoints are: ' + str(self.StartPoints))
        print('EndPoints are: ' + str(self.EndPoints))