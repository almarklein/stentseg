import visvis as vv

class VolViewer:
    """ VolViewer. View (CT) volume while scrolling through slices (z)
    """
    
    def __init__(self, vol):
        # Store vol and init
        self.vol = vol
        self.z = 0
        # Prepare figure and axex
        self.f = vv.figure(1001)
        self.f.Clear()
        self.a = vv.gca()
        # Create slice in 2D texture
        self.t = vv.imshow(vol[self.z,:,:])
        # Bind
        self.f.eventScroll.Bind(self.on_scroll)
        self.a.eventScroll.Bind(self.on_scroll)
    
    def on_scroll(self, event):
        self.z += int(event.verticalSteps)
        self.z = max(0, self.z)
        self.z = min(self.vol.shape[0], self.z)
        self.show()
        return True
    
    def show(self):
        self.t.SetData(self.vol[self.z])
    
    
# #todo: incorporate option to draw line profile to get intensities and option to display intensity under mouse
# from skimage.measure import profile_line
# img = vol[0,:,:]
# vv.imshow(img)
# plt.imshow(img)
# 
# im = vv.imread('lena.png')
# new_viewer = skimage.viewer.ImageViewer(img) 
# from skimage.viewer.plugins import lineprofile
# new_viewer += lineprofile.LineProfile() 
# new_viewer.show()     
# 
# 
# import numpy as np
# import matplotlib.pyplot as plt
# 
# class Formatter(object):
#     def __init__(self, im):
#         self.im = im
#     def __call__(self, x, y):
#         z = self.im.get_array()[int(y), int(x)]
#         return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)
# 
# fig, ax = plt.subplots()
# im = ax.imshow(img)
# ax.format_coord = Formatter(im)
# plt.show()
