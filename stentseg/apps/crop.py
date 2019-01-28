""" Use modified cropper tool from visvis to crop data.
Bypass and error in cropper.crop3d
"""

import imageio
import visvis as vv
from visvis.utils import cropper
from visvis.utils.cropper import Cropper3D
from visvis import ssdf
import time

def crop3d(vol, fig=None):
    """ crop3d(vol, fig=None)
    Manually crop a volume. In the given figure (or a new figure if None),
    three axes are created that display the transversal, sagittal and
    coronal MIPs (maximum intensity projection) of the volume. The user
    can then use the mouse to select a 3D range to crop the data to.
    """
    vv.use()
    
    # Create figure?
    if fig is None:
        fig = vv.figure()
        figCleanup = True
    else:
        fig.Clear()
        figCleanup = False
    
    # Create three axes and a wibject to attach text labels to
    a1 = vv.subplot(221)
    a2 = vv.subplot(222)
    a3 = vv.subplot(223)
    a4 = vv.Wibject(fig)
    a4.position = 0.5, 0.5, 0.5, 0.5
    
    # Set settings
    for a in [a1, a2, a3]:
        a.showAxis = False
    
    # Create cropper3D instance
    cropper3d = Cropper3D(vol, a1, a3, a2, a4)
    
    # Enter a mainloop
    while not cropper3d._finished:
        vv.processEvents()
        time.sleep(0.01)
    
    # Clean up figure (close if we opened it)
    fig.Clear()
    fig.DrawNow()
    if figCleanup:
        fig.Destroy()
    
    # Obtain ranges
    rx = cropper3d._range_transversal._rangex
    ry = cropper3d._range_transversal._rangey
    rz = cropper3d._range_coronal._rangey
    
    # Perform crop
    # make sure we have int not float
    rzmin, rzmax = int(rz.min), int(rz.max)
    rymin, rymax = int(ry.min), int(ry.max)
    rxmin, rxmax = int(rx.min), int(rx.max)
    vol2 = vol[rzmin:rzmax, rymin:rymax, rxmin:rxmax]
    # vol2 = vol[rz.min:rz.max, ry.min:ry.max, rx.min:rx.max]
    
    # Done
    return vol2


def cropvol(vol, fig=None):
    """ run crop3d and check if volume was cropped
    """
    vol2 = crop3d(vol, fig) # use local while error with cropper.crop3d
    # check crop shape
    if vol2.shape == vol.shape:
        print('User did not crop')
    return vol2





