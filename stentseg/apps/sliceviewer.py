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
    
    