"""
flycam = alt + 1
-keys to move = A D W S
-keys to turn J L I K
-keys up down F C 
"""

# import imageio
# import visvis as vv
# 
# # record fig
# f = vv.gcf()
# r = vv.record(f)
# 
# ## record axes
# r = vv.record(vv.gca())
# 
# ## to stop recording
# 
# r.Stop()
# 
# ## save
# imageio.mimsave(r'D:\Profiles\koenradesma\Desktop.gif', r.GetFrames())
# imageio.mimsave(r'D:\Profiles\koenradesma\Desktop\manual_pick_hookpoints_002_1M.avi', r.GetFrames())
# imageio.mimsave(r'C:\Users\Maaike\Desktop\test.avi', r.GetFrames())
# 
# imageio.help('.avi')

# todo: WARNING:root:IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from

##

import os
import imageio
import visvis as vv
import datetime
from stentseg.utils.datahandling import select_dir

class recordMovie:
    """ Record a figure
    """
    def __init__(self, fig=None, filename=None, dirsave=None, fileformat='avi'):
        """ fig or axes can be given. gif swf or avi possible
        """
        # import os
        # import imageio
        # import visvis as vv
        # import datetime
        # from stentseg.utils.datahandling import select_dir
    
        # self.os = os
        # self.imageio = imageio
        # self.vv = vv
        # self.datetime = datetime
        self.r = None
        if fig is None:
            self.fig = vv.gcf()
        else:
            self.fig = fig
        self.filename = filename
        self.fileformat = fileformat
        if dirsave is None:
            self.dirsave = select_dir(r'C:\Users\Maaike\Desktop',
                            r'D:\Profiles\koenradesma\Desktop')
        else:
            self.dirsave = dirsave
        # Bind eventhandler record
        self.fig.eventKeyUp.Bind(self.record)
    
    def record(self, event): 
        """ keys to record, stop, continue save  clear r of figure
                      r       t      u      s     q
        """
        if event.text == 'r':
            self.r = vv.record(self.fig)
            print('recording..')
        if event.text == 't':
            self.r.Stop()
            print('stop recording')
        if event.text == 'u':
            self.r.Continue()
            print('continue recording')
        if event.text == 'q':
            self.r.Clear()
            print('clear recording')
        if event.text == 's': # save
            if self.filename is None:
                now = datetime.datetime.now()
                self.filename = now.strftime("%Y-%m-%d_%H.%M")+'_recorded'+'.'+self.fileformat
            imageio.mimsave(os.path.join(self.dirsave,self.filename), self.r.GetFrames())
            print('Recorded movie stored')
    
    # def bindRecord(self):
    #     self.fig.eventKeyUp.Bind(self.record)

