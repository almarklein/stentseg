"""
flycam = alt + 1
-keys to move = A D W S
-keys to turn J L I K
-keys up down F C 
"""

import imageio
import visvis as vv

r = vv.record(vv.gca())

## to stop recording

r.Stop()

## save
imageio.mimsave(r'D:\Profiles\koenradesma\Desktop.gif', r.GetFrames())
imageio.mimsave(r'D:\Profiles\koenradesma\Desktop\test.avi', r.GetFrames())

imageio.help('.avi')

# todo: how to record including colorbar and title? 
# todo: WARNING:root:IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16