""" Segment legs as seeds

Store graph in _save_segmentation
"""

import os

import numpy as np
import visvis as vv
from visvis import ssdf
from visvis import Pointset

from stentseg.utils import PointSet, _utils_GUI, visualization
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel
from stentseg.stentdirect import stentgraph, getDefaultParams, initStentDirect
from stentseg.utils.picker import pick3d, get_picked_seed, label2worldcoordinates
from stentseg.utils.visualization import DrawModelAxes
from stentseg.utils.visualization import show_ctvolume

# Select the ssdf basedir
basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                     r'D:\LSPEAS\LSPEAS_ssdf',
                     r'F:\LSPEAS_ssdf_backup',r'G:\LSPEAS_ssdf_backup')


dirsave = select_dir(r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Leg angulation',
                     r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Leg angulation')

# Select dataset to register
ptcode = 'LSPEAS_021'
ctcode = '6months'
cropname = 'stent'
what = 'avgreg' # avgreg
normalize = True

# Load volumes
s = loadvol(basedir, ptcode, ctcode, cropname, what)
vol = s.vol

# h = vv.hist(vol, bins = 1000)
# h.color = 'y'
# vv.surf(vol[:,:,150])
# f = vv.figure()
# t0 = vv.volshow(vol, clim=(0,2500))
# label = pick3d(vv.gca(), vol)
# vv.gca().daspect = 1,1,-1

# Vis volume
clim = (0,2500)
showVol = 'MIP'
meshColor = None # or give FaceColor

fig = vv.figure(3); vv.clf()
fig.position = 9.00, 38.00,  1140.00, 985.00

a0 = vv.subplot(121)
label1 = DrawModelAxes(vol, graph=None, ax=a0, clim=clim, showVol=showVol, climEditor=True)

## Initialize segmentation parameters
stentType = 'nellix'  # use nellix to pick seeds above seed threshold

p = getDefaultParams(stentType)
p.seed_threshold = [600]        # step 1 [lower th] or [lower th, higher th]

## Perform segmentation

# Instantiate stentdirect segmenter object
sd = initStentDirect(stentType, vol, p)
cleanNodes = True

# Normalize vol to certain limit
if normalize:
    sd.Step0(3071)
    vol = sd._vol

# Perform the first step of stentDirect but as threshold, not local max
sd.Step1()

# # other option, threshold (maybe also skeletonize) and create mesh
# threshold = 300
# binaryVolTh = (vol > threshold).astype(int)

## Vis seeds

# Show model Step 1
a1 = vv.subplot(122)
label = DrawModelAxes(vol, sd._nodes1, a1, clim=clim, showVol=showVol, climEditor=True, removeStent=False) # lc, mc


def on_key(event):
    """KEY commands for user interaction
        'DELETE   = remove seed in nodes1 closest to [picked point]'
        'PageDown = remove graph posterior (y-axis) to [picked point] (use for spine seeds)'
        'n = add [picked point] (SHIFT+R-click) as seed'
    """
    global label
    global sd
    if event.key == vv.KEY_DELETE:
        if len(selected_nodes) == 0:
            # remove node closest to picked point
            node = _utils_GUI.snap_picked_point_to_graph(sd._nodes1, vol, label, nodesOnly=True)
            sd._nodes1.remove_node(node)
            view = a1.GetView()
            a1.Clear()
            label = DrawModelAxes(vol, sd._nodes1, a1, clim=clim, showVol=showVol, removeStent=False)
            a1.SetView(view)
    if event.text == 'n':
        # add picked seed to nodes_1
        coord2 = get_picked_seed(vol, label)
        sd._nodes1.add_node(tuple(coord2))
        view = a1.GetView()
        point = vv.plot(coord2[0], coord2[1], coord2[2], mc= 'b', ms = 'o', mw= 8, alpha=0.5, axes=a1)
        a1.SetView(view)
    if event.key == vv.KEY_PAGEDOWN:
        # remove false seeds posterior to picked point, e.g. for spine
        label = _utils_GUI.remove_nodes_by_selected_point(sd._nodes1, vol, a1, label, clim, showVol=showVol)
    if event.text == 't':
        # redo step1
        view = a1.GetView()
        a1.Clear()
        sd._params = p
        sd.Step1()
        label = DrawModelAxes(vol, sd._nodes1, a1, clim=clim, showVol=showVol, removeStent=False) # lc, mc
        a1.SetView(view)
    if event.key == vv.KEY_ESCAPE:
        model = sd._nodes1
        # Build struct
        s2 = vv.ssdf.new()
        # We do not need croprange, but keep for reference
        s2.sampling = s.sampling
        s2.origin = s.origin
        s2.stenttype = s.stenttype
        s2.croprange = s.croprange
        for key in dir(s):
                if key.startswith('meta'):
                    suffix = key[4:]
                    s2['meta'+suffix] = s['meta'+suffix]
        s2.what = what
        s2.params = p
        s2.stentType = stentType
        # Store model
        s2.model = model.pack()
        # get filename
        filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, 'stentseeds'+what)
        # check if file already exists to not overwrite automatically
        if os.path.exists(os.path.join(dirsave, ptcode, filename)):
            #todo: shell jams after providing answer?
            print('File already exists')
            answer = str(input("\nWould you like to"
                        " overwrite (y), save different (u) or cancel (c)? "))
            if answer == 'u':
                filename = '%s_%s_%s_%s_2.ssdf' % (ptcode, ctcode, cropname, 'stentseeds'+what)
                ssdf.save(os.path.join(dirsave, ptcode, filename), s2)
                print("Ssdf saved to:")
                print(os.path.join(dirsave, ptcode))
            elif answer == 'y':
                ssdf.save(os.path.join(dirsave, ptcode, filename), s2)
                print("Ssdf saved to:")
                print(os.path.join(dirsave, ptcode))  
        else:
            try: # check folder existance
                ssdf.save(os.path.join(dirsave, ptcode, filename), s2)
            except FileNotFoundError: # if dirsave does not exist, create
                os.makedirs(os.path.join(dirsave, ptcode))
                ssdf.save(os.path.join(dirsave, ptcode, filename), s2)
            print("Ssdf saved to:")
            print(os.path.join(dirsave, ptcode))


# Init list for nodes
selected_nodes = list()
# Bind event handlers
fig.eventKeyDown.Bind(on_key)
fig.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [a1]) )
fig.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [a1]) )

# Print user instructions
print('')
print('n = add [picked point] (SHIFT+R-click) as seed')
print('PageDown = remove graph posterior (y-axis) to [picked point] (spine seeds)')
print('t = redo step 1')
print('x/a/d = axis invisible/visible/rotate')
print('DELETE = remove seed in nodes1 closest to [picked point]')
print('ESCAPE = save seeds in ssdf')
print('')

