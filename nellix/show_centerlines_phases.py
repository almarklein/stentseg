""" Show centerlines at relative positions during the cardiac cycle

"""

import os
from stentseg.utils.datahandling import select_dir, loadmodel, loadvol
from nellix._do_analysis_centerline import _Do_Analysis_Centerline
from stentseg.utils.centerline import points_from_nodes_in_graph
import visvis as vv
from stentseg.utils import PointSet, _utils_GUI
from stentseg.utils.visualization import DrawModelAxes, show_ctvolume
from stentseg.utils.utils_graphs_pointsets import get_graph_in_phase
from stentseg.utils.picker import pick3d

basedir = select_dir(r'E:\Nellix_chevas\CT_SSDF\SSDF_automated',
r'D:\Nellix_chevas_BACKUP\CT_SSDF\SSDF_automated')

ptcode = 'chevas_09_thin'
ctcode = '12months'
phases =  range(10)  # all 10 phases

showmodelavgreg = True
showvol = True

# get vol for reference
s = loadvol(basedir, ptcode, ctcode, 'prox', what='avgreg')
# set sampling for cases where this was not stored correctly
s.vol.sampling = [s.sampling[1], s.sampling[1], s.sampling[2]]
vol = s.vol

s_modelcll = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='centerline_total_modelavgreg_deforms')
# get graph
modelcll = s_modelcll.model

s_modelcll_vessel = loadmodel(basedir, ptcode, ctcode, 'prox', modelname='centerline_modelvesselavgreg_deforms')
# get graph
models = []
for key in s_modelcll_vessel:
    if key.startswith('model'):
        cllv = s_modelcll_vessel[key]
        models.append(cllv)

# combine graphs
for model in models:
    modelcll.add_edges_from(model.edges(data=True))
#todo: correct concerning coordinates when different crops were used for stents and vessels?

color = 'cgmrcgywmb'  # r op 30%, b op 90%

f = vv.figure(); vv.clf()
f.position = 0.00, 22.00,  1920.00, 1018.00
ax = vv.gca()
ax.daspect = 1,1,-1
ax.axis.axisColor = 0,0,0
ax.bgcolor = 1,1,1
vv.xlabel('x (mm)');vv.ylabel('y (mm)');vv.zlabel('z (mm)') 

# draw
showVol = 'ISO'
t = show_ctvolume(vol, showVol=showVol, removeStent=False, 
                    climEditor=True, isoTh=300)
if showmodelavgreg:
    # show model and CT mid cycle
    mw = 5
    modelcll.Draw(mc='b', mw = mw, lc='b', alpha = 0.5)
    label = pick3d(ax, vol)
if not showvol:
    t.visible = False
    
# get centerline models in different phases
for phasenr in phases:
    model_phase = get_graph_in_phase(modelcll, phasenr = phasenr)
    model_phase.Draw(mc=color[phasenr], mw = 10, lc=color[phasenr])

#bind view control
f.eventKeyDown.Bind(lambda event: _utils_GUI.RotateView(event, [ax], axishandling=False) )
f.eventKeyDown.Bind(lambda event: _utils_GUI.ViewPresets(event, [ax]) )
