""" Make centerline dynamic with deformation fields from registration and store
Input: ssdf and mat of centerline
Output: ssdf and mat of centerline stored as _dynamic; deforms are added as var
"""

import pirt
from stentseg.motion.dynamic import incorporate_motion_nodes, incorporate_motion_edges 
from visvis import ssdf 
import os
import scipy.io
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel, loadmodel_location


# Select the ssdf basedir
basedir = select_dir(r'D:\LSPEAS\LSPEAS_ssdf',
                    r'F:\LSPEAS_ssdf_backup', r'G:\LSPEAS_ssdf_backup')
basedirstl = select_dir(r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS\Analysis\Leg angulation',
                        r'C:\Users\Maaike\SURFdrive\UTdrive\LSPEAS\Analysis\Leg angulation')

# Select dataset
ptcode = 'LSPEAS_008'
ctcode = 'discharge'
cropname = 'stent'
what = 'centerline'

# Load deforms
s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
deformkeys = []
for key in dir(s):
    if key.startswith('deform'):
        deformkeys.append(key)
deforms = [s[key] for key in deformkeys]
deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]
paramsreg = s.params

# Load centerline mat
matName = '%s_%s_%s_%s.mat' % (ptcode, ctcode, cropname, what)
centerlineMat = scipy.io.loadmat(os.path.join(basedirstl, ptcode, matName))

# Load model
s = loadmodel(basedirstl, ptcode, ctcode, cropname, what)
for key in dir(s):
    if key.startswith('model'):
        model = s[key]

        # Combine ...
        incorporate_motion_nodes(model, deforms, s.origin) # adds deforms PointSets
        incorporate_motion_edges(model, deforms, s.origin) # adds deforms PointSets
        
        # get deforms of centerline path as array
        assert model.number_of_edges() == 1 # cll should be 1 path
        centerlineEdge = model.edges()[0] # the two nodes that define the edge
        centerlineDeforms = model.edge[centerlineEdge[0]][centerlineEdge[1]]['pathdeforms']
        
        # store in dict of centerlineMat
        if key == 'model':
            centerlineMat['centerline_deforms'] = centerlineDeforms # list with PointSet arrays
        elif key == 'model_2':
            centerlineMat['centerline2_2_deforms'] = centerlineDeforms # list with PointSet arrays
        else:
            print('No model found with known name')
        
        # pack model for save
        s[key] = model.pack()
        
# Save back ssdf
filename = '%s_%s_%s_%s.ssdf' % (ptcode, ctcode, cropname, what+'_dynamic')
s.paramsreg = paramsreg
ssdf.save(os.path.join(basedirstl, ptcode, filename), s)
print('saved dynamic to disk in {} as {}.'.format(basedirstl, filename) )

# Save back mat
newMatName = '%s_%s_%s_%s.mat' % (ptcode, ctcode, cropname, what+'_dynamic')
scipy.io.savemat(os.path.join(basedirstl, ptcode, newMatName),centerlineMat)
print('')
print('Dynamic centerline was stored as {}'.format(newMatName))
print('')