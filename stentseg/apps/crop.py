""" Use cropper tool from visvis to crop data.
"""

# ARGS
fname_in = '/home/almar/data/dicom/stent_LPEAS/test_anaconda1'
fname_out = '/home/almar/data/test.ssdf'


import imageio
from visvis.utils import cropper
from visvis import ssdf

# Read data
vols = imageio.mvolread(fname_in, 'DICOM')
vol1 = vols[1]  # 1 to 4

# Crop
vol2 = cropper.crop3d(vol1)

# Export
s = ssdf.new()
s.vol = vol2
s.sampling = 1.0, vol1.meta.sampling[1], vol1.meta.sampling[2]
ssdf.save(fname_out, s)
