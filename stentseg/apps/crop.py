""" Use cropper tool from visvis to crop data.
"""

# ARGS
fname_in = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\DICOM\Discharge\LSPEAS_003\1.2.392.200036.9116.2.6.1.48.1214833767.1398645614.547701'
fname_out = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\ssdf\LSPEAS_003\LSPEAS_003_discharge_10.ssdf'


import imageio
import visvis as vv
from visvis.utils import cropper
from visvis import ssdf

# Read data
vols = imageio.mvolread(fname_in, 'DICOM')
vol1 = vols[2]  # 1 to 4

# Crop
vol2 = cropper.crop3d(vol1)

# Export
s = ssdf.new()
s.vol = vol2
s.sampling = 1.0, vol1.meta.sampling[1], vol1.meta.sampling[2]
ssdf.save(fname_out, s)



## Load ssdf as vol
BASEDIR = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\ssdf\LSPEAS_003\\'

# Load volume data, use Aarray class for anisotropic volumes
s = ssdf.load(BASEDIR+'LSPEAS_003_discharge_99.ssdf') 
vol = vv.Aarray(s.vol, s.sampling)
vol.meta = s.meta


## To crop from ssdf volume using the cropper

# ARGS
fname_out = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\ssdf\LSPEAS_003\LSPEAS_003_discharge_0.ssdf'

# Crop
vol1 = vol  # load volume data as vol when vol not yet in workspace
vol2 = cropper.crop3d(vol1)

# Export
s = ssdf.new()
s.vol = vol2
s.sampling = vol1.sampling
ssdf.save(fname_out, s)


## To crop from ssdf volume by known cropsize

# ARGS
fname_out = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\ssdf\LSPEAS_003\LSPEAS_003_discharge_99.ssdf'

# Crop
vol1 = vol
vol2 = vol1[34:334,43:299,145:401] # zyx

# Export and overwrite
s = ssdf.new()
s.vol = vol2
s.sampling = vol1.meta.sampling
ssdf.save(fname_out, s)



## Visualize
fig = vv.figure(1); vv.clf()
fig.position = 0, 22, 1366, 706
#fig.position = -1413.00, -2.00,  1366.00, 706.00
t = vv.volshow(vol2)
t.clim = 0, 2500

