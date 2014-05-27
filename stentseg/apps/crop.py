""" Use cropper tool from visvis to crop data.
"""

# ARGS
fname_in = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\DICOM\LSPEAS_003\LSPEAS_003_Discharge\1.2.392.200036.9116.2.6.1.48.1214833767.1398645614.547701'
fname_out = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\ssdf\LSPEAS_003\LSPEAS_003_discharge.ssdf'


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




## Crop from ssdf volume using the cropper

# ARGS
fname_out = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\ssdf\LSPEAS_003\LSPEAS_003_discharge_0_cropring.ssdf'

# Crop
vol1 = vol  # load volume data as vol when vol not yet in workspace
vol2 = cropper.crop3d(vol1)

# Export
s = ssdf.new()
s.vol = vol2
s.sampling = vol.sampling
ssdf.save(fname_out, s)


## Crop from ssdf volume by preset cropsize

# ARGS
fname_out = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\ssdf\LSPEAS_003\LSPEAS_003_discharge_90_cropring.ssdf'

# Crop
vol1 = vol
vol2 = vol1[35:104,57:216,85:240] # zyx

# Export and overwrite
s = ssdf.new()
s.vol = vol2
s.sampling = vol1.sampling
ssdf.save(fname_out, s)


## Load ssdf as vol
BASEDIR = r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_data\ssdf\LSPEAS_003\\'

# Load volume data, use Aarray class for anisotropic volumes
s = ssdf.load(BASEDIR+'LSPEAS_003_discharge_90.ssdf') 
vol = vv.Aarray(s.vol, s.sampling)
#vol.meta = s.meta


## Visualize
fig = vv.figure(1); vv.clf()
fig.position = 0, 22, 1366, 706
#fig.position = -1413.00, -2.00,  1366.00, 706.00
t = vv.volshow(vol2)
t.clim = 0, 2500
vv.xlabel('x')
vv.ylabel('y')
vv.zlabel('z')

