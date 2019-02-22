""" Module to analyze vessel pulsatility during the heart cycle in ecg-gated CT
radius change - area change - volume change

"""

import openpyxl
from stentseg.utils.datahandling import select_dir, loadvol, loadmodel, loadmesh
import sys, os, time
import numpy as np
from stentseg.stentdirect.stentgraph import create_mesh
from stentseg.motion.vis import create_mesh_with_abs_displacement
from lspeas.utils.vis import showModelsStatic
from lspeas.utils.deforminfo import DeformInfo
import visvis as vv
from stentseg.utils.centerline import points_from_mesh
from stentseg.utils import PointSet, fitting
import pirt

assert openpyxl.__version__ < "2.4", "Do pip install openpyxl==2.3.5"


#todo: move function to lspeas utils?
def load_excel_centerline(basedirCenterline, vol, ptcode, ctcode, filename=None):
    """ Load centerline data from excel
    Centerline exported from Terarecon; X Y Z coordinates in colums
    """
    if filename is None:
        filename = '{}_{}_centerline.xlsx'.format(ptcode,ctcode)
    excel_location = os.path.join(basedirCenterline,ptcode)
    #read sheet
    try:
        wb = openpyxl.load_workbook(os.path.join(excel_location,filename),read_only=True)
    except FileNotFoundError:
        wb = openpyxl.load_workbook(os.path.join(basedirCenterline,filename),read_only=True)
    sheet = wb.get_sheet_by_name(wb.sheetnames[0]) # data on first sheet
    colStart = 2 # col C
    rowStart = 1 # row 2 excel
    coordx = sheet.columns[colStart][rowStart:] 
    coordy = sheet.columns[colStart+1][rowStart:]
    coordz = sheet.columns[colStart+2][rowStart:]  
    #convert to values
    coordx = [obj.value for obj in coordx]
    coordy = [obj.value for obj in coordy]
    coordz = [obj.value for obj in coordz]
    #from list to array
    centerlineX = np.asarray(coordx, dtype=np.float32)
    centerlineY = np.asarray(coordy, dtype=np.float32)
    centerlineZ = np.asarray(coordz, dtype=np.float32)
    centerlineZ = np.flip(centerlineZ, axis=0) # z of volume is also flipped
    
    # convert centerline coordinates to world coordinates (mm)
    origin = vol1.origin # z,y,x
    sampling = vol1.sampling # z,y,x
    
    centerlineX = centerlineX*sampling[2] +origin[2]
    centerlineY = centerlineY*sampling[1] +origin[1]
    centerlineZ = (centerlineZ-0.5*vol1.shape[0])*sampling[0] + origin[0]
    
    return centerlineX, centerlineY, centerlineZ


# Select the ssdf basedir
basedir = select_dir(
    os.getenv('LSPEAS_BASEDIR', ''),
    r'D:\LSPEAS\LSPEAS_ssdf',
    r'F:\LSPEAS_ssdf_backup')
                     
basedirMesh = select_dir(
    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\MedDataMimics\LSPEAS_Mimics',
    r'C:\Users\Maaike\SURFdrive\UTdrive\MedDataMimics\LSPEAS_Mimics',
    r"C:\stack\data\lspeas\vaatwand")

basedirCenterline = select_dir(
    r"C:\stack\data\lspeas\vaatwand",
    r'D:\Profiles\koenradesma\SURFdrive\UTdrive\LSPEAS_centerlines_terarecon',
    r"C:\stack\data\lspeas\vaatwand")

# Select dataset to register
ptcode = 'LSPEAS_003'
ctcode1 = 'discharge'
cropname = 'ring'
modelname = 'modelavgreg'
cropvol = 'stent'

drawModelLines = False  # True or False
drawRingMesh, ringMeshDisplacement = True, False
meshColor = [(1,1,0,1)]
removeStent = True # for iso visualization
dimensions = 'xyz'
showAxis = False
showVol  = 'ISO'  # MIP or ISO or 2D or None
showvol2D = False
drawVessel = True

clim = (0,2500)
clim2D = -200,500 # MPR
clim2 = (0,2)
isoTh = 180 # 250


## Load data

# Load CT image data for reference, and deform data to measure motion
try:
    # If we run this script without restart, we can re-use volume and deforms
    vol1
    deforms
except NameError:
    vol1 = loadvol(basedir, ptcode, ctcode1, cropvol, 'avgreg').vol
    s_deforms = loadvol(basedir, ptcode, ctcode1, cropvol, 'deforms')
    deforms = [s_deforms[key] for key in dir(s_deforms) if key.startswith('deform')]
    deforms = [pirt.DeformationFieldBackward(*fields) for fields in deforms]

# Load vesselmesh (mimics)
# We make sure that it is a mesh without faces, which makes our sampling easier
try:
    ppvessel
except NameError:
    filename = '{}_{}_neck.stl'.format(ptcode,ctcode1)
    vessel1 = loadmesh(basedirMesh,ptcode[-3:],filename) #inverts Z
    vv.processing.unwindFaces(vessel1)
    ppvessel = PointSet(vessel1._vertices)  # Yes, this has duplicates, but thats ok

# Load ring model
try:
    modelmesh1
except NameError:
    s1 = loadmodel(basedir, ptcode, ctcode1, cropname, modelname)
    if drawRingMesh:
        if not ringMeshDisplacement:
            modelmesh1 = create_mesh(s1.model, 0.7)  # Param is thickness
        else:
            modelmesh1 = create_mesh_with_abs_displacement(s1.model, radius = 0.7, dim=dimensions)

# Load vessel centerline (excel terarecon) (is very fast)
centerline = PointSet(np.column_stack(
    load_excel_centerline(basedirCenterline, vol1, ptcode, ctcode1, filename=None)))


## Setup visualization

# Show ctvolume, vessel mesh, ring model - this uses figure 1 and clears it
axes1, cbars = showModelsStatic(ptcode, ctcode1, [vol1], [s1], [modelmesh1], 
              [vessel1], showVol, clim, isoTh, clim2, clim2D, drawRingMesh, 
              ringMeshDisplacement, drawModelLines, showvol2D, showAxis, 
              drawVessel, vesselType=1,
              climEditor=True, removeStent=removeStent, meshColor=meshColor)
axes1 = axes1[0]

# Show or hide the volume (showing is nice, but also slows things down)
tex3d = axes1.wobjects[1]
tex3d.visible = False

# Show the centerline
vv.plot(centerline, ms='.', ls='', mw=8, mc='b', alpha=0.5)

# Initialize 2D view
axes2 = vv.Axes(vv.gcf())
axes2.position = 0.7, 200, 0.25, 0.5
axes2.daspectAuto = False
axes2.camera = '2d'
axes2.axis.axisColor = 'w'

# Create label to show measurements
label = vv.Label(axes1)
label.bgcolor = "w"
label.position = 0, 0, 1, 25

# Initialize sliders
slider_ref = vv.Slider(axes2, fullRange=(1, len(centerline)-2), value=10)
slider_ves = vv.Slider(axes2, fullRange=(1, len(centerline)-2), value=10)
slider_ref.position = 10, -70, -20, 25
slider_ves.position = 10, -40, -20, 25

# Initialize line objects for showing the plane orthogonal to centerline
slider_ref.line_plane = vv.plot([], [], [], axes=axes1, ls='-', lw=3, lc='w', alpha = 0.9)
slider_ves.line_plane = vv.plot([], [], [], axes=axes1, ls='-', lw=3, lc='y', alpha = 0.9)
# Initialize line objects for showing selected points close to that plane
slider_ref.line_3d = vv.plot([], [], [], axes=axes1, ms='.', ls='', mw=8, mc='w', alpha = 0.9)
slider_ves.line_3d = vv.plot([], [], [], axes=axes1, ms='.', ls='', mw=8, mc='y', alpha = 0.9)

# Initialize line objects for showing selected points and ellipse in 2D
line_2d = vv.plot([], [], axes=axes2,  ms='.', ls='', mw=8, mc='y')
line_ellipse1 = vv.plot([], [], axes=axes2,  ms='', ls='-', lw=2, lc='b')
line_ellipse2 = vv.plot([], [], axes=axes2,  ms='', ls='+', lw=2, lc='b')


## Functions to update visualization and do measurements


def get_plane_points_from_centerline_index(i):
    """ Get a set of points that lie on the plane orthogonal to the centerline
    at the given index. The points are such that they can be drawn as a line for
    visualization purposes. The plane equation can be obtained via a plane-fit.
    """
    
    if True:
        # Cubic fit of the centerline
        
        i = max(1.1, min(i, centerline.shape[0] - 2.11))
        
        # Sample center point and two points right below/above, using
        # "cardinal" interpolating (C1-continuous), or "basic" approximating (C2-continious).
        pp = []
        for j in [i - 0.1, i, i + 0.1]:
            index = int(j)
            t = j - index
            coefs = pirt.get_cubic_spline_coefs(t, "basic")
            samples = centerline[index - 1], centerline[index], centerline[index + 1], centerline[index + 2]
            pp.append(samples[0] * coefs[0] + samples[1] * coefs[1] + samples[2] * coefs[2] + samples[3] * coefs[3])
        
        # Get center point and vector pointing down the centerline
        p = pp[1]
        vec1 = (pp[2] - pp[1]).normalize()
    
    else:
        # Linear fit of the centerline
        
        i = max(0, min(i, centerline.shape[0] - 2))
        
        index = int(i)
        t = i - index
        
        # Sample two points of interest
        pa, pb = centerline[index], centerline[index + 1]
        
        # Get center point and vector pointing down the centerline
        p = t * pb + (1 - t) * pa
        vec1 = (pb - pa).normalize()
    
    # Get two orthogonal vectors that define the plane that is orthogonal
    # to the above vector. We can use an arbitrary vector to get the first,
    # but there is a tiiiiiny chance that it is equal to vec1 so that the
    # normal collapese.
    vec2 = vec1.cross([0, 1, 0])
    if vec2.norm() == 0:
        vec2 = vec1.cross((1, 0, 0))
    vec3 = vec1.cross(vec2)
    
    # Sample some point on the plane and get the plane's equation
    pp = PointSet(3)
    radius = 6
    pp.append(p)
    for t in np.linspace(0, 2 * np.pi, 12):
        pp.append(p + np.sin(t) * radius * vec2 + np.cos(t) * radius * vec3)
    return pp


def get_vessel_points_from_plane_points(pp):
    """ Select points from the vessel points that are very close to the plane
    defined by the given plane points. Returns a 2D and a 3D point set.
    """
    abcd = fitting.fit_plane(pp)
    
    # Get 2d and 3d coordinates of points that lie (almost) on the plane
    pp2 = fitting.project_to_plane(ppvessel, abcd)
    pp3 = fitting.project_from_plane(pp2, abcd)
    above_below = np.sign(ppvessel[:, 2] - pp3[:, 2])  # Note: we're only looking in z-axis
    distances = (ppvessel - pp3).norm()
    
    # Select points to consider. This is just to reduce the search space somewhat.
    selection = np.where(distances < 5)[0]
    
    # We assume that each tree points in ppvessel makes up a triangle (face)
    # We make sure of that when we load the mesh.
    # Select first index of each face (every 3 vertices is 1 face), and remove duplicates
    selection_faces = set(3 * (selection // 3))
    
    # Now iterate over the faces (triangles), and check each edge. If the two
    # points are on different sides of the plane, then we interpolate on the
    # edge to get the exact spot where the edge intersects the plane.
    t0 = time.time()
    sampled_pp3 = PointSet(3)
    visited_edges = set()
    for fi in selection_faces:  # for each face index
        for edge in [(fi + 0, fi + 1), (fi + 0, fi + 2), (fi + 1, fi + 2)]:
            if above_below[edge[0]] * above_below[edge[1]] < 0:
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    d1, d2 = distances[edge[0]], distances[edge[1]]
                    w1, w2 = d2 / (d1 + d2), d1 / (d1 + d2)
                    p = w1 * ppvessel[edge[0]] + w2 * ppvessel[edge[1]]
                    sampled_pp3.append(p)
    
    return fitting.project_to_plane(sampled_pp3, abcd), sampled_pp3


def get_distance_along_centerline():
    i1 = slider_ref.value
    i2 = slider_ves.value
    
    index1 = int(np.ceil(i1))
    index2 = int(np.floor(i2))
    t1 = i1 - index1  # -1 < t1 <= 0
    t2 = i2 - index2  # 0 <= t2 < 1
    
    dist = 0
    dist += -t1 * (centerline[index1] - centerline[index1 - 1]).norm()
    dist += +t2 * (centerline[index2] - centerline[index2 + 1]).norm()
    
    for index in range(index1, index2):
        dist += (centerline[index + 1] - centerline[index]).norm()
    
    return float(dist)


def triangle_area(p1, p2, p3):
    # Use Heron's formula to calculate a triangle's area
    # https://www.mathsisfun.com/geometry/herons-formula.html
    a = p1.distance(p2)
    b = p2.distance(p3)
    c = p3.distance(p1)
    s = (a + b + c) / 2
    return (s * (s - a) * (s - b) * (s - c)) ** 0.5


def deform_points_2d(pp2, plane):
    """ Given a 2D pointset (and the plane that they are on), return
    a list with the deformed versions of that pointset.
    """
    pp3 = fitting.project_from_plane(pp2, plane)
    deformed = []
    for phase in range(len(deforms)):
        deform = deforms[phase]
        dx = deform.get_field_in_points(pp3, 0)
        dy = deform.get_field_in_points(pp3, 1)
        dz = deform.get_field_in_points(pp3, 2)
        deform_vectors = PointSet(np.stack([dx, dy, dz], 1))
        pp3_deformed = pp3 + deform_vectors
        deformed.append(fitting.project_to_plane(pp3_deformed, plane))
    return deformed


def take_measurements():
    """ This gets called when the slider is releases. We take measurements and
    update the corresponding texts and visualizations.
    """
    slider = slider_ves
    pp = get_plane_points_from_centerline_index(slider.value)
    pp2, pp3 = get_vessel_points_from_plane_points(pp)
    plane = pp2.plane
    
    # Collect measurements in a dict. That way we can process it in one step at the end
    measurements = {}
    
    # Early exit?
    if len(pp2) == 0:
        label.text = ""
        line_2d.SetPoints(pp2)
        line_ellipse1.SetPoints(pp2)
        line_ellipse2.SetPoints(pp2)
        return
    
    # Get ellipse, and sample it so we can calculate its area in different phases
    ellipse = fitting.fit_ellipse(pp2)
    p0 = PointSet([ellipse[0], ellipse[1]])
    ellipse_points2 = fitting.sample_ellipse(ellipse, 256)  # results in N + 1 points
    area = 0
    for i in range(len(ellipse_points2)-1):
        area += triangle_area(p0, ellipse_points2[i], ellipse_points2[i + 1])
    measurements["reference area"] = "{:0.2f} cm^2".format(float(area / 100))
    # Do a quick check to be sure that this triangle-approximation is close enough
    assert abs(area - fitting.area(ellipse)) < 2, "area mismatch"  # mm2  typically ~ 0.1 mm2
    
    # Measure ellipse area changes
    areas = DeformInfo()
    for ellipse_points2_deformed in deform_points_2d(ellipse_points2, plane):
        area = 0
        for i in range(len(ellipse_points2_deformed)-1):
            area += triangle_area(p0, ellipse_points2_deformed[i], ellipse_points2_deformed[i + 1])
        areas.append(area)
    measurements["min-max area"] = "{:0.2f} - {:0.2f} cm^2 ({:0.1f}%)".format(areas.min / 100, areas.max / 100, areas.percent)
    
    # Get ellipse axis
    ellipse_points = fitting.sample_ellipse(ellipse, 4)
    major_minor = PointSet(2)
    major_minor.append(p0); major_minor.append(ellipse_points[0])  # major axis
    major_minor.append(p0); major_minor.append(ellipse_points[2])  # other major
    major_minor.append(p0); major_minor.append(ellipse_points[1])  # minor axis
    major_minor.append(p0); major_minor.append(ellipse_points[3])  # other minor
    radii_major, radii_minor = DeformInfo(), DeformInfo()
    for ellipse_points_deformed in deform_points_2d(ellipse_points, plane):
       radii_major.append(float( ellipse_points_deformed[0].distance(ellipse_points_deformed[2]) ))
       radii_minor.append(float( ellipse_points_deformed[1].distance(ellipse_points_deformed[3]) ))
    measurements["min-max radius major axis"] = "{:0.2f} - {:0.2f} cm ({:0.1f}%)".format(radii_major.min / 10, radii_major.max / 10, radii_major.percent)
    measurements["min-max radius minor axis"] = "{:0.2f} - {:0.2f} cm ({:0.1f}%)".format(radii_minor.min / 10, radii_minor.max / 10, radii_minor.percent)
    
    # More measurements
    measurements["distance"] = "{:0.1f} mm".format(get_distance_along_centerline())
    
    # Update line objects
    line_2d.SetPoints(pp2)
    line_ellipse1.SetPoints(fitting.sample_ellipse(ellipse))
    line_ellipse2.SetPoints(major_minor)
    axes2.SetLimits(margin=0.12)
    
    # Show measurements. Now the results are shown in a label object, but we could do anything here ...
    texts = []
    print("Measurements:")
    for key, value in measurements.items():
        texts.append(key + ": " + value)
        print(key.rjust(16) + ": " + value)
    label.text = " " + "  |  ".join(texts)


def on_sliding(e):
    """ When the slider is moved, update the centerline position indicator.
    """
    slider = e.owner
    pp = get_plane_points_from_centerline_index(slider.value)
    slider.line_plane.SetPoints(pp)


def on_sliding_done(e):
    """ When the slider is released, update the whole thing.
    """
    slider = e.owner
    pp = get_plane_points_from_centerline_index(slider.value)
    pp2, pp3 = get_vessel_points_from_plane_points(pp)
    slider.line_plane.SetPoints(pp)
    slider.line_3d.SetPoints(pp3)
    take_measurements()


# Connect!
slider_ref.eventSliding.Bind(on_sliding)
slider_ves.eventSliding.Bind(on_sliding)
slider_ref.eventSliderChanged.Bind(on_sliding_done)
slider_ves.eventSliderChanged.Bind(on_sliding_done)


#todo: measure radius change in minor/major axis, asymmetry.
#todo: measure distance from center to ellipse-points to get "direction-sensitive expansion".
#todo: make it easy to export results/data.
#todo: measure volume change at/between selected level(s)
#todo: measure how centerline segment changes (longitudinal strain)
#todo: measure (change of) curvature of centerline
#todo: measure (change of) curvature of stent rings
#todo: visualize mesh with colors, each vertice representing axis change from COM/centerline
