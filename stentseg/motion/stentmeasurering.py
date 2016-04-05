import visvis as vv
import stentModel
import numpy as np
import OpenGL.GLU as glu


def fit_cirlce(pp, warnIfIllDefined=True):
    """ fit_cirlce(pp, warnIfIllDefined=True)
    Calculate the circle (x - c.x)**2 + (y - x.y)**2 = c.r**2
    From the set of points pp. Returns a point instance with an added
    attribute "r" specifying the radius.
    
    In case the three points are on a line, the algorithm will fail, and
    return 0 for x,y and r. This waring can be suppressed.
    
    The solution is a Least Squares fit. The method as describes in [1] is
    called Modified Least Squares (MLS) and poses a closed form solution
    which is very robust.

    [1]
    Dale Umbach and Kerry N. Jones
    2000
    A Few Methods for Fitting Circles to Data
    IEEE Transactions on Instrumentation and Measurement
    """
    
    # Init error point
    ce = vv.Point(0,0)
    ce.r = 0.0
    
    def cov(a, b):
        n = len(a)
        Ex = a.sum() / n
        Ey = b.sum() / n
        return ( (a-Ex)*(b-Ey) ).sum() / (n-1)
    
    # Get x and y elements
    X = pp[:,0]
    Y = pp[:,1]
    
    # In the paper there is a factor n*(n-1) in all equations below. However,
    # this factor is removed by devision in the equations in the following cell
    A = cov(X,X)
    B = cov(X,Y)
    C = cov(Y,Y)
    D = 0.5 * ( cov(X,Y**2) + cov(X,X**2) )
    E = 0.5 * ( cov(Y,X**2) + cov(Y,Y**2) )
    
    # Calculate denumerator
    denum = A*C - B*B
    if denum==0:
        if warnIfIllDefined:
            print "Warning: can not fit a circle to the given points."
        return ce
    
    # Calculate point
    c = vv.Point( (D*C-B*E)/denum, (A*E-B*D)/denum )
    
    # Calculate radius
    c.r = c.distance(pp).sum() / len(pp)
    
    # Done
    return c


def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
 
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    
    Each tuple in points may contain additional elements which happilly move
    along, but only the first 2 elements (x,y) are considered.
    """
    
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(points, key=lambda x:x[:2])
 
    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points
 
    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
 
    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
 
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
 
    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]



class MeasuringRing(vv.OrientableMesh):
    
    def __init__(self, mvertices, mverticesDeltas, mangleChanges):
        """ Allows measureing on the motions of a stent by placing a ring
        around it. The z-position of the ring can be changed using a slider.
        
        The ring is oriented purely in the xy plane; it is not orthogonal to
        the stent's centerline!
        
        """
        
        # For interacting
        self._interact_over = False
        self._interact_down = False
        self._screenVec = None
        self._refPos = (0,0)
        self._screenVec = None
        self._refZ = 0
        
        # Get axes
        axes = vv.gca()
        
        # Get vertices etc.
        vertices, indices, normals, texcords = self._CalculateDonut()
        
        # Initialize
        vv.OrientableMesh.__init__(self, axes, 
                vertices, indices, normals, values=texcords, verticesPerFace=4)   
        self.faceColor = 'c'
        
        # Make hittable
        self.hitTest = True
        
        # Bind events
        self.eventEnter.Bind(self._OnMouseEnter)
        self.eventLeave.Bind(self._OnMouseLeave)
        self.eventMouseDown.Bind(self._OnMouseDown)
        self.eventMouseUp.Bind(self._OnMouseUp)
        self.eventMotion.Bind(self._OnMouseMotion)
        
        # Variables for "child" objects
        self._slider = None
        self._line = None
        
        # Store data per vertex
        self._mvertices = mvertices
        self._mverticesDeltas = mverticesDeltas
        self._mangleChanges = mangleChanges
        
        # Get limits and halfway position
        limits = self._getVertexLimits(mvertices)[2]
        halfway = limits.min+limits.range*0.5
        
        # Create slider
        if True:
            self._slider = vv.Slider(self.GetFigure(), limits, halfway)
            self._slider.eventSliderChanged.Bind(self.onSliderChanged)
            self._slider.eventSliding.Bind(self.onSliderSliding)
        
        # Init
        self.performMeasurements(halfway)
    
    
    def _getVertexLimits(self, vertices):
        ranges = []
        for i in range(3):
            X = vertices[:,i]
            I, = np.where( np.logical_not(np.isnan(X) | np.isinf(X)) )
            R = vv.Range(X[I].min(), X[I].max())
            ranges.append(R)
        return ranges
    
    
    def _CalculateDonut(self, N=32, M=32, thickness=0.2):
        
        # Quick access
        pi2 = np.pi*2
        cos = np.cos
        sin = np.sin
        sl = M+1
        
        # Calculate vertices, normals and texcords
        vertices = vv.Pointset(3)
        normals = vv.Pointset(3)
        texcords = vv.Pointset(2)
        # Cone
        for n in range(N+1):
            v = float(n)/N
            a = pi2 * v        
            # Obtain outer and center position of "tube"
            po = vv.Point(sin(a), cos(a), 0)
            pc = po * (1.0-0.5*thickness)
            # Create two vectors that span the the circle orthogonal to the tube
            p1 = (pc-po)
            p2 = vv.Point(0, 0, 0.5*thickness)
            # Sample around tube        
            for m in range(M+1):
                u = float(m) / (M)
                b = pi2 * (u) 
                dp = cos(b) * p1 + sin(b) * p2
                vertices.append(pc+dp)
                normals.append(dp.normalize())
                texcords.append(v,u)
        
        # Calculate indices
        indices = []
        for j in range(N):
            for i in range(M):
                indices.extend([(j+1)*sl+i, (j+1)*sl+i+1, j*sl+i+1, j*sl+i])
        
        # Make indices a numpy array
        indices = np.array(indices, dtype=np.uint32)
        
        return vertices, indices, normals, texcords
    
    
    def Destroy(self):
        """ Clean up the Measurement tool.
        """
        if self._slider is not None:
            self._slider.Destroy()
        if self._line is not None:
            self._line.Destroy()
        vv.OrientableMesh.Destroy(self)
    
    
    ## For interaction with donut itself
    
    def OnDraw(self):
        vv.OrientableMesh.OnDraw(self)
        
        if self._screenVec is None:
            pos1 = [int(s/2) for s in reversed(self.translation.data.flatten())]
            pos2 = [s for s in pos1]
            pos2[0] += 1
            #
            screen1 = glu.gluProject(pos1[2], pos1[1], pos1[0])
            screen2 = glu.gluProject(pos2[2], pos2[1], pos2[0])
            #
            self._screenVec = screen2[0]-screen1[0], screen1[1]-screen2[1]
    
    
    def _OnMouseEnter(self, event):
        self._interact_over = True
        self.faceColor = (0,0.8,1)
        self.Draw()
    
    
    def _OnMouseLeave(self, event):
        self._interact_over = False
        self.faceColor = 'c'
        self.Draw()
    
    
    def _OnMouseDown(self, event):
        
        if event.button == 1:
            
            # Signal that its down
            self._interact_down = True
            
            # Make the screen vector be calculated on the next draw
            self._screenVec = None
            
            # Store position and index for reference
            self._refPos = event.x, event.y
            self._refZ = self.translation.z
            
            # Redraw
            self.Draw()
            
            # Handle the event
            return True
    
    
    def _OnMouseUp(self, event):
        self._interact_down = False
        self.Draw()
        self.performMeasurements(self.translation.z, False)
        if self._slider is not None:
            self._slider._range.max = float(self.translation.z)
            self._slider._limitRangeAndSetText()
            self._slider.Draw()
    
    
    def _OnMouseMotion(self, event):
        
        # Handle or pass?
        if not (self._interact_down and self._screenVec):
            return
        
        # Get vector relative to reference position
        refPos = vv.Point(self._refPos)
        pos = vv.Point(event.x, event.y)
        vec = pos - refPos
        
        # Length of reference vector, and its normalized version
        screenVec = vv.Point(self._screenVec)
        L = screenVec.norm()
        V = screenVec.normalize()
        
        # Number of indexes to change
        n = vec.dot(V) / L
        
        # Apply!
        # scale of 100 is approximately half the height of a stent
        #delta = (self._refZ + n*50.0) - self.translation.z
        #self.translation += vv.Point(0, 0, delta)
        self.performMeasurements(self._refZ + n*50, True)
        if self._slider is not None:
            self._slider._range.max = float(self._refZ + n*50)
            self._slider._limitRangeAndSetText()
            self._slider.Draw()
    
    
    ## For interaction with slider
    
    
    def onSliderSliding(self, event):
        self.performMeasurements(event.owner.value, True)
    
    
    def onSliderChanged(self, event):
        self.performMeasurements(event.owner.value, False)
    
    
    ## The measurements
    
    def performMeasurements(self, z, quick=False):
        
        # Get points that are near the given z
        I1 = self.getIndicesNearZ(z, quick=quick)
        # Get contour
        I2 = self.getStentContourAtZ(z, I1)
        
        # Store
        self._I1, self._I2 = I1, I2
        
        # Make points and calculate centre and radius
        pp = vv.Pointset(self._mvertices[I2,:])
        pp2 = vv.Pointset(pp.data[:,:2])
        c = fit_cirlce(pp2)
        
        # Show contour
        if self._line is None:
            self._line = vv.plot(pp, lw=4, mw=8, ms='.', axesAdjust=False,
                    axes=self.GetAxes())
        else:
            self._line.SetPoints(pp)
        
        # Translate the donut
        self.translation = c.x, c.y, z
        self.scaling = c.r*2.0, c.r*2.0, c.r*4.0
        
        # Calculate motion
        if not quick:
            motions = self.calculateMotion(I2)
            minArea, maxArea, minCirc, maxCirc = self.calculatePulsation(pp, motions)
            distalMotion = self.calculateDistalMotion(pp, motions)
            motionMag = self.calculateMotionMagnitude(pp, motions)
            aMax, a75, a95 = self.calculateAngleChanges(I2)
            
            print('=== Ring measurements ===')
            print('Area change: %1.1f%%' % (100*maxArea/minArea))
            #print('Circumference change: %1.1f%%' % (100*maxCirc/minCirc))
            print('Distal motion: %1.2f mm' % distalMotion)
            print('Motion magnitude: %1.2f mm' % motionMag)
            print('Angular change: %1.2f / %1.2f / %1.2f degrees' % (aMax, a75, a95))
    
    
    def getIndicesNearZ(self, z, d=1.25, quick=False):
        """ Get the the indices of the vertices that are close to 
        the specified z position. 
        """
        # Get decimation
        decimation = 1
        if quick:
            decimation = 5
        # Get z values
        zz = self._mvertices[::decimation,2]
        # Select the ones that are close to our reference
        I, = np.where((zz < z+d) & (zz> z-d))
        I = I*decimation
        # Done (perform decimation twice)
        return I[::decimation]
        
    
    def getStentContourAtZ(self, z, I):
        """ Get the contour of the stent at a specified z position, by
        applying the convex hull on the points provided by I, an array 
        of indices to the vertices in the mesh.
        Returns a new list of indices
        """
        # Make 2d tuple-points (for the convex hull alg)
        # Put the index in the 3th element
        v = self._mvertices
        points = [(v[i,0], v[i,1], i) for i in I]
        # Get convex hull and make into pointset
        points = convex_hull(points)
        I_ = [p[2] for p in points]
        return np.array(I_)
    
    
    def calculateMotion(self, I):
        """ Given a mesh and the indices of vertices in the mesh, calculates
        the motion of all vertices.
        Returns a list of pointsets that contain the motion vectors.
        """
        deltas = []
        for delta in self._mverticesDeltas:
            dd = vv.Pointset(delta[I,:])
            deltas.append(dd)
        return deltas
    
    
    def calculatePulsation(self, contour, deltas):
        """ Given the contour points and their motion vectors, calculate
        the minimum and maximum of the area and circumference.
        """
        
        # Init
        areas = []
        circs = []
        
        # For each time instance in the motion ...
        for delta in deltas:
            
            # Get closed contour points for this motion phase
            pp = contour + delta
            pp.append(pp[0])
            
            # Get a point in the centre
            cp = vv.Point(0,0,0)
            for p in pp[:-1]:
                cp += p
            cp *= 1.0 / (len(pp)-1)
            
            # Area
            if True:
                area = 0.0
                for i in range(len(contour)):
                    # Get triangle points and lengths of the sides
                    p1, p2 = pp[i], pp[i+1]
                    a, b, c = cp.distance(p1), cp.distance(p2), p1.distance(p2)
                    # Approximate area of this triangle
                    area += 0.5 * 0.5*(a+b) * c
                areas.append(float(area))
            
            # Circumference
            if True:
                circ = 0.0
                for i in range(len(contour)):
                    # Get triangle points and length between them
                    p1, p2 = pp[i], pp[i+1]
                    circ += p1.distance(p2)
                circs.append(float(circ))
        
        
        # Post process: look for smallest and largest
        return min(areas), max(areas), min(circs), max(circs)
    
    
    def calculateDistalMotion(self, contour, deltas):
        """ Calculate the mean distal motion. 
        """
        
        # Init a list that contains the mean z-position for each time unit
        # Note that this is a relative z-position, but that doesnt matter
        meanZPositions = []
        
        for delta in deltas:
            meanZPosition = 0.0
            for p in delta:
                meanZPosition += p.z
            meanZPosition *= 1.0 / len(delta)
            meanZPositions.append(float(meanZPosition))
        
        return max(meanZPositions) - min(meanZPositions)
    
    
    def calculateMotionMagnitude(self, contour, deltas):
        """ Calculate the mean motion magnitude (i.e. amplitude in principal
        direction).
        """
        
        # Init a list that contains the mean z-position for each time unit
        # Note that this is a relative z-position, but that doesnt matter
        meanPositions = vv.Pointset(3)
        
        # delta is a pointset, there is such a pointsets for each time unit
        for delta in deltas: 
            meanPosition = vv.Point(0,0,0)
            for p in delta:
                meanPosition += p
            meanPosition *= 1.0 / len(delta)
            meanPositions.append(meanPosition)
        
        return self._calculateMagnitude(meanPositions)
    
    
    def _calculateMagnitude(self, pp):
        """ Given a cloud of points, calculate the largest distance
        between any two points.
        """
        dmax = 0.0
        for i in range(len(pp)):
            for j in range(len(pp)):
                d = pp[i].distance(pp[j])
                dmax = max(dmax, d)
        return dmax
    
    
    def calculateAngleChanges(self, I):
        """ Calculate the maximum angle change found, and also
        the 75 and 95 percentile
        """
        
        # First, get list of arrays of angles
        anglesList = []
        for angleChange in self._mangleChanges:
            anglesList.append(angleChange[I])
        
        anglesMax = []
        angles75 = []
        angles95 = []
        
        # Calculate the 75 percentile
        for angles in anglesList:
            angles = sorted(angles)
            i75 = int(len(angles) * 0.75)
            i95 = int(len(angles) * 0.95)
            #
            anglesMax.append(float(max(angles)))
            angles75.append(float(angles[i75]))
            angles95.append(float(angles[i95]))
        
        return max(anglesMax), max(angles75), max(angles95)


if __name__ == '__main__':
    
    # Select the ssdf basedir
    basedir = select_dir(os.getenv('LSPEAS_BASEDIR', ''),
                        r'C:\Users\Maaike\Documents\UT MA3\LSPEAS_ssdf',)

    # Select dataset to register
    ptcode = 'LSPEAS_002'
    #ctcode, nr = 'discharge', 1
    ctcode, nr = '1month', 2
    cropname = 'ring'
    
    # Load deformations
    s = loadvol(basedir, ptcode, ctcode, cropname, 'deforms')
    deforms = [s['deform%i'%(i*10)] for i in range(10)]
    deforms = [[field[::2,::2,::2] for field in fields] for fields in deforms]
    
    # Load the stent model and mesh
    s = loadmodel(basedir, ptcode, ctcode, cropname)
    model = s.model
    modelmesh = create_mesh(model, 0.9)  # Param is thickness
    
    # Prepare axes
    a = vv.gca()
    a.daspectAuto = False
    a.daspect = 1, -1, -1
    a.bgcolors = (0.2, 0.4, 0.6), 'k'
    
    # Create deformable mesh
    dm = DeformableMesh(a, modelmesh)
    dm.SetDeforms(*[list(reversed(deform)) for deform in deforms)
    dm.clim = 0, 5
    dm.colormap = vv.CM_JET
    vv.colorbar()
    
    # Instantiate measure ring
    mr = MeasuringRing(modelmesh._vertices, , )
    
    ## Old
    # Load data
    patnr = 1
    dataDir = 'c:/almar/data/stentMotionData'
    s = vv.ssdf.load(dataDir + '/motionMesh_pat%02i.bsdf' % patnr)
    
    # Prepare axes
    a = vv.gca()
    a.daspectAuto = False
    a.daspect = 1, -1, -1
    a.bgcolors = (0.2, 0.4, 0.6), 'k'
        
    # Create mesg object
    m = vv.Mesh(a, s.vertices, s.faces, s.values)
    m.colormap = (0.1,1,0.1), (0.1,0.8,0.3), (0.8,0.3,0.1)
    a.SetLimits()
    
    # Instantiate measure ring
    mr = MeasuringRing(m._vertices, s.verticesDeltas, s.valuesList)
