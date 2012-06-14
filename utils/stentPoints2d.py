""" Module stentPoints2d

Implements functions to detect points on a stent in 2D slices sampled
 orthogonal to the stent's centerline.

"""

import os, sys, time
import numpy as np
import visvis as vv
from points import Point, Pointset, Aarray
#import subpixel deprecated module, use interp instead

from pirt import gaussfun


## Point detection

# HU (if a pixel next to the candidate is this low, its probably a streak)
th_streHU = 0.0;  

def detect_points(slice, th_gc=2000, th_minHU=300, sigma=1.6):
    """ Detect points
    Detects points on a stent in the given slice. Slice 
    should be a numpy array.        
    - The Gaussian Curvature should be above a threshold 
      (needle detection) (th_gc)
    - An absolute (weak) threshold is used based on the 
      Houndsfield units (th_minHU)    
    - Streak artifacts are suppressed
    - sigma is the used scale at which the GC is calculated.    
    """
    
    # Make sure that the slice is a float
    if slice.dtype not in [np.float32, np.float64]:
        slice = slice.astype(np.float32)
    
    # Create slice to supress streak artifacts
    # Where streak artifacts are present, the image is inverted, anywhere else
    # its zero. By also calculating derivatives on this image and than adding
    # it, the threshold will never be reached where the streak artifacts are.
    sliceStreak = th_streHU - slice
    sliceStreak[sliceStreak<0] = 0
    
    # Create new point set to store points
    pp = Pointset(2)
    
    # Calculate Gaussian curvature    
    if True:
        Lxx = gaussfun.gfilter(slice, sigma, [0,2])
        Ltmp = gaussfun.gfilter(sliceStreak, sigma, [0,2])
        Lxx = Lxx+2*Ltmp
        Lxx[Lxx>0]=0;
        
        Lyy = gaussfun.gfilter(slice, sigma, [2,0])
        Ltmp = gaussfun.gfilter(sliceStreak, sigma, [2,0])
        Lyy = Lyy+2*Ltmp
        Lyy[Lyy>0]=0;
        
        Lgc = Lxx * Lyy
    
    # Make a smoothed version
    slice_smoothed = gaussfun.gfilter(slice, 0.5, 0)
    
    # Make a selection of candidate pixels
    Iy,Ix = np.where( (slice > th_minHU) & (Lgc > th_gc) )
    
    # Mask to detect clashes
    clashMask = np.zeros(slice.shape, dtype=np.bool)
    
    # Detect local maxima
    for x,y in zip(Ix,Iy):
        if x==0 or y==0 or x==slice.shape[1]-1 or y==slice.shape[0]-1:
            continue
        
        # Select patch        
        patch1 = slice[y-1:y+2,x-1:x+2]
        patch2 = slice_smoothed[y-1:y+2,x-1:x+2]
     
        if slice[y,x] == patch1.max():# and slice_smoothed[y,x] == patch2.max():
            # Found local max (allowing shared max)
            
            # Not if next to another found point
            if clashMask[y,x]:
                continue
            
            # Not a streak artifact
            if patch2.min() <= th_streHU:
                continue
            
            # Subpixel
            #dx,dy = subpixel.fitLQ2_5(patch1)
            
            # Store
            pp.append( x, y )
            clashMask[y-1:y+2,x-1:x+2] = 1
    
    # Express points in world coordinates and return
    if isinstance(slice, Aarray):
        ori = [i for i in reversed(slice.origin)]
        sam = [i for i in reversed(slice.sampling)]
        pp *= Point(sam)
        pp += Point(ori)
    return pp
    

def detect_points2(slice, y_x, spacing, width=10):
    """ Alternative version of detect_points, which uses a reference point.
    Used for measurements on our phantom.
    """
    
    # create slice as double (makes a copy)
    slice = slice.astype(np.float64)
            
    # create new point set to store points
    pp = Pointset(2)
    
    th_minHU = 300
        
    refy, refx = y_x
    refy -= spacing # because we start with incrementing it
    
    while 1:
        # next
        refy += spacing
        if refy > slice.shape[0]:
            break
        
        # get patch
        y1, y2 = refy - spacing//4, refy + spacing//4
        x1, x2 = refx - width//2, refx + width//2
        if y1<0: y1=0
        patch = slice[y1:y2+1, x1:x2+1]
        
        # detect 
        Iy, Ix = np.where( (patch == patch.max()) & (patch > th_minHU) )
        try:
            Ix = Ix[0]
            Iy = Iy[0]
        except IndexError:            
            continue # if no points found...  
        y, x = y1+Iy, x1+Ix
        if y<=0 or y>=slice.shape[0]-1:
            continue
        
        # get subpixel and store
        patch2 = slice[y-1:y+2,x-1:x+2]
        dx,dy = subpixel.fitLQ2_5(patch2)
        pp.append( x+dx, y+dy ) 
        
        
    return pp


## Clustering


eps = 0.00000001
def chopstick_criteria(c, p1, p2, pp, param=0.25):
    
    ## Calculate intrinsic properties
    # 1 means point one (if relative measure, relative to c). 2 idem
    # plural means the point set, relative to c.
    # plural and 1 means for all points, relative to point 1.
    
    # Calculate vectors
    vec1 = c - p1
    vec2 = c - p2
    vecs = c - pp
    
    # Calculate angles
    ang12 = abs( float(vec1.angle(vec2)) )
    angs1 = np.abs( vec1.angle(vecs) )
    angs2 = np.abs( vec2.angle(vecs) )
    
    # Calculate distance
    dist1 = float( vec1.norm() )
    dist2 = float( vec2.norm() )
    dists = vecs.norm()
    dist12 = p1.distance(p2) + eps
    #dists1 = p1.distance(pp)
    #dists2 = p2.distance(pp)
    
    # Get the point between two points
    p12 = (p1+p2) * 0.5
    
    
    ## Criterion 1: focus
    # Find subset of points that is on the proper side of the centre.
    # For this we get the angle between the line p1-c, to p-c. We need to
    # check the angle p1-p3, to see if the result should me smaller or larger
    # than zero.  The result is a kind of FAN from the centre, spanned by
    # the two points.
    M1 = (angs1 < ang12) * (angs2 < ang12)
    
    
    ## Criterion 2: ellipse
    # There are two things to be determined. The point p3, (p4 in the paper)
    # and the ellipsepoints on the lines p1-p3 and p2-p3. 
    # Note the change in behaviour when d<0:
    # - the ellipsepoints are both in p3
    # - the point p3 is moved differently.
    
    # Get distance vector
    d = p12.distance(c)
    f = 1 - np.exp(-d/dist12)
    if f < eps:
        f = eps
    f = float(f)
    
    # Get normal to line p1-p2
    n = (p1-p2).normal()
    
    # Flip normal if its in the direction of the center
    if (p12 + n*d).distance(c) < d:
        n = -1 * n
    
    # Go from p12, a bit in the direction of the normal
    p3 = p12
    ratio = param # 0.25
    if d>0: d3 = ratio*d/f
    else: d3 = ratio*dist12-d
    p3 = p12 + n*d3
    
    # Ellipse points    
    e1 = f*p1 + (1-f)*p3
    e2 = f*p2 + (1-f)*p3
    
    # Ellipse. Make sure the length of the string is a bit too short so
    # that p1 and p2 themselves will not be included.
    d1 = e1.distance(pp) + e2.distance(pp)
    d = e1.distance(p1) + p1.distance(e2)
    M2 = d1 < d*0.99
    
    return M1, M2

def add_angles(ang1, ang2):
    """ add_angles(ang1, ang2)
    Add two angles, returning a result that's always between
    -pi and pi. Each angle can be a scalar or a numpy array.
    """
    # Get pi and subtract angles
    pi = np.pi
    dang = ang1 + ang2   
    if isinstance(dang, np.ndarray): 
        # Limit, wrap around
        while True:
            I, = np.where(dang < -pi)
            if len(I):
                dang[I] += 2*pi
            else:
                break
        while True:
            I, = np.where(dang > pi)
            if len(I):
                dang[I] -= 2*pi
            else:
                break
    else:
        while dang < -pi:
            dang += 2*pi
        while dang > pi:
            dang -= 2*pi
            
    # Done
    return dang


def subtract_angles(ang1, ang2):
    """ subtract_angles(ang1, ang2)
    Subtract two angles, returning a result that's always between
    -pi and pi. Each angle can be a scalar or a numpy array.
    """
    # Get pi and subtract angles
    pi = np.pi
    dang = ang1 -ang2   
    if isinstance(dang, np.ndarray): 
        # Limit, wrap around
        while True:
            I, = np.where(dang < -pi)
            if len(I):
                dang[I] += 2*pi
            else:
                break
        while True:
            I, = np.where(dang > pi)
            if len(I):
                dang[I] -= 2*pi
            else:
                break
    else:
        while dang < -pi:
            dang += 2*pi
        while dang > pi:
            dang -= 2*pi
            
    # Done
    return dang


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
    ce = Point(0,0)
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
    c = Point( (D*C-B*E)/denum, (A*E-B*D)/denum )
    
    # Calculate radius
    c.r = c.distance(pp).sum() / len(pp)
    
    # Done
    return c


def sample_circle(c, N=32):
    """ sample_circle(c, N=32)
    Sample a circle represented by point c (having attribute "r") using
    N datapoints. Returns a pointset.
    """
    
    # Get radius
    r = 1.0
    if hasattr(c, 'r'):
        r = c.r
    
    # Sample N points, but add one to close the loop
    d = 2*np.pi / N
    a = np.linspace(0,2*np.pi, N+1)
    
    # Prepare array
    pp = np.empty((len(a), 2), dtype=np.float32)
    
    # Apply polar coordinates
    pp[:,0] = np.cos(a) * r + c.x
    pp[:,1] = np.sin(a) * r + c.y
    
    # Return as a pointset
    return Pointset(pp)


def converge_to_centre(pp, c, nDirections=5, maxRadius=50, pauseTime=0):
    """ converge_to_centre(pp, c)
    Given a set of points and an initial center point c, 
    will find a better estimate of the center.
    Returns (c, L), with L indices in pp that were uses to fit 
    the final circle.
    """
    
    # Shorter names
    N = nDirections
    pi = np.pi
    
    # Init point to return on error
    ce = Point(0,0)
    ce.r = 0
    
    # Are there enough points?
    if len(pp) < 3:
        return ce, []
    
    # Init a pointset of centers we've has so far
    cc = Pointset(2) 
    
    # Init list with vis objects (to be able to delete them)
    showObjects = []
    if pauseTime:
        fig = vv.gcf()
    
    
    while c not in cc:
        
        # Add previous center
        cc.append(c)
        
        # Calculate distances and angles
        dists = c.distance(pp)
        angs = c.angle2(pp)
        
        # Get index of closest points
        i, = np.where(dists==dists.min())
        i = iClosest = int(i[0])
        
        # Offset the angles with the angle relative to the closest point.
        refAng = angs[i]
        angs = subtract_angles(angs, refAng)
        
        # Init L, the indices to the closest point in each direction
        L = []
        
        # Get closest point on N directions
        for angle in [float(angnr)/N*2*pi for angnr in range(N)]:
            
            # Get indices of points that are in this direction
            dangs = subtract_angles(angs, angle)
            I, = np.where(np.abs(dangs) < pi/N )
            
            # Select closest
            if len(I):
                distSelection = dists[I]
                minDist = distSelection.min()
                J, = np.where(distSelection==minDist)
                if len(J) and minDist < maxRadius:
                    L.append( int(I[J[0]]) )
        
        # Check if ok
        if len(L) < 3:
            return ce, []
        
        # Remove spurious points (points much furter away that the 3 closest)
        distSelection = dists[L]
        tmp = [d for d in distSelection]
        d3 = sorted(tmp)[2] # Get distance of 3th closest point
        I, = np.where(distSelection < d3*2)
        L = [L[i] for i in I]
        
        # Select points
        ppSelect = Pointset(2)
        for i in L:
            ppSelect.append(pp[i])
        
        # Refit circle
        cnew = fit_cirlce(ppSelect,False)
        if cnew.r==0:
            return ce, []
        
        # Show
        if pauseTime>0:
            # Delete
            for ob in showObjects:
                ob.Destroy()
            # Plot center and new center
            ob1 = vv.plot(c, ls='', ms='x', mc='r', mw=10, axesAdjust=0)
            ob2 = vv.plot(cnew, ls='', ms='x', mc='r', mw=10, mew=0, axesAdjust=0)
            # Plot selection points            
            ob3 = vv.plot(ppSelect, ls='', ms='.', mc='y', mw=10, axesAdjust=0)
            # Plot lines dividing the directions
            tmpSet1 = Pointset(2)
            tmpSet2 = Pointset(2)            
            for angle in [float(angnr)/N*2*pi for angnr in range(N)]:
                angle = -subtract_angles(angle, refAng)
                dx, dy = np.cos(angle), np.sin(angle)
                tmpSet1.append(c.x, c.y)
                tmpSet1.append(c.x+dx*d3*2, c.y+dy*d3*2)
            for angle in [float(angnr+0.5)/N*2*pi for angnr in range(N)]:
                angle = -subtract_angles(angle, refAng)
                dx, dy = np.cos(angle), np.sin(angle)
                tmpSet2.append(c.x, c.y)
                tmpSet2.append(c.x+dx*d3*2, c.y+dy*d3*2)
            ob4 = vv.plot(tmpSet1, lc='y', ls='--', axesAdjust=0)
            ob5 = vv.plot(tmpSet2, lc='y', lw=3, axesAdjust=0)
            # Store objects and wait
            showObjects = [ob1,ob2,ob3,ob4,ob5]
            fig.DrawNow()
            time.sleep(pauseTime)
        
        # Use new
        c = cnew
    
    # Done    
    for ob in showObjects:
        ob.Destroy()
    return c, L


    
def cluster_points(pp, c, pauseTime=0, showConvergeToo=True):
    """ cluster_points(pp, c, pauseTime=0) 
    
    Given a set of points, and the centreline position, this function 
    returns a set of points, which is a subset of pp. The returned pointset
    is empty on error.
    
    This algorithm uses the chopstick clustering implementation.
    
    The first step is the "stick" method. We take the closest point from the
    centre and attach one end of a stick to it. The stick has the length of
    the radius of the fitted circle. We rotate the stick counterclockwise
    untill it hits a point, or untill we rotate too far without finding a
    point, thus failing. When it fails, we try again with a slighly larger
    stick. This is to close gaps of up to almost 100 degrees. When we reach
    a point were we've already been, we stop. Afterwards, the set of points
    is sorted by angle.   
    
    In the second step we try to add points: we pick
    two subsequent points and check what other points are closer than "stick"
    to both these points, where "stick" is the distance between the points. We
    will break this stick in two and take the best point in the cluster, thus
    the name: chopstick. BUT ONLY if it is good enough! We draw two lines
    from the points under consideration to the centre. The angle that these
    two lines make is a reference for the allowed angle that the two lines
    may make that run from the two points to the new point. To be precise:
    ang > (180 - refang) - offset
    offset is a parameter to control the strictness. 
    
    The third part of this step consists of removing points. We will only
    remove points that are closer to the centre than both neighbours. We
    check each point, comparing it with its two neighbours on each side,
    applying the same criterion as  above. This will remove outliers that lie
    withing the stent, such as points found due to the contrast fluid...
    
    The latter two parts are repeated untill the set of points does not change.
    Each time the centre is recalculated by fitting a circle.
    
    """
    
    # Get better center
    if showConvergeToo:
        c, I = converge_to_centre(pp, c, pauseTime=pauseTime)
    else:
        c, I = converge_to_centre(pp, c)
    if not I:
        return Pointset(2)
    
    # Init list with vis objects (to be able to delete them)
    showObjects = []
    if pauseTime:
        fig = vv.gcf()
    
    # Short names
    pi = np.pi
    
    # Minimum and maximum angle that the stick is allowed to make with the line
    # to the circle-centre. Pretty intuitive...
    # it is a relative measure, we will multiply it with a ratio: 
    # radius/distance_current_point_to_radius. This allows ellipses to be
    # segmented, while remaining strict for circles.
    difAng1_p = 0.0*pi
    difAng2_p = 0.7*pi
    
    
    ## Step 1, stick method to find an initial set of points
    
    # Select start point (3th point returned by converge_to_centre)
    icurr = I[2]
    
    # Init list L, at the beginning only contains icurr
    L = [icurr]
    
    # Largest amount of iterations that makes sense. Probably the loop
    # exits sooner.
    maxIter = len(pp)
    
    # Enter loop
    for iter in range(maxIter):
        
        # We can think of it as fixing the stick at one end at the current
        # point and then rotating it in a direction such that a point is
        # found in the clockwise direction of the current point. We thus
        # need the angle between the next and current point to be not much
        # more than the angle between the current point and the circle
        # cenrre + 90 deg. But it must be larger than the angle between the
        # current point and the circle centre.
        
        # Calculate distances
        dists = pp.distance(pp[icurr])
        
        # Do the next bit using increasing stick length, untill success
        for stick in [c.r*1.0, c.r*1.5, c.r*2.0]:
            
            # Find subset of points that be "reached by the stick"
            Is, = np.where(dists<stick)
            
            # Calculate angle with circle centre
            refAng = c.angle2(pp[icurr])
            
            # Culcuate angles with points that are in reach of the stick
            angs = pp[Is].angle2(pp[icurr])
            
            # Select the points that are in the proper direction
            # There are TWO PARAMETERS HERE (one important) the second TH can
            # not be 0.5, because we allow an ellipse.
            # Use distance measure to make sure not to select the point itself.
            difAngs = subtract_angles(angs, refAng)
            difAng2 = difAng2_p # pp[icurr].distance(c) / c.r
            
            # Set current too really weid value
            icurr2, = np.where(Is==icurr)
            if len(icurr2):
                difAngs[icurr2] = 99999.0
            
            # Select. If a selection, we're good!
            II, = np.where( (difAngs > difAng1_p) + (difAngs < difAng2))
            if len(II):
                break
            
        else:
            # No success
            _objectClearer(showObjects)
            return Pointset(2)
        
        # Select the point with the smallest angle
        tmp = difAngs[II]
        inext, = np.where(tmp == tmp.min())
        
        # inext is index in subset. Make it apply to global set
        inext = Is[ II[ inext[0] ] ]
        inext = int(inext)
        
        # Show
        if pauseTime>0:
            # Delete
            _objectClearer(showObjects)
            # Show center
            ob1 = vv.plot(c, ls='', ms='x', mc='r', mw=10, axesAdjust=0)
            # Show all points
            ob2 = vv.plot(pp, ls='', ms='.', mc='g', mw=6, axesAdjust=0)
            # Show selected points L
            ob3 = vv.plot(pp[L], ls='', ms='.', mc='y', mw=10, axesAdjust=0)
            # Show next
            ob4 = vv.plot(pp[inext], ls='', ms='.', mc='r', mw=12, axesAdjust=0)
            # Show stick
            vec = ( pp[inext]-pp[icurr] ).normalize()
            tmp = Pointset(2)
            tmp.append(pp[icurr]); tmp.append(pp[icurr]+vec*stick)            
            ob5 = vv.plot(tmp, lw=2, lc='b')
            # Store objects and wait
            showObjects = [ob1,ob2,ob3,ob4,ob5]
            fig.DrawNow()
            time.sleep(pauseTime)
        
        
        # Check whether we completed a full round already 
        if inext in L:
            break
        else:
            L.append(inext)
        
        # Prepare for next round
        icurr = inext
    
    
    # Sort the list by the angles
    tmp = zip( pp[L].angle2(c), L )
    tmp.sort(key=lambda x:x[0])
    L = [i[1] for i in tmp]
    
    # Clear visualization
    _objectClearer(showObjects)
    
    
    ## Step 2 and 3, chopstick algorithm to find more points and discard outliers
    
    # Init
    L = [int(i) for i in L] # Make Python ints
    Lp = []
    round = 0
    
    # Iterate ...
    while Lp != L and round < 20:
        round += 1
        #print 'round', round
        
        # Clear list (but store previous)
        Lp = [i for i in L]
        L = []
        
        # We need at least three points
        if len(Lp)<3:
            _objectClearer(showObjects)
            print 'oops: len(LP)<3' 
            return []
        
        # Recalculate circle
        c = fit_cirlce(pp[Lp], False)
        if c.r == 0.0:
            print 'oops: c.r==0' 
            _objectClearer(showObjects)
            return []
        
        # Step2: ADD POINTS
        for iter in range(len(Lp)):
            
            # Current point
            icurr = Lp[iter]
            if iter < len(Lp)-1:
                inext = Lp[iter+1]
            else:
                inext = Lp[0]
            
            # Prepare, get p1 and p2
            p1 = pp[icurr]
            p2 = pp[inext]
            
            # Apply masks to points in pp
            M1, M2 = chopstick_criteria(c, p1, p2, pp)
            
            # Combine measures. I is now the subset (of p) of OK points
            I, = np.where(M1*M2)
            if not len(I):
                L.append(icurr)
                continue
            elif len(I)==1:
                ibetw = int(I)
            else:
                # Multiple candidates: find best match
                pptemp = pp[I]
                dists = p1.distance(pptemp) + p2.distance(pptemp)
                II, = np.where( dists==dists.min() )
                ibetw = int( I[II[0]] )
            
            # Add point            
            L.append(icurr)
            if not ibetw in L:
                L.append(ibetw)
            
            # Check
            assert ibetw not in [icurr, inext]
            
            # Draw
            if pauseTime>0:
                # Delete
                _objectClearer(showObjects)
                # Show center
                ob1 = vv.plot(c, ls='', ms='x', mc='r', mw=10, axesAdjust=0)
                # Show all points
                ob2 = vv.plot(pp, ls='', ms='.', mc='g', mw=6, axesAdjust=0)
                # Show selected points L
                ob3 = vv.plot(pp[L], ls='', ms='.', mc='y', mw=10, axesAdjust=0)
                # Show between and vectors
                ob4 = vv.plot(pp[ibetw], ls='', ms='.', mc='r', mw=12, axesAdjust=0)
                ob5 = vv.plot(pp[[icurr, ibetw, inext]], ls='-', lc='g', axesAdjust=0)
                ob6 = vv.plot(pp[[icurr, inext]], ls=':', lc='g', axesAdjust=0) 
                # Store objects and wait
                showObjects = [ob1,ob2,ob3,ob4,ob5,ob6]
                fig.DrawNow()
                time.sleep(pauseTime)
        
        # Lpp stores the set of points we have untill now, we will refill the
        # set L, maybe with less points
        Lpp = [int(i) for i in L]
        L = []
        
        # Step3: REMOVE POINTS 
        for iter in range(len(Lpp)):
            
            # Current point and neighbours
            ibetw = Lpp[iter]
            if iter<len(Lpp)-1:
                inext = Lpp[iter+1]
            else:
                inext = Lpp[0]
            if iter>0:
                icurr = Lpp[iter-1]
            else:
                icurr = Lpp[-1]
            
            # Test
#             print icurr, ibetw, inext
            assert ibetw not in [icurr, inext]
            
            # Prepare, get p1 and p2 and p3
            p1 = pp[icurr]
            p2 = pp[inext]
            p3 = pp[ibetw]
            
            # Apply masks to points in pp
            M1, M2 = chopstick_criteria(c, p1, p2, p3)
            M = M1*M2
            
            # Do we keep the point?           
            if M.sum():
                L.append(ibetw)
            
            # Draw
            if pauseTime>0 and not M.sum():
                # Delete
                _objectClearer(showObjects)
                # Show center
                ob1 = vv.plot(c, ls='', ms='x', mc='r', mw=10, axesAdjust=0)
                # Show all points
                ob2 = vv.plot(pp, ls='', ms='.', mc='g', mw=6, axesAdjust=0)
                # Show selected points L
                ob3 = vv.plot(pp[L], ls='', ms='.', mc='y', mw=10, axesAdjust=0)
                # Show between and vectors
                ob4 = vv.plot(pp[ibetw], ls='', ms='.', mc='r', mw=12, axesAdjust=0)
                ob5 = vv.plot(pp[[icurr, ibetw, inext]], ls='-', lc='r', axesAdjust=0)
                ob6 = vv.plot(pp[[icurr, inext]], ls='-', lc='g', axesAdjust=0)
                # Store objects and wait
                showObjects = [ob1,ob2,ob3,ob4,ob5,ob6]
                fig.DrawNow()
                time.sleep(pauseTime)
    
    
    # Done
    if round == 20:
        print 'Warning: chopstick seemed not to converge.'
    _objectClearer(showObjects)
    #print 'cluster end', len(L)
    return pp[L]


def _objectClearer(showObjects):
    for ob in showObjects:
        ob.Destroy()
    showObjects[:] = []

## Testers

class ChopstickCriteriaTester:
    """ Small app to test the chopstick criteria visually.
    """
    def __init__(self, c=None, p1=None, p2=None):
        
        # Init visualization
        fig = vv.figure(101); vv.clf()
        a = vv.gca()
        
        # Init patch
        self._patchSize = patchSize = 64
        self._im = np.zeros((patchSize,patchSize), dtype=np.float32)
        self._t = vv.imshow(self._im, clim=(0,10))
        
        # Init points
        if c is None: c = Point(14,12)
        if p1 is None: p1 = Point(12,16)
        if p2 is None: p2 = Point(16,16)
        
        #
        self._c = vv.plot(c, ls='', ms='+', mw=10, mc='k')
        self._p1 = vv.plot(p1, ls='', ms='.', mw=10, mc='r')
        self._p2 = vv.plot(p2, ls='', ms='.', mw=10, mc='b')
        
        
        
        # Init object being moved
        self._movedPoint = None
        
        # Enable callbacks
        for line in [self._c, self._p1, self._p2]:
            line.hitTest = True
            line.eventMouseDown.Bind(self.OnDown)
            line.eventMotion.Bind(self.OnMotion)
            line.eventMouseUp.Bind(self.OnUp)
        a.eventMotion.Bind(self.OnMotion)
        a.eventMouseUp.Bind(self.OnUp)
        
        # Start
        self.Apply()
    
    
    def OnDown(self, event):
        self._movedPoint = event.owner
        #self.OnMotion()
    
    def OnMotion(self, event):
        
        if self._movedPoint:
            # Update point
            self._movedPoint._points[0,0] = event.x2d
            self._movedPoint._points[0,1] = event.y2d
            # Update
            self.Apply()
    
    def OnUp(self, event):
        self._movedPoint = None
    
    def Apply(self):
        
        # Create big list of points
        pp = Pointset(2)
        for y in range(self._patchSize):
            for x in range(self._patchSize):            
                pp.append(x,y)
        
        # Get point locations
        c = Point( self._c._points[0].x, self._c._points[0].y )
        p1 = Point( self._p1._points[0].x, self._p1._points[0].y )
        p2 = Point( self._p2._points[0].x, self._p2._points[0].y )
        
        # Get masks
        M1, M2 = chopstick_criteria(c, p1, p2, pp)
        M1.shape = self._patchSize, self._patchSize
        M2.shape = self._patchSize, self._patchSize
        
        # Update image
        self._im[:] = 9
        self._im[M1] -= 3 
        self._im[M2] -= 3 
        self._t.Refresh()
        
        # Draw
        self._c.Draw()


class FitCircleTester:
    """ Small app to test the circlefit.
    """
    def __init__(self):
        
        # Init visualization
        fig = vv.figure(102); vv.clf()
        a = vv.gca()
        
        # Init points
        pp = Pointset(2)
        pp.append(14,12)
        pp.append(12,16)
        pp.append(16,16)
        self._pp = vv.plot(pp, ls='', ms='.', mw=10, mc='g')
        
        # Init line representing the circle
        self._cc = vv.plot(pp, lc='r', lw=2, ms='.', mw=5, mew=0, mc='r')
        
        # Set limits
        a.SetLimits((0,32), (0,32))
        a.daspectAuto = False
        
        # Init object being moved
        self._movedPoint = None
        
        # Enable callbacks
        self._pp.hitTest = True
        self._pp.eventMouseDown.Bind(self.OnDown)
        self._pp.eventMouseUp.Bind(self.OnUp)
        a.eventMotion.Bind(self.OnMotion)
        a.eventDoubleClick.Bind(self.OnDD)
        
        # Start
        self.Apply()
    
    
    def OnDD(self, event):
        
        # Add point
        x, y = event.x2d, event.y2d
        self._pp._points.append(x,y, 0.2)
    
    
    def OnDown(self, event):
        
        # Get distances
        mousePos = Point(event.x2d, event.y2d, 0)
        dists = self._pp._points.distance(mousePos)
        if not len(dists):
            self._movedPoint = None
            return 
        
        # Get closest point
        i, = np.where(dists == dists.min())
        self._movedPoint = i[0]
    
    
    def OnMotion(self, event):
        
        if self._movedPoint is not None:
            # Update point
            i = self._movedPoint
            self._pp._points[i,0] = event.x2d
            self._pp._points[i,1] = event.y2d
            # Update
            self.Apply()
    
    def OnUp(self, event):
        self._movedPoint = None
    
    def Apply(self):
        
        # Get 2D pointset
        pp = Pointset(2)
        for p in self._pp._points:
            pp.append(p.x, p.y)
        
        # Fit circle and sample and show
        c = fit_cirlce(pp)
        cc = sample_circle(c, 32)
        self._cc.SetPoints(cc)
        
        # Draw
        self._cc.Draw()


## Tests
if __name__ == "__main__":
    import visvis as vv     
    tester1 = ChopstickCriteriaTester()
#     tester2 = FitCircleTester()

    if False:
        pp = Pointset(2)
        pp.append(1,1)
        pp.append(10,3)
        pp.append(8,8)
        pp.append(2,6)
        pp.append(1,2)
        
        fig = vv.figure(103)
        fig.Clear()
        a = vv.gca()
        a.daspectAuto = False
        fig.position = -799.00, 368.00,  544.00, 382.00
        vv.plot(pp, ls='', ms='.', mc='g')
        c2 = converge_to_centre(pp, Point(6,5), 5, pauseTime=0.3)
