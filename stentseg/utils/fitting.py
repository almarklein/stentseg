""" Code for fitting circles, ellipses, planes, etc.
"""


import numpy as np
from numpy.linalg import eig, inv

from stentseg.utils.new_pointset import PointSet


def fit_circle(pp, warnIfIllDefined=True):
    """ Fit a circle on the given 2D points
    
    Returns a tuple (x, y, r).
    
    In case the three points are on a line, the algorithm will fail, and
    return (0, 0, 0). A warning is printed, but this can be suppressed.
    
    The solution is a Least Squares fit. The method as describes in [1] is
    called Modified Least Squares (MLS) and poses a closed form solution
    which is very robust.

    [1]
    Dale Umbach and Kerry N. Jones
    2000
    A Few Methods for Fitting Circles to Data
    IEEE Transactions on Instrumentation and Measurement
    """
    
    # Check
    if pp.ndim != 2:
        raise ValueError('Circle fit needs an Nx2 array.')
    if pp.shape[1] != 2:
        raise ValueError('Circle fit needs 2D points.')
    if pp.shape[0] < 2:
        raise ValueError('Circle fit needs at least two points.')
    
    def cov(a, b):
        n = len(a)
        Ex = a.sum() / n
        Ey = b.sum() / n
        return ( (a-Ex)*(b-Ey) ).sum() / (n-1)
    
    # Get x and y elements
    X = pp[:,0]
    Y = pp[:,1]
    xoffset = X.mean()
    yoffset = Y.mean()
    X = X - xoffset
    Y = Y - yoffset
    
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
            print("Warning: can not fit a circle to the given points.")
        return 0, 0, 0
    
    # Calculate point
    x = (D*C-B*E)/denum + xoffset
    y = (A*E-B*D)/denum + yoffset
    c = PointSet([x, y])
    
    # Calculate radius
    r = c.distance(pp).sum() / len(pp)
    
    # Done
    return x, y, r


def fit_ellipse(pp):
    """ Fit an ellipse to the given 2D points
    
    Returns a tuple (x, y, r1, r2, phi).
    
    Algorithm derived from:
    From http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html.
    Based on approach suggested by Fitzgibbon et al., Direct least squares 
    fitting of ellipsees, 1996.
    """
    # Check
    if pp.ndim != 2:
        raise ValueError('Ellipse fit needs an Nx2 array.')
    if pp.shape[1] != 2:
        raise ValueError('Ellipse fit needs 2D points.')
    if pp.shape[0] < 3:
        raise ValueError('Ellipse fit needs at least three points.')
    
    # Get x and y and subtract offset to avoid inaccuracied during
    # eigenvalue decomposition.
    x = pp[:,0]
    y = pp[:,1]
    xoffset = x.mean()
    yoffset = y.mean()
    x = x - xoffset
    y = y - yoffset
    
    # Do the math
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    
    # Calculate position
    num = b*b-a*c
    x0 = (c*d-b*f)/num + xoffset
    y0 = (a*f-b*d)/num + yoffset
    
    # Calculate radii
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1 = np.sqrt(up/down1)
    res2 = np.sqrt(up/down2)
    
    # Calculate direction vector
    phi = 0.5*np.arctan(2*b/(a-c))
    
    # Ensure that first radius is the largers
    if res1 < res2:
        res2, res1 = res1, res2
        phi += 0.5 * np.pi
    
    # Ensure that phi is between 0 and pi
    while phi < 0:
        phi += np.pi
    while phi > np.pi:
        phi -= np.pi
    
    return x0, y0, res1, res2, phi


def area(circle_or_ellipse):
    """ Calculate the area of the given circle or ellipse
    """
    
    if len(circle_or_ellipse) == 3:
        r1 = r2 = circle_or_ellipse[2]
    elif len(circle_or_ellipse) == 5:
        r1, r2 = circle_or_ellipse[2], circle_or_ellipse[3]
    else:
        raise ValueError('Input of area() is not a circle nor an ellipse.')
    
    return np.pi * r1 * r2


def sample_circle(c, N=32):
    """ Sample points on a circle c
    
    Returns a 2D PointSet with N points
    """
    
    assert len(c) == 3
    
    # Get x, y and radius
    x, y, r = c
    
    # Sample N points, but add one to close the loop
    a = np.linspace(0,2*np.pi, N+1)
    
    # Prepare array
    pp = np.empty((len(a), 2), dtype=np.float32)
    
    # Apply polar coordinates
    pp[:,0] = np.cos(a) * r + x
    pp[:,1] = np.sin(a) * r + y
    
    # Return as a pointset
    return PointSet(pp)


def sample_ellipse(e, N=32):
    """ Sample points on a ellipse e
    
    Returns a 2D PointSet with N+1 points
    """
    
    assert len(e) == 5
    
    # Get x, y, radii and phi
    x, y, r1, r2, phi = e
    
    # Sample N points, but add one to close the loop
    a = np.linspace(0, 2*np.pi, N+1)
    
    # Prepare array
    pp = np.empty((len(a), 2), dtype=np.float32)
    
    # Apply polar coordinates
    pp[:,0] = x + r1 * np.cos(a) * np.cos(phi) - r2 * np.sin(a) * np.sin(phi)
    pp[:,1] = y + r1 * np.cos(a) * np.sin(phi) + r2 * np.sin(a) * np.cos(phi)
    
    # Return as a pointset
    return PointSet(pp)


def fit_plane(pp):
    """ Fit a plane through a set of 3D points
    
    Returns a tuple (a, b, c, d) which represents the plane mathematically
    as ``a*x + b*y + c*z = d``.
    
    This method uses singular value decomposition. It is the SVD method
    plublished here: http://stackoverflow.com/questions/15959411
    """
    
    # Check
    if pp.ndim != 2:
        raise ValueError('Plane fit needs an Nx3 array.')
    if pp.shape[1] != 3:
        raise ValueError('Plane fit needs 3D points.')
    if pp.shape[0] < 3:
        raise ValueError('Plane fit needs at least three points.')
    
    rows, cols = pp.shape
    # Set up constraint equations of the form  AB = 0,
    # where B is a column vector of the plane coefficients
    # in the form b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
    p = np.ones((rows, 1))
    AB = np.hstack([pp, p])
    [u, d, v] = np.linalg.svd(AB, 0)
    B = v[3, :]  # Solution is last column of v.
    # Normalize
    nn = np.linalg.norm(B[0:3])
    B = B / nn
    # Return a b c d
    return B[0], B[1], B[2], B[3]


def project_to_plane(pp, plane):
    """ Project given 3D points to a plane to make them 2D
    
    Returns a 2D PointSet. We assume that the plane represents a grid
    that is aligned with the world grid, but rotated over the x and y
    axis.
    """
    
    # Check
    if pp.ndim != 2:
        raise ValueError('project_to_plane needs an Nx3 array.')
    if pp.shape[1] != 3:
        raise ValueError('project_to_plane needs 3D points.')
    
    # Prepare
    a, b, c, d = plane
    norm = a**2 + b**2 + c**2
    common = (a*pp[:,0] + b*pp[:,1] + c*pp[:,2] + d) / norm
    
    # Calculate angles
    phix = np.arctan(a/c)
    phiy = np.arctan(b/c)
    
    # Project points to the plane. Points are still in world
    # coordinates, but are moved so that thet lie on the plane. The
    # movement is such that they are now on the closest point to the
    # plane.
    pp3 = pp.copy()
    pp3[:,0] = pp[:,0] - a * common
    pp3[:,1] = pp[:,1] - b * common
    pp3[:,2] = pp[:,2] - c * common
    
    # Rotate the points
    pp2 = PointSet(pp3[:,:2])
    pp2[:,0] = pp3[:,0] / np.cos(phix)
    pp2[:,1] = pp3[:,1] / np.cos(phiy)
    
    # Add some information so we can reconstruct the points
    pp2.plane = a, b, c, d
    
    return pp2


def project_from_plane(pp, plane):
    """ Project 2D points on a plane to the original 3D coordinate frame
    
    Returns a 3D PointSet.
    """
    
     # Check
    if pp.ndim != 2:
        raise ValueError('project_from_plane needs an Nx2 array.')
    if pp.shape[1] != 2:
        raise ValueError('project_from_plane needs 2D points.')
    
    # Prepare
    pp2 = pp
    a, b, c, d = plane
    phix = np.arctan(a/c)
    phiy = np.arctan(b/c)
    
    # Init 3D points
    pp3 = PointSet(np.zeros((pp2.shape[0], 3), 'float32'))
    
    # Rotate the points
    pp3[:,0] = pp2[:,0] * np.cos(phix)
    pp3[:,1] = pp2[:,1] * np.cos(phiy)
    
    # Find the z value for all points
    pp3[:,2] = -(pp3[:,0]*a + pp3[:,1]*b + d) / c
    
    return pp3


def convex_hull(points):
    """Computes the convex hull of a set of 2D points
 
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



if __name__ == '__main__':
    
    from stentseg.utils.new_pointset import PointSet
    
    # Create some data, 2D and 3D
    pp2 = PointSet(2)
    pp3 = PointSet(3)
    for r in np.linspace(0, 2*np.pi):
        x = np.sin(r) + 10
        y = np.cos(r) * 1.33 + 20
        z = 0.17*x + 0.79*y + 30
        pp2.append(x, y)
        pp3.append(x, y, z)
    # With noise
    pp2 += np.random.normal(0, 0.15, size=pp2.shape)
    pp3 += np.random.normal(0, 0.15, size=pp3.shape)
    
    # Fit 2D 
    c2 = fit_circle(pp2)
    e2 = fit_ellipse(pp2)
    print('area circle 2D: % 1.2f' % area(c2))
    print('area ellipse 2D: % 1.2f' % area(e2))
    
    # Fit 3D. We first fit a plane, then project the points onto that
    # plane to make the points 2D, and then we fit the ellipse.
    # Further down, we sample the ellipse and project them to 3D again 
    # to be able to visualize the result.
    plane = fit_plane(pp3)
    pp3_2 = project_to_plane(pp3, plane)
    c3 = fit_circle(pp3_2)
    e3 = fit_ellipse(pp3_2)
    print('area circle 3D: % 1.2f' % area(c3))
    print('area ellipse 3D: % 1.2f' % area(e3))
    
    # For visualization, calculate 4 points on rectangle that lies on the plane
    x1, x2 = pp3.min(0)[0]-0.3, pp3.max(0)[0]+0.3
    y1, y2 = pp3.min(0)[1]-0.3, pp3.max(0)[1]+0.3
    p1 = x1, y1, -(x1*plane[0] + y1*plane[1] + plane[3]) / plane[2]
    p2 = x2, y1, -(x2*plane[0] + y1*plane[1] + plane[3]) / plane[2]
    p3 = x2, y2, -(x2*plane[0] + y2*plane[1] + plane[3]) / plane[2]
    p4 = x1, y2, -(x1*plane[0] + y2*plane[1] + plane[3]) / plane[2]
    
    # Init visualization
    import visvis as vv
    fig = vv.clf()
    fig.position = 300, 300, 1000, 600
    
    # 2D vis
    a = vv.subplot(121)
    a.daspectAuto = False
    a.axis.showGrid = True
    vv.title('2D fitting')
    vv.xlabel('x'); vv.ylabel('y')
    # Plot
    vv.plot(pp2, ls='', ms='.', mc='k')
#     vv.plot(sample_circle(c2), lc='r', lw=2)
    vv.plot(sample_ellipse(e2), lc='b', lw=2)
#     vv.legend('2D points', 'Circle fit', 'Ellipse fit')
    vv.legend('2D points', 'Ellipse fit')
    
    # 3D vis
    a = vv.subplot(122)
    a.daspectAuto = False
    a.axis.showGrid = True
    vv.title('3D fitting')
    vv.xlabel('x'); vv.ylabel('y'); vv.zlabel('z')
    # Plot
    vv.plot(pp3, ls='', ms='.', mc='k')
    vv.plot(project_from_plane(pp3_2, plane), lc='r', ls='', ms='.', mc='r', mw=4)
#     vv.plot(project_from_plane(sample_circle(c3), plane), lc='r', lw=2)
    vv.plot(project_from_plane(sample_ellipse(e3), plane), lc='b', lw=2)
    vv.plot(np.array([p1, p2, p3, p4, p1]), lc='g', lw=2)
#     vv.legend('3D points', 'Projected points', 'Circle fit', 'Ellipse fit', 'Plane fit')
    vv.legend('3D points', 'Projected points', 'Ellipse fit', 'Plane fit')
    