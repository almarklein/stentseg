"""
Provides pick3d().

Example:

    vol = vv.Aarray(vv.volread('stent'))
    a = vv.gca()
    t = vv.volshow(vol)
    pick3d(a, vol)

"""

import math
import numpy as np
import visvis as vv


def ortho(left, right, bottom, top, znear, zfar):
   
    assert(right != left)
    assert(bottom != top)
    assert(znear != zfar)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 / (right - left)
    M[3, 0] = -(right + left) / float(right - left)
    M[1, 1] = +2.0 / (top - bottom)
    M[3, 1] = -(top + bottom) / float(top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 2] = -(zfar + znear) / float(zfar - znear)
    M[3, 3] = 1.0
    return M


def translate(offset, dtype=None):
    assert len(offset) == 3
    x, y, z = offset
    M = np.array([[1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [x, y, z, 1.0]], dtype)
    return M


def scale(s, dtype=None):
    assert len(s) == 3
    return np.array(np.diag(np.concatenate([s, (1.,)])), dtype)


def rotate(angle, axis, dtype=None):
    """The 3x3 rotation matrix for rotation about a vector.

    Parameters
    ----------
    angle : float
        The angle of rotation, in degrees.
    axis : ndarray
        The x, y, z coordinates of the axis direction vector.
    """
    angle = np.radians(angle)
    assert len(axis) == 3
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = np.array([[cx * x + c, cy * x - z * s, cz * x + y * s, .0],
                  [cx * y + z * s, cy * y + c, cz * y - x * s, 0.],
                  [cx * z - y * s, cy * z + x * s, cz * z + c, 0.],
                  [0., 0., 0., 1.]], dtype).T
    return M


def get_camera_matrix(a):
    """ Replicate the transform of the camera.
    """
    MM = []
    p = a.GetView()
    
    fx = fy =  abs( 1.0 / p['zoom'] )
    w, h = [float(i) for i in a.position.size]
    if w / h > 1:
        fx *= w/h
    else:
        fy *= h/w
    
    MM.append(ortho( -0.5*fx, 0.5*fx, -0.5*fy, 0.5*fy, -1, 1))
    
    MM.append(rotate(p['roll'], [0.0, 0.0, 1.0]))
    MM.append(rotate(270+p['elevation'], [1.0, 0.0, 0.0]))
    MM.append(rotate(-p['azimuth'], [0.0, 0.0, 1.0]))
    
    ndaspect = a.camera.daspectNormalized
    MM.append(scale([ndaspect[0], ndaspect[1] , ndaspect[2] ]))
    
    loc = p['loc']
    MM.append(translate([-loc[0], -loc[1], -loc[2]]))
    
    # Build single matrix
    M = np.eye(4)
    for M2 in reversed(MM):
        M = np.dot(M, M2)
    return M


def sample(vol, p):
    try:
        return vol.sample(p)
    except IndexError:
        return None


def pick3d(axes, vol):
    """ Enable picking intensities in a 3D volume.

    Given an Axes object and a volume (an Aarray), will show the value,
    index and position of the voxel with maximum intentity under the
    cursor when doing a SHIFT+RIGHTCLICK.
    
    Returns the label object, so that it can be re-positioned.
    """
    assert hasattr(vol, 'sampling'), 'Vol must be an Aarray.'
    
    line = vv.plot(vv.Pointset(3), ms='o', mw=12, lc='c', mc='c', alpha=0.4, axesAdjust=False)
    label = vv.Label(axes)
    label.position = 0, 0, 1, 20
    
    @axes.eventMouseDown.Bind
    def onClick(event):
        if event.button == 2 and vv.KEY_SHIFT in event.modifiers:
            # Get clicked location in NDC
            w, h = event.owner.position.size
            x, y = 2 * event.x / w - 1, -2 * event.y / h + 1
            # Apply inverse camera transform to get two points on the clicked line
            M = np.linalg.inv(get_camera_matrix(event.owner))
            p1 = vv.Point(np.dot((x, y, -100, 1), M)[:3])
            p2 = vv.Point(np.dot((x, y, +100, 1), M)[:3])
            # Calculate center point and vector
            pm = 0.5 * (p1 + p2)
            vec = (p2 - p1).normalize()
            # Prepare for searching in two directions
            pp = [pm-vec, pm+vec]
            status = 0 if sample(vol, pm) is None else 1
            status = [status, status]
            hit = None
            max_sample = -999999
            step = min(vol.sampling)
            # Look in two directions simulaneously, search for volume, collect samples
            for i in range(10000):  # Safe while-loop
                for j in (0, 1):
                    if status[j] < 2:
                        s = sample(vol, pp[j])
                        inside = s is not None
                        if inside:
                            if s > max_sample:
                                max_sample, hit = s, pp[j]
                            if status[j] == 0:
                                status[j] = 1
                                status[not j] = 2  # stop looking in other direction
                        else:
                            if status[j] == 1:
                                status[j] = 2
                        pp[j] += (j*2-1) * step * vec
                if status[0] == 2 and status[1] == 2:
                    break
            else:
                print('Warning: ray casting to collect samples did not stop')  # clicking outside the volume
            # Draw
            pp2 = vv.Pointset(3);
            text = 'No point inside volume selected.'
            if hit:
                pp2.append(hit)
                ii = vol.point_to_index(hit)
                text = 'At z=%1.1f, y=%1.1f, x=%1.1f -> X[%i, %i, %i] -> %1.2f' % (hit.z, hit.y, hit.x, ii[0], ii[1], ii[2], max_sample)
            line.SetPoints(pp2)
            label.text = text
            return True  # prevent default mouse action
    
    return label


if __name__ == '__main__':
    vol = vv.Aarray(vv.volread('stent'), (0.5, 0.5, 0.5), (100, 40, 10))
    a = vv.gca()
    t = vv.volshow(vol)
    pick3d(a, vol)
