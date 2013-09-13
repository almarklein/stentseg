""" MODULE WALKERS
The walkers framework.
Provides a manager class to manage multiple walkers.
Also provides various base (abstract) walker classes for 
2D and 3D data.

"""


import visvis as vv
import numpy as np
import scipy as sp, scipy.ndimage
from visvis.pypoints import Point, Pointset, Aarray

from .. import mcp
from stentseg import gaussfun


class Parameters:
    def __init__(self):
        
        # The scale of the walker, this should match with the diameter
        # if the vessel. It should be at least 1.1 otherwise the kernel
        # to look around is too small. It should not be too large, 
        # otherwise the walker may cut corners or want to go towards
        # structures next to the wire....
        self.scale = 1.5
        
        # Two thresholds. The first should be below which we should start
        # to worry and should probably between the estimated value for the
        # wire and background. The second is the value for which the walker
        # should emediately be killed if it encounteres such a value.
        # See the threshold and food mechanism in walker.Move() for more
        # information.
        self.th1 = 0.5
        self.th2 = 0.0
        
        # The type of kernel to use. I think the full cosine is the best.
        # See manager.GetWeightKernel() for more information.
        self.kernelType = 0
        
        # The weights of the different direction components to apply to 
        # calculate the viewdir.
        self.comWeight = 1.0
        self.ringWeight = 0.5
        self.historyWeight = 0.0
        
        # The angle by which the viewdir is limited. The angle with the last
        # viewdir is limited by "limitingAngle". The next by 
        # "limitingAngle + limitingAngle*limitingAngleMultiplier", etc.
        # The limiting angle is also used to determine the candidate pixels
        # for taking a step, s that we can guarantee we do not hop between
        # two pixels.
        self.limitingAngle = 20
        self.limitingAngle2 = 280
        self.limitingAngleMultiplier = 99
        
        # The amount of steps that the regionGrowing Walker can step back.
        # When zero, it signifies a normal regiongrowing method.
        self.maxStepsBack = 0
        
        # For the mcp, the amount of ct values when a doubling of speed occurs.
        self.ctvalues2double = 50
        
        # The minimum distance to have travelled in order to stop.
        self.mcpDistance = 5 # mm
        
        # Test angle limit. If True, comDir is made normal to the previous one
        # to make the shortest corner as possible.
        self.testAngleLimit = False
        
        # A test param to quickly test new things without creating a new param.
        # Bugfix tip: Check if it is used elsewhere!
        self.test1 = 0.0
        self.test2 = 0.0
        


class WalkerManager:
    """ WalkerManager(data, walkerClass, verbose=True)
    
    Generic walker manager. Works for 2D and 3D data. Data should be
    a 2D or 3D np.ndarray or points.Aarray (anisotropic array).    
    
    Also manages drawing the data and walkers.
    """
    
    def __init__(self, data, walkerClass, verbose=True):
        
        # check
        if not isinstance(data, Aarray) and isinstance(data, np.ndarray):
            data = Aarray(data)
        if not isinstance(data, Aarray) or not data.ndim in [2,3]:
            raise ValueError("data should be a 2D or 3D Aarray")
        
        # parameters        
        self.params = Parameters()
        
        # list of walkers
        self._walkers = []
        
        # store stuff
        self.data = data
        self.walkerClass = walkerClass
        self.verbose = verbose
        
        # for cashing kernels
        self._gaussianKernels = {}
        self._weightKernels = {}
        
        # for debugging...
        self.walkerToInspect = None
        
        # setup visualization?
        if verbose:
            self._initVisualization()
        else:
            self._f = None
            self._t = None

    
    def _initVisualization(self):
        
        # clear
        self._f = f = vv.figure(1)
        f.Clear()
        
        # draw image or volume
        data = self.data
        if data.ndim == 2:
            self._t = vv.imshow(data)
        elif data.ndim == 3:
            self._t = vv.volshow(data)
        
        # make button to stop
        l=vv.Label(f,'Stop')
        l.position = -50,0,50,15
        l.eventMouseDown.Bind(self._OnClick)
        l.hitTest = True
        self._clicked = False
        
        # tune visualization        
        a = vv.gca()
        a.daspect = 1,-1,-1 # texture is scaled using the Aarray
    
    
    def _OnClick(self, event):
        self._clicked = True
    
    
    def __len__(self):
        return len(self._walkers)
    
    def __iter__(self):
        tmpList =  [walker for  walker in self._walkers]        
        return tmpList.__iter__()
    
    
    def Reset(self, data=None):
        """ Reset        
        - Remove all walkers (not kill). 
        - Resetup visualisation.
        - Reset data (if given)"""
        
        # clear walkers
        for w in self._walkers:
            w.__del__()
        self._walkers[:] = []
        
        # reset data
        if data is not None:
            if not isinstance(data, Aarray) and isinstance(data, np.ndarray):
                data = Aarray(data)
            if not isinstance(data, Aarray) or not data.ndim in [2,3]:
                raise ValueError("data should be a 2D or 3D Aarray")
            self.data = data
        
        # reset food
        if hasattr(self, 'food'):
            del self.food
        if hasattr(self, 'mcp'):
            del self.mcp
        
        # clear visualisation
        if self.verbose:
            self._initVisualization()            
    
    
    def Spawn(self, pos):
        """ Spawn a walker at the specified position. 
        The walker instance is returned.
        """
        return self.walkerClass(self, pos )
    
    
    def Sprinkle(self, distance):
        """ Sprinkle walkers in the data in a regular grid. 
        distance is the distance in Array units (mm for CT data).
        The given distance is stored in .distance so the walkers
        can use this value to determine as a hint how far they 
        should look around them.
        """
        
        # store distance
        self.distance = d = float(distance)
        
        # clear any old walkers
        for walker in self:
            walker.Kill("re-sprinkling")
        
        # spawn walkers
        data = self.data
        starts = data.get_start()
        ends = data.get_end()
        if starts.ndim == 2:            
            for y in        np.arange(starts.y+d, ends.y-d/2, d):
                for x in    np.arange(starts.x+d, ends.x-d/2, d):
                    s = self.Spawn( (x,y) )
        
        elif starts.ndim == 3:
            for z in            np.arange(starts.z+d, ends.z-d/2, d):
                for y in        np.arange(starts.y+d, ends.y-d/2, d):
                    for x in    np.arange(starts.x+d, ends.x-d/2, d):
                        s = self.Spawn( (x,y,z) )
        
        # draw
        self.DrawIfVerbose()
    
    
    def Snap(self):
        """ Snap all walkers to the pixel/voxel with largest
        intensity. The region in which to look is chosen to be 
        slightly more OR LESS than half the sprinkle distance.
        """
        #region = self.distance * 0.6 # small overlap
        region = self.distance * 0.4 # no overlap
        for walker in self:                        
            walker.GoToMaxInRegion(region)
            walker._history.clear()
            walker.history.clear()
        self.DrawIfVerbose()
    
    
    def DrawIfVerbose(self):
        if self.verbose:
            self.Draw()
    
    
    def Draw(self):
        """ Draw all walkers. """
        
        # figure exists?
        if self._f is None or self._t is None or self._f._destroyed:
            self._initVisualization()
        
        # init walker appearance
        for walker in self:
            walker._l1.mec = 'k'
            walker._l1.mew = 1
        
        # draw specific walkers
        colors = [None, None, (0,1,0), (0,0,1), (0,1,1),(1,0,1),(1,0,0)]        
        w = self.walkerToInspect
        if isinstance(w,int):
            if w > len(self._walkers):
                w = None
            w = self._walkers[w]
        if w:
            # give figure            
            f = vv.figure(2); f.Clear()
            f.bgcolor = colors[2]
            w._l1.mec = colors[2]
            w._l1.mew = 2
            w.DrawMore(f)
        
        # let walkers do their thing
        for walker in self:
            walker.Draw()        
        self._f.DrawNow()
    
    
    def Walk(self, iterations=1):
        """ Let the walkers walk. This is the main function to use,
        after having sprinkled. """
        for i in range(iterations):
            for walker in self:
                walker.Move()
        self.DrawIfVerbose()
    
    def WalkInspectedWalker(self, iterations=1):
        """ Let the inspected walker walk."""
        for i in range(iterations):
            if self.walkerToInspect:
                walker.Move()
        self.DrawIfVerbose()
    
    def Dance(self):
        """ Let's dance! """
        
        def steps(i, factor):
            for walker in self:
                walker.pos[i] += factor
            self.Draw()
        
        def twostep(i, step):
            step = step/3.0
            steps(i,step);    steps(i,step);    steps(i,step)
            steps(i,-step);   steps(i,-step);   steps(i,-step)
            steps(i,-step);   steps(i,-step);   steps(i,-step)
            steps(i,step);    steps(i,step);    steps(i,step)
        
        twostep(0, 9)
        twostep(1, 9)
        if self.data.ndim == 3:
            twostep(2, 4)
    
    
    def GetGaussianKernel(self, sigma, sigma2szeRatio=3, orders=(0,0) ):
        """ Get a Gaussian kernel. If the requested kernel was
        requested earlier, the cashed version is used. Otherwise
        it is calculated and cashed for reuse.
        
        This method is aware of anisotropy; sigma is expressed in 
        th units of the anisotropic array.
        """
        
        # create unique key for this kernel
        key = [ sigma, sigma2szeRatio ] + list(orders)
        key = tuple(key)
        
        # was this kernel calculated earlier?
        if key in self._gaussianKernels:
            return self._gaussianKernels[key]
        
        # ... calculate it
        
        # calculate sigmas, given the anisotropy of the data        
        sigmas = [sigma/s for s in self.data.sampling]
        if len(orders) > len(sigmas):
            raise ValueError("The requested kernel has more dims than data.")
        
        # create 1D kernels
        kernels = []
        for i in range(len(orders)):
            tail = sigma2szeRatio * sigmas[i]
            tmp = gaussfun.gaussiankernel( sigmas[i], orders[i], -tail)
            kernels.append(tmp)
        
        # init higher dimensional kernel
        shape = tuple([len(k) for k in kernels])
        k = np.zeros(shape,dtype=np.float32)
        
        # well, we only have 2D or 3D kernels
        if k.ndim==2:
            for y in range(k.shape[0]):
                for x in range(k.shape[1]):
                    k[y,x] = kernels[0][y] * kernels[1][x]
        if k.ndim==3:
            for z in range(k.shape[0]):
                for y in range(k.shape[1]):
                    for x in range(k.shape[2]):
                        k[z,y,x] = kernels[0][z] * kernels[1][y] * kernels[2][x]
        
        # store and return
        self._gaussianKernels[key] = k
        return k
    
    
    def GetWeightKernel(self, szes, direction):
        """ Get the weighting kernels for the dimensions
        given in the tuple of szes.
        self.kernelType determines what kernel to use.
        """
        
        # get key
        key = szes
        if isinstance(key, list):
            key = tuple(key)
        
        # normalize direction
        direction = direction.normalize()
        
        if key in self._weightKernels:
            # the first bit was calculated earlier
            kwz, kwy, kwx = self._weightKernels[key]
        else:
            # calculate and store the first bit
            
            # create kernels with directional vectors
            if len(szes)==2:
                kwy, kwx = np.mgrid[ -szes[0]:szes[0]+1, -szes[1]:szes[1]+1 ]
                kwz = np.array(0.0,dtype=np.float32)
                kwy = kwy.astype(np.float32) * self.data.sampling[0]
                kwx = kwx.astype(np.float32) * self.data.sampling[1]
            elif len(szes)==3:
                kwz, kwy, kwx = np.mgrid[ -szes[0]:szes[0]+1,
                    -szes[1]:szes[1]+1, -szes[2]:szes[2]+1 ]
                kwz = kwz.astype(np.float32) * self.data.sampling[0]
                kwy = kwy.astype(np.float32) * self.data.sampling[1]
                kwx = kwx.astype(np.float32) * self.data.sampling[2]
            else:
                raise Exception("Given szes tuple should be length 2 or 3")
            # normalize                
            kwl = np.sqrt(kwx**2 + kwy**2 + kwz**2)
            kwl[kwl==0] = 99.0 #center point
            kwz, kwy, kwx = kwz/kwl, kwy/kwl, kwx/kwl            
            # store
            self._weightKernels[key] = kwz, kwy, kwx
            
        # calculate kernel now ...  
        
        # Is the cosine of the angle of all vectors in the kernel (and 0 if
        # that number is negative):
        #   max(0, cos(angle(direction,v)))
        # The angle between two vectors can be calculated using the 
        # arccosine of the inproduct of the normalized vectors.
        # Because we want the cosine of that result, the cosine/arccosine
        # can both be left out :)
        
        # inproduct
        if len(szes) == 2:
            kw = kwy * direction.y + kwx * direction.x 
        elif len(szes) == 3:
            kw = kwz * direction.z + kwy * direction.y + kwx * direction.x 
        
        if self.params.kernelType <= 0:
            # do it the quick way because cos and acos fall out.            
            return kw*0.5 + 0.5
            
        else:
            # first calculate actual angle
            kw[kw>1.0] = 1.0
            kw[kw<-1.0] = -1.0
            kw = np.arccos(kw)
            
            if self.params.kernelType == 1:
                # half moon
                kw = abs(kw)
                pi2 = np.pi/2
                kw[kw<pi2] = 1
                kw[kw>=pi2] = 0
                
            elif self.params.kernelType == 2:
                # half circle
                pi2 = np.pi/2
                kw = (pi2**2-kw**2)**0.5 /pi2
                kw[np.isnan(kw)] = 0
            
            elif self.params.kernelType == 3:
                # half cosine
                kw = np.cos(kw)
                kw[kw<0] = 0
                
            else:
                # full cosine
                kw = np.cos(kw)*0.5 + 0.5
            
            return kw
    
    
class BaseWalker(object):
    """ An agent to walk the stent. 
    An abstract class that implements basic functionality. 
    The most important method to overload is Move(), which is called
    each iteration. Draw() can also be overloaded to provide more info.
    """
    
    
    def __init__(self, manager, p):
        
        # store position
        self.pos = Point(p)
        
        # register
        self._manager = manager
        self._manager._walkers.append(self)
        
        # keep track of where we've been
        self._history = Pointset(manager.data.ndim)
        self.history = Pointset(manager.data.ndim)
        self._viewdirHistory = Pointset(manager.data.ndim)
        
        # initialize the line objects        
        if manager.verbose:
            self._initVisualization()
        else:
            self._l1, self._l2, self._l3 = None, None, None
        
        # current direction (must be something)
        if manager.data.ndim==2:
            self.dir = Point(1,1)
            self.viewDir = Point(1,1) # you have to look *somewhere*
            self.walkDir = Point(0,0)
        elif manager.data.ndim==3:
            self.dir = Point(1,0,0)
            self.viewDir = Point(1,1,1)
            self.walkDir = Point(0,0,0)
        
        # store this
        self._killReason = ''
        
    def _initVisualization(self):
        # current pos
        self._l1 = vv.plot([0,1],[0,1], lw=0,ms='o',mw=10,mc='g',mew=1,mec='b',
                axesAdjust=False)
        #self._l1.points = Pointset(self._manager.data.ndim)
        self._l1.alpha = 0.6
        # direction
        self._l2 = vv.plot([0,1],[0,1], lw=2,lc='r',ms='', axesAdjust=False)
        #self._l2.points = Pointset(self._manager.data.ndim)
        self._l2.alpha = 0.6
        # history
        self._l3 = vv.plot([0,1],[0,1], lw=1,lc='y',ms='', axesAdjust=False)
        #self._l3.points = Pointset(self._manager.data.ndim)
        self._l3.alpha = 0.4
        
        # make inspection callback
        self._l1.eventMouseDown.Bind(self._OnClick)
    
    
    def _OnClick(self, event):
        self._manager.walkerToInspect = self
        self._manager.Draw()
    
    
    def Draw(self):
        """ Update lines. Called by manager. Don't call this. """ 
        
        # init visualisation
        if None in [self._l1, self._l2, self._l3]:
            self._initVisualization()
        
        # set position
        pp = Pointset(self.pos.ndim)
        pp.append(self.pos)
        self._l1.SetPoints(pp)
        
        # set direction
        if self.dir is not None:
            pos = self.pos + self.dir*4
        else:
            pos = self.pos
        pp = Pointset(self.pos.ndim)        
        pp.append(self.pos)
        pp.append(pos)
        self._l2.SetPoints(pp)
        
        # markercolor
        if self._manager.data.sample(self.pos) < self._manager.params.th1:
            self._l1.mc=(1,0.5,0.0)
        else:
            self._l1.mc='g'
        
        # set history
        pp = self.history.copy()
        pp.append(self.pos)
        self._l3.SetPoints(pp)
        
    
    
    def DrawMore(self, f):
        """ The walker should draw more information, like used patches
        in the specified figure. Subfigures can be used to display
        even more information. """
        pass
    
    
    def Move(self):
        """ This is called every iteration. Do your thing here. """
        raise NotImplemented()
    
    def GetPatch(self, sze, data=None):
        """ Implemented by the 2D and 3D base walker classes. """
        raise NotImplemented()
    
    
    def GetMaxInRegion(self, region):
        """ Implemented by the 2D and 3D base walker classes. """
        raise NotImplemented()
    
    
    def GoToMaxInRegion(self, region):
        """ Look in a certain region for the highest intensity voxel. 
        This becomes the new position.
        """     
        # get threshold        
        th = self._manager.params.th1
        # get pos
        pos = self.GetMaxInRegion(region)
        if pos is None:
            self.Kill("Too near the edge in looking for stent.")
            return
        # set
        self.SetPos(pos)
        # kill ourselves?
        data = self._manager.data
        if data.point_to_index(pos,True) is None:
            self.Kill("In looking for stent, new pos out of bounds.")
        if data.sample(pos) < th:
            self.Kill("In going to max in region, intensity too low.")
    
    
    def SetPos(self, pos):
        """ Set the position, making the old position history.
        """
        # add to history and smooth
        self._history.append(self.pos)
        self._viewdirHistory.append(self.viewDir)
        self.history.clear()
        h = self._history
        self.history.append(h[0])
        for i in range(1,len(h)-1):
            self.history.append( (h[i-1] + h[i]+ h[i+1])/3 )
        self.history.append(h[-1])
        
        if pos is None:
            self.Kill("Position beyond array.")
            return
        self.pos = pos
    
    
    def StepBack(self, nsteps=1):
        """ Set back the position and viewDir. """
        # put back
        self.pos = self._history[-nsteps]
        self.viewDir = self._viewdirHistory[-nsteps]        
        # remove history
        for i in range(nsteps):
            self._history.pop()
            self._viewdirHistory.pop()
    
    
    def Kill(self, reason="unknown reason"):
        self._killReason = reason        
        # notify
        if self._manager.verbose:
            print("walker killed: " + reason)
        # remove from list
        L = self._manager._walkers
        if self in L:
            L.remove(self)
        # clear lines
        self.__del__()

    
    def __del__(self):
        if self._l1 is not None:
            self._l1.parent = None
        if self._l2 is not None:
            self._l2.parent = None
        if self._l3 is not None:
            if True:#self._killReason.count('Out of bounds'):
                #pass # leave line
                self._l3.ls = ':'
            else:
                self._l3.parent = None

    def _LimitAngle(self, vec1, vec2, angleLimit):
        """ Correct vec1 such that the angle between vec1
        and vec2 is smaller or equal to angleLimit (in degrees). 
        The math is pretty hard. (Harder than it might seem)
        Therefore I solve this by estimating it iteratively...
        """
        vec1 = vec0 = vec1.normalize()        
        vec2 = vec2.normalize()
        angleLimit = angleLimit * np.pi / 180.0
        while abs(vec1.angle(vec2)) > angleLimit:
            vec1 = ( vec1*10 + vec2 ).normalize()
        #print(vec0.angle(vec2)*180/np.pi, vec1.angle(vec2)*180/np.pi)
        return vec1


class BaseWalker2D(BaseWalker):
    """ An abstract walker for 2D data. """
    
    def GetPatch(self, sze, data=None):
        """ GetPatch(sze, data=None):
        
        Get a pach from the data around the current position.
        For this, the current position is first translated to data
        coordinates.
        
        If data is None, uses self._manager.data.
        sze is the amount of voxels in each direction, or a tuple 
        to specify the amount of voxels (in each direction) for 
        the y, and x dimension seperately.
        
        returns None if too near the edge or beyond.
        """
        
        if data is None:
            data = self._manager.data
        
        # get szes
        if isinstance(sze,tuple):
            sy, sx = sze
        else:
            sy = sx = sze
        # make sure they are integer
        sx, sy = int(sx+0.5), int(sy+0.5)
        
        # get indices for this point
        try:
            iy,ix = data.point_to_index( self.pos )
        except IndexError:
            return None
            #raise # simply reraise
        
        # check if too near the edge
        shape = data.shape
        if ix-sx < 0 or iy-sy < 0:
            return None
            #raise IndexError("Too near the edge to sample patch.")
        if ix+sx >= shape[1] or iy+sy >= shape[0]:
            return None
            #raise IndexError("Too near the edge to sample patch!")
        
        # select patch and return
        patch = data[iy-sy:iy+sy+1, ix-sx:ix+sx+1]
        if patch.size != (sx*2+1) * (sy*2+1):
            raise Exception("This should not happen.")
        return patch


    def GetMaxInRegion(self, region):
        """ Get the maximum position (in "global" coordinates)
        in a region around the walker position. Returns None
        if the region could not be sampled because we are too
        near the edge.
        """   
        # calculate sze in each dimension
        sam = self._manager.data.sampling
        sze = int(region/sam[0]), int(region/sam[1])
        # get patch
        patch = self.GetPatch(sze)        
        if patch is None:
            return None
        # find max in it
        Iy,Ix = np.where( patch == patch.max() )
        # get position
        scale = Point(sam[1],sam[0])
        dp = Point( Ix[0]-sze[1], Iy[0]-sze[0])
        return self.pos + dp * scale


class BaseWalker3D(BaseWalker):
    """ An abstract walker for 3D data. """
    
    def GetPatch(self, sze, data=None):
        """ GetPatch(sze, data=None):
        
        Get a pach from the data around the current position.
        For this, the current position is first translated to data
        coordinates.
        
        If data is None, uses self._manager.data.
        sze is the amount of voxels in each direction, or a tuple 
        to specify the amount of voxels (in each direction) for 
        the z, y, and x dimension seperately.
        
        returns None if too near the edge or beyond.
        """
        
        if data is None:
            data = self._manager.data
        
        # get szes
        if isinstance(sze,tuple):
            sz, sy, sx = sze
        else:
            sz = sy = sx = sze
        # make sure they are integer
        sx, sy, sz = int(sx+0.5), int(sy+0.5), int(sz+0.5)
        
        # get indices for this point
        try:
            iz,iy,ix = data.point_to_index( self.pos )
        except IndexError:
            return None
            #raise # simply reraise
        
        # check if too near the edge
        shape = data.shape
        if ix-sx < 0 or iy-sy < 0 or iz-sz < 0:
            return None
            #raise IndexError("Too near the edge to sample patch.")
        if ix+sx >= shape[2] or iy+sy >= shape[1] or iz+sz >= shape[0]:
            return None
            #raise IndexError("Too near the edge to sample patch!")
        
        # select patch and return
        patch = data[ iz-sz:iz+sz+1, iy-sy:iy+sy+1, ix-sx:ix+sx+1]
        if patch.size != (sx*2+1) * (sy*2+1) * (sz*2+1):            
            raise Exception("This should not happen.")
        return patch
    
    
    def GetMaxInRegion(self, region):
        """ Get the maximum position (in "global" coordinates)
        in a region around the walker position. Returns None
        if the region could not be sampled because we are too
        near the edge.
        """   
        # calculate sze in each dimension
        sam = self._manager.data.sampling
        sze = int(region/sam[0]), int(region/sam[1]), int(region/sam[2])
        # get patch
        patch = self.GetPatch(sze)        
        if patch is None:
            return None
        # find max in it
        Iz,Iy,Ix = np.where( patch == patch.max() )
        # get position
        scale = Point(sam[2],sam[1],sam[0])
        dp = Point( Ix[0]-sze[2], Iy[0]-sze[1], Iz[0]-sze[0])
        return self.pos + dp * scale


## Specific implementations

class NewRegionGrowingWalker2D(BaseWalker2D):
    """ The new region growing walker. 
    I was implementing stuff that made the walker search for the minimum
    cost path. But while this was in progress, I implemented the real
    mcp algorithm. This class is therefore depreciated.
    """
    
    def __init__(self, manager, p):
        BaseWalker2D.__init__(self, manager, p)
        
        # create mask
        if not hasattr(manager, 'mask'):
            manager.mask = Aarray(manager.data.shape, dtype=np.uint16,
                sampling=manager.data.sampling, fill=0)
        
        # to determine if we encountered an 
        self._encounteredEdge = False
        
        # to be able to "dilate" the mask
        self._dilatepos = Point(0,0)
    
    
    def SetPos(self, pos):        
        # dilate first
        self.RememberPos(self._dilatepos,1)
        # set what to dilate next round (the previous pos)
        self._dilatepos = self.pos
        # set this pos
        if pos is not None:
            self.RememberPos(pos)
            BaseWalker2D.SetPos(self, pos)
    
    def RememberPos(self, pos, sze=0):
        """ Remember this pos as a position where we've been,
        so we cannot go there again. """
        mask = self._manager.mask
        iy, ix = self._manager.data.point_to_index(pos)
        for dy in range(-sze,sze+1):
            for dx in range(-sze,sze+1):
                y, x = iy+dy, ix+dx
                if y<0 or x<0 or y>=mask.shape[0] or x>=mask.shape[1]:
                    continue
                mask[y,x] = 1
        
    
    def GetBestNeighbor(self, pos):
        """ look around if we can find a direct neighbor. """
        
        # get arrays 
        data = self._manager.data
        mask = self._manager.mask
        scale = Point(data.sampling[1],data.sampling[0])
        
        # get indices
        iy, ix = data.point_to_index(pos)
        
        # init
        bestdelta, bestval = None, -9999999
        self._encounteredEdge = False
        
        # go!
        for delta in candidates2:
            # sample value
            try:
                val = data[iy+delta.y, ix+delta.x]
            except IndexError:
                self._encounteredEdge = True
                continue
            # highest?
            if val > bestval:                
                # have we been there before?
                if mask[iy+delta.y, ix+delta.x] == 0:
                    bestdelta, bestval = delta, val
                else:
                    pass # already been there
        
        # done
        if bestdelta is None:
            return None, bestval
        else:
            newpos = pos + bestdelta * scale
            return newpos, bestval
    
    
    def MinimalCostPath(self):
        
        # init cost and visited patch
        sze = 3
        size = sze*2+1
        alot = 999999
        visited = np.zeros((size,size,size),dtype=np.uint8)
        cost = np.ones(frozen.shape, dtype=np.float32) * alot
        
        # get data and indices
        data = self._manager.data
        iy,ix = data.point_to_index(self.pos)
        patch = data[iy-sze:iy+sze+1, ix-sze:ix+sze+1]
        
        # todo: check if on a border
        
        cost[sze,sze] = 0
        while visited.min() == 0:
            # select point
            y,x = np.where(cost < alot & visisted==0)
            #if t
        
    
    def Move(self):
        
        
#         # todo: make method
#         def ProduceOrderedCandidates( pos, distance):
#             # prepare
#             sam = data.sampling
#             ny, nx = int(distance / sam[0]), int(distance / sam[1])
#             pp = []
#             # select all voxels in range
#             for iy in range(-ny, ny+1):
#                 for ix in range(-nx, nx+1):
#                     if ix==0 and iy ==0:
#                         continue
#                     delta = Point(ix*sam[1], iy*sam[0])
#                     pp.append( (pos + delta, delta.norm()) )
#             # sort
#             pp.sort(key=lambda x:x[1])
#             return pp
#         
#         bestpos, bestval, bestdist = None, -99999, 99999
#         edge = False
#         for p,d in ProduceOrderedCandidates(pos, ringDistance):
#             # is this point further than the best so far?
#             if d > bestdist:
#                 break
#             # sample value and test
#             val = data.sample(p,-10000)
#             if val == -10000:
#                 edge = True
#                 continue
#             if val < th:                
#                 continue
#             if val > bestval:
#                 # test if we are allowed to go there ...
#                 if food.sample(p) == 1:
#                     # if so, this is our best option so far
#                     bestpos, bestval, bestdist = p, val, d
#                 else:
#                     continue

        # get threshold parameters
        th1, th2 = self._manager.params.th1, self._manager.params.th2
        
        # check direct neighbours
        pos, val = self.GetBestNeighbor(self.pos)
        
        if pos is None:
            if self._encounteredEdge:
                self.Kill('Fell off the edge of the data.')
            else:
                self.Kill('Encountered another walker.')
            return
        
        if val >= th1:
            self.SetPos(pos)
        
        else:
            
            # init road
            road = Pointset(self._manager.data.ndim)
            road.append(pos)
            
            # explore ...            
            while pos is not None and val < th1:
                # (re)set                
                pos = self.pos.copy()
                road = [self.pos]
                distance = 0
                while distance < 4.5:#self._manager.params.test1:
                    pos, val = self.GetBestNeighbor(pos)
                    if pos is None:
                        break
                    distance += road[-1].distance(pos)
                    road.append(pos)
                    self.RememberPos(pos)
                    print(pos)
                    if val > th1:
                        break # finished!
                else:
                    self.dir = pos - self.pos
                    self._manager.Draw()
            if pos is None:
                self.Kill('Could not find a voxel with sufficient intensity.')
            else:
                for pos in road[1:]:
                    self.SetPos(pos)
    
    
    

class RegionGrowingWalker2D(BaseWalker2D):
    """ This things is unaware of anisotropy. 
    It choses one of its neighbours as the next position based on 
    pixel intensities. To overcome getting into little "sidebranches"
    due to noise, I implemented a backtracking algorithm that kicks in
    when we cannot proceed because the intensities are too low. It will
    try again from a previous step, at max traversing X steps back in
    history. This actually works pretty good, but the gaps in corners
    are still a major problem.
    
    This already starts to look like the MCP method a bit. In the 
    NewRegionGrowingWalker we took this further, untill I realized that
    MCP is what I wa looking for. Which resulted in the MCPwalker classes.
    """
    
    def __init__(self, manager, p):
        BaseWalker2D.__init__(self, manager, p)
        self._foodsupply = 1.0
        self._forbidden = Pointset(manager.data.ndim)
        
        # create food
        if not hasattr(manager, 'food'):
            manager.food = Aarray(manager.data.shape, dtype=np.uint8,
                sampling=manager.data.sampling, fill=1)
        
        # to keep track of the backtracking
        self._stepsBack = 0
        
        # dont show a vector
        self.dir = None
    
    
    def SetPos(self, pos):
        """ Override. """
        prevpos = self.pos
        BaseWalker2D.SetPos(self, pos)
        if pos is not None:
            food = self._manager.food
            # eat from newfound location 
            iy, ix = self._manager.data.point_to_index(pos)
            food[iy,ix] = 0
            # eat some more one position back            
            iy, ix = self._manager.data.point_to_index(prevpos)
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    yy, xx = iy+y, ix+x
                    if yy<0 or xx<0 or yy>=food.shape[0] or xx>=food.shape[1]:
                        continue
                    food[iy+y,ix+x] = 0
    
    
    def _ProduceCandidates(self, dist, yIndex, xIndex):
        candidates = Pointset(2)        
        for y in range(-dist, dist+1):
            for x in range(-dist, dist+1):
                if dist in [abs(y), abs(x)]:
                    candidates.append(xIndex+x,yIndex+y)
        return candidates
    
    
    def Move(self):
        """ Take my turn. """
        
        # get data, position expressed as index
        food = self._manager.food
        data = self._manager.data
        pos = self.pos.copy()
        
        # get threshold parameter.
        th1, th2 = self._manager.params.th1, self._manager.params.th2
        
        # get neighbor with highest intensity
        iy, ix = data.point_to_index(self.pos)
        bestp, bestval = Point(0,0), -999 # th
        edge = False
        for p in candidates2:
            try:
                val = data[ iy+p.y, ix+p.x ]
            except IndexError:
                edge = True
                continue
            if val > bestval:
                if food[ iy+p.y, ix+p.x ]:
                    bestval = val
                    bestp = p
        bestpos = data.index_to_point(iy+bestp.y, ix+bestp.x)
        
#         # should we worry?
#         worry = False
#         val = self._manager.data.sample(self.pos)
#         th1, th2 = self._manager.params.th1, self._manager.params.th2
#         if  val < th1:
#             portion = (th1 - val ) / (th1-th2)
#             self._foodsupply -= portion
#             if self._foodsupply <= 0:
#                 worry = True
#                 #self.Kill("Ran in too low intensity pixels")
#         else:
#             self._foodsupply = 1.0
        
        
        # Maybe we cannot find anything ...
        if bestval < th1:
            
            maxSteps = self._manager.params.maxStepsBack
            
            # if on the edge, nothing we can do...
            if edge:
                self.Kill('Fell off the edge of the data.')
            
            # step back and try again            
            elif self._stepsBack < maxSteps and len(self._history):
                self._stepsBack += 1
                self.StepBack()
                self.Move()
            
            # we tried too often
            else:            
                self.Kill('Cannot find a voxel with sufficient intensity.')
            
            # always return here ...
            return
        
        # go there (Don't forget to go back to mm!)
        #pos = data.index_to_point(bestpos.yi, bestpos.xi)
        
        try:
            self.SetPos(bestpos)
        except IndexError:
            # I dont know why, but this happens sometimes.
            self.Kill('Fell off the edge of the data.')            
        self._stepsBack = 0
    


class RegionGrowingWalker3D(BaseWalker3D):
    """ A walker that walks based on intensity and keeping
    a volume of bools to determine where a walker has walked
    (thus "eaten" the stent) to prevent going there twice.
    """
    
    
    def __init__(self, manager, p):
        BaseWalker3D.__init__(self, manager, p)
        
        # create food
        if not hasattr(manager, 'food'):
            manager.food = Aarray(manager.data.shape, dtype=np.uint8,
                sampling=manager.data.sampling, fill=1)
        
        # to keep track of the backtracking
        self._stepsBack = 0
        
        # dont show a vector
        self.dir = None  
    
    
    def SetPos(self, pos):
        """ Override. """        
        prevpos = self.pos
        BaseWalker3D.SetPos(self, pos)
        if pos is not None:
            food = self._manager.food
            shape = food.shape
            # eat from newfound location 
            iz, iy, ix = self._manager.data.point_to_index(pos)
            food[iz,iy,ix] = 0
            # eat some more one position back            
            iz, iy, ix = self._manager.data.point_to_index(prevpos)
            for z in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    for x in [-1, 0, 1]:
                        zz, yy, xx = iz+z, iy+y, ix+x
                        if (yy<0 or xx<0 or zz<0 or 
                            zz>=shape[0] or yy>=shape[1] or xx>=shape[2]):
                            continue
                        food[zz,yy,xx] = 0
   
    
    def GetMaxInRegion(self, region):
        """ Get the maximum position (in "global" coordinates)
        in a region around the walker position. Returns None
        if the region could not be sampled because we are too
        near the edge.
        """  
        # calculate sze in each dimension
        sam = self._manager.data.sampling
        sze = int(region/sam[0]), int(region/sam[1]), int(region/sam[2])
        # get patch
        patch = self.GetPatch(sze)
        tmp = self.GetPatch(sze, self._manager.food)        
        if patch is None or tmp is None:
            return None
        # apply mask
        patch = patch.copy() * tmp
        # find max in it
        Iz,Iy,Ix = np.where( patch == patch.max() )
        # get position
        scale = Point(sam[2],sam[1],sam[0])
        dp = Point( Ix[0]-sze[2], Iy[0]-sze[1], Iz[0]-sze[0])
        return self.pos + dp * scale
    
    
    def Move(self):
        """ Take my turn. """
        
        # get data, position expressed as index
        food = self._manager.food
        data = self._manager.data        
        pos = self.pos.copy()
        
        # get threshold parameter.
        th1, th2 = self._manager.params.th1, self._manager.params.th2
        
        # get neighbor with highest intensity
        iz, iy, ix = data.point_to_index(self.pos)
        bestp, bestval = Point(0,0,0), -999 # th
        edge = False
        for p in candidates3:
            try:
                val = data[ iz+p.z, iy+p.y, ix+p.x ]
            except IndexError:
                edge = True
                continue
            if val > bestval:
                if food[ iz+p.z, iy+p.y, ix+p.x ]:
                    bestval = val
                    bestp = p
        bestpos = data.index_to_point(iz+bestp.z, iy+bestp.y, ix+bestp.x)
        
        # Maybe we cannot find anything ...
        if bestval < th1:
            
            maxSteps = self._manager.params.maxStepsBack
            
            # if on the edge, nothing we can do...
            if edge:
                self.Kill('Fell off the edge of the data.')
            
            # step back and try again            
            elif self._stepsBack < maxSteps and len(self._history):
                self._stepsBack += 1
                self.StepBack()
                self.Move()
            
            # we tried too often
            else:            
                self.Kill('Cannot find a voxel with sufficient intensity.')
            
            # always return here ...
            return
        
        # go there
        try:
            self.SetPos(bestpos)
        except IndexError:
            # I dont know why, but this happens sometimes.
            self.Kill('Fell off the edge of the data.')            
        self._stepsBack = 0



# create set of points for candidates
candidates2 = Pointset(2)
candidates3 = Pointset(3)
for x in [-1,0,1]:
    for y in [-1,0,1]:                
        if x**2 + y**2 > 0: # to remove center
            candidates2.append(x,y)
for z in [-1,0,1]:
    for x in [-1,0,1]:
        for y in [-1,0,1]:  
            if x**2 + y**2 + z**2 > 0: # to remove center
                candidates3.append(z,x,y)


class DirComWalker2D(BaseWalker2D):
    """ A walker that uses the directional center of mass method to
    establish the next direction. The found direction is limited to the
    previous direction by a predetermined angle.
    """
    
    def __init__(self, *args):
        BaseWalker2D.__init__(self, *args)
        self._foodsupply = 1.0
    
    
    def GoToMaxInRegion(self, region):
        """ Overloaded version. Calls the original method and
        then determines an initial orientation. """
        BaseWalker2D.GoToMaxInRegion(self, region)
        
        # init dir
        for i in range(5):
            comdir = self.DirCom(self.viewDir)
            if comdir  is not None:
                self.viewDir = comdir
        
        # spawn walker in opposite direction
        spawn = DirComWalker2D(self._manager, self.pos)
        spawn.viewDir = self.viewDir*-1
        
    
    def Move(self):
        """ Do our move... """
        
        # self.dir is the direction that is visualized.
        
        # apply directional center of mass operator
        comDir = self.DirCom(self.viewDir)
        if comDir is None:
            return
        
        # store previous viewdir
        oldViewDir = self.viewDir
        
        # for testing
        if self._manager.params.testAngleLimit:
            comDir = self.viewDir.normal()
        
        
        # keep running direction and limit using history            
        
        # apply
        self.viewDir = comDir
        
        # make sure viewDir history exists
        if not hasattr(self,'_viewDirHistory'):
            self._viewDirHistory = []
        
        # limit (remember: 180 degrees never occurs!)
        # The limit must be increased the further we go back in history, but
        # it must not be increased linearly, otherwise it has no effect since
        # the limit imposed is the already guaranteed by the testing of the 
        # previous history dir. Therefore we apply the multiplier.
        # Note that if we change the amount of history taken into account can
        # have a serious effect on the optimal parameters.
        limitingAngle = 0
        extra = self._manager.params.limitingAngle
        tmp = []
        for dir in self._viewDirHistory:                
            limitingAngle += extra
            extra *= self._manager.params.limitingAngleMultiplier
            if limitingAngle > 180:
                break
            tmp.append(limitingAngle)
            ang = abs(self.viewDir.angle(dir)) * 180.0 / np.pi
            if ang > limitingAngle:
                # Correct viewdir. Draw this out on paper to understand it.
                oldAng = abs(oldViewDir.angle(dir)) * 180.0 / np.pi
                a1 = abs(limitingAngle - ang)
                a2 = abs(limitingAngle - oldAng)
                a = a2 / (a1+a2)
                self.viewDir = self.viewDir * a + oldViewDir * (1-a)
                self.viewDir = self.viewDir.normalize()
        
        # store and limit history
        self._viewDirHistory.insert(0,self.viewDir)
        self._viewDirHistory[10:] = []            
       
        
        # Do a step in that direction
        self.viewDir = self.viewDir.normalize()        
        stepDir = self.DoStep(self.viewDir)
        
        # what do we visualize?
        self.dir = self.viewDir
        #self.dir = self.walkDir (not used anymore)
        
        # test if we are ok here...
        # There are two thresholds. th1 says below which intensity we
        # should start to worry. th2 says below which intensity we can
        # be sure it is background. An error measure is calculated 
        # which indicate where between th2 and th1 the value is now.
        # The square of the value is subtracted from a foodsupply.
        # when this supply reaches 0, the walker is killed.
        # Each time we encounter a sample above th1, the food supply
        # is reset to 1.0.
        val = self._manager.data.sample(self.pos)
        th1, th2 = self._manager.params.th1, self._manager.params.th2
        if  val < th1:
            portion = (th1 - val ) / (th1-th2)
            self._foodsupply -= portion**2
            if self._foodsupply <= 0:
                self.Kill("Ran in too low intensity pixels")
        else:
            self._foodsupply = 1.0
    
    
    def DirCom(self, viewdir, normalize=True):
        """ Apply the directional center of mass operator.
        The result depends on a general view direction 
        and the resulting (normalized) direction (comDir) 
        is returned. """
        
        # get data and its scale vector
        data = self._manager.data
        sam = self._manager.data.sampling
        scale = Point(sam[1],sam[0])
        
        # get Gaussian derivative kernels
        sigma = self._manager.params.scale
        sigma2size = 2
        if True:
            ky = -self._manager.GetGaussianKernel(sigma, sigma2size, (1,0) )
            kx = -self._manager.GetGaussianKernel(sigma, sigma2size, (0,1) )
        else:
            g = self._manager.GetGaussianKernel(sigma, sigma2size, (0,0) )
            c = [(i-1)/2 for i in g.shape]
            kx = np.zeros(g.shape,dtype=np.float32)
            ky = np.zeros(g.shape,dtype=np.float32)
            kx[:,:c[1]], kx[:,c[1]+1:] = -1, 1
            ky[:c[0],:], ky[c[0]+1:,:] = -1, 1
            kx, ky = kx*g/scale[1], ky*g/scale[0]
        
        # calculate sze's
        szes = [(s-1)/2 for s in kx.shape]
        sze_y, sze_x = szes[0], szes[1]
        
        # get patch
        patch = self.GetPatch( tuple(szes) )
        if patch is None:
            self.Kill("Out of bounds in getting patch for dirCom.")
            return
        
        # normalize patch (required because kw is asymetric)
        #patch = patch - self._manager.params.th2
        #patch = patch - patch.min()
        patch = patch - sp.ndimage.filters.minimum_filter(patch,3)
        
        # get weighting kernel 
        kw = self._manager.GetWeightKernel(szes, viewdir)
        
        # apply kernels
        dx = patch * kx * kw
        dy = patch * ky * kw
        
        # get center-of-mass and store direction
        # com is initially in voxel coordinates and
        # should be scaled to transform to world coordinates.
        # But not if the gaussian kernels are scaled..
        com = Point(dx.sum(), dy.sum())
        if com.norm()==0:
            com = viewDir
        dir = (com/scale)
        if normalize:
            dir = dir.normalize()
        
        # store stuff for debugging...
        self._kw = Aarray(kw, self._manager.data.sampling)
        self._patch = Aarray(patch, self._manager.data.sampling)
        self._dx = Aarray(dx, self._manager.data.sampling)
        self._dy = Aarray(dy, self._manager.data.sampling)
        self._kx = Aarray(kx, self._manager.data.sampling)
        self._ky = Aarray(ky, self._manager.data.sampling)
        
        return dir
    
    
    def DoStep(self, dir):
        """ Do a step in the direction pointed to by dir. 
        Taking into account pixel values.
        Returns the vector representing the direction in which we
        stepped.
        """
        
        # get data and its scale vector
        data = self._manager.data
        sam = self._manager.data.sampling
        scale = Point(sam[1],sam[0])
        
        # create list of candidates        
        candidates = candidates2.copy()
        
        if self._manager.params.testAngleLimit:
            # select best candidate
            th1, th2 = self._manager.params.th1, self._manager.params.th2
            iy, ix = data.point_to_index(self.pos)
            bestp, bestval = Point(0,0), -99999 # th        
            for p in candidates:
                val = ( data[ iy+p.y, ix+p.x ] - th2 ) / (th1-th2)
                val = max(val,0) * np.cos(dir.angle(p))
                if val > bestval:
                    bestval = val
                    bestp = p
        elif False:            
            # use patch intensities, as they have been normalized
            # can jump between two values
            bestp, bestval = Point(0,0), -999 # th        
            patch = self._patch
            iy, ix = (patch.shape[0]-1)/2, (patch.shape[1]-1)/2
            for p in candidates:
                val = patch[ iy+p.y, ix+p.x ]
                val = val * (np.cos(dir.angle(p))+0.5)
                if val > bestval:
                    bestval = val
                    bestp = p
        else:            
            # Select best candidate. To make sure that we cannot go back to
            # the previous pixel, we use limitingAngle to determine the
            # candidate Angle.
            # 2*candidateAng + limitingAngle < 180
            # candidateAng < 90 - limitingAngle/2
            iy, ix = data.point_to_index(self.pos)
            bestp, bestval = Point(0,0), -99999 # th        
            candidateAng = 89.0 - self._manager.params.limitingAngle/2.0
            candidateAng *= np.pi / 180 # make radians
            for p in candidates:
                if abs( dir.angle(p) ) > candidateAng:
                    continue
                val = data[ iy+p.y, ix+p.x ]            
                if val > bestval:
                    bestval = val
                    bestp = p
        
        # now go there (keep sampling into account)...       
        bestp = bestp * scale
        self.SetPos( self.pos + bestp)
        
        # return step vector (stepdir)
        return bestp
    
    
    def DrawMore(self, f):
        # make current
        vv.figure(f.nr)
        
        if not hasattr(self, '_patch') or not hasattr(self, '_kw'):
            return
        
        vv.subplot(311)
        vv.imshow(self._patch)
        vv.subplot(312)
        vv.imshow(self._kw)
        vv.subplot(313)
        vv.imshow(self._ky)
        


    
class DirComWalker3D(BaseWalker3D):
    """ A walker that walks based on a direction. It keeps
    walking more or less in that direction. By taking small steps
    we prevent it from going of "track" (the stent).    
    """
    
    def __init__(self, manager, p):
        BaseWalker3D.__init__(self, manager, p)
        self._foodsupply = 1.0
    
    
    def GoToMaxInRegion(self, region):
        """ Overloaded version. Calls the original method and
        then determines an initial orientation. """
        BaseWalker3D.GoToMaxInRegion(self, region)
        
        # init dir
        for i in range(5):
            comdir = self.DirCom(self.viewDir)
            if comdir  is not None:
                self.viewDir = comdir
        
        # spawn walker in opposite direction
        spawn = DirComWalker3D(self._manager, self.pos)
        spawn.viewDir = self.viewDir*-1
    
    
    def Move(self):
        # "inherit" from walker2D
        DirComWalker2D.Move.im_func(self)
    
    def DirCom(self, viewdir):
        
        # get data and its scale vector
        data = self._manager.data        
        sam = self._manager.data.sampling
        scale = Point(sam[2],sam[1],sam[0])
        
        # get (anisotropic) Gaussian derivative kernels
        sigma = self._manager.params.scale
        sigma2size = 2
        kz = -self._manager.GetGaussianKernel(sigma, sigma2size, (1,0,0) )
        ky = -self._manager.GetGaussianKernel(sigma, sigma2size, (0,1,0) )
        kx = -self._manager.GetGaussianKernel(sigma, sigma2size, (0,0,1) )
        
        # normalize kernels (if not commented remove scaling below)
        #kz, ky, kx = kz / kz.max(), ky / ky.max(), kx / kx.max()
        
        # calculate sze's
        szes = [(s-1)/2 for s in kx.shape]        
        sze_z, sze_y, sze_x = szes[0], szes[1], szes[2]
        
        # get patch
        patch = self.GetPatch( tuple(szes) )
        if patch is None:
            self.Kill("Out of bounds in getting patch for dirCom.")
            return
        
        # normalize patch (required because kw is asymetric)
        #patch = patch - patch.min()
        patch = patch - sp.ndimage.filters.minimum_filter(patch,3)
        
        # get weighting kernel 
        kw = self._manager.GetWeightKernel(szes, viewdir)
        
        # apply kernels
        dx = patch * kx * kw
        dy = patch * ky * kw
        dz = patch * kz * kw
        
        # get center-of-mass and store direction
        # com is initially in voxel coordinates and
        # should be scaled to transform to world coordinates.
        # But not if the gaussian kernels are scaled..
        com = Point(dx.sum(), dy.sum(), dz.sum())        
        if com.norm()==0:
            com = viewDir
        dir = (com/scale).normalize()
        
        # store stuff for inspection...
        self._kw = Aarray(kw, self._manager.data.sampling)
        self._patch = Aarray(patch, self._manager.data.sampling)
        self._patch2 = Aarray(patch*kw, self._manager.data.sampling)
        self._com = com
        
        return dir
    
    
    def DoStep(self, dir):
        
        # get data and its scale vector
        data = self._manager.data        
        sam = self._manager.data.sampling
        scale = Point(sam[2],sam[1],sam[0])
        
        # create list of candidates
        # represent position change in voxels
        candidates = candidates3.copy()
        
        # Select best candidate. To make sure that we cannot go back to
        # the previous pixel, we use limitingAngle to determine the
        # candidate Angle.
        # 2*candidateAng + limitingAngle < 180
        # candidateAng < 90 - limitingAngle/2
        iz, iy, ix = data.point_to_index(self.pos)
        bestp, bestval = Point(0,0,0), -99999 # th
        candidateAng = 89.0 - self._manager.params.limitingAngle/2.0
        candidateAng *= np.pi / 180 # make radians
        for p in candidates:
            if abs( dir.angle(p) ) > candidateAng:
                continue
            val = data[ iz+p.z, iy+p.y, ix+p.x ]            
            if val > bestval:
                bestval = val
                bestp = p
        
        # now go there...                
        bestp =  bestp * scale
        self.SetPos( self.pos + bestp )
        return bestp
    
    
    def DrawMore(self, f):
        # make current
        vv.figure(f.nr)
        
        if not hasattr(self, '_patch') or not hasattr(self, '_kw'):
            return
        
        a=vv.subplot(311)
        vv.volshow(self._patch)
        a.daspect = 1,-1,-1
        a=vv.subplot(312)
        vv.volshow(self._kw)
        a.daspect = 1,-1,-1
        a=vv.subplot(313)
        vv.volshow(self._patch2)
        a.daspect = 1,-1,-1
        tmp = Pointset(3)
        sam = self._manager.data.sampling
        shape = self._patch.shape        
        c = Point(shape[2],shape[1],shape[0]) * Point(sam[2],sam[1],sam[0]) * 0.5
        tmp.append(c)
        tmp.append(c+self.dir*4)
        p=vv.plot(tmp)
        p.alpha = 0.5



class DirComWithRingWalker2D(DirComWalker2D):
    """ A better version of the DirComWalker.
    It uses a second direction term based on sampling local maxima
    in a ring around the current position.
    The final direction is a linear combination of the walkDir, ringDir 
    and ringDir, and is thereafter limited.
    Also, the kernel is used differently, but for this one needs to
    change the if-statement in DirComWalker2D.DirCom().
    """
    
    def DirRing(self):
        """ Find the position of the wire penetrating a ring
        around the current position. If there are two such
        positions, return the sum of the vectors to them. """
        
        # get data and its scale vector
        data = self._manager.data
        sam = self._manager.data.sampling
        scale = Point(sam[1],sam[0])
        
        # get COM kernels
        sigma = self._manager.params.scale
        sigma2size = 2
        g = self._manager.GetGaussianKernel(sigma, sigma2size, (0,0) )
        
        # calculate sze's
        szes = [(s-1)/2 for s in g.shape]
        sze_y, sze_x = szes[0], szes[1]
        
        # get patch
        patch = self.GetPatch( tuple(szes) )
        if patch is None:
            self.Kill("Out of bounds in getting patch for DirRing.")
            return
        
        # only keep the edges
        # todo: this mask can be calculated beforehand
        patch = Aarray(patch+0, sampling=sam)
        patch.origin = -sze_y * sam[0], -sze_x * sam[1]
        dref = patch.index_to_point(sze_y,0).norm()
        for y in range(patch.shape[0]):
            for x in range(patch.shape[1]):
                d = patch.index_to_point(y, x).norm()
                if d < dref-1 or d >= dref:
                    patch[y,x] = 0
        
        # get high local maxima. 
        mask = ( patch - sp.ndimage.filters.maximum_filter(patch,3) ) == 0
        patch[mask==0] = 0
        patch[patch<self._manager.params.th1] = 0
        
        # show
        self._patch2 = patch
        
        # if there are two pixels, create a vector!
        p = Point(0,0)
        Iy, Ix = np.where(patch>0)
        if len(Iy) >= 2 and len(Iy) < 3:
            for i in range(len(Ix)):
                tmp = patch.index_to_point(Iy[i], Ix[i])
                p = p + tmp.normalize()
        
        # Done
        return p
    
    
    def Move(self):
        """ Overloaded move method. """
        
        # apply directional center of mass operator
        comDir = self.DirCom(self.viewDir)
        if comDir is None:
            return
        
        # get full center of mass
        ringDir = self.DirRing()
        
        # get walkdir        
        if not self.history:
            walkdir = self.viewDir
        else:
            refpos = self.history[-2:][0]
            walkdir =  self.pos - refpos 
            if walkdir.norm()>0:
                walkdir = walkdir.normalize()
        
        # combine
        oldViewDir = self.viewDir
        params = self._manager.params
        w0, w1, w2 = params.comWeight, params.ringWeight, params.historyWeight
        self.viewDir = comDir*w0 + ringDir*w1 + walkdir*w2
        self.viewDir = self.viewDir.normalize()
        
        # apply limit to angle
        limitingAngle = self._manager.params.limitingAngle
        self.viewDir = self._LimitAngle(self.viewDir, oldViewDir, limitingAngle)
        limitingAngle = self._manager.params.limitingAngle2
        self.viewDir = self._LimitAngle(self.viewDir, walkdir, limitingAngle)
        
        # Do a step in that direction        
        self.viewDir = self.viewDir.normalize()        
        stepDir = self.DoStep(self.viewDir)
        
        # combining walkdir and fullcom: the walkdir is "reset" each time
        # by rounding to the voxel, and therefore does not bend along.
        
        # what do we visualize?
        self.dir = self.viewDir        
        #self.dir = self.DirCom(self.viewDir) * 0.01
        #self.dir = walkdir #(not used anymore)
        
        # test if we are ok here...
        # There are two thresholds. th1 says below which intensity we
        # should start to worry. th2 says below which intensity we can
        # be sure it is background. An error measure is calculated 
        # which indicate where between th2 and th1 the value is now.
        # The square of the value is subtracted from a foodsupply.
        # when this supply reaches 0, the walker is killed.
        # Each time we encounter a sample above th1, the food supply
        # is reset to 1.0.
        val = self._manager.data.sample(self.pos)
        th1, th2 = self._manager.params.th1, self._manager.params.th2
        if  val < th1:
            portion = (th1 - val ) / (th1-th2)
            self._foodsupply -= portion**2
            if self._foodsupply <= 0:
                self.Kill("Ran in too low intensity pixels")
        else:
            self._foodsupply = 1.0

    
class DirComWithRingWalker3D(DirComWalker3D):
    
    def DirRing(self):
        """ Find the position of the wire penetrating a ring
        around the current position. If there are two such
        positions, return the sum of the vectors to them. """
        
        # get data and its scale vector
        data = self._manager.data
        sam = self._manager.data.sampling
        scale = Point(sam[2], sam[1],sam[0])
        
        # get COM kernels
        sigma = self._manager.params.scale
        sigma2size = 2
        g = self._manager.GetGaussianKernel(sigma, sigma2size, (0,0,0) )
        
        # calculate sze's
        szes = [(s-1)/2 for s in g.shape]
        sze_z, sze_y, sze_x = szes[0], szes[1], szes[2]
        
        # get patch
        patch = self.GetPatch( tuple(szes) )
        if patch is None:
            self.Kill("Out of bounds in getting patch for DirRing.")
            return
        
        # only keep the edges
        # todo: this mask can be calculated beforehand
        patch = Aarray(patch+0, sampling=sam)
        patch.origin = -sze_z * sam[0], -sze_y * sam[1], -sze_x * sam[2]
        dref = patch.index_to_point(sze_y,0,0).norm()
        for z in range(patch.shape[0]):
            for y in range(patch.shape[1]):
                for x in range(patch.shape[2]):
                    d = patch.index_to_point(z,y,x).norm()
                    if d < dref-1 or d >= dref:
                        patch[z,y,x] = 0
        
        # get high local maxima. 
        mask = ( patch - sp.ndimage.filters.maximum_filter(patch,3) ) == 0
        patch[mask==0] = 0
        patch[patch<self._manager.params.th1] = 0
        
        # show
        self._patch2 = Aarray(patch, self._manager.data.sampling)
        
        # if there are two pixels, create a vector!
        p = Point(0,0,0)
        Iz, Iy, Ix = np.where(patch>0)
        if len(Iy) >= 2 and len(Iy) < 3:
            for i in range(len(Ix)):
                tmp = patch.index_to_point(Iz[i], Iy[i], Ix[i])
                p = p + tmp.normalize()
        
        # Done
        return p
    
    
    def Move(self):
        # "inherit" from walker2D
        Testing2D.Move.im_func(self)


class MPCWalker2D(BaseWalker2D):
    
    def __init__(self, manager, p):
        BaseWalker2D.__init__(self, manager, p)
        
        # get params
        ctvalues2double = self._manager.params.ctvalues2double
        mcpDistance = self._manager.params.mcpDistance 
        
        # create mcp object if required
        if not hasattr(self._manager, 'mcp'):
            speed = 1/2**(self._manager.data/ctvalues2double)
            self._manager.mcp = mcp.McpDistance(speed, 0, mcpDistance)
        
        # dont show a vector
        self.dir = None
        
        # keep a path to walk
        self._future = Pointset(self._manager.data.ndim)
        self._distance = 0
    
    def SetPos(self, pos):
        # set this pos
        if pos is not None:
            #self.RememberPos(pos)
            BaseWalker2D.SetPos(self, pos)
    
    def RememberPos(self, pos, sze=0):
        """ Remember this pos as a position where we've been,
        so we cannot go there again. """
        mask = self._manager.mask
        iy, ix = self._manager.data.point_to_index(pos)
        for dy in range(-sze,sze+1):
            for dx in range(-sze,sze+1):
                y, x = iy+dy, ix+dx
                if y<0 or x<0 or y>=mask.shape[0] or x>=mask.shape[1]:
                    continue
                mask[y,x] = 1
    
    def Move(self):
        
        # todo: only in patch (but nasty to take edges into account...)
        
        # do we have some path to walk left over?
        maxdist = self._manager.params.mcpDistance/2.0
        if len(self._future) and self._distance < maxdist:
            p = self._future.pop()
            self._distance += p.distance(self.pos)
            self.SetPos( p )
            return
        else:
            self._distance = 0
            
        
        m = self._manager.mcp
        
        # reset mcp object
        m.Reset(self.pos)
        
        # freeze the voxels that we came from
        if self._history:
            for pos in self._history[-20:]:
                ii = m.MakeIntPos(pos)
                m.nindex_f[ ii ] = - abs(m.nindex_f[ ii ])
            ii = m.MakeIntPos(self._history[-1])
            m.nindex_f[ ii ] = - abs(m.nindex_f[ ii ])
            for n in m.GetNeighbors(ii):                
                m.nindex_f[ n ] = - abs(m.nindex_f[ n ])
                
        
        # lets go!
        m.EvolveFront()
        if m._endpoint is None:
            self.Kill("No stent to follow.")
            return
        path = m.GetPathAsPoints(m._endpoint)
        
        # store
        self._future = path[:-1]
        self.Move() # do one step
        
        # add to history
#         d = 0        
#         for p in reversed(path[:-1]):
#             if p == self.pos:
#                 print('same one')
#             self.SetPos(p)
#             d += p.distance(self.pos)
#             if d > self._manager.params.mcpDistance/2.0:
#                 break
    

class MPCWalker3D(BaseWalker3D):
    
    def __init__(self, manager, p):
        BaseWalker3D.__init__(self, manager, p)
        
        # get params
        ctvalues2double = self._manager.params.ctvalues2double
        mcpDistance = self._manager.params.mcpDistance 
        
        # create mcp object if required
        if not hasattr(self._manager, 'mcp'):
            speed = 1/2**(self._manager.data/ctvalues2double)
            self._manager.mcp = mcp.McpDistance(speed, 0, mcpDistance)
        
        # don't show a direction vector
        self.dir = None
        
        # keep a path to walk
        self._future = Pointset(self._manager.data.ndim)
        self._distance = 0
    
    def SetPos(self, pos):
        # set this pos
        if pos is not None:
            #self.RememberPos(pos)
            BaseWalker3D.SetPos(self, pos)
    
    def Move(self):
        m = self._manager.mcp
        
        # todo: only in patch (but nasty to take edges into account...)
        
        # do we have some path to walk left over?
        maxdist = self._manager.params.mcpDistance/2.0
        if len(self._future) and self._distance < maxdist:
            p = self._future.pop()
            self._distance += p.distance(self.pos)
            self.SetPos( p )
            return
        else:
            self._distance = 0
        
        # reset mcp object        
        m.Reset(self.pos)
        
        # freeze the voxels that we came from
        # todo: this can probably be done more efficiently
        if self._history:
            for pos in self._history[-50:]:
                ii = m.MakeIntPos(pos)
                m.nindex_f[ ii ] = - abs(m.nindex_f[ ii ])
            ii = m.MakeIntPos(self._history[-1])
            m.nindex_f[ ii ] = - abs(m.nindex_f[ ii ])
            for n in m.GetNeighbors(ii):
                m.nindex_f[ n ] = - abs(m.nindex_f[ n ])
        
        # look around
        th1 = self._manager.params.th1 / 2
        tmp = 2**(th1/self._manager.params.ctvalues2double)
        m.EvolveFront(-1,tmp)
        if m._endpoint is None:
            self.Kill("No stent to follow.")
            return
        
        # backtrack
        path = m.GetPathAsPoints(m._endpoint)
        
        # store
        self._future = path[:-1]
        self.Move() # do one step
        