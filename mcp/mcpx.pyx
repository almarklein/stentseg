
from __future__ import division

# cython specific imports
import numpy as np
cimport numpy as np
import cython 

# more imports. 
# There is an absolute import here; Relative cimport is not supported yet
#from . cimport heap # cimport FastUpdateBinaryHeap
cimport heap
# from stentseg.mcp 
from visvis.pypoints import Aarray, Point, Pointset
import inspect

# cdef extern from "heap2.c":
#     ctypedef struct Heap:
#         int count
#         int levels
#         int popped_ref
#         int pushed
#     Heap* heap_init_wcr(int initial_capcity, int max_reference)
#     void heap_destruct(Heap *self)
#     int heap_push(Heap *self, double value, int reference)
#     int heap_push_wcr(Heap *self, double value, int reference)
#     float heap_pop(Heap *self)
#     void heap_destruct(Heap *self)

 
# determine datatypes for MCP
ctypedef np.float32_t FLOAT_T
FLOAT = np.float32
ctypedef np.int32_t INT_T
INT = np.int32
ctypedef np.int8_t NINDEX_T
NINDEX = np.int8

# this is handy
cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline int int_abs(int a): return -a if a < 0 else a

cdef class McpBase:
    """ McpBase(costs, *args) 
    
    An MPC object instance is an object in which the different
    arrays for calculating the minimum cost path are stored. Also, it
    has several convenience functions, and contains a few methods that
    can be overloaded (by inheriting from this class) to change the
    algorithms behaviour.
    
    The arrays are:
    - costs: the given cost map.
    - cumCosts: the resulting cumulative cost map after calling EvolveFront().
    - traceback: a traceback array for easy backtracking (in theory the
      cumCosts map could be used for this, but due to round-off errors this
      will not always work (the cumCosts map may not monotonically decrease).
    - nindex: a map used to handle the edges correctly (and efficiently).
    
    This class makes use of three different ways to represent a position:
    - point: a Point instance, expressed in the same units as the Aarray
    - tuple: a tuple index, expressed in voxel units
    - int: an integer index in the flat array
    
    To create a new McpObject class, inherit from McpBase and implement
    the _Allocate() and _Reset() methods. To change the behaviour of the
    algorithm, reimplement GoalReached(), Update() and Examine().
    """
    
    cdef readonly object costs, cumCosts, traceback, nindex
    cdef readonly object costs_f, cumCosts_f, traceback_f, nindex_f
    cdef readonly object front
    cdef public object _neighborStuff
    cdef object _args, _kwargs
    cdef readonly object debugInfo
    
    def __init__(self, costs, *args):
        # allocate
        self._Allocate(costs)
        # store args and call reset
        self._args = [arg for arg in args]
        self.Reset()
    
    
    def Reset(self, *args):
        """ Reset(*args), where args are the arguments given to __init__
        minus the costs-argument.
        Reset the MCP object, setting cumCosts array to inf, etc.
        This method will call _Reset() but reuses arguments that 
        were given during initialization (so only arguments that
        you want to change have to be given). KW arguments are not
        allowed.
        Automatically called on __init__().
        """
        
        # get stored args
        args2 = [arg for arg in self._args]
        # insert new args
        for i in range(len(args)):
            args2[i] = args[i]
        # call reset
        self._Reset(*args2)    
    
    
    def _Reset(self):
        """ Reset the MCP object, setting cumCosts array to inf, etc.
        """
        
        # reset cumCosts array
        self.cumCosts_f.fill(np.inf)
        self.traceback_f.fill(-1)
        
        # prepare array of neighbour indices
        self._PrepareNeighbourIndexArray()
        
        # empty front
        while self.front.count: # heap2
            self.front.pop()
        
        # reset neighbourstuff buffer
        self._neighborStuff = {}
   
    
    def _Allocate(self, costs):
        """ _Allocate(costs)
        Allocate the arrays. Called in __init__, before _Reset.
        Overload this if necessary.
        """
        
        # make sure data is float32
        if not isinstance(costs, np.ndarray):
            raise ValueError('Costs must be a float32 numpy array.')
        if not costs.dtype == FLOAT:
            raise ValueError('Costs must be a float32 numpy array.')
        
        # make sure costs is an anisotropic array
        if not isinstance(costs, Aarray):
            costs = Aarray(costs)
        
        # store costs and stuff
        self.costs = costs
        shape, flatshape = costs.shape, costs.size, 
        sampling, origin = costs.sampling, costs.origin
        
        # Make other arrays. cumCosts is the result of the algorithm, nindex
        # is a helper array that for each pixel/voxel, contains the 
        # neighbour index (see docstring of GetNeighbourStuff() for more
        # information).
        self.cumCosts = Aarray(shape, sampling, origin, dtype=FLOAT)
        self.traceback = Aarray(shape, sampling, origin, dtype=INT)
        self.nindex = Aarray(shape, sampling, origin, dtype=NINDEX)
        
        # make flat versions        
        self.costs_f = self.costs.reshape(flatshape)
        self.cumCosts_f = self.cumCosts.reshape(flatshape)
        self.traceback_f = self.traceback.reshape(flatshape)
        self.nindex_f = self.nindex.reshape(flatshape)
        
        # create front: list of integer-positions
        self.front = heap.FastUpdateBinaryHeap(128, costs.size)
    
    ## Positional methods
    
    def MakeIntPos(self, pos):
        """ MakeIntPos(pos)
        Given any kind of pos, return the position as an integer. 
        """
        
        # is this easy?
        if isinstance(pos, (int, np.int32, np.int64)):
            return pos
        
        # make sure to have a tuple
        if isinstance(pos, Point):
            pos = self.costs.point_to_index(pos)
        
        # check
        if not isinstance(pos, tuple):
            tmp = str(pos.__class__)
            raise ValueError('position should be a (point,tuple,int) got '+tmp)
        
        # check dimensions
        shape = self.costs.shape
        if len(shape) > 3 or len(shape)<2:
            raise ValueError('Dimension should be 2d or 2d.')
        if len(shape) != len(pos):
            raise ValueError('Dimension of position does not match data shape.')
        
        for i in range(len(shape)):
            if pos[i] < 0 or pos[i] >= shape[i]:
                return -1
        
        # convert to single index
        if len(shape)==2:
            return pos[1] + shape[1] * pos[0] 
        elif len(shape)==3:
            return pos[2] + shape[2] * ( pos[1] + shape[1] * pos[0] )
    
    
    def MakeTuplePos(self, pos):
        """ MakeTuplePos(pos)
        Given any kind of pos, return the position as a tuple. 
        """
        
        # is this easy?
        if isinstance(pos, tuple):
            return pos
        
        # is this not too hard?
        if isinstance(pos, Point):
            return self.costs.point_to_index(pos)
        
        # or is this a bit of work
        if isinstance(pos, (int, np.int32, np.int64)):
            ii = pos
            
            # translate the singular index to real coords
            shape = self.costs.shape
            if len(shape)==2:
                y = int(ii / shape[1])
                x = ii - y * shape[1]
                pos = y,x
            elif len(shape)==3:
                z = int(ii / (shape[1]*shape[2]))
                ii = ii - z * (shape[1]*shape[2])
                y = int(ii / shape[2])
                x = ii - y * shape[2]
                pos = z,y,x
            else:
                raise ValueError("Invalid dimension.")
            return pos
        else:
            raise ValueError('position should be a point, tuple or int!')
    
    
    def MakePointPos(self, pos):
        """ MakePointPos(pos)
        Given any kind of pos, return the position as an point. 
        """
        
        # is this easy?
        if isinstance(pos, Point):
            return pos
        
        # make sure we have a tuple
        if isinstance(pos, (int, np.int32, np.int64)):
            pos = self.MakeTuplePos(pos)
        
        # check
        if not isinstance(pos, tuple):
            raise ValueError('position should be a point, tuple or int!')
        
        # convert and return
        return self.costs.index_to_point(*pos)
    
    
    ## Methods to handle edges and neighbours
    
    def _getIndex(self, x, y, z=None):
        """ Given the neighbour, represented as either -1,0,1, 
        returns the neighbour index. """
        imap = {-1:1, 0:3, 1:2, None:0}        
        return (imap[x]<<0)+(imap[y]<<2)+(imap[z]<<4) 
    
    def _getAllIndices(self, x, y, z=None):
        """ Given the neighbour, represented as either -1,0,1, 
        return the indices of the lists in which this neighbour
        should be visited. """
        imap = {-1:[2,3], 0:[1,2,3], 1:[1,3], None:[0]}
        return [(bbx<<0)+(bby<<2)+(bbz<<4) 
            for bby in imap[y] for bbx in imap[x] for bbz in imap[z]]
    
    
    def _PrepareNeighbourIndexArray(self):
        """ _PrepareNeighbourIndexArray()
        Set the edges of the nindex array to the appropriate values. 
        """
        
        # Init. Note that the minimum value will always be >0
        self.nindex.fill(0)
        
        # get shape
        shape = self.costs.shape
        
        # init dicts and function
        sliceMap = {-1:0, 0:slice(1,-1), 1:-1}
        
        # set indices
        if len(shape)==2:
            for y in [-1, 0, 1]:
                for x in [-1, 0, 1]:
                    sy,sx = sliceMap[y], sliceMap[x]
                    self.nindex[sy,sx] = self._getIndex(x,y)
        elif len(shape)==3:
            for z in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    for x in [-1, 0, 1]:
                        sz, sy,sx = sliceMap[z], sliceMap[y], sliceMap[x]
                        self.nindex[sz,sy,sx] = self._getIndex(x,y,z)
        else:
            raise ValueError("Can only handle 2D and 3D data.")
    
    
    def GetNeighbors(self, pos):
        """ A convenience method, not to be used when it should go fast!
        """
        ii = self.MakeIntPos(pos)
        NA, WA, offset = self.GetNeighborStuff()
        ni = self.nindex_f[ii]
        return [ii+NA[i] for i in range(offset[ni], offset[ni+1])]
    
    
    def GetNeighborStuff(self):
        """ (NA, WA, offset) = GetNeighborStuff()
        
        Multiple lists of neighbours are created. These lists correspond to
        the different sets of neighbours that we can visit, depending on 
        where we are in the volume (on which edge, or not at edge).
        
        For each dimension there are three options: 
        - the neighbours to the LEFT may be visited 
        - the neighbours to the RIGHT may be visited 
        - the neighbours on both sides may be visited
        Therefore, for 2D and 3D there are 3**ndim is 9 and 27 lists, resp.
        
        Each list is assigned an index. I chose the index to be calculated
        as follows below. The index is applied in this function, but also at 
        _PrepareNeighbourIndexArray, the point where the nindex array initial 
        values are assigned. 
        
        The total index is a number (a word) of 2*ndim bits, with two bits
        reserved for each dimension. These two bits are used to code which
        one of the three options (see list above) is valid.
        - 00 (0) not used
        - 01 (1) left may be visited
        - 10 (2) right may be visited
        - 11 (3) both may be visited
        To calculate the total index for, for example 1,3,2:
        (1<<0) + (3<<2) + (2<<4) = 45
        The maximum value for 3D (which represents not being at any edge):
        (3<<0) + (3<<2) + (3<<4) = 63
        Which fits nicely in a 8 bit integer, so our nindex array can be int8.
        
        Note that this way to produce an index results in maximum indices
        of 4**ndim-1.
        
        The produced lists are concatenated to form a single array. Two 
        index arrays are created to map index->start and index->end in 
        this array. 
        
        This function produces 3 arrays:
        NA neighbours array (concatination of the neighbour lists)
        WA weights array (dito for weights)
        offset map (length 4**ndim+1)
        
        To iterate over the neighbours, given current location ii:
        index = int_abs(nindex_f[ii])
        for i in range(offset[index], offset[index+1]):
            n = ii + NA[i]
            w = WA[i]
        
        """
       
        # init
        sam = self.costs.sampling
        shape = self.costs.shape
        hash = 'x'.join([str(i) for i in shape+sam])
        
        # can we do this quickly?
        if hash in self._neighborStuff:
            return self._neighborStuff[hash]
        
        # calculate entries
        if len(shape)==2:
            
            # init lists of lists (only 3**ndim lists will be filled)
            NA = [[] for i in range(4**2)]
            WA = [[] for i in range(4**2)]
            
            for y in [-1, 0, 1]:                
                for x in [-1, 0, 1]:                    
                    if y==0 and x==0:
                        continue
                    # calculate weight and neighbour as intPos
                    n = y*shape[1] + x
                    w = (y*sam[0])**2 + (x*sam[1])**2
                    # insert in all lists that they belong
                    for index in self._getAllIndices(x,y):
                        NA[index].append(n)
                        WA[index].append(w**0.5)
        
        elif len(shape)==3:
            
            # init lists of lists (only 3**ndim lists will be filled)
            NA = [[] for i in range(4**3)]
            WA = [[] for i in range(4**3)]
            
            for z in [-1, 0 , 1]:
                for y in [-1, 0, 1]:
                    for x in [-1, 0, 1]:
                        if z==0 and y==0 and x==0:
                            continue
                        # calculate weight and neighbour as intPos
                        n = z*shape[1]*shape[2] + y*shape[2] + x
                        w = (z*sam[0])**2 + (y*sam[1])**2 + (x*sam[2])**2
                        # insert in all lists that they belong
                        for index in self._getAllIndices(x,y,z):
                            NA[index].append(n)
                            WA[index].append(w**0.5)
        
        else:
            raise RuntimeError("Data should be 2D or 3D!")
        
        # create full arrays and mapping arrays
        
        # count number of elements in total
        count = 0
        for L in NA:
            count += len(L)
        
        # create full arrays and mapping arrays
        NA2 = np.zeros((count,), dtype=np.int32)
        WA2 = np.zeros((count,), dtype=np.float32)
        offset = np.zeros((len(NA)+1,), dtype=np.int32)
        
        # fill the arrays (offset[0] is 0)
        count = 0        
        for index in range(len(NA)): # i from 0 to 4**ndim
            for n,w in zip(NA[index],WA[index]):
                NA2[count] = n
                WA2[count] = w
                count += 1
            offset[index+1] = count
        
        # done
        self._neighborStuff[hash] = NA2, WA2, offset
        return NA2, WA2, offset
    
    
    def _NeighbourTester(self):
        """ _NeighbourTester()
        Does a kind of simulation to test whether the neighbour 
        system works well, specifically at the edges. It is a pretty dumb 
        and inefficient test, only to apply when the algorithm was changed
        with respect to the handling of neighbours or edges.
        """
        
        NA, WA, offset = self.GetNeighborStuff()
        
        if self.costs.ndim==2:
            for y in range(self.costs.shape[0]):
                for x in range(self.costs.shape[1]):
                    ii = self.MakeIntPos((y,x))
                    
                    # get possible neighbours
                    ni = abs(self.nindex[y,x])
                    Ln = [NA[i]+ii for i in range(offset[ni], offset[ni+1])]
                    for dy in [-1,0,1]:
                        for dx in [-1,0,1]:
                            if dy==0 and dx==0:
                                continue
                            nn = self.MakeIntPos((y+dy,x+dx))
                            # get 
                            tmp = "(%i,%i) (%i,%i)" % (y,x, dy,dx)
                            
                            s = self.costs.shape
                            if y+dy<0 or y+dy>=s[0] or x+dx<0 or x+dx>=s[1]:
                                if nn in Ln:
                                    print tmp,"Found neighbour that should not be there."
                            else:
                                if nn not in Ln:
                                    print tmp, "Missing a neighbour" 
                            
        elif self.costs.ndim==3:
            for z in range(self.costs.shape[0]):
                for y in range(self.costs.shape[1]):
                    for x in range(self.costs.shape[2]):
                        ii = self.MakeIntPos((z,y,x))
                        
                        # get possible neighbours
                        ni = abs(self.nindex[z,y,x])
                        Ln = [NA[i]+ii for i in range(offset[ni], offset[ni+1])]
                        for dz in [-1,0,1]:
                            for dy in [-1,0,1]:
                                for dx in [-1,0,1]:
                                    if dz==0 and dy==0 and dx==0:
                                        continue
                                    nn = self.MakeIntPos((z+dz,y+dy,x+dx))
                                    # get 
                                    tmp = "(%i,%i,%i) (%i,%i,%i)" % (z,y,x, dz,dy,dx)
                                    
                                    s = self.costs.shape
                                    if z+dz<0 or z+dz>=s[0] or y+dy<0 or y+dy>=s[1] or x+dx<0 or x+dx>=s[2]:
                                        if nn in Ln:
                                            print tmp,"Found neighbour that should not be there."
                                    else:
                                        if nn not in Ln:
                                            print tmp, "Missing a neighbour" 
        
    
    ## Methods that are called during the algorithm
    
    
    cpdef int GoalReached(self, int ii, float cumCost):
        """ int GoalReached(self, int ii, float cumCost)
        This method is called each iteration after popping it from
        the heap and freezing it, and before examining the neighbours 
        of this current voxel. 
        
        This method should return 1 if the algorithm should not
        check the current point's neighbours and 2 if the algorithm
        is now done, for example an end point is reached. 
        
        Overload this method to adapt the behaviour of the algorithm.
        """        
        return 0

    cpdef Update(self, int i1, int i2, float w):
        """ Update(self, int i1, int i2, float w)
        This method is called when a node is updated. 
        
        Overload this method to adapt the behaviour of the algorithm.
        """
        pass
    
    cpdef Examine(self, int i1, int i2, float w):
        """ Examine(self, int i1, int i2, float w)
        This method is called for every neighbor examined, even before
        checking whether it is frozen.
        
        Overload this method to adapt the behaviour of the algorithm. 
        """
        pass
    
    
    ## The methods where it's all about
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def EvolveFront(McpBase self, int maxiter=-1):
        """ EvolveFront(maxiter=-1):
        Calculate the map of cumulative costs  to go from the seed point 
        to any point in the grid. 
        If costs is an anisotropic array (Aarray), the anisotropy is taken
        into account in calculating the transition costs.
        """
        
        # get shorter names for object, which are typed
        cdef np.ndarray[FLOAT_T, ndim=1, mode="c", negative_indices=False] costs = self.costs_f        
        cdef np.ndarray[FLOAT_T, ndim=1, mode="c", negative_indices=False] cumCosts = self.cumCosts_f
        cdef np.ndarray[INT_T, ndim=1, mode="c", negative_indices=False] traceback = self.traceback_f
        cdef np.ndarray[NINDEX_T, ndim=1, mode="c", negative_indices=False] nindex = self.nindex_f
        
        # get neighbors stuff
        tmp = self.GetNeighborStuff()
        cdef np.ndarray[INT_T, ndim=1, mode="c", negative_indices=False] NA = tmp[0]
        cdef np.ndarray[FLOAT_T, ndim=1, mode="c",negative_indices=False] NW = tmp[1]
        cdef np.ndarray[INT_T, ndim=1, mode="c", negative_indices=False] offset = tmp[2]
        
        # get short reference to binary heap
        cdef heap.FastUpdateBinaryHeap front = self.front
        
        # define nice types
        cdef int iter1, iter2 # iteretors for current and neighbor voxel
        cdef int i1, i2 # index for current and neighbor voxel
        cdef int ni # neighbor-index (see GetNeighborStuff docstring)
        cdef int tmpi
        #
        cdef float w # weight of transition from current to neighbor
        cdef float cost1, cost2 # cost at current and neighbor voxel
        cdef float cumCost1, cumCost2 # cumCost at current and neighbor voxel
        cdef float inf = np.inf
        
        # start loop
        maxiter += 1
        if maxiter<=0:
            maxiter = 2**30
        try:
            for iter1 in range(maxiter):
                
                # still nodes to process in the front?
                if front.count==0:
                    break
                
                # select voxel in front with smallest cost
                cumCost1 = front.pop_fast()  # heap2
                i1 = front._popped_ref
                
                # get neighbour index: index in the list of neighbour-lists
                ni = int_abs(nindex[i1])
                
                # mark this node as frozen
                nindex[i1] = -ni
                
                # check if goal reached        
                tmpi = self.GoalReached(i1, cumCost1)
                if tmpi>0:
                    if tmpi==1:
                        continue
                    else:
                        #print "Goal reached"
                        break
                
                # visit all neighbors
                for iter2 in range(offset[ni],offset[ni+1]):
                    i2 = i1 + NA[iter2]
                    w = NW[iter2]
                    
                    # let object examine this neighbour
                    self.Examine(i1,i2,w)
                    
                    # frozen voxels do not need to be processed
                    if nindex[i2]>=0:
                    
                        # calculate cumCost and update if smaller
                        cost1 = costs[i1]
                        cost2 = costs[i2]
                        if cost2 < inf:
                            cumCost2 = cumCost1 + 0.5*w*( cost1 + cost2 )
                            if cumCost2 < cumCosts[i2]:
                                # Update
                                cumCosts[i2] = cumCost2
                                traceback[i2] = i1
                                self.Update(i1,i2,w)
                                front.push_fast(cumCost2, i2)
                        else:
                            # Let cumCost[i2] be inf
                            pass
        except Exception:
            self.debugInfo = iter1, iter2, i1, i2
            raise
        # return number of iterations
        return iter1

    
    def GetPath(self, startPoint):
        """ GetPath(self, startPoint)
        Calculate the minimum cost path from the startPoint 
        to the seed point (or to the 'closest' seed point if there were 
        more than one).
        The traceback map in the mcpObject is used to calculate the path (via
        backtracking). Therefore, first run EvolveFront() to obtain it.
        Returns a list of integers.
        """
        
        # init        
        if startPoint is None:
            raise RuntimeError('No valid startPoint given.')
        
        # get flat map
        cdef np.ndarray[INT_T] traceback = self.traceback_f
        
        cdef int i1, next # index for current and next voxel
        
        # init point and start history
        i1 = self.MakeIntPos(startPoint)
        history = []
        
        while True:   
            
            # keep track
            history.append(i1)
            
            # get next index
            next = traceback[i1]
            
            # examine
            if next == i1:
                # we're at the source
                break
            elif next<0:
                self.debugInfo = startPoint, i1, next
                raise Exception('Traceback info not available.')
            else:
                i1 = next
        
        # done
        return history
    
    
    def GetPathAsPoints(self, startPoint):
        """ GetPathAsPoints(self, startPoint)
        Calculate the fastest (minimum cost) path from the startPoint 
        to the seed point (or to the 'closest' seed point if there were 
        more than one).
        The traceback map in the mcpObject is used to calculate the path (via
        backtracking). Therefore, first run EvolveFront() to obtain it.
        Returns a Pointset object.
        """
        
        # get path
        indices = self.GetPath(startPoint)
        
        # make points and return
        pp = Pointset(self.costs.ndim)
        for ii in indices:
            pp.append( self.MakePointPos(ii) )
        return pp
    
    
    def GetPathAndMinCost(self, startPoint):
        """ GetPathAndMinCost(self, startPoint)
        Calculate the minimum cost path from the startPoint 
        to the seed point (or to the 'closest' seed point if there were 
        more than one).
        The backtrack map in the mcpObject is used to calculate the path (via
        backtracking). Therefore, first run EvolveFront() to obtain it.
        Returns tuple (a list of indices, the minimum cost on that path).
        """
        
        # get flat map
        cdef np.ndarray[FLOAT_T] costs = self.costs_f
        
        # get path
        indices = self.GetPath(startPoint)
        
        # find minimum cost
        minCost = np.inf
        for ii in indices:
            tmp = costs[<int>ii]
            if tmp < minCost:
                minCost = tmp
        
        # done
        return indices, minCost
    
    
## Implementations of McpBase

cdef class McpSimple(McpBase):
    """ McpSimple(costs, seedPoints)
    This MCP object provides an initialization method to give a set 
    of seed points. Additionally, it provides a reset method. 
    """
    
    def _Reset(self, seedPoints):
        """ Reset( seedPoints) """
        McpBase._Reset(self)
        
        # if one seedPoint, make list
        if not isinstance(seedPoints, (list, Pointset)):
            seedPoints = [seedPoints]
        
        # discard points beyond edge
        seedPoints2 = []
        for pos in seedPoints:
            pos = self.MakeTuplePos(pos)
            shape = self.costs.shape
            ok = True
            for i in range(len(pos)):
                if pos[i] < 0 or pos[i] >= shape[i]:
                    ok = False
                    break
            if ok:
                seedPoints2.append( pos )
        
        # make integers
        seedPoints2 = [self.MakeIntPos(p) for p in seedPoints2]
        
        # insert seedpoints
        for ii in seedPoints2:
            self.cumCosts_f[ii] = 0
            self.traceback_f[ii]=ii
            self.front.push(0,ii) # heap2

cdef class McpWithEndPoints(McpSimple):
    """ McpWithEndPoints(costs, seedPoints, endPoints)
    This object provides endpoints. The algorithm is stopped if 
    during evolving the front, an endpoint is encountered. 
    """
    
    cdef public object _endPoints
    cdef public object _endpoint
    
    
    def _Reset(self, seedPoints, endPoints=None):
        """ Reset(seedPoints, endPoints) """
        McpSimple._Reset(self, seedPoints)
        
        # process endPoints
        if endPoints is not None:      
            # if single point given, make list
            if not isinstance(endPoints, list):
                endPoints = [endPoints]
            # store as integers
            self._endPoints = [self.MakeIntPos(p) for p in endPoints]
        
        # init endpoint
        self._endpoint = None
    
    
    cpdef int GoalReached(self, int ii, float cumCost):
        """ Stop if we encounter an endpoint """
        if ii in self._endPoints:
            self._endpoint = ii
            return 2
        else:
            return 0


cdef class McpDistance(McpSimple):
    """ McpDistance(costs, seedPoints, distance)
    This object will stop the algorithm if the current point on the
    front has travelled a large enough distance. 
    """
    
    cdef readonly object distance, distance_f    
    cdef float _distance
    cdef public object _endpoint
    
    def _Allocate(self, costs):
        McpSimple._Allocate(self, costs)
        
        # get properties
        shape, flatshape = self.costs.shape, self.costs.size, 
        sampling, origin = self.costs.sampling, self.costs.origin
        
        # create distance array        
        self.distance = Aarray(shape, sampling, origin, 0, dtype=np.float32)
        self.distance_f = self.distance.reshape(flatshape)
    
    
    def _Reset(self, seedPoints, distance):
        McpSimple._Reset(self, seedPoints)
        
        # clear distance array
        self.distance_f.fill(0)
        
        # init reference distance and endpoint
        self._distance = <float>distance
        self._endpoint = None
    
    
    cpdef int GoalReached(self, int ii, float cumCost):
        """ Stop if we have travelled enough distance. """
        if self.distance_f[ii] >= self._distance:
            self._endpoint = ii
            return 2
        else:
            return 0
    
    
    cpdef Update(self, int i1, int i2, float w):
        """ Called when a neighbor is set. """
        cdef np.ndarray[FLOAT_T] distance = self.distance_f
        distance[i2] = distance[i1] + w


cdef class McpConnectedSourcePoints(McpSimple):
    """ McpConnectedSourcePoints(costs, connectedNodes, cumCostThreshold=inf)
    This class is to find the paths that connect the source points.
    To build the resulting geometrical model, a ConnectedNodes object 
    is used (which should already contain the seed points).
    The cumCostThreshold can be used to stop the evolving of the front if the 
    cumulative cost is large enough. 
    """
    
    # declare attributes
    cdef readonly object idmap, idmap_f    
    cdef readonly float _cumCostThreshold
    cdef readonly object nodes
    
    def _Allocate(self, costs):
        McpSimple._Allocate(self, costs)
        
        # get properties
        shape, flatshape = self.costs.shape, self.costs.size, 
        sampling, origin = self.costs.sampling, self.costs.origin
        
        # make idmap array        
        self.idmap = Aarray(shape, sampling, origin, dtype=np.int32)
        self.idmap_f = self.idmap.reshape(flatshape)
    
    
    def _Reset(self, connectedNodes, cumCostThreshold=0):
        McpSimple._Reset(self, connectedNodes)
        
        # also clear idmap
        self.idmap_f.fill(-1)
        
        # init cumCosts threshold
        if cumCostThreshold <= 0:
            cumCostThreshold = np.inf
        self._cumCostThreshold = cumCostThreshold
        
        # init idmap from seed points
        self.nodes = connectedNodes
        self.nodes.ClearEdges()
        for iter in range(len(connectedNodes)):
            pos = connectedNodes[iter]
            self.idmap_f[ self.MakeIntPos(pos) ] = iter
    
    
    cpdef int GoalReached(self, int ii, float cumCost):
        """ Stop if we have travelled enough. """        
        if cumCost > self._cumCostThreshold:            
            return 2
        else:
            return 0
    
    
    cpdef Examine(self, int i1, int i2, float w):
        """ Examine the data, check for ids. """
        # define stuff
        cdef np.ndarray[INT_T] idmap = self.idmap_f
        cdef np.ndarray[FLOAT_T] cumCosts = self.cumCosts_f
        
        # get ids
        cdef int id1 = int(idmap[i1])
        cdef int id2 = int(idmap[i2])
        
        # do we have a match?
        if id2>=0 and id2 != id1:
            #print "found edge", id1, id2
            # we reached the 'front' of another seed point!
            # use backtracking on both sides to establish paths
            # NOTE that the path is traced even if the cost is higher!
            cost = max(cumCosts[i1], cumCosts[i2])
            ii1 = self.GetPath(i1)
            ii2 = self.GetPath(i2)
            self.nodes.CreatePotentialEdge(id1,id2, cost, ii1, ii2)
    
    
    cpdef Update(self, int i1, int i2, float w):
        """ Update the idmap array. """        
        cdef np.ndarray[INT_T] idmap = self.idmap_f
        idmap[i2] = idmap[i1]
        # because idmap and traceback are updated simultaneously,
        # the traceback always runs to the source with the 
        # corresponding id.
        # Since i2 is pushed on the front right after this, examine
        # is bound to be called with i2 resp i1 as arguments.
        # Note that Examine should also be called for frozen neighbours, 
        # since a voxel's id can change if from another source the cumCosts
        # can be smaller.
        
        # IOW: When backtracking, you always end up in the source
        # corresponding to the id of the start voxel. However, because id's 
        # can change, you need to trace the path when detecting it. Id's 
        # should be able to change, because the cumCosts should be able to
        # change (to less), and if not also changing the id, the cumCosts 
        # calculated for next nodes will not be correct.
    
