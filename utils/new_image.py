class Image(np.ndarray):
    
    def __new__(cls, array):
        return array.view(cls)
    
    def __repr__(self):
        n = 'x'.join([str(i) for i in self.shape])
        dtype = self.dtype
        ndim = self.ndim
        if self.shape[-1] in (1,3,4):
            ndim -= 1
        return '<%iD image: numpy array with %s elements of dtype %s>' % (ndim, n, dtype)
    
    def __array_wrap__(self, out, context=None):
        """ So that we return a native numpy array (or scalar) when a
        reducting ufunc is applied (such as sum(), std(), etc.)
        """
        if not out.shape:
            return out.dtype.type(out)  # Scalar
        elif out.shape != self.shape:
            return np.asarray(out)
        else:
            return out  # Type Image
    
    def __getitem__(self, index):
        """ Get a point or part of the pointset. """
        # Single index from numpy scalar
        if isinstance(index, np.ndarray) and index.size==1:
            index = int(index)
        
        if isinstance(index, tuple):
            # Multiple indexes: return as array
            return np.asarray(self)[index]
        elif isinstance(index, slice):
            # Slice: return subset
            return np.ndarray.__getitem__(self, index)
        elif isinstance(index, int):
            ndim = self.ndim
            if ndim and self.shape[-1] in (1,3,4):
                ndim -= 1
            if ndim > 1:
                return np.ndarray.__getitem__(self, index)
            else:
                return np.asarray(self)[index]  # Back to normal
        else:
            # Probably some other form of subslicing
            return np.asarray(self)[index]
