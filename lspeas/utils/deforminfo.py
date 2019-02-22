
class DeformInfo:
    """ Little class that enables storing per-phase information and show
    aggregates of this.
    """
    
    def __init__(self, *values):
        self.values = [float(v) for v in values]
    
    def __repr__(self):
        stringvalues = ["{:.3g}".format(v) for v in self.values]
        return "<Deforminfo: {}>".format(", ".join(stringvalues))
    
    def append(self, value):
        self.values.append(float(value))
    
    @property
    def min(self):
        return min(self.values)
    
    @property
    def max(self):
        return max(self.values)
    
    @property
    def mean(self):
        return sum(self.values) / len(self.values)
    
    @property
    def percent(self):
        return 100 * (self.max - self.min) / self.min


if __name__ == "__main__":
    
    x = DeformInfo(3, 4.123, 5, 6)
    print(x)