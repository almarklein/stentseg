
class DeformInfo(list):
    """ Little class that enables storing a collection of numbers and show
    aggregates of it.
    """

    def __init__(self, values=None, unit=""):
        self.unit = unit
        for v in values or []:
            self.append(v)

    def __repr__(self):
        r = super().__repr__()
        return "<Deforminfo {} {}>".format(r, self.unit)

    def append(self, value):
        super().append(float(value))

    @property
    def min(self):
        return min(self)

    @property
    def max(self):
        return max(self)

    @property
    def mean(self):
        return sum(self) / len(self)

    @property
    def percent(self):
        return 100 * (self.max - self.min) / self.min

    @property
    def summary(self):
        if self.min < 1000:
            s = "{:0.4g} - {:0.4g}".format(self.min, self.max)
        else:
            s = "{:0.0f} - {:0.0f}".format(self.min, self.max)
        if self.unit:
            s += " " + self.unit
        if self.min > 0:
            s += " ({:0.1f}%)".format(self.percent)

        return s


if __name__ == "__main__":

    x = DeformInfo(3, 4.123, 5, 6, unit="cm")
    print(x)