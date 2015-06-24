class PEPoint(complex):
    """
    A python complex number, with hints for the plotter.
    """
    def __new__(cls, *args, **kwargs):
        leave_gap = kwargs.pop('leave_gap', False)
        marker = kwargs.pop('marker', '')
        obj = complex.__new__(cls, *args, **kwargs )
        obj.leave_gap = leave_gap
        obj.marker = marker
        return obj

    def __add__(self, other):
        return PEPoint(complex.__add__(self, other),
                       leave_gap=self.leave_gap,
                       marker=self.marker)
    def __radd__(self, other):
        return PEPoint(complex.__radd__(self, other),
                       leave_gap=self.leave_gap,
                       marker=self.marker)
    def __sub__(self, other):
        return PEPoint(complex.__sub__(self, other),
                       leave_gap=self.leave_gap,
                       marker=self.marker)
    def __rsub__(self, other):
        return PEPoint(complex.__rsub__(self, other),
                       leave_gap=self.leave_gap,
                       marker=self.marker)

