import collections

keyword_defaults = collections.OrderedDict(
    [('index', None), ('marker', ''), ('leave_gap', False)])
    

class PEPoint(complex):
    """
    A python complex number, with hints for the plotter.
    """
    def __new__(cls, *args, **kwargs):
        attrs = dict()
        for kw, default in keyword_defaults.items():
            attrs[kw] = kwargs.pop(kw, default)
        obj = complex.__new__(cls, *args, **kwargs )
        for kw, val in attrs.items():
            setattr(obj, kw, val)
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

    def __str__(self):
        parts = [complex.__repr__(self)]
        for kw, default in keyword_defaults.items():
            val = getattr(self, kw)
            if val != default:
                parts.append(kw + '=' + repr(val))
        return 'PEPoint(' + ', '.join(parts) + ')'


