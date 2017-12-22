from __future__ import print_function
try:
    from .tkplot import MatplotFigure, Tk, ttk
except ImportError:
    pass

from .point import PEPoint
from .input import user_input
import collections
try:
    from collections import Sequence
except ImportError:
    from collections.abc import Sequence
import numpy

def attribute_map(listlike, attribute):
    return numpy.asarray([getattr(x, attribute, None) for x in listlike])

def expand_leave_gaps_to_nones(points):
    ans = []
    for p in points:
        ans.append(p)
        if p.leave_gap:
            ans.append(None)
    return ans

class PlotBase(object):
    """
    Plot a vector or list of vectors.

    A vector is a sequence in which each non-None element has the same
    type (float, complex, or PEPoint).  (A PEPoint is a complex number
    with additional attributes, such as 'marker' and 'leave_gap'.)
    A None value indicates that the points on either side of the None
    should not be connected in the plot.
    """
    def __init__(self, data, number_type=complex, **kwargs):
        self.linewidth = kwargs.get('linewidth', 1.0)
        self.style = kwargs.get('style', 'lines')
        self.color_dict = kwargs.get('colors', {})
        self.args = kwargs
        self.type = number_type
        if not isinstance(data[0], (Sequence, numpy.ndarray)):
            data = [data]
        if self.type == PEPoint:
            data = [expand_leave_gaps_to_nones(d) for d in data]
        elif self.type != complex:
            data = [[complex(n, z) if z is not None else None for n, z in enumerate(d)]
                    for d in data]
        self.data = data
        self.arcs = []
        self.arc_views = []
        self.vertex_sets = []
        self.scatter_point_to_raw_data = collections.OrderedDict()
        self.start_plotter()
        self.create_plot()

    def __repr__(self):
        return ''

    def init_backend(self):
        # Backend-specific initialization: subclasses override.
        pass
        

    def on_hover(self, event):
        # Subclasses override this handler.
        pass

    def on_pick(self, event):
        # Subclasses override this handler.
        pass
    
    def start_plotter(self):
        self.figure = MF = MatplotFigure(add_subplot=False)
        MF.axis = axis = MF.figure.add_axes([0.07, 0.07, 0.8, 0.9])
        for i, component in enumerate(self.data):
            color = self.color(self.color_dict.get(i, i))
            X = attribute_map(component, 'real')
            Y = attribute_map(component, 'imag')
            arc = axis.plot(X, Y, color=color, label='%d' % i)
            self.arcs.append(arc[0])
            arc_items = [arc[0]]
            verts = axis.scatter(X, Y, s=0.01, color=color, marker='.', picker=3)
            arc_items.append(verts)
            self.vertex_sets.append(verts)
            self.scatter_point_to_raw_data[verts] = component
            if self.args.get('show_group', False):
                markers = attribute_map(component, 'marker')
                for marker in set(markers):
                    if marker is not None:
                        mask = markers != marker
                        marks = axis.scatter(numpy.ma.masked_where(mask, X),
                                             numpy.ma.masked_where(mask, Y),
                                             marker=marker, color=color)
                        arc_items.append(marks)
            self.arc_views.append(arc_items)
        self.configure()
        self.init_backend()

    def configure(self):
        """Configure the plot based on keyword arguments."""
        figure = self.figure
        axis = figure.axis
        window = figure.window
        limits = self.args.get('limits', None)
        xlim, ylim = limits if limits else (axis.get_xlim(), axis.get_ylim())

        margin_x, margin_y = self.args.get('margins', (0.1, 0.1))
        sx = (xlim[1] - xlim[0])*margin_x
        xlim = (xlim[0] - sx, xlim[1] + sx)
        sy = (ylim[1] - ylim[0])*margin_y
        ylim = (ylim[0] - sy, ylim[1] + sy)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)

        axis.set_aspect(self.args.get('aspect', 'auto'))
        axis.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

        self.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)

    @staticmethod
    def color(i):
        from matplotlib.cm import gnuplot2
        return gnuplot2(i/8.0 - numpy.floor(i/8.0))

    def create_plot(self, dummy_arg=None):
        axis = self.figure.axis

        # Configure the plot based on keyword arguments
        margin_x, margin_y = self.args.get('margins', (0.1, 0.1))
        limits = self.args.get('limits', None)
        xlim, ylim = limits if limits else (axis.get_xlim(), axis.get_ylim())
        sx = (xlim[1] - xlim[0])*margin_x
        sy = (ylim[1] - ylim[0])*margin_y
        xlim = (xlim[0] - sx, xlim[1] + sx)
        ylim = (ylim[0] - sy, ylim[1] + sy)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)

        axis.set_aspect(self.args.get('aspect', 'auto'))
        axis.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

        title = self.args.get('title', None)
        if title:
            self.figure.window.title(title)

        extra_lines = self.args.get('extra_lines', None)
        if extra_lines:
            extra_line_args = self.args.get('extra_line_args', {})
            for xx, yy in extra_lines:
                self.draw_line(xx, yy, **extra_line_args)

        self.figure.draw()

    def draw_line(self, xx, yy, **kwargs):
        ax = self.figure.axis
        if 'color' not in kwargs:
            kwargs['color'] = 'black'
        ax.plot(xx, yy, **kwargs)
        ax.plot(xx, yy, **kwargs)
    
    def save(self, filename):
        self.figure.save(filename)

    def save_tikz(self, filename, path='plots/'):
        self.figure.save_tikz(filename, path='plots/')

class Plot(PlotBase):
    
    def init_backend(self):
        self.arc_vars = collections.OrderedDict()
        for view in self.arc_views:
            var = Tk.BooleanVar(self.figure.window, value=True)
            var.trace('w', self.arc_button_callback)
            var.arc = view 
            self.arc_vars[str(var)] = var
        title = self.args.get('title', None)
        if title:
            figure, axis, window = self.figure, self.figure.axis, self.figure.window
            window.title(title)
            axis.text(0.02, 0.98, title,
                      horizontalalignment='left', verticalalignment='top',
                      transform=axis.transAxes, fontsize=15)
        func_selector_frame = ttk.Frame(window)
        for i, var in enumerate(self.arc_vars):
            button = ttk.Checkbutton(func_selector_frame,
                                     text='%d'% i, variable=var)
            button.grid(column=0, row=i, sticky=(Tk.N, Tk.W))
        func_selector_frame.grid(column=1, row=0, sticky=(Tk.N))
        window.columnconfigure(1, weight=0)

    def on_pick(self, event):
        num_points = len(event.ind)
        print(num_points, 'points; choosing', end=' ')
        median = event.ind[num_points // 2]
        print(self.scatter_point_to_raw_data[event.artist][median])

    def on_hover(self, event):
        for verts in self.vertex_sets:
            if verts.hitlist(event):
                self.figure.set_cursor('hand1')
                return
        self.figure.unset_cursor()

    def arc_button_callback(self, var_name, *args):
        var = self.arc_vars[var_name]
        if var.get():
            for subarc in var.arc:
                self.figure.axis.add_artist(subarc)
        else:
            for subarc in var.arc:
                subarc.remove()
        self.figure.draw()


if __name__ == "__main__":
    scattered = numpy.random.random((30, 2))
    zs = [complex(a, b) for a, b in scattered]
    P = Plot(zs[:10])
    data0 = [PEPoint(z, marker='D', index=i) for i, z in enumerate(zs[:10])]
    data1 = [PEPoint(z, marker='x', index=i) for i, z in enumerate(zs[10:])]
    data1[5].leave_gap = True
    data1[15].leave_gap = True
    Q = Plot([data0, data1], show_group=True)
    R = Plot(numpy.random.random(10))
    Tk.mainloop()
