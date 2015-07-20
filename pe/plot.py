from .tkplot import MatplotFigure, Tk, ttk
try:
    from .tkplot import MatplotFigure, Tk, ttk
except ImportError:
    pass

from .point import PEPoint
import collections
import numpy as np
import time

def attribute_map(listlike, attribute):
    return np.asarray([getattr(x, attribute, None) for x in listlike])

def expand_leave_gaps_to_nones(points):
    ans = []
    for p in points:
        ans.append(p)
        if p.leave_gap:
            ans.append(None)
    return ans

class Plot(object):
    """
    Plot a vector or list of vectors. Assumes that all vectors in the
    list are the same type (float, complex, or PEPoint). Prompts for
    which ones to show.
    """
    def __init__(self, data, **kwargs):
        self.quiet = kwargs.get('quiet', True)
        self.linewidth = kwargs.get('linewidth', 1.0)
        self.style = kwargs.get('style', 'lines')
        self.color_dict = kwargs.get('colors', {})
        self.args = kwargs
        if isinstance(data, list) and len(data) == 0:
            self.type = None
        else:
            if not (isinstance(data[0], list) or isinstance(data[0], np.ndarray)):
                data = [data]
            duck = data[0][0]
            self.type = type(duck)
        if self.type == PEPoint:
            data = [expand_leave_gaps_to_nones(d) for d in data]
        elif self.type != complex:
            data = [[complex(*z) for z in enumerate(d)] for d in data]
        self.data = data
        self.start_plotter()
        if len(self.data) > 0:
            self.show_plots()
        else:
            self.create_plot([0])
            time.sleep(1)

    def __repr__(self):
        return ''

    def show_plots(self):
        if not self.quiet:
            print 'There are %d functions.'%len(self.data)
            print 'Which ones do you want to see?'
        else:
            self.create_plot(range(len(self.data)))
        while 1:
            try:
                stuff = raw_input('plot> ')
                items = stuff.split()
                if len(items) and items[0] == 'all':
                    funcs = range(len(self.data))
                else:
                    funcs = [int(item)%len(self.data) for item in items]
                if len(funcs) == 0:
                    break
            except ValueError:
                break
            print funcs
            self.create_plot(funcs)
        return

    def start_plotter(self):
        """Stub for starting up the plotting window."""

    def create_plot(self, funcs):
        """Stub for drawing the plot itself."""

class MatplotPlot(Plot):
    def start_plotter(self):
        self.figure = MF = MatplotFigure(add_subplot=False)
        MF.axis = axis = MF.figure.add_axes([0.07, 0.07, 0.8, 0.9])
        self.arcs = []
        self.vertex_sets = []
        self.arc_vars = collections.OrderedDict()
        self.scatter_point_to_raw_data = collections.OrderedDict()
        for i, component in enumerate(self.data):
            color = self.color(self.color_dict.get(i, i))
            X = attribute_map(component, 'real')
            Y = attribute_map(component, 'imag')
            arc = axis.plot(X, Y, color=color, linewidth=self.linewidth, label='%d' % i)
            self.arcs.append(arc[0])
            verts = axis.scatter(X, Y, s=0.01, color=color, marker='.', picker=3)
            self.vertex_sets.append(verts)
            self.scatter_point_to_raw_data[verts] = component
            var = Tk.BooleanVar(MF.window, value=True)
            var.trace('w', self.arc_button_callback)
            var.arc = arc + [verts]
            self.arc_vars[str(var)] = var

            if self.args.get('show_group', False):
                markers = attribute_map(component, 'marker')
                for marker in set(markers):
                    if marker is not None:
                        mask = markers != marker
                        marks = axis.scatter(np.ma.masked_where(mask, X),
                                             np.ma.masked_where(mask, Y),
                                             marker=marker, color=color)
                        var.arc.append(marks)
        self.configure()

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

        title = self.args.get('title', None)
        if title:
            figure.window.title(title)
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
        self.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)

    def on_pick(self, event):
        num_points = len(event.ind)
        print num_points, 'points; choosing',
        median = event.ind[num_points // 2]
        print self.scatter_point_to_raw_data[event.artist][median]
        
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

    @staticmethod
    def color(i):
        from matplotlib.cm import gnuplot2
        return gnuplot2(i/8.0 - np.floor(i/8.0))

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

    def show_plots(self):
        self.create_plot()

if __name__ == "__main__":
    scattered = np.random.random((30, 2))
    zs = [complex(a, b) for a, b in scattered]
    P = MatplotPlot(zs[:10])
    data0 = [PEPoint(z, marker='D', index=i) for i, z in enumerate(zs[:10])]
    data1 = [PEPoint(z, marker='x', index=i) for i, z in enumerate(zs[10:])]
    data1[5].leave_gap = True
    data1[15].leave_gap = True
    Q = MatplotPlot([data0, data1], show_group=True)
    R = MatplotPlot(np.random.random(10))
    Tk.mainloop()
