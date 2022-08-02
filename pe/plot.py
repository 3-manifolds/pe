from __future__ import print_function

import matplotlib
backend = matplotlib.get_backend()
if  backend == 'TkAgg':
    matplotlib.use('tkagg')
    from .figure import MatplotFigure, Tk, ttk
elif backend.endswith('module://ipympl.backend_nbagg'):
    from .figure import MatplotFigure
else:
    matplotlib.use('nbagg')
    from .figure import MatplotFigure

from .point import PEPoint
from .input import user_input
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
        self.color_dict = kwargs.get('colors', {})
        self.button_dict = kwargs.get('buttons', {})
        if self.color_dict:
            self.num_colors = len(set(self.color_dict.values()))
        else:
            self.num_colors = len(self.data)
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
        groups = collections.OrderedDict()
        for i, component in enumerate(self.data):
            color = self.color(i)
            X = attribute_map(component, 'real')
            Y = attribute_map(component, 'imag')
            arc = axis.plot(X, Y, color=color, label='%d' % i)
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
            if color in groups:
                groups[color] += arc_items
            else:
                groups[color] = arc_items
        self.arc_views = list(groups.values())
        self.configure()
        self.init_backend()

    def configure(self):
        """Configure the plot based on keyword arguments."""
        figure = self.figure
        axis = figure.axis
        xlim, ylim = self.args.get('limits', (None, None))
        xlim = axis.get_xlim() if xlim is None else xlim
        ylim = axis.get_ylim() if ylim is None else ylim
        margin_x, margin_y = self.args.get('margins', (0.1, 0.1))
        sx = (xlim[1] - xlim[0])*margin_x
        xlim = (xlim[0] - sx, xlim[1] + sx)
        sy = (ylim[1] - ylim[0])*margin_y
        ylim = (ylim[0] - sy, ylim[1] + sy)
        position = self.args.get('position', None)
        if position:
            axis.set_position(position)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect(self.args.get('aspect', 'auto'))
        self.create_legend(axis)
        self.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_hover)

    def create_legend(self, axis):
        handles, labels = [], []
        for n, group in enumerate(self.arc_views):
            handles.append(group[0])
            labels.append('%d'%n)
        self.legend = axis.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.0, 1.0))

    def color(self, i):
        n = self.color_dict.get(i, i)
        return matplotlib.cm.brg(float(1+n)/self.num_colors)

    def create_plot(self, dummy_arg=None):
        axis = self.figure.axis
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

class TkPlot(PlotBase):

    def init_backend(self):
        self.arc_vars = collections.OrderedDict()
        for view in self.arc_views:
            var = Tk.BooleanVar(self.figure.window, value=True)
            var.trace('w', self.arc_button_callback)
            var.arc = view
            self.arc_vars[str(var)] = var
        title = self.args.get('title', None)
        figure, axis, window = self.figure, self.figure.axis, self.figure.window
        if title:
            window.title(title)
            figure.set_title(title)
            #axis.title.set_position(0.5, 1.1)
        func_selector_frame = ttk.Frame(window)
        uncheck = Tk.Button(func_selector_frame, text='X', fg='red',
                            padx=3, pady=0, command=self.clearall)
        uncheck.grid(column=0, row=0, sticky=(Tk.N, Tk.W))
        for i, var in enumerate(self.arc_vars):
            label = self.button_dict.get(i, '%d'%i)
            button = ttk.Checkbutton(func_selector_frame,
                                     text=label, variable=var)
            button.grid(column=0, row=i+1, sticky=(Tk.N, Tk.W))
        func_selector_frame.grid(column=1, row=0, sticky=(Tk.N))
        window.columnconfigure(1, weight=0)

    def on_pick(self, event):
        num_points = len(event.ind)
        print(num_points, 'points; choosing', end=' ')
        median = event.ind[num_points // 2]
        print(self.scatter_point_to_raw_data[event.artist][median])

    def on_hover(self, event):
        pass
#       The hitlist was removed in matplotlib 3.1.0 with no replacement.    
#        for verts in self.vertex_sets:
#            if verts.hitlist(event):
#                self.figure.set_cursor('hand1')
#                return
#        self.figure.unset_cursor()

    def arc_button_callback(self, var_name, *args):
        var = self.arc_vars[var_name]
        if var.get():
            for subarc in var.arc:
                self.figure.axis.add_artist(subarc)
        else:
            for subarc in var.arc:
                subarc.remove()
        self.figure.draw()

    def clearall(self, *args):
        for var in self.arc_vars.values():
            if var.get():
                var.set(False)

class NbPlot(PlotBase):

    def init_backend(self):
        title = self.args.get('title', None)
        if title:
            self.figure.set_title(title)
        self.init_clickable_legend()
        ax = self.figure

    def init_clickable_legend(self):
        """
        Setup legend to turn off and on arcs.
        """
        self.legend_to_arc_view = leg_to_view = dict()
        for leg_line, arc_view in zip(self.legend.get_lines(), self.arc_views):
            leg_to_view[leg_line] = arc_view
            leg_line.set_picker(True)
            leg_line.set_pickradius(5)
        self.picklog = []

    def on_legend_click(self, artist):
        views = self.legend_to_arc_view[artist]
        new_vis = not views[0].get_visible()
        for view in views:
            view.set_visible(new_vis)
        if new_vis:
            artist.set_alpha(1.0)
        else:
            artist.set_alpha(0.2)
        self.figure.draw()

    def on_pick(self, event):
        artist = event.artist
        if artist in self.legend_to_arc_view:
            self.on_legend_click(artist)
        else:
            num_points = len(event.ind)
            median = event.ind[num_points // 2]
            index = self.scatter_point_to_raw_data[artist][median].index
            message = 'Point index ' + repr(index)
            if num_points > 1:
                message += ' (%d others nearby)' % (num_points - 1)
            self.figure.toolbar.set_message(message)


if  backend == 'TkAgg':
    Plot = TkPlot
else:
    Plot = NbPlot

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
