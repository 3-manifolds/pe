import time, sys, os, Tkinter, numpy, math, collections
from subprocess import Popen, PIPE
try:
    from tkplot import MatplotFigure, Tk, ttk
except ImportError:
    pass
from point import PEPoint
from collections import defaultdict

class Plot:
    """
    Plot a vector or list of vectors. Assumes that all vectors in the list
    are the same type (Float or Complex) Prompts for which ones to show.
    """
    def __init__(self, data, **kwargs):
        self.quiet = kwargs.get('quiet', True)
        self.linewidth=kwargs.get('linewidth', 1.0)
        self.style = kwargs.get('style', 'lines')
        self.color_dict = kwargs.get('colors', {})
        self.args = kwargs
        if isinstance(data, list) and len(data) == 0:
            self.data = data
            self.type = None
        else:
            if isinstance(data[0], list) or isinstance(data[0], numpy.ndarray):
                self.data = data
            else:
                self.data = [data]
            duck = self.data[0][0]
            self.type = type(duck)
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
            self.create_plot( range(len(self.data)) )
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
        """
        Stub for starting up the plotting window. 
        """

    def create_plot(self, funcs):
        """
        Stub for drawing the plot itself.
        """

class SagePlot(Plot):
    def create_plot(self, funcs):
        from sage.all import Graphics, line, colormaps, floor
        cm = colormaps['gnuplot2']

        G = Graphics()
        for f in funcs:
            if self.type == complex:
                points = [(d.real, d.imag) for d in self.data[f]]
            else:
                points = [ (i,d) for i, d in enumerate(self.data[f])]
            G += line(points, color=cm( f/8.0 - floor(f/8.0) )[:3],
                      thickness=self.linewidth, legend_label='%d' % f)
        G.show()

class MatplotPlot(Plot):
         
    def start_plotter(self):
        self.figure = MF = MatplotFigure(add_subplot=False)
        MF.axis = axis = MF.figure.add_axes( [0.07, 0.07, 0.8, 0.9] )
        self.arcs = []
        self.arc_vars = collections.OrderedDict()
        groups = dict()
        for i, component in enumerate(self.data):
            color = self.color_dict.get(i, i)
            if color not in groups:
                groups[color] = []
            arc = groups[color]
            segments = self.split_data(component)
            for X, Y in segments:
                # axis.plot returns a list of line2D objects.
                # we only assign a label to the first thing in each group
                if len(arc) == 0:
                    arc += axis.plot(X, Y, color=self.color(color),
                                     linewidth=self.linewidth, label='%d' % color)
                else:
                    arc += axis.plot(X, Y, color=self.color(color),
                                     linewidth=self.linewidth)
            if self.args.get('show_group', False):
                point_dict = defaultdict(list)
                for p in component:
                    if p.marker:
                        point_dict[p.marker].append(p)
                for marker in point_dict:
                    arc.append(axis.scatter([p.real for p in point_dict[marker]],
                                            [p.imag for p in point_dict[marker]],
                                            c=self.color(color), marker=marker))
                                    
        for color in groups:
            var = Tk.BooleanVar(MF.window, value=True)
            var.trace('w', self.arc_button_callback)
            var.arc = groups[color]
            self.arc_vars[var._name] = var
        self.configure()
                    
    def configure(self):
        """
        Configure the plot based on keyword arguments.
        """
        figure = self.figure
        axis = figure.axis
        window = figure.window
        limits = self.args.get('limits', None)
        xlim, ylim = limits if limits else (axis.get_xlim(), axis.get_ylim())

        margin_x, margin_y = self.args.get('margins', (0.1, 0.1))
        sx = ( xlim[1] - xlim[0])*margin_x
        xlim = (xlim[0] - sx, xlim[1] + sx)
        sy = (ylim[1] - ylim[0])*margin_y
        ylim = (ylim[0] - sy, ylim[1] + sy)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)

        axis.set_aspect(self.args.get('aspect', 'auto'))
        legend = axis.legend(loc='upper left', bbox_to_anchor = (1.0, 1.0))

        title = self.args.get('title', None)
        if title:
            figure.window.title(title)
            axis.text(0.02, 0.98, title, 
                      horizontalalignment='left', verticalalignment='top',
                      transform=axis.transAxes, fontsize=15)
            


        n = len(self.data)
        func_selector_frame = ttk.Frame(window)
        for i, var in enumerate(self.arc_vars):
            button = ttk.Checkbutton(func_selector_frame,
                                     text='%d'% i, variable=var)
            button.grid(column=0, row=i, sticky=(Tk.N, Tk.W))
        func_selector_frame.grid(column=1, row=0, sticky=(Tk.N))
        window.columnconfigure(1, weight=0)

    def arc_button_callback(self, var_name, *args):
        var = self.arc_vars[var_name]
        if var.get():
            for subarc in var.arc:
                self.figure.axis.add_artist(subarc)
        else:
            for subarc in var.arc:
                subarc.remove()
        self.figure.draw()

    def test(self):
        return [v.get() for v in self.funcs_to_show]

    def color(self, i):
        from matplotlib.cm import gnuplot2
        return gnuplot2(i/8.0 - math.floor(i/8.0))

    def split_data(self, data):
        """
        The data has None entries between points which should not
        be connected by arcs in the picture.  For example, in the case of a curve
        on a pillowcase the breaks are inserted when the curve wraps over an
        edge of the pillowcase.
        This method splits the data at the None entries, and builds
        the x and y lists for the plotter.

        """
        result = []
        x_list, y_list = [], []
        if self.type == PEPoint:
            for d in data:
                x_list.append(d.real)
                y_list.append(d.imag)
                if d.leave_gap and len(x_list) > 1:
                    result.append( (x_list, y_list) )
                    x_list, y_list = [], []
        elif self.type == complex:
            for d in data:
                if d is None and len(x_list) > 1:
                    result.append( (x_list, y_list) )
                    x_list, y_list =[], []
                else:
                    x_list.append(d.real)
                    y_list.append(d.imag)
        else:
            for n, d in enumerate(data):
                if d is None and len(x_list) > 1:
                    result.append( (x_list, y_list) )
                    x_list, y_list =[], []
                else:
                    x_list.append(n)
                    y_list.append(d)
        result.append( (x_list, y_list) )
        return result
                    
    def create_plot(self, dummy_arg=None):
        axis = self.figure.axis

        # Configure the plot based on keyword arguments
        margin_x, margin_y = self.args.get('margins', (0.1, 0.1))
        limits = self.args.get('limits', None)
        xlim, ylim = limits if limits else (axis.get_xlim(), axis.get_ylim())
        sx = ( xlim[1] - xlim[0])*margin_x
        sy = (ylim[1] - ylim[0])*margin_y
        xlim = (xlim[0] - sx, xlim[1] + sx)
        ylim = (ylim[0] - sy, ylim[1] + sy)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)

        axis.set_aspect(self.args.get('aspect', 'auto'))
        legend = axis.legend(loc='upper left', bbox_to_anchor = (1.0, 1.0))

        title = self.args.get('title', None)
        if title:
            self.figure.window.title(title)

        extra_lines = self.args.get('extra_lines', None)
        if extra_lines:
            extra_line_args = self.args.get('extra_line_args', {})
            for xx, yy in extra_lines:
                self.draw_line(xx, yy, **extra_line_args)
            
        self.figure.draw()

    def draw_line(self, xx, yy,  **kwargs):
        ax = self.figure.axis
        if 'color' not in kwargs:
            kwargs['color'] = 'black'
        ax.plot( xx, yy, **kwargs)
        ax.plot( xx, yy, **kwargs)

    def show_plots(self):
        self.create_plot()
    

if __name__ == "__main__":
    MyPlot = MatplotPlot
    #float_data = numpy.random.random( (10,) )
    #MyPlot(float_data)
    P = MyPlot([[ a+b*1j for a, b in numpy.random.random( (10,2) )] for i in range(15)])
