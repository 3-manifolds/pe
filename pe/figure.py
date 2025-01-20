"""
This module exports the MatplotFigure class, which comes in different flavors
depending on the matplotlib backend.  Before importing this module, the
backend should have been selected using matplotlib.use().

Plotting using matplotlib and Tkinter.
---------------------------------------

Matplotlib does all the 2D graphics for Sage, but unfortunately none of
its GUI backends are compiled by default.  The following suffices to
compile the Tk backend on Linux, provided you have the tk-dev(el)
package installed:

  export SAGE_MATPLOTLIB_GUI=yes; sage -f matplotlib

This doesn't work on OS X because in addition to TkAgg, Sage will try to
compile the native Mac backend, which fails since Sage doesn't include
an Objective-C compiler.  For OS X, download the source tarball for
matplotlib, and then do

  cd matplotlib-*
  sage -sh echo '[gui_support]' >> setup.cfg
  echo 'macosx=false' >> setup.cfg
  python setup.py install

Also, if one has freetype or libpng installed via brew, one should temporarily
unlink them to avoid conflicting with Sage's internal version.

Note that the TkAgg backend will be compiled against one's current
version of Tk, which might not be the one that Sage is linked against.
So inaddition one may need to recompile the Tkinter module via

   sage -f python
"""

import matplotlib
import matplotlib.pyplot
backend = matplotlib.get_backend()

if backend == 'TkAgg':
    print('Using TkAgg as the matplotlib backend.')
    from IPython import get_ipython
    ip = get_ipython()
    if ip:
        ip.magic("%gui tk")
    import tkinter as Tk
    from tkinter import ttk
    from matplotlib.figure import Figure
    import matplotlib.backends.backend_tkagg as tkagg
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)
elif backend == 'nbAgg':
    print('Using nbAgg as the matplotlib backend.')
elif backend in ['module://ipympl.backend_nbagg', 'widget']:
    print("Using ipympl's nbAgg as the matplotlib backend.")
elif backend == 'inline':
    print('Using inline as the matplotlib backend.')

from matplotlib.figure import Figure

class FigureBase(object):

    def draw(self):
        # Subclasses override this message.
        pass

    def set_title(self, title):
        self.figure.suptitle(title)

    def clear(self):
        self.axis.clear()
        self.draw()

    def set_cursor(self, cursor_name):
        toolbar = self.toolbar
#        tkagg.cursord[1] = cursor_name
        if not toolbar._active:
            toolbar.set_cursor(cursor_name)

    def unset_cursor(self):
        toolbar = self.canvas.toolbar
#        tkagg.cursord[1] = self.default_cursor
        if not toolbar._active:
            toolbar.set_cursor(self.default_cursor)

    def save(self, filename):
        self.figure.savefig(filename, bbox_inches='tight', transparent='true')

    def save_tikz(self, filename, path='plots/'):
        import nplot.tikzplot
        nplot.tikzplot.save_matplotlib_for_paper(self.figure, filename, path)


class NbFigure(FigureBase):
    def __init__(self, add_subplot=True, root=None, size=(10, 6), dpi=72, **kwargs):
        figure = matplotlib.pyplot.figure(figsize=size, dpi=dpi, facecolor='white')
        axis = figure.add_subplot(111) if add_subplot else None
        self.figure, self.axis = figure, axis
        self.canvas = figure.canvas
        self.toolbar = figure.canvas.toolbar
        self.manager = figure.canvas.manager

    def draw(self):
        self.canvas.draw()
        self.manager.show()

class TkFigure(FigureBase):
    def __init__(self, add_subplot=True, root=None, size=(10, 6), dpi=100, **kwargs):
        figure = Figure(figsize=size, dpi=dpi, facecolor='white')
        axis = figure.add_subplot(111) if add_subplot else None
        self.figure, self.axis = figure, axis

        window = Tk.Tk() if root is None else Tk.Toplevel(root)
        figure_frame = ttk.Frame(window)
        W, H = size
        canvas = FigureCanvasTkAgg(figure, master=figure_frame)
        canvas._tkcanvas.config(highlightthickness=0, width=W*dpi, height=H*dpi)
        toolbar = NavigationToolbar2Tk(canvas, figure_frame)
        toolbar.pack(side=Tk.TOP, fill=Tk.X)
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
        toolbar.update()

        figure_frame.grid(column=0, row=0, sticky=(Tk.N, Tk.S, Tk.E, Tk.W))
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        self.window, self.canvas, self.toolbar = window, canvas, toolbar
        self.figure_frame = figure_frame
        self.default_cursor = 'arrow'

    def draw(self):
        self.canvas.draw()

if backend == 'TkAgg':
    MatplotFigure = TkFigure
else:
    MatplotFigure = NbFigure

if __name__ == "__main__":
    from numpy import arange, sin, pi
    import matplotlib
    matplotlib.use('tkagg')
    MF = MatplotFigure()
    t = arange(0.0, 3.0, 0.01)
    s = sin(2*pi*t)
    ans = MF.axis.plot(t, s)
    MF.draw()
    Tk.mainloop()
