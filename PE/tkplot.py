"""
Plotting using matplotlib and Tkinter.

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

# Load Tkinter
import sys, os
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import ttk

# Fix Sage issue; breaks attach.
try:
    import IPython.lib.inputhook as ih
    ih.clear_inputhook()
except:
    pass

# Load MatplotLib
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg

class MatplotFigure:
    def __init__(self, add_subplot=True, root=None, **kwargs):
        args = kwargs
        figure = matplotlib.figure.Figure(figsize=(10,6), dpi=100)
        figure.set_facecolor('white')
        axis = figure.add_subplot(111) if add_subplot else None
        self.figure, self.axis = figure, axis
        
        window = Tk.Tk() if root is None else Tk.Toplevel(root)
        figure_frame = ttk.Frame(window)
        canvas = FigureCanvasTkAgg(figure, master=figure_frame)
        canvas._tkcanvas.config(highlightthickness=0, width=1000, height=600)
        toolbar = NavigationToolbar2TkAgg(canvas, figure_frame)
        toolbar.pack(side=Tk.TOP, fill=Tk.X)
        canvas._tkcanvas.pack(side=Tk.TOP,  fill=Tk.BOTH, expand=1)
        toolbar.update()
        
        figure_frame.grid(column=0, row=0, sticky=(Tk.N, Tk.S, Tk.E, Tk.W))
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        self.window, self.canvas, self.toolbar = window, canvas, toolbar
        self.figure_frame = figure_frame

    def draw(self):
        self.canvas.draw()

    def clear(self):
        self.axis.clear()
        self.draw()
        

if __name__ == "__main__":
    from numpy import arange, sin, pi
    MF = MatplotFigure()
    t = arange(0.0,3.0,0.01)
    s = sin(2*pi*t)
    ans = MF.axis.plot(t,s)
    Tk.mainloop()
