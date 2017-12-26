"""
The PolyViewer class displays the Newton polygon of a 2-variable polynomial,
with vertices colored by the log of the absolute value of the coefficient.
"""
import colorsys, matplotlib
from matplotlib import units, ticker
from matplotlib.cbook import iterable
from .figure import MatplotFigure
    
def color_string(h, s, v):
    """
    Return an html style color string from HSV values in [0.0, 1.0]
    """
    r, g, b = colorsys.hsv_to_rgb((.03125 + h)%1.0, s, v)
    return "#%.2x%.2x%.2x"%(int(255*r), int(255*g), int(255*b))

class PolyViewerBase(object):
    dpi=72
    min_width=2.0
    default_height=5.0
    
    def __init__(self, newton_poly, title=None, gridsize=None):
        self.NP, self.title = newton_poly, title
        self.columns = 1 + self.NP.support[-1][0]
        self.rows = 1 + max([d[1] for d in self.NP.support])
        if gridsize == None:
            gridsize = self.default_height/max(self.rows, self.columns)
        self.gridsize = gridsize
        self.width = self.columns*self.gridsize
        self.height = self.rows*self.gridsize
        self.dot_radius = min(8, int(self.dpi*gridsize/3))
        size = W, H = (self.min_width + self.width, 0.5 + self.height)
        l, w = (W - self.width)/(2*W), self.width/W
        b, h = (H - self.height)/(2*H), self.height/H
        self.figure = MF = MatplotFigure(add_subplot=False, size=size, dpi=100)
        self.axis = axis = MF.figure.add_axes([l, b, w, h])
        axis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))    
        axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))    
        axis.margins(x=0.2, y=0.2)
        self.init_backend()

    def init_backend(self):
        # Override to provide backend-specific code
        pass
    
    def show_dots(self):
        for i, j in self.NP.support:
            color = self.NP.color_dict[(j, i)]
            self.axis.plot(i, j, marker='o', ms=self.dot_radius, color=color,
                           markeredgecolor=color)
        axis = self.axis
        axis.set_xlim(xmin=-0.75, xmax=self.columns - 0.25)
        axis.set_ylim(ymin=-0.75, ymax=self.rows - 0.25)
        axis.locator_params(axis='x', integer=True)
        axis.locator_params(axis='y', integer=True)
        self.figure.draw()
            
    def show_sides(self):
        r = 2 + self.dot_radius
        upper = list(self.NP.upper_vertices)
        upper.reverse()
        x1, y1 = first = self.NP.lower_vertices[0]
        vertices = self.NP.lower_vertices + upper + [first]
        for vertex in vertices:
            self.axis.plot(x1, y1, marker='o', ms=r, color='black')
            x2, y2 = vertex
            self.axis.plot([x1, x2], [y1, y2], color='black')
            x1, y1 = x2, y2
        self.figure.draw()

class TkPolyViewer(PolyViewerBase):
    dpi=100
    min_width=2.0
    default_height=8.0
                           
    def __init__(self, newton_poly, title=None, gridsize=None):
        PolyViewerBase.__init__(self, newton_poly, title, gridsize)
                         
    def init_backend(self):
        if self.title:
            self.figure.window.title(self.title)
        self.figure.window.wm_geometry('+400+20')

class NbPolyViewer(PolyViewerBase):
    dpi=72
    min_width=2.0
    default_height=5.0

    def __init__(self, newton_poly, title=None, gridsize=None):
        PolyViewerBase.__init__(self, newton_poly, title, gridsize)

class Unsupported:
    def __init__(self, newton_poly, title=None, gridsize=None):
        raise RuntimeError ('PolyViewer does not support this matpotlib backend (%s).'%backend)

backend = matplotlib.get_backend()
if backend == 'TkAgg':
    PolyViewer = TkPolyViewer
elif backend == 'nbAgg':
    PolyViewer = NbPolyViewer
else:
    PolyViewer = Unsupported
    
