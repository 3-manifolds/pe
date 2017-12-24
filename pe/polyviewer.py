"""
The PolyViewer class displays the Newton polygon of a 2-variable polynomial,
with vertices colored by the log of the absolute value of the coefficient.
"""
import colorsys, matplotlib

backend = matplotlib.get_backend()
if backend == 'TkAgg':
    try:
        import Tkinter as tkinter
    except ImportError:
        import tkinter

def color_string(h, s, v):
    """
    Return an html style color string from HSV values in [0.0, 1.0]
    """
    r, g, b = colorsys.hsv_to_rgb((.03125 + h)%1.0, s, v)
    return "#%.2x%.2x%.2x"%(int(255*r), int(255*g), int(255*b))

class TkPolyViewer:
    def __init__(self, newton_poly, title=None, scale=None, margin=10):
        self.NP = newton_poly
        self.columns = 1 + self.NP.support[-1][0]
        self.rows = 1 + max([d[1] for d in self.NP.support])
        if scale == None:
            scale = 600//max(self.rows, self.columns)
        self.scale = scale
        self.margin = margin
        self.width = (self.columns - 1)*self.scale + 2*self.margin
        self.height = (self.rows - 1)*self.scale + 2*self.margin
        self.window = tkinter.Tk()
        if title:
            self.window.title(title)
        self.window.wm_geometry('+400+20')
        self.canvas = tkinter.Canvas(self.window,
                                     bg='white',
                                     height=self.height,
                                     width=self.width)
        self.canvas.pack(expand=True, fill=tkinter.BOTH)
        self.font = ('Helvetica','18','bold')
        self.dots=[]
        self.text=[]
        self.sides=[]

        self.grid = (
            [ self.canvas.create_line(
                0, self.height - self.margin - i*scale,
                self.width, self.height - self.margin - i*scale,
                fill=self.gridfill(i))
              for i in range(self.rows)] +
            [ self.canvas.create_line(
                self.margin + i*scale, 0,
                self.margin + i*scale, self.height,
                fill=self.gridfill(i))
              for i in range(self.columns)])
        #self.window.mainloop()

    def write_psfile(self, filename):
        self.canvas.postscript(file=filename)
            
    def gridfill(self, i):
        if i:
            return '#f0f0f0'
        else:
            return '#d0d0d0'
          
    def point(self, pair):
        i,j = pair
        return (self.margin+i*self.scale,
                self.height - self.margin - j*self.scale)
      
    def show_dots(self):
        r = 2 + self.scale//20
        for i, j in self.NP.support:
            x,y = self.point((i,j))
            color = self.NP.color_dict[(j, i)]
            self.dots.append(self.canvas.create_oval(
                x-r, y-r, x+r, y+r, fill=color, outline=color))

    def erase_dots(self):
        for dot in self.dots:
            self.canvas.delete(dot)
        self.dots = []

    def show_text(self):
        r = 2 + self.scale/20
        for i, j in self.NP.support:
            x,y = self.point((i,j))
            self.sides.append(self.canvas.create_oval(
                x-r, y-r, x+r, y+r, fill='black'))
            self.text.append(self.canvas.create_text(
                2*r+x,-2*r+y,
                text=str(self.NP.coeff_dict[(j,i)]),
                font=self.font,
                anchor='c'))
              
    def erase_text(self):
        for coeff in self.text:
            self.canvas.delete(coeff)
        self.text=[]

    def show_sides(self):
        r = 3 + self.scale//20
        first = self.NP.lower_vertices[0]
        x1, y1 = self.point(first)
        upper = list(self.NP.upper_vertices)
        upper.reverse()
        vertices = self.NP.lower_vertices + upper + [first]
        for vertex in vertices:
            self.sides.append(self.canvas.create_oval(
                x1-r, y1-r, x1+r, y1+r, fill='black', outline='black'))
            x2, y2 = self.point(vertex)
            self.sides.append(self.canvas.create_line(
                x1, y1, x2, y2,
                fill='black'))
            x1, y1 = x2, y2

    def erase_sides(self):
        for object in self.sides:
            self.canvas.delete(object)
        self.sides=[]

class Unsupported:
    def __init__(self, newton_poly, title=None, scale=None, margin=10):
        raise RuntimeError ('PolyViewer does not support this matpotlib backend (%s).'%backend)

if backend == 'TkAgg':
    PolyViewer = TkPolyViewer
else:
    PolyViewer = Unsupported
    
