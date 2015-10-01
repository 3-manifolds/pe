import snappy
import taskdb2
from sage.all import ComplexField

def initial_database():
    names = []
    for M in snappy.OrientableCuspedCensus(cusps=1):
        M.dehn_fill((1,0))
        if M.homology().order() == 1:
            names.append(M.name())

    cols = [('volume', 'double'), ('tets', 'int'), ('inS3', 'tinyint'), 
            ('alex', 'text'), ('alex_deg', 'int'), ('alex_monic', 'tinyint'),
            ('num_uniroots', 'int'), ('num_mult_uniroots', 'int') ]
    db = taskdb2.ExampleDatabase('ZHCircles', names, cols, overwrite=True)
    db.new_task_table('task0')
    return db

def basic(task):
    M = snappy.Manifold(task['name'])
    task['volume'] = M.volume()
    task['tets'] = M.num_tetrahedra()
    M.dehn_fill((1,0))
    if M.fundamental_group().num_generators() == 0:
        task['inS3'] = True
    else:
        task['inS3'] = False
    task['done'] = True

def on_unit_circle(z):
    prec = z.parent().prec()
    return abs(abs(z) - 1) < 2.0**(-0.8*prec)
    
def unimodular_roots(p):
    CC = ComplexField(212)
    return [(z, e) for z, e in p.roots(CC) if on_unit_circle(z)]
    
def alexander(task):
    M = snappy.Manifold(task['name'])
    p = M.alexander_polynomial()
    task['alex'] = repr(p)
    task['alex_deg'] = p.degree()
    task['alex_monic'] = p.is_monic()
    uniroots = unimodular_roots(p)
    task['num_uniroots'] = len(uniroots)
    task['num_mult_uniroots'] = len([z for z, e in uniroots if e > 1])
    task['done'] = True

if __name__ == '__main__':
    #initial_database()
    db = taskdb2.ExampleDatabase('ZHCircles')
    #task = db.get_row(100)
    db.run_function('task0', alexander, num_tasks=10)
