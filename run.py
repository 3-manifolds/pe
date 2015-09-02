#! /bin/env sage-python
#
#SBATCH --partition m
#SBATCH --tasks=1
#SBATCH --mem-per-cpu=4096
#SBATCH --nice=10000
#SBATCH --time=1-00:00
#SBATCH --output=slurm_out/%j
#SBATCH --error=slurm_out/%j

# Some problem cases:
#
# m036
# m044


import slurm
import taskdb2
import snappy
import traceback
import pe
import bz2
import cPickle as pickle


def create_database():
    names = [M.name() for M in snappy.OrientableCuspedCensus(cusps=1)]
    taskdb2.create_task_db(names, 'PEChar', overwrite=True)
    db = taskdb2.TaskDatabase('PEChar')
    db.add_column('lifter', 'mediumblob')
    return db

def save_pdf(task):
    try:
        V = pe.PECharVariety(task['name'])
        L = pe.SL2RLifter(V)
        F = L.show(True)
        F.save('/pkgs/tmp/pe_pdfs/' + task['name'] + '.pdf')
        task['done'] = True
    except:
        print('<apoly-error>')
        print('task: ' + task['name'])
        print(traceback.format_exc())
        print('</apoly-error>')


if __name__ == '__main__':
    db = taskdb2.TaskDatabase('PEChar')
    db.run_function(save_pdf, num_tasks=1)

