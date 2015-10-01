#! /bin/env sage-python
#
#SBATCH --partition m
#SBATCH --tasks=1
#SBATCH --mem-per-cpu=4096
#SBATCH --nice=10000
#SBATCH --time=1-00:00
#SBATCH --output=slurm_out/%j
#SBATCH --error=slurm_out/%j

import slurm
import taskdb2
import traceback
import zhcircles


def save_pdf(task):
    try:
        V = pe.PECharVariety(task['name'], radius=1.04)
        L = pe.SL2RLifter(V)
        if L.nonempty():
            F = L.show(True)
            F.save('/pkgs/tmp/pe_pdfs/' + task['name'] + '.pdf')
        task['done'] = True
    except:
        print('<apoly-error>')
        print('task: ' + task['name'])
        print(traceback.format_exc())
        print('</apoly-error>')


if __name__ == '__main__':
    db = zhcircles.ZHCircles()
    db.run_function('task0', zhcircles.save_plot_data, num_tasks=1)

