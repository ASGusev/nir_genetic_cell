import numpy as np


class StepWriter:
    def __init__(self, filename):
        self.filename = filename
        self.output_file = open(filename, 'wt')
        self.output_file.close()

    def write_step(self, step, fitness, densities):
        with open(self.filename, 'at') as fout:
            fout.write('begin step {} \n'.format(step))
            fout.write('fitness: {}\n'.format(fitness))
            for line in densities:
                fout.write('\t'.join([' '.join(map(str, column)) for column in line]) + '\n')
            fout.write('end step\n')


def read_step(fin):
    line = ''
    while not line.startswith('begin'):
        line = fin.readline()
        if line == '':
            return None
    step_number = int(line.split()[2])
    fitness = float(fin.readline().split()[1])
    line = ''
    densities = []
    while not line.startswith('end'):
        line = fin.readline()
        if not line.startswith('end'):
            cols = line.split('\t')
            densities.append([[float(i) for i in col.split(' ')] for col in cols])
    return step_number, fitness, np.array(densities)


def read_steps(filename):
    with open(filename, 'rt') as fin:
        while True:
            step = read_step(fin)
            if step is None:
                raise StopIteration
            yield step


__all__ = ['read_steps', 'StepWriter']
