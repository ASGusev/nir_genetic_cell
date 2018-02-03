import numpy as np


def write_specimen(fout, densities, fitness, desc):
    fout.write('begin{}\n'.format(desc))
    fout.write(' '.join(map(str, fitness)) + '\n')
    for line in densities:
        fout.write('\t'.join([' '.join(map(str, column)) for column in line]) + '\n')
    fout.write('end\n')


class StepWriter:
    def __init__(self, filename, clear):
        self.filename = filename
        if clear:
            self.output_file = open(filename, 'wt')
            self.output_file.close()

    def write_step(self, step, fitness, densities):
        with open(self.filename, 'at') as fout:
            desc = ' step ' + str(step)
            write_specimen(fout, densities, fitness, desc)


def read_specimen(fin):
    line = ''
    while not line.startswith('begin'):
        line = fin.readline()
        if line == '':
            return None
    desc = line[5:]
    fitness = tuple(float(i) for i in fin.readline().split())
    line = ''
    densities = []
    while not line.startswith('end'):
        line = fin.readline()
        if not line.startswith('end'):
            cols = line.split('\t')
            densities.append([[float(i) for i in col.split(' ')] for col in cols])
    densities = np.array(densities)
    return densities, fitness, desc


def read_steps(filename):
    with open(filename, 'rt') as fin:
        while True:
            step = read_specimen(fin)
            if step is None:
                raise StopIteration
            densities, fitness, desc = step
            step_number = int(desc.split()[1])
            yield step_number, fitness, densities


def write_population(population, iterations_passed, filename):
    with open(filename, 'wt') as fout:
        fout.write('{}\n'.format(iterations_passed))
        for elem in population:
            write_specimen(fout, elem[0], elem[1], '')


def read_population(filename):
    with open(filename, 'rt') as fin:
        steps = int(fin.readline())
        population = []
        element = read_specimen(fin)
        while element is not None:
            population.append(element[:2])
            element = read_specimen(fin)
    return population, steps
