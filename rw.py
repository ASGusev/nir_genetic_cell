import numpy as np


class StepWriter:
    def __init__(self, filename, clear):
        self.filename = filename
        if clear:
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


def write_population(population, iterations_passed, filename):
    with open(filename, 'wt') as fout:
        fout.write('{}\n'.format(iterations_passed))
        for elem in population:
            fout.write('begin\n')
            fout.write(' '.join(map(str, elem[1])))
            fout.write('\n')
            for line in elem[0]:
                fout.write('\t'.join([' '.join(map(str, column)) for column in line]) + '\n')
            fout.write('end\n')


def read_element(fin):
    line = ''
    while not line.startswith('begin'):
        line = fin.readline()
        if line == '':
            return None
    fitness = tuple(float(i) for i in fin.readline().split())
    densities = []
    while not line.startswith('end'):
        line = fin.readline()
        if not line.startswith('end'):
            cols = line.split('\t')
            densities.append([[float(i) for i in col.split(' ')] for col in cols])
    densities = np.array(densities)
    return densities, fitness


def read_population(filename):
    with open(filename, 'rt') as fin:
        steps = int(fin.readline())
        population = []
        element = read_element(fin)
        while element is not None:
            population.append(element)
            element = read_element(fin)
    return population, steps
