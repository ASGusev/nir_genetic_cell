import sys
import operator
import matplotlib.pyplot as plt


def parse_front(front):
    return [[float(j) for j in i.split()]
            for i in front.split('\t')]


def sort_front(front):
    front = sorted(front, key=operator.itemgetter(0))
    res = []
    best_yet = float('inf')
    for fitness in front:
        if fitness[2] < best_yet:
            res.append(fitness)
            best_yet = fitness[2]
    return res


input_filename = sys.argv[1]
output_filename = sys.argv[2]
with open(input_filename, 'rt') as fin:
    read_fronts = [parse_front(front) for front in fin.readlines()]
fronts = [(i, sort_front(read_fronts[i])) for i in range(0, len(read_fronts), 200)]
for number, front in fronts:
    plt.plot([i[0] for i in front], [i[2] for i in front], 'o', markersize=3,
             label='Generation {}'.format(number + 1))
plt.xlabel('Length fitness function')
plt.ylabel('Approximation coefficient fitness function')
plt.legend()
plt.savefig(output_filename)
