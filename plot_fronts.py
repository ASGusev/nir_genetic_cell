import sys
import operator
import matplotlib.pyplot as plt


def parse_front(front):
    return [[float(j) for j in i.split()]
            for i in front.split('\t')]


def sort_front(front):
    front = sorted(front, key=operator.itemgetter(0))
    res = []
    res.append(front[0])
    for i in range(1, len(front)):
        if front[i][2] < res[len(res) - 1][2]:
            res.append(front[i])
    print([i[2] for i in res])
    return res


input_filename = sys.argv[1]
output_filename = sys.argv[2]
with open(input_filename, 'rt') as fin:
    read_fronts = [parse_front(front) for front in fin.readlines()]
#fronts = list(filter(lambda x: x[0] % 200 == 0, enumerate(fronts)))
fronts = [sort_front(read_fronts[i]) for i in range(0, len(read_fronts), 200)]
for front in fronts:
    plt.plot([i[0] for i in front], [i[2] for i in front], 'o', markersize=3)
plt.savefig(output_filename)
