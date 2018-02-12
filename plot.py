import sys
import numpy as np
import matplotlib.pyplot as plt


FONT_SIZE = 16
N = 3


def plot_func(vals, pos, title):
    plt.subplot(1, N, pos)
    plt.title(title, size=FONT_SIZE)
    plt.plot(vals)
    plt.tick_params('both', labelsize=FONT_SIZE)


data_file = sys.argv[1]
output_file = sys.argv[2]
mins = []
with open(data_file, 'rt') as fin:
    for line in fin.readlines():
        gen_front = np.array([list(map(float, fitness.split(' '))) for fitness in line.split('\t')])
        mins.append(gen_front.min(axis=0))
mins = np.array(mins).T
fig = plt.figure(figsize=(N * 5, 5))

plot_func(mins[0], 1, 'Length fitness')
plot_func(mins[1], 2, 'Grade')
plot_func(mins[2], 3, 'Coefficient')

fig.tight_layout()
plt.savefig(output_file)