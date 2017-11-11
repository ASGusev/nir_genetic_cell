import sys
import numpy as np
import matplotlib.pyplot as plt
import rw


def make_relevant_indexes(n):
    r = n / 2

    def is_relevant(ind):
        x, y = ind
        return min(abs(x - r), abs(x + 1 - r)) ** 2 + \
               min(abs(y - r), abs(y + 1 - r)) ** 2 <= r ** 2

    indexes = [(i, j) for i in range(n) for j in range(n)]
    return np.array([i for i in filter(is_relevant, indexes)]).T


def plot_map(dens, step_number, fitness, ax, relevant_indexes):
    avg_dens = np.ones((len(dens), len(dens))) * np.average(dens)
    ind0, ind1 = relevant_indexes[0], relevant_indexes[1]
    avg_dens[ind0, ind1] = np.average(dens, axis=2)[ind0, ind1]
    ax.imshow(avg_dens, cmap='Greys')
    ax.set_title('Step {}, fitness: {}'.format(step_number, fitness))


def main(input_filename, output_filename):
    maps_number = 1
    steps_iter = rw.read_steps(input_filename)
    map_side = len(next(steps_iter)[2])
    for i in steps_iter:
        maps_number += 1
    relevant_indexes = make_relevant_indexes(map_side)

    dens_iter = rw.read_steps(input_filename)
    fig = plt.figure(figsize=(10, 10 * maps_number))
    for i, (step_number, fitness, dens) in enumerate(dens_iter):
        cur_subplot = plt.subplot(maps_number, 1, i + 1)
        plot_map(dens, step_number, fitness, cur_subplot, relevant_indexes)
    fig.tight_layout()
    fig.savefig(output_filename)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])