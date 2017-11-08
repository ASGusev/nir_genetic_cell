import sys
import numpy as np
import matplotlib.pyplot as plt
import rw
from optimize import R as RADIUS


def plot_map(dens, step_number, fitness, ax):
    avg_dens = np.array([[np.average(col) for col in line] for line in dens])
    ax.imshow(avg_dens, cmap='Greys')
    ax.set_title('Step {}, fitness: {}'.format(step_number, fitness))


def main(input_filename, output_filename):
    n = 0
    for i in rw.read_steps(input_filename):
        n += 1
    dens_iter = rw.read_steps(input_filename)
    fig = plt.figure(figsize=(10, 10 * n))
    for i, (step_number, fitness, dens) in enumerate(dens_iter):
        cur_subplot = plt.subplot(n, 1, i + 1)
        plot_map(dens, step_number, fitness, cur_subplot)
    fig.tight_layout()
    fig.savefig(output_filename)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])