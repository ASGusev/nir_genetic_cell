import sys
from math import *
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from operator import itemgetter
from multiprocessing import Pool
import scipy.optimize
from data_parser import read_data


type_particle = [('x', float), ('y', float), ('z', float), ('r', float)]


TRACK_SIZE = 50
R = 10 * 1e-6
H = 1.5 * 1e-5
DELTA_T = 1.8 * 1e-6
F_CNT = 10
PART_CNT = 500
kB = 1.38e-23
T = 293
SIZES = [0, 7 * 0.05 * 1e-6, 11 * 0.05 * 1e-6, float('inf')]
GEN_SIZE = 10
BOXES_ALONG = 50
BOX_SIDE = 2 * R / BOXES_ALONG
BOXES_IN_HEIGHT = int(H / BOX_SIDE)
BOX_BORDERS_H = np.linspace(-R, R, BOXES_ALONG + 1)
BOX_BORDERS_V = np.linspace(0, H,  + 1)
EPS = 1e-12


def calc_dist(t_vector, r_vector):
    return np.linalg.norm((t_vector - r_vector) / np.max(t_vector, axis=0)) ** 2


def low_data_std_values(coords, length_factor=1):
    xs = np.arange(1, int(coords.shape[0] * length_factor))
    ys = [np.mean([np.linalg.norm(coords[j + i, 1:] - coords[j, 1:])**2 for j in range(coords.shape[0] - i)]) for i in xs]
    return xs, ys


def large_data_std_values(tracks, length_factor=1):
    xs = np.arange(1, tracks[0].shape[0])
    ys = [np.mean(np.linalg.norm(tracks[:, i, :] - tracks[:, 0, :], axis=1)**2) for i in xs]
    return xs, ys


def t_function(x, alpha, beta):
    return beta * x**alpha


def std_to_curve_params(xs, ys, popt=[1, 1]):
    popt, _ = sp.optimize.curve_fit(t_function, xs, ys, popt)
    return popt


def sort_with_fitness(generation, fitness_function):
    with Pool(5) as pool:
        fitnesses = pool.map(fitness_function, generation)
    sorted_zipped = sorted(zip(fitnesses, generation), key=itemgetter(0))
    sorted_beings = [being for (fitness, being) in sorted_zipped]
    sorted_fitnessess = [fitness for (fitness, being) in sorted_zipped]
    return sorted_beings, sorted_fitnessess


def generate_particles(number, min_r, max_r):
    particles = np.zeros(number, dtype=type_particle)
    particles['r'] = np.random.uniform(min_r, max_r)
    angles = np.random.random(number) * 2 * np.pi
    rads = np.sqrt(np.random.random(number) * (R - particles['r']) ** 2)
    particles['x'] = np.cos(angles) * rads
    particles['y'] = np.sin(angles) * rads
    particles['z'] = np.random.random(number) * H
    return particles


def cross_boxes(father, mother):
    from_father = np.random.choice((False, True), (BOXES_ALONG, BOXES_ALONG, BOXES_IN_HEIGHT))
    from_mother = np.logical_not(from_father)
    child = np.zeros((BOXES_ALONG, BOXES_ALONG, BOXES_IN_HEIGHT))
    child[from_father] = father[from_father]
    child[from_mother] = mother[from_mother]
    child *= np.random.normal(1, 0.02, (BOXES_ALONG, BOXES_ALONG, BOXES_IN_HEIGHT))
    return child


def drive_particles(viscs, steps, min_r, max_r):
    parts = generate_particles(PART_CNT, min_r, max_r)
    tracks = np.zeros([PART_CNT, steps, 3])
    pre_coeffs = kB * T / (3 * np.pi * viscs)
    for j in range(steps):
        tracks[:, j, 0] = parts['x']
        tracks[:, j, 1] = parts['y']
        tracks[:, j, 2] = parts['z']
        box_xes = np.searchsorted(BOX_BORDERS_H, parts['x'] + EPS) - 1
        box_ys = np.searchsorted(BOX_BORDERS_H, parts['y'] + EPS) - 1
        box_zs = np.searchsorted(BOX_BORDERS_V, parts['z'] + EPS) - 1
        coeffs = pre_coeffs[box_xes, box_ys, box_zs] / parts['r']

        shifts = np.random.normal(0, np.sqrt(2 * coeffs * DELTA_T), (3, len(parts))).T
        bad = (parts['x'] + shifts[:, 0]) ** 2 + (parts['y'] + shifts[:, 1]) ** 2 > R ** 2
        parts['x'][~bad] += shifts[~bad, 0]
        parts['y'][~bad] += shifts[~bad, 1]
        parts['z'] += shifts[:, 2]
        shifts[~bad] = 0
        shifts[:, 2] = 0

#       Refuting from sides
        while np.count_nonzero(bad) > 0:
            a = shifts[bad, 0] ** 2 + shifts[bad, 1] ** 2
            b = 2 * (shifts[bad, 0] * parts['x'][bad] + shifts[bad, 1] * parts['y'][bad])
            c = parts['x'][bad] ** 2 + parts['y'][bad] ** 2 - R ** 2
            k = (-b + np.sqrt(np.abs(b ** 2 - 4 * c * a))) / (2 * a)
            parts['x'][bad] += k * shifts[bad, 0]
            parts['y'][bad] += k * shifts[bad, 1]
            shifts[bad, 0] *= (1 - k)
            shifts[bad, 1] *= (1 - k)
            proj = np.copy(parts[bad])
            fpabs = np.sqrt(parts['x'][bad] ** 2 + parts['y'][bad] ** 2)
            proj['x'] /= fpabs
            proj['y'] /= fpabs
            shabs = shifts[bad, 0] * proj['x'] + shifts[bad, 1] * proj['y']
            proj['x'] *= shabs
            proj['y'] *= shabs
            shifts[bad, 0] -= 2 * proj['x']
            shifts[bad, 1] -= 2 * proj['y']
            good = (parts['x'] + shifts[:, 0]) ** 2 + (parts['y'] + shifts[:, 1]) ** 2 <= R ** 2
            parts['x'][good] += shifts[good, 0]
            parts['y'][good] += shifts[good, 1]
            shifts[good] = 0
            bad = (parts['x'] + shifts[:, 0]) ** 2 + (parts['y'] + shifts[:, 1]) ** 2 > R ** 2
        parts['x'][bad] += shifts[bad, 0]
        parts['y'][bad] += shifts[bad, 1]

#       Refuting from floor and ceil
        too_high = parts['z'] > H
        too_low = parts['z'] < 0
        while np.count_nonzero(too_high) or np.count_nonzero(too_low):
            parts['z'][too_high] = 2 * H - parts[too_high]['z']
            parts['z'][too_low] = -parts[too_low]['z']
            too_low = parts['z'] < 0
            too_high = parts['z'] > H
    return tracks


def calculate_parameters(viscs, r_means, r_stds):
    parameters = []
    for r_mean, r_std in zip(r_means, r_stds):
        tracks = drive_particles(viscs, TRACK_SIZE, r_mean - r_std, r_mean + r_std)
        parameters.append(std_to_curve_params(*large_data_std_values(tracks)))
    return np.array(parameters)


class FitnessCalculator:
    def __init__(self, target=None, means=None, stds=None):
        self.target = target
        self.means = means
        self.stds = stds

    def set_target(self, target):
        self.target = target

    def set_means(self, means):
        self.means = means

    def set_stds(self, stds):
        self.stds = stds

    def calculate_fitness(self, viscs):
        result_params = np.zeros((len(SIZES) - 1, 2), dtype=float)
        for j in range(F_CNT):
            result_params += calculate_parameters(viscs, self.means, self.stds)
        result_params /= F_CNT
        result = calc_dist(self.target, result_params)
        return result


def next_generation(generation):
    return [cross_boxes(j, k) for j in generation for k in generation]


def prepare_data(data):
    params = np.array([[np.mean(track[:, 0]), *std_to_curve_params(*low_data_std_values(track[:, 1:], 0.5))]
                       for track in data])
    r_means = []
    r_stds = []
    alphas = []
    betas = []
    for i in range(1, len(SIZES)):
        indices = np.logical_and(SIZES[i - 1] <= params[:, 0], params[:, 0] < SIZES[i])
        r_means.append(np.mean(params[indices][:, 0]))
        r_stds.append(np.std(params[indices][:, 0]))
        alphas.append(np.mean(params[indices][:, 1]))
        betas.append(np.mean(params[indices][:, 2]))
    target_vector = np.array(list(zip(alphas, betas)))
    return r_means, r_stds, target_vector


fitness_calculator = FitnessCalculator()


def __main__(argv):
    iterations_number = int(argv[1])
    plot_filename = argv[2]
    write_step = int(argv[3])
    if write_step > 0:
        raw_filename = argv[4]
    data = read_data()
    r_means, r_stds, target_vector = prepare_data(data)
    generation = [np.ones((BOXES_ALONG, BOXES_ALONG, BOXES_IN_HEIGHT)) * 2.390041077895209e-06 for j in range(GEN_SIZE)]
    best_fitnesses = []
    fitness_calculator.set_target(target_vector)
    fitness_calculator.set_means(r_means)
    fitness_calculator.set_stds(r_stds)

    for j in range(iterations_number):
        print('Iteration {}'.format(j + 1))
        sorted_generation, sorted_fitnesses = sort_with_fitness(generation,
                                                                fitness_calculator.calculate_fitness)
        best_fitness = sorted_fitnesses[0]

        best_fitnesses.append(best_fitness)

        if j % write_step == 0:
            with open(raw_filename, 'a') as fout:
                fout.write('begin step {} \n'.format(j))
                fout.write('fitness: {}\n'.format(sorted_fitnesses[0]))
                for line in sorted_generation[0]:
                    fout.write('\t'.join([' '.join(map(str, column)) for column in line]) + '\n')
                fout.write('end step\n')

        generation = next_generation(sorted_generation[:GEN_SIZE]) + sorted_generation[:GEN_SIZE]
    plt.plot(np.arange(len(best_fitnesses)), best_fitnesses)
    plt.savefig(plot_filename)


if __name__ == '__main__':
    __main__(sys.argv)
