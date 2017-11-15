import numpy as np
import scipy as sp
from operator import itemgetter
from multiprocessing import Pool
import scipy.optimize
from data_parser import read_data
import rw


type_particle = [('x', float), ('y', float), ('z', float), ('r', float)]


EPS = 1e-12
TRACK_SIZE = 50
R = 10 * 1e-6
H = 1.5 * 1e-5
DELTA_T = 1.8 * 1e-6
F_CNT = 10
PART_CNT = 500
kB = 1.38e-23
T = 293
SIZES = [0, 7 * 0.05 * 1e-6, 11 * 0.05 * 1e-6, float('inf')]
GEN_SIZE = 200
REPLACED_SHARE = 0.2
TO_REPLACE = int(GEN_SIZE * REPLACED_SHARE + EPS)
TO_KEEP = GEN_SIZE - TO_REPLACE
BOXES_ALONG = 50
BOX_SIDE = 2 * R / BOXES_ALONG
COORD_MULT = 1 / BOX_SIDE
BOXES_IN_HEIGHT = int(H / BOX_SIDE + 1 - EPS)
R_SQR = R ** 2


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
    with Pool() as pool:
        fitnesses = pool.map(fitness_function, generation)
    sorted_zipped = sorted(zip(fitnesses, generation), key=itemgetter(0))
    sorted_beings = [being for (fitness, being) in sorted_zipped]
    sorted_fitnesses = [fitness for (fitness, being) in sorted_zipped]
    return sorted_beings, sorted_fitnesses


def generate_particles(number, min_r, max_r):
    particles = np.zeros(number, dtype=type_particle)
    particles['r'] = np.random.uniform(min_r, max_r)
    angles = np.random.random(number) * 2 * np.pi
    rads = np.sqrt(np.random.random(number) * (R - particles['r']) ** 2)
    particles['x'] = np.cos(angles) * rads
    particles['y'] = np.sin(angles) * rads
    particles['z'] = np.random.random(number) * H
    return particles


def cross(father, mother):
    from_father = np.random.choice((False, True), (BOXES_ALONG, BOXES_ALONG, BOXES_IN_HEIGHT))
    from_mother = np.logical_not(from_father)
    child = np.zeros((BOXES_ALONG, BOXES_ALONG, BOXES_IN_HEIGHT))
    child[from_father] = father[from_father]
    child[from_mother] = mother[from_mother]
    child *= np.random.normal(1, 0.02, (BOXES_ALONG, BOXES_ALONG, BOXES_IN_HEIGHT))
    return child


def refute_sides(parts, shifts, bad):
    while np.count_nonzero(bad) > 0:
        bad_parts = parts[bad]
        bad_shifts = shifts[bad]
        a = bad_shifts[:, 0] ** 2 + bad_shifts[:, 1] ** 2
        b = 2 * (bad_shifts[:, 0] * bad_parts['x'] + bad_shifts[:, 1] * bad_parts['y'])
        c = bad_parts['x'] ** 2 + bad_parts['y'] ** 2 - R ** 2
        k = (-b + np.sqrt((b ** 2 - 4 * c * a))) / (2 * a)
        parts['x'][bad] += k * bad_shifts[:, 0]
        parts['y'][bad] += k * bad_shifts[:, 1]
        shifts[bad, 0] *= (1 - k)
        shifts[bad, 1] *= (1 - k)

        proj = parts[bad]
        shabs = bad_shifts[:, 0] * proj['x'] + bad_shifts[:, 1] * proj['y']
        shifts[bad, 0] -= 2 * shabs * proj['x'] / R_SQR
        shifts[bad, 1] -= 2 * shabs * proj['y'] / R_SQR
        good = (parts['x'] + shifts[:, 0]) ** 2 + (parts['y'] + shifts[:, 1]) ** 2 <= R ** 2
        parts['x'][good] += shifts[good, 0]
        parts['y'][good] += shifts[good, 1]
        shifts[good] = 0
        bad = np.logical_not(good)
    parts['x'][bad] += shifts[bad, 0]
    parts['y'][bad] += shifts[bad, 1]


def refute_bases(parts):
    too_high = parts['z'] > H
    too_low = parts['z'] < 0
    while np.count_nonzero(too_high) or np.count_nonzero(too_low):
        parts['z'][too_high] = 2 * H - parts[too_high]['z']
        parts['z'][too_low] = -parts[too_low]['z']
        too_low = parts['z'] < 0
        too_high = parts['z'] > H


def drive_particles(viscs, steps, min_r, max_r, particles_number):
    parts = generate_particles(particles_number, min_r, max_r)
    tracks = np.zeros([particles_number, steps, 3])
    pre_coeffs = kB * T / (3 * np.pi * viscs)
    for j in range(steps):
        extend_tracks(j, parts, tracks)

        box_xes = np.int32((R + parts['x']) * COORD_MULT)
        box_ys = np.int32((R + parts['y']) * COORD_MULT)
        box_zs = np.int32(parts['z'] * COORD_MULT)

        coeffs = pre_coeffs[box_xes, box_ys, box_zs] / parts['r']

        shifts = calc_shifts(coeffs, parts)
        bad = (parts['x'] + shifts[:, 0]) ** 2 + (parts['y'] + shifts[:, 1]) ** 2 > R ** 2
        parts['x'][~bad] += shifts[~bad, 0]
        parts['y'][~bad] += shifts[~bad, 1]
        parts['z'] += shifts[:, 2]
        shifts[~bad] = 0
        shifts[bad, 2] = 0

        refute_sides(parts, shifts, bad)
        refute_bases(parts)
    return tracks


def extend_tracks(j, parts, tracks):
    tracks[:, j, 0] = parts['x']
    tracks[:, j, 1] = parts['y']
    tracks[:, j, 2] = parts['z']


def calc_shifts(coeffs, parts):
    return np.random.normal(0, np.sqrt(2 * coeffs * DELTA_T), (3, len(parts))).T


def calculate_parameters(viscs, r_means, r_stds, repeats):
    parameters = []
    for r_mean, r_std in zip(r_means, r_stds):
        tracks = drive_particles(viscs, TRACK_SIZE, r_mean - r_std, r_mean + r_std, PART_CNT * repeats)
        parameters.append([std_to_curve_params(*large_data_std_values(tracks[i * PART_CNT: (i + 1) * PART_CNT]))
                           for i in range(repeats)])
    parameters = np.array(parameters)
    parameters = parameters.mean(axis=1)
    return parameters


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
        result_params = calculate_parameters(viscs, self.means, self.stds, F_CNT)
        result = calc_dist(self.target, result_params)
        return result


def next_generation(generation):
    children = []
    for father, mother in zip(generation[:TO_REPLACE:2],
                              generation[1:TO_REPLACE:2]):
        children.append(cross(father, mother))
        children.append(cross(father, mother))
    return generation[:TO_KEEP] + children


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


def calc(iterations_number, write_step, raw_filename, print_progress=True):
    fitness_calculator = FitnessCalculator()
    data = read_data()
    r_means, r_stds, target_vector = prepare_data(data)
    generation = [np.ones((BOXES_ALONG, BOXES_ALONG, BOXES_IN_HEIGHT)) * 2.390041077895209e-06 for j in range(GEN_SIZE)]
    best_fitnesses = []
    fitness_calculator.set_target(target_vector)
    fitness_calculator.set_means(r_means)
    fitness_calculator.set_stds (r_stds)
    step_writer = rw.StepWriter(raw_filename)

    for j in range(iterations_number):
        if print_progress:
            print('Iteration {}'.format(j + 1))
        sorted_generation, sorted_fitnesses = sort_with_fitness(generation,
                                                                fitness_calculator.calculate_fitness)
        best_fitness = sorted_fitnesses[0]

        best_fitnesses.append(best_fitness)

        if (j + 1) % write_step == 0:
            step_writer.write_step(j, best_fitnesses[0], sorted_generation[0])

        generation = next_generation(sorted_generation[:GEN_SIZE])
    return np.array(best_fitnesses)
