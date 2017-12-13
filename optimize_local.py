import numpy as np
from operator import itemgetter
from multiprocessing import Pool
from data_parser import read_data
import rw
import scipy.optimize as soptimize
import math


type_particle = [('x', float), ('y', float), ('z', float)]


EPS = 1e-12
TRACK_SIZE = 50
MAX_DIST = 1e-6
DELTA_T = 1.8 * 1e-6  # 1.89 sec
PART_CNT = 50
kB = 1.38e-23
T = 293
GEN_SIZE = 100
REPLACED_SHARE = 0.2
TO_REPLACE = int(GEN_SIZE * REPLACED_SHARE + EPS)
TO_KEEP = GEN_SIZE - TO_REPLACE
BOXES_ALONG = 12
BOX_SIDE = 2 * MAX_DIST / BOXES_ALONG
COORD_MULT = 1 / BOX_SIDE


def drive_particles(viscs, steps, radius, particles_number):
    parts = np.zeros(particles_number, dtype=type_particle)
    tracks = np.zeros([particles_number, steps, 3])
    moves = np.zeros([particles_number, steps, 3])
    box_coeffs = kB * T / (3 * np.pi * radius * viscs)
    average_visc = viscs.mean()
    outside_coeff = kB * T / (3 * np.pi * average_visc * radius)
    for j in range(steps):
        box_xs = np.int32((MAX_DIST + parts['x']) * COORD_MULT)
        box_ys = np.int32((MAX_DIST + parts['y']) * COORD_MULT)
        box_zs = np.int32((MAX_DIST + parts['z']) * COORD_MULT)
        good_xs = np.logical_and(0 <= box_xs, box_xs < BOXES_ALONG)
        good_ys = np.logical_and(0 <= box_ys, box_ys < BOXES_ALONG)
        good_zs = np.logical_and(0 <= box_zs, box_zs < BOXES_ALONG)
        not_far = np.logical_and(good_xs, np.logical_and(good_ys, good_zs))

        part_coeffs = np.zeros(particles_number) + outside_coeff
        part_coeffs[not_far] = box_coeffs[box_xs[not_far], box_ys[not_far], box_zs[not_far]]

        shifts = calc_shifts(part_coeffs, particles_number)
        parts['x'] += shifts[:, 0]
        parts['y'] += shifts[:, 1]
        parts['z'] += shifts[:, 2]

        tracks[:, j, 0] = parts['x']
        tracks[:, j, 1] = parts['y']
        tracks[:, j, 2] = parts['z']

        moves[:, j] = shifts
    return tracks, moves


def calc_shifts(coeffs, n):
    return np.random.normal(0, np.sqrt(2 * coeffs * DELTA_T), (3, n)).T


def prepare_avg_lengths(data):
    horizontal_lengths = []
    rads = []
    step_numbers = []
    for t in data:
        steps = t[1:, 1:] - t[:-1, 1:]
        horizontal_lengths.append(np.sum(np.sqrt(np.sum(steps ** 2, axis=1))))
        rads.append(t[:, 0].mean())
        step_numbers.append(t.shape[0])
    return horizontal_lengths, rads, step_numbers


def track_to_params(track):
    time_disps = []
    for i in range(1, len(track)):
        displacements = np.sum((track[i:] - track[:-i]) ** 2, axis=1) ** 0.5
        time_disps.append(displacements.mean())
    time_disps = np.array(time_disps)
    times = np.arange(1, len(track))

    def f(x, a, b):
        return b * x ** a

    c, _ = soptimize.curve_fit(f, times, time_disps, [1e1, 1e-9])
    return c


def prepare_disp_params(data):
    return [track_to_params(t[:, 1:]) for t in data]


def calc_length_fitness(shifts, length):
    shifts_hor_lengths = np.sqrt(np.sum((shifts ** 2)[:, :, :-1], axis=2))
    track_lengths = np.sum(shifts_hor_lengths, axis=1)
    length_ratios = track_lengths / length
    exp_fitnesses = (length_ratios - 1) ** 2
    return exp_fitnesses.mean()


def calc_param_fitness(tracks, disp_params):
    p1, p2 = [], []
    for t in tracks:
        params = track_to_params(t[:, :2])
        p1.append(params[0])
        p2.append(params[1])
    p1 = np.abs(np.array(p1) / disp_params[0] - 1)
    p2 = np.abs(np.array(p2) / disp_params[1] - 1)
    return p1.mean(), p2.mean()


class FitnessCalculator:
    def __init__(self, data):
        self.exps_count = len(data)
        self.lengths, self.rads, self.steps_numbers = prepare_avg_lengths(data)
        self.disp_params = prepare_disp_params(data)

    def calculate_fitness(self, viscs):
        length_diff = []
        params = []
        for i in range(self.exps_count):
            tracks, shifts = drive_particles(viscs, self.steps_numbers[i],
                                             self.rads[i], PART_CNT)
            length_diff.append(calc_length_fitness(tracks, self.lengths[i]))
            params.append(calc_param_fitness(tracks, self.disp_params[i]))
        length_diff = np.array(length_diff)
        params = np.array(params).T
        return length_diff.mean(), params[0].mean(), params[1].mean()


def cross(father, mother):
    from_father = np.random.choice((False, True), (BOXES_ALONG, BOXES_ALONG, BOXES_ALONG))
    from_mother = np.logical_not(from_father)
    child = np.zeros((BOXES_ALONG, BOXES_ALONG, BOXES_ALONG))
    child[from_father] = father[from_father]
    child[from_mother] = mother[from_mother]
    mutations = np.random.normal(1, 0.02, (BOXES_ALONG, BOXES_ALONG, BOXES_ALONG))
    child *= mutations
    return child


def children(parents, fitness_function):
    kids = []
    for father, mother in zip(parents[::2], parents[1::2]):
        kids.append(cross(father[0], mother[0]))
        kids.append(cross(father[0], mother[0]))
    with Pool() as pool:
        fitnesses = list(pool.map(fitness_function, kids))
    return list(zip(kids, fitnesses))


def make_first_gen(fitness_function_calculator):
    rep = np.ones((BOXES_ALONG, BOXES_ALONG, BOXES_ALONG)) * 2.390041077895209e-06
    fitness = fitness_function_calculator(rep)
    return [(rep, fitness) for i in range(GEN_SIZE)]


def normalize(arr):
    avg = arr.mean()
    while abs(avg) > EPS:
        arr = arr - avg
        avg = arr.mean()
    disp = (arr ** 2).mean()
    if abs(disp) > EPS:
        arr /= math.sqrt(disp)
    return arr


def sort_generation(generation):
    cum_func = np.array([i[1] for i in generation]).T
    keys = np.vstack([normalize(i) for i in cum_func]).T.sum(axis=1)
    return [i[0] for i in sorted(zip(generation, keys), key=itemgetter(1))]


def calc(iterations_number, write_step, raw_filename=None, show_progress=True):
    fitness_calculator = FitnessCalculator(read_data())
    best_fitnesses = []
    if write_step > 0:
        step_writer = rw.StepWriter(raw_filename)

    if show_progress:
        print('Preparing first generation')
    generation = make_first_gen(fitness_calculator.calculate_fitness)
    for j in range(iterations_number):
        if show_progress:
            print('Iteration {}'.format(j + 1))
        generation = sort_generation(generation)
        best_fitnesses.append(generation[0][1])

        if write_step > 0 and (j + 1) % write_step == 0:
            step_writer.write_step(j + 1, generation[0][1], generation[0][0])

        generation = generation[:TO_KEEP] + children(generation,
                                                     fitness_calculator.calculate_fitness)
        if show_progress:
            print(generation[0])
    return np.array(best_fitnesses)
