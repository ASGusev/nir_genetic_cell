import numpy as np
from operator import itemgetter
from multiprocessing import Pool
from data_parser import read_data
import rw
import math


type_particle = [('x', float), ('y', float), ('z', float)]


EPS = 1e-12
TRACK_SIZE = 50
MAX_DIST = 7 * 1e-7
DELTA_T = 1.8 * 1e-6
PART_CNT = 100
kB = 1.38e-23
T = 293
GEN_SIZE = 100
REPLACED_SHARE = 0.2
TO_REPLACE = int(GEN_SIZE * REPLACED_SHARE + EPS)
TO_KEEP = GEN_SIZE - TO_REPLACE
BOXES_ALONG = 20
BOX_SIDE = 2 * MAX_DIST / BOXES_ALONG
COORD_MULT = 1 / BOX_SIDE


def sort_with_fitness(generation, fitness_function):
    with Pool() as pool:
        fitnesses = pool.map(fitness_function, generation)
    sorted_zipped = sorted(zip(fitnesses, generation), key=itemgetter(0))
    sorted_beings = [being for (fitness, being) in sorted_zipped]
    sorted_fitnesses = [fitness for (fitness, being) in sorted_zipped]
    return sorted_beings, sorted_fitnesses


def cross(father, mother):
    from_father = np.random.choice((False, True), (BOXES_ALONG, BOXES_ALONG, BOXES_ALONG))
    from_mother = np.logical_not(from_father)
    child = np.zeros((BOXES_ALONG, BOXES_ALONG, BOXES_ALONG))
    child[from_father] = father[from_father]
    child[from_mother] = mother[from_mother]
    mutations = np.random.normal(1, 0.02, (BOXES_ALONG, BOXES_ALONG, BOXES_ALONG))
    child *= mutations
    return child


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


class FitnessCalculator:
    def __init__(self, data):
        self.rads = data[0]
        self.lengths = data[1]

    def diff(self, lengths):
        return abs(math.log(np.mean(np.array(lengths) / self.lengths)))

    def calculate_fitness(self, viscs):
        exp_lengths = []
        for rad, length in zip(self.rads, self.lengths):
            tracks, shifts = drive_particles(viscs, TRACK_SIZE, rad, PART_CNT)
            shifts_hor_lengths = np.sqrt(np.sum((shifts ** 2)[:, :, :-1], axis=2))
            track_lengths = np.sum(shifts_hor_lengths, axis=1)
            exp_lengths.append(track_lengths.mean())
        return self.diff(np.array(exp_lengths))


def next_generation(generation):
    children = []
    for father, mother in zip(generation[:TO_REPLACE:2],
                              generation[1:TO_REPLACE:2]):
        children.append(cross(father, mother))
        children.append(cross(father, mother))
    return generation[:TO_KEEP] + children


def prepare_data(data):
    horizontal_lengths = []
    rads = []
    for t in data:
        steps = t[1:, 1:] - t[:-1, 1:]
        horizontal_lengths.append(np.sum(np.sqrt(np.sum(steps ** 2, axis=1))))
        rads.append(t[0, 0])
    horizontal_lengths = np.array(horizontal_lengths)
    rads = np.array(rads)
    return np.vstack([rads, horizontal_lengths])


def calc(iterations_number, write_step, raw_filename, show_progress=True):
    data = read_data()
    fitness_calculator = FitnessCalculator(prepare_data(data))
    generation = [np.ones((BOXES_ALONG, BOXES_ALONG, BOXES_ALONG)) * 2.390041077895209e-06 for j in range(GEN_SIZE)]
    best_fitnesses = []
    step_writer = rw.StepWriter(raw_filename)

    for j in range(iterations_number):
        if show_progress:
            print('Iteration {}'.format(j + 1))
        sorted_generation, sorted_fitnesses = sort_with_fitness(generation,
                                                                fitness_calculator.calculate_fitness)
        best_fitness = sorted_fitnesses[0]
        best_fitnesses.append(best_fitness)

        if (j + 1) % write_step == 0:
            step_writer.write_step(j + 1, best_fitnesses[0], sorted_generation[0])

        generation = next_generation(sorted_generation[:GEN_SIZE])
        if show_progress:
            print(best_fitness)
    return np.array(best_fitnesses)
