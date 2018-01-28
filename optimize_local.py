import numpy as np
from operator import itemgetter
from multiprocessing import Pool
from data_parser import read_data
import rw
import scipy.optimize as soptimize
import random


type_particle = [('x', float), ('y', float), ('z', float)]


EPS = 1e-12
TRACK_SIZE = 50
MAX_DIST = 1e-6
DELTA_T = 1.8 * 1e-6  # 1.89 sec
PART_CNT = 100
kB = 1.38e-23
T = 293
GEN_SIZE = 50
REPLACED_SHARE = 0.2
TO_REPLACE = int(GEN_SIZE * REPLACED_SHARE + EPS)
TO_KEEP = GEN_SIZE - TO_REPLACE
BOXES_ALONG = 12
BOX_SIDE = 2 * MAX_DIST / BOXES_ALONG
COORD_MULT = 1 / BOX_SIDE
MSD_MAX_TIME = 10
MSD_TIMES = np.arange(1, MSD_MAX_TIME + 1)


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


def target_f(x, a, b):
    return b * x ** a


def track_to_params(track):
    time_disps = []
    for i in MSD_TIMES:
        displacements = np.sum((track[i:] - track[:-i]) ** 2, axis=1)
        time_disps.append(displacements.mean())
    time_disps = np.array(time_disps)
    c, _ = soptimize.curve_fit(target_f, MSD_TIMES, time_disps, [1, 1e-9])
    return c


def calc_length_fitness(shifts, length):
    shifts_hor_lengths = np.sqrt(np.sum((shifts ** 2)[:, :, :-1], axis=2))
    track_lengths = np.sum(shifts_hor_lengths, axis=1)
    exp_fitnesses = (track_lengths - length) ** 2
    return exp_fitnesses.mean()


def calc_param_fitness(tracks, disp_params):
    time_disps = []
    for i in MSD_TIMES:
        displacements = np.sum((tracks[:, i:, :2] - tracks[:, :-i, :2]) ** 2, axis=2)
        time_disps.append(displacements.mean())
    time_disps = np.array(time_disps)
    (p1, p2), _ = soptimize.curve_fit(target_f, MSD_TIMES, time_disps, [1, 1e-9])
    return (p1 - disp_params[0]) ** 2, (p2 - disp_params[1]) ** 2


def prepare_disp_params(data):
    return [track_to_params(t[:, 1:]) for t in data]


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


def mutate(visc):
    return visc * np.random.normal(1, 0.02, (BOXES_ALONG, BOXES_ALONG, BOXES_ALONG))


def cross(father, mother):
    from_father = np.random.choice((False, True), (BOXES_ALONG, BOXES_ALONG, BOXES_ALONG))
    from_mother = np.logical_not(from_father)
    child = np.zeros((BOXES_ALONG, BOXES_ALONG, BOXES_ALONG))
    child[from_father] = father[from_father]
    child[from_mother] = mother[from_mother]
    return mutate(child)


def dominates(a, b):
    return all(i < j for i, j in zip(a, b))


def make_fronts(generation):
    n = len(generation)
    dominators = [set() for i in generation]
    for i in range(n):
        for j in range(i):
            a = generation[i][1]
            b = generation[j][1]
            if dominates(a, b):
                dominators[j].add(i)
            elif dominates(a, b):
                dominators[i].add(j)
    fronts = []
    used = set()
    while True:
        new_front = set(filter(lambda x: len(dominators[x]) == 0 and x not in used, list(range(n))))
        if len(new_front) == 0:
            break
        front_set = set(new_front)
        used.update(front_set)
        for doms in dominators:
            doms.difference_update(front_set)
        new_front = [generation[i] for i in new_front]
        fronts.append(new_front)
    return fronts


def crowding_distance_sort(fronts):
    def sort_front(f):
        front = [i + ([float('inf'), float('inf'), float('inf')],) for i in f]
        n = len(front)
        for obj_id in range(3):
            front = sorted(front, key=lambda x: x[1][obj_id])
            diapason = front[n - 1][1][obj_id] - front[0][1][obj_id]
            for i in range(1, n - 1):
                front[i][2][obj_id] = (front[i + 1][1][obj_id] - front[i - 1][1][obj_id]) / diapason
        front = [(visc, fitness, sum(cds)) for visc, fitness, cds in front]
        return sorted(front, key=itemgetter(2))

    return [sort_front(f) for f in fronts]


def create_initial_population(fitness_function_calculator):
    ref = np.ones((BOXES_ALONG, BOXES_ALONG, BOXES_ALONG)) * 2.390041077895209e-06
    first_gen = [mutate(ref) for i in range(GEN_SIZE)]
    with Pool() as pool:
        fitness = pool.map(fitness_function_calculator, first_gen)
    first_gen = list(zip(first_gen, fitness))
    first_gen = make_fronts(first_gen)
    return crowding_distance_sort(first_gen)


def cut_gen(new_gen):
    i = 0
    rest = GEN_SIZE
    while rest > 0:
        if len(new_gen[i]) > rest:
            new_gen[i] = new_gen[i][:rest]
        rest -= len(new_gen[i])
        i += 1
    new_gen = new_gen[:i]
    return new_gen


def update_generation(parent_generation, fitness_calculator):
    parents = []
    for ind, front in enumerate(parent_generation):
        parents.extend([(visc, fitness, crowding_dist, ind)
                        for visc, fitness, crowding_dist in front])

    def choose_parent():
        candidate_1, candidate_2 = random.choice(parents), random.choice(parents)
        if candidate_1[3] == candidate_2[3] and candidate_1[2] > candidate_2[2] \
                or candidate_1[3] < candidate_2[3]:
            return candidate_1
        return candidate_2

    kids = []
    for i in range(TO_REPLACE):
        kids.append(cross(choose_parent()[0], choose_parent()[0]))
    with Pool() as pool:
        kid_fitnesses = pool.map(fitness_calculator, kids)
    kids = list(zip(kids, kid_fitnesses))
    parents = [(visc, fitness) for visc, fitness, _, _ in parents]
    new_gen = make_fronts(parents + kids)
    new_gen = crowding_distance_sort(new_gen)
    return cut_gen(new_gen)


def run(initial_population, iterations_number, write_step, final_population_filename=None,
        raw_filename=None, iterations_before=0, show_progress=True):
    fitness_calculator = FitnessCalculator(read_data())
    best_fitnesses = []
    if write_step > 0:
        step_writer = rw.StepWriter(raw_filename, iterations_before == 0)

    population = initial_population
    for j in range(iterations_number):
        if show_progress:
            print('Iteration {}'.format(j + 1))
        best_fitnesses.append([i for _, i, _ in population[0]])

        population = update_generation(population, fitness_calculator.calculate_fitness)

        if write_step > 0 and (j + 1) % write_step == 0:
            step_writer.write_step(j + 1, population[0][1], population[0][0])
    if final_population_filename is not None:
        final_population = []
        for front in population:
            final_population.extend(front)
        rw.write_population(final_population, iterations_before + iterations_number, final_population_filename)
    return np.array(best_fitnesses)
