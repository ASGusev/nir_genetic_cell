import sys
import optimize_local
import rw
from data_parser import read_data


if __name__ == '__main__':
    if sys.argv[1] == 'create':
        print('Preparing first generation')
        initial_population = optimize_local.create_initial_population(
            optimize_local.FitnessCalculator(read_data()).calculate_fitness)
        iterations_ran = 0
    else:
        initial_population, iterations_ran = rw.read_population(sys.argv[1])
        initial_population = optimize_local.make_fronts(initial_population)
        initial_population = optimize_local.crowding_distance_sort(initial_population)
    iterations = int(sys.argv[2])
    output_filename = sys.argv[3]
    if len(sys.argv) > 4:
        final_population_filename = sys.argv[4]
    else:
        final_population_filename = None
    if len(sys.argv) > 5:
        raw_write_period = int(sys.argv[5])
        raw_filename = sys.argv[6]
    else:
        raw_write_period = -1
        raw_filename = None

    best_fitnesses = optimize_local.run(initial_population, iterations, raw_write_period,
                                        final_population_filename, raw_filename, iterations_ran)
    if iterations_ran == 0:
        fout_params = 'wt'
    else:
        fout_params = 'at'
    with open(output_filename, fout_params) as fout:
            for gen in best_fitnesses:
                fout.write('\t'.join([' '.join([str(i) for i in fitness]) for fitness in gen]))
                fout.write('\n')
