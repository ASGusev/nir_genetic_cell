import sys
import optimize_local
import rw
from data_parser import read_data


if __name__ == '__main__':
    data_source = sys.argv[1]
    iterations = int(sys.argv[2])
    output_filename = sys.argv[3]
    final_population_filename = None
    raw_write_period = -1
    raw_filename = None
    single_objective = False
    pos = 4
    while pos < len(sys.argv):
        if sys.argv[pos] == '--so':
            single_objective = True
        if sys.argv[pos] == '--pop':
            pos += 1
            final_population_filename = sys.argv[pos]
        if sys.argv[pos] == '--snaps':
            raw_write_period = int(sys.argv[pos + 1])
            raw_filename = sys.argv[pos + 2]
            pos += 2
        pos += 1
    if data_source == 'create':
        print('Preparing first generation')
        initial_population = optimize_local.create_population(
            optimize_local.FitnessCalculator(read_data()).calculate_fitness)
        iterations_ran = 0
    else:
        initial_population, iterations_ran = rw.read_population(sys.argv[1])
    if not single_objective:
        initial_population = optimize_local.make_fronts(initial_population)
        initial_population = optimize_local.crowding_distance_sort(initial_population)

    if single_objective:
        best_fitnesses = optimize_local.run_so(initial_population, iterations, raw_write_period,
                                               final_population_filename, raw_filename, iterations_ran)
    else:
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
