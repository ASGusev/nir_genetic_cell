import sys
import optimize
import optimize_local


if __name__ == '__main__':
    if sys.argv[1] == 'global':
        output_filename = sys.argv[3]
        best_fitnesses = optimize.calc(int(sys.argv[2]), int(sys.argv[4]), sys.argv[5])
        with open(output_filename, 'wt') as fout:
            fout.write(' '.join(map(str, best_fitnesses)))
    elif sys.argv[1] == 'local':
        output_filename = sys.argv[3]
        best_fitnesses = optimize_local.calc(int(sys.argv[2]), int(sys.argv[4]), sys.argv[5])
        with open(output_filename, 'wt') as fout:
            fout.write('\n'.join(map(lambda x: ' '.join(x), best_fitnesses)))
