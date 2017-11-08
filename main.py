import sys
import optimize


if __name__ == '__main__':
    output_filename = sys.argv[2]
    best_fitnesses = optimize.calc(int(sys.argv[1]), int(sys.argv[3]), sys.argv[4])
    with open(output_filename, 'wt') as fout:
        fout.write(' '.join(map(str, best_fitnesses)))
