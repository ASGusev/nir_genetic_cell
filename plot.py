import sys
import numpy as np
import matplotlib.pyplot as plt


data_file = sys.argv[1]
mins = []
with open(data_file, 'rt') as fin:
    for line in fin.readlines():
        gen_front = np.array([list(map(float, fitness.split(' '))) for fitness in line.split('\t')])
        mins.append(gen_front.min(axis=0))
mins = np.array(mins).T
n = 3
fig = plt.figure(figsize=(n * 5, 5))
plt.subplot(1, n, 1)
plt.title('Length fitness')
plt.plot(mins[0])
plt.subplot(1, n, 2)
plt.title('Grade')
plt.plot(mins[1])
plt.subplot(1, n, 3)
plt.title('Coefficient')
plt.plot(mins[2])
fig.tight_layout()
plt.show()