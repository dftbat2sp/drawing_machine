import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=[6, 6])
ax = fig.add_subplot(111, frameon=True)
ax.set_aspect('equal')

lim = 10
ax.set_xlim(left=-lim, right=lim)
ax.set_ylim(bottom=-lim, top=lim)

line, = ax.plot([], [], linewidth=0.7)

data = np.array([-5-5j, -2+1j, 3+4j])

A = data[1] - data[0]
B = data[2] - data[1]
print(f'A: {A}')
print(f'B: {B}')

circle = plt.Circle((4, 3), 4, fill=False)
ax.add_artist(circle)

line.set_data(data[:].real,data[:].imag)


plt.show()