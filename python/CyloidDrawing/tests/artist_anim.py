import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import itertools

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
ax.set_aspect('equal')

class MyCircle:

    def __init__(self):
        self.c = plt.Circle((5,-5), 0.75, fc='y')
        self.o = plt.Circle((5,-5), 0.75, fc='g')

    def update_frame(self, frame):

        x, y = self.c.center
        x = 5 + 3 * np.sin(np.radians(frame))
        y = 5 + 3 * np.cos(np.radians(frame))
        self.c.center = (x, y)

        x, y = self.c.center
        x = 5 + 3 * -1 * np.cos(np.radians(frame))
        y = 5 + 3 * -1 * np.sin(np.radians(frame))
        self.o.center = (x, y)

    def return_drawers(self):
        return self.c, self.o,

cir = MyCircle()

patch = cir.return_drawers()

def init():
    # patch.center = (5, 5)
    for obj in patch:
        ax.add_patch(obj)
        joy = obj
    return itertools.chain(patch, patch)

def animate(i):
    # x, y = patch.center
    # x = 5 + 3 * np.sin(np.radians(i))
    # y = 5 + 3 * np.cos(np.radians(i))
    # patch.center = (x, y)
    cir.update_frame(i)
    return patch

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=360,
                               interval=1,
                               blit=True)

plt.show()
