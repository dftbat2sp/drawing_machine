"""
==================
Animated line plot
==================

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

# matplotlib.use("TkAgg")
fig = plt.figure()
ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))

x = np.arange(0, (2 * np.pi) - 0.5, 0.01)

sinx = np.sin(x)
cosx = np.cos(x)

line, = ax.plot([], [])


# line, = ax.plot(np.cos(x), np.sin(x))
# nonthing = ax.plot([0], [0])
# line = matplotlib.lines.Line2D(np.cos(x), np.sin(x))
# print(line)


def init():  # only required for blitting to give a clean slate.
    # line.set_ydata([np.nan] * len(x))
    line.set_data([], [])
    return line,
    # return matplotlib.lines.Line2D(np.cos(x[:1]), np.sin(x[:1])),


def animate(i):
    # y = np.arange(0 + (i/np.pi), (2 * np.pi) - 0.5 + (i/np.pi), 0.01)
    # line.set_ydata(np.sin(y))  # update the data.
    # line.set_xdata(np.cos(y))
    # line2, = ax.plot(np.cos(x[:i]), np.sin(x[:i]))
    i=i*2
    line.set_data(sinx[:i], cosx[:i])
    return line,
    # return matplotlib.lines.Line2D(np.cos(x[:i]), np.sin(x[:i])),


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=1, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
