"""
================
pyplot animation
================

Generating an animation by calling `~.pyplot.pause` between plotting commands.

The method shown here is only suitable for simple, low-performance use.  For
more demanding applications, look at the :mod:`animation` module and the
examples that use it.

Note that calling `time.sleep` instead of `~.pyplot.pause` would *not* work.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

np.random.seed(19680801)
data = np.random.random((50, 50, 50))

fig = plt.figure()

gs = GridSpec(nrows=4, ncols=8)

plt.subplot(gs[0,3], title="0,0")

plt.subplot(gs[1,3], title="1,0")

plt.subplot(gs[2,3], title="2,0")

plt.subplot(gs[3,3], title="3,0")

plt.subplot(gs[3,0], title="3,1")

plt.subplot(gs[3,1], title="6")

plt.subplot(gs[3,2], title="3,3")

plt.subplot(gs[:, 4:], title="3,3")

plt.subplot(gs[:3,:3])

# plt.subplot(132, title="4")


plt.show()
