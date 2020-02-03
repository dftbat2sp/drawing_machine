# sphinx_gallery_thumbnail_number = 3
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib

"""
example 1
"""
# matplotlib.use("TkAgg")
#
# fig = plt.figure()  # an empty figure with no axes
# fig.suptitle('No axes on this figure')  # Add a title so we know which it is
#
# fig, ax_lst = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
#
# plt.show()
"""
/ex1
"""

"""
ex2
"""
# x = np.linspace(0, 2, 100)
#
# plt.plot(x, x, label='linear')
# plt.plot(x, x ** 2, label='quadratic')
# plt.plot(x, x ** 3, label='cubic')
# plt.plot(x, x ** 4, label="4's")
#
# plt.xlabel('x label')
# plt.ylabel('y label')
#
# plt.title('Simple Plot')
# plt.legend()
#
# plt.show()
"""
/ex2
"""

"""
ex3
"""
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# y = np.random.rand(100000)
# y[50000:] *= 2
# y[np.logspace(1, np.log10(50000), 400).astype(int)] = -1
# mpl.rcParams['path.simplify'] = True
#
# mpl.rcParams['path.simplify_threshold'] = 0.0
# plt.plot(y)
# plt.show()
#
# mpl.rcParams['path.simplify_threshold'] = 1.0
# plt.plot(y)
# plt.show()
"""
/ex3
"""

"""
ex4
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

y = np.random.rand(100000)
y[50000:] *= 2
y[np.logspace(1, np.log10(50000), 400).astype(int)] = -1
mpl.rcParams['path.simplify'] = True

mpl.rcParams['agg.path.chunksize'] = 0
plt.plot(y)
plt.show()

mpl.rcParams['agg.path.chunksize'] = 10000
plt.plot(y)
plt.show()
"""
/ex4
"""
