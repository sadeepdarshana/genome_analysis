from matplotlib import pyplot as plt
from celluloid import Camera
import numpy as np


# create figure object
fig = plt.figure()
# load axis box

camera = Camera(fig)
for i in range(10):
    plt.scatter(i, np.random.random())
    plt.scatter(i, np.random.random())
    plt.scatter(i, np.random.random())
    plt.scatter(i, np.random.random())
    camera.snap()
    plt.scatter(i, np.random.random())
    plt.scatter(i, np.random.random())
    plt.scatter(i, np.random.random())
    plt.show()
camera.snap()
plt.show()
animation = camera.animate()
animation.save('test/animation.gif', writer='PillowWriter', fps=2)