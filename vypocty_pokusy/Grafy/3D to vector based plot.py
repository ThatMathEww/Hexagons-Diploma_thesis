import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams["figure.figsize"] = [7.00, 3.50]
#plt.rcParams["figure.autolayout"] = True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

ax.plot_wireframe(x, y, z, color="red")

ax.set_title("Sphere")

fig.savefig("test.pdf", format="pdf", bbox_inches='tight')

plt.show()