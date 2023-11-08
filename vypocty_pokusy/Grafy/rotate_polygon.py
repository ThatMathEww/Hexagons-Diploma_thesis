import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as tr

# define figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('auto', adjustable='box')

plt.grid()

# rectangle spec
polygon = np.array([[1, 1], [2, 1], [1.5, 2]])

print('rotate_deg_around')
# rectangle1
rect = plt.Polygon(polygon, closed=True, fill=False, ec="green", lw=2.0)
ax.add_patch(rect)

# rectangle3(rotate_deg_around)
CENTER = np.mean(polygon, axis=0)
plt.scatter(CENTER[0], CENTER[1])
rect_tr2 = tr.Affine2D().rotate_deg_around(CENTER[0], CENTER[1], 180) + ax.transData

rect3 = plt.Polygon(polygon, closed=True, fill=False, ec="red", lw=2.0, transform=rect_tr2)
ax.add_patch(rect3)

plt.show()
