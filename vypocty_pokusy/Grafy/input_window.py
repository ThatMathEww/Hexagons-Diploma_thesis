import matplotlib.pyplot as plt
import numpy as np
# import cv2

# import matplotlib.transforms as tr
from matplotlib.widgets import TextBox

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

polygon_cor = np.array([[-2, -2], [2, -2], [0, 4]], dtype=np.float64)
polygon = plt.Polygon(polygon_cor, closed=True, fill=False, ec="green", lw=2.0)
center_cor = np.mean(polygon_cor, axis=0, dtype=np.float64)
s = ax.scatter(*center_cor)
ax.add_patch(polygon)


def submit_r(_):
    global polygon_cor, center_cor
    try:
        rot = round(float(eval(text_box3.text.replace(",", "."), {'np': np})), 5)
        rot = int(rot) if rot.is_integer() else rot
        if not 0 <= rot < 359:
            text_box3.set_val(f"{rot % 360}")
        else:
            text_box3.set_val(f"{rot}")
        rot = np.deg2rad(rot)
        scale = round(max(float(text_box4.text.replace(",", ".")), 0), 5)
        scale = int(scale) if scale.is_integer() else scale
        text_box4.set_val(f"{scale}")

        # Vytvoření transformační matice pro rotaci
        rotation_matrix = np.array([[np.cos(rot), -np.sin(rot)],
                                    [np.sin(rot), np.cos(rot)]]) * scale
        """rotation_matrix = cv2.getRotationMatrix2D(center_cor, -rot, scale)
        rotated_points = cv2.transform(polygon_cor.reshape(-1, 1, 2), rotation_matrix).reshape(-1,2)"""
        polygon.set_xy(np.dot(polygon_cor - center_cor, rotation_matrix.T) + center_cor)
        s.set_offsets(center_cor)
        """tr_mat = tr.Affine2D().rotate_deg_around(*center, rot)
        polygon.set_xy([tr_mat.transform_point(p) for p in polygon_cor])
        print(polygon_cor)"""
        # polygon.remove()
        # polygon = plt.Polygon(polygon_cor, closed=True, fill=False, ec="red", lw=2.0, transform=rect_tr)
        # ax.add_patch(polygon)
        """ax.relim()
        ax.autoscale_view()"""
        fig.canvas.draw()
    except (ValueError, SyntaxError):
        pass


def submit_x(num):
    global polygon_cor, center_cor
    try:
        points = polygon.get_path().vertices
        c = float(num.replace(",", ".")) - center_cor[0]
        polygon_cor[:, 0] = polygon_cor[:, 0] + c
        points[:, 0] = points[:, 0] + c
        center_cor[0] = center_cor[0] + c
        polygon.set_xy(points)
        s.set_offsets(center_cor)
        """ax.relim()
        ax.autoscale_view()"""
        fig.canvas.draw()
    except (ValueError, SyntaxError):
        pass


def submit_y(num):
    global polygon_cor, center_cor
    try:
        points = polygon.get_path().vertices
        c = float(num.replace(",", ".")) - center_cor[1]
        points[:, 1] = points[:, 1] + c
        polygon_cor[:, 1] = polygon_cor[:, 1] + c
        center_cor[1] = center_cor[1] + c
        polygon.set_xy(points)
        s.set_offsets(center_cor)
        """ax.relim()
        ax.autoscale_view()"""
        fig.canvas.draw()
    except (ValueError, SyntaxError):
        pass


ax_box1 = fig.add_axes([0.1, 0.05, 0.15, 0.05])
ax_box2 = fig.add_axes([0.1 + 0.15 + 0.05, 0.05, 0.15, 0.05])
ax_box3 = fig.add_axes([0.1 + 0.5, 0.05, 0.15, 0.05])
ax_box4 = fig.add_axes([0.1 + 0.75, 0.05, 0.1, 0.05])
text_box1 = TextBox(ax_box1, "X: ", textalignment="center",
                    initial=f"{int(center_cor[0])}" if center_cor[0].is_integer() else f"{center_cor[0]:.2f}")
text_box2 = TextBox(ax_box2, "Y: ", textalignment="center",
                    initial=f"{int(center_cor[1])}" if center_cor[1].is_integer() else f"{center_cor[1]:.2f}")
text_box3 = TextBox(ax_box3, "Rotation: ", textalignment="center", initial="0")
text_box4 = TextBox(ax_box4, "Scale: ", textalignment="center", initial="1")
# text_box3.set_val("0")  # Trigger `submit` with the initial string.
text_box1.on_submit(submit_x)
text_box2.on_submit(submit_y)
text_box3.on_submit(submit_r)
text_box4.on_submit(submit_r)

ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

plt.show()
