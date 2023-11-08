import cv2
import numpy as np
import keyboard

circles = np.zeros((4, 2), int)
counter = 0


def mouse_points(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN and keyboard.is_pressed('shift'):
        print("Bod", counter + 1, "  x:", x, ",  y:", y)
        if counter == 0:
            cv2.putText(img, "1", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
        if counter < 4:
            cv2.circle(img, (x, y), 5, (0, 0, 255), 3)
            circles[counter] = x, y
        counter += 1


def calculate_angle(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    return np.arctan2(y2 - y1, x2 - x1)


def reorder(points):
    # Vypočítání středu bodů
    center = np.mean(points, axis=0)

    # Výpočet úhlů bodů od středu
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # Seřazení bodů podle úhlů
    sorted_indices = np.argsort(angles)

    # Přeuspořádání bodů
    sorted_points = points[sorted_indices]

    # Přesunutí prvního bodu na začátek
    first_index = np.where(sorted_indices == 0)[0][0]
    sorted_points = np.roll(sorted_points, -first_index, axis=0)
    sorted_points = np.array(sorted_points, dtype=np.float32).reshape((4, 1, 2))

    return sorted_points


path = "book.jpg"

img = cv2.imread(path)
input_img = img.copy()

height, width, _ = img.shape

while True:
    if keyboard.is_pressed('escape') or keyboard.is_pressed('enter'):
        cv2.destroyAllWindows()
        break

    if keyboard.is_pressed('r'):
        img = cv2.imread(path)
        input_img = img.copy()
        print("")
        if counter > 3:
            cv2.destroyWindow("Output image")
        counter = 0
        circles = np.zeros((4, 2), int)

    if counter == 4:
        w, h = 250, 350
        pts1 = reorder(circles)
        pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        output_img = cv2.warpPerspective(input_img, matrix, (w, h))
        cv2.namedWindow('Output image', cv2.WINDOW_NORMAL)
        cv2.imshow("Output image", output_img)

    cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original image', int(0.55 * width), int(0.55 * height))
    cv2.setMouseCallback('Original image', mouse_points)
    cv2.imshow("Original image", img)
    cv2.waitKey(1)
