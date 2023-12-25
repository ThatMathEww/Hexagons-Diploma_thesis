import cv2
import numpy as np
import os
import glob
import yaml

# Defining the dimensions of checkerboard
CHECKERBOARD = (6, 9)
n = "1280x960"   # "1280x960" # "1920x1080" # "2560x1440" # "canon"

output_file = f"calibration_{n}.yaml"
images = glob.glob(f'./{n}/*.jpg')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
obj_points = []
# Creating vector to store vectors of 2D points for each checkerboard image
img_points = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
for frame in images:
    img = cv2.imread(frame)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret:
        obj_points.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        img_points.append(corners2)

        im = cv2.cvtColor(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        # Nastavte tloušťku čáry
        line_thickness = 3

        # Nastavte velikost a tloušťku markeru
        marker_size = 20
        marker_thickness = 2

        # Nastavte velikost a tloušťku kolečka
        circle_radius = 10
        circle_thickness = 2

        # Vykreslete čáry spojující rohy
        for i in range(len(corners) - 1):
            start_point = np.int32(np.round(corners[i][0]))
            end_point = np.int32(np.round(corners[(i + 1) % len(corners)][0]))
            cv2.line(im, start_point, end_point, (255, 144, 30), line_thickness)

            cv2.circle(im, start_point, circle_radius, (30, 144, 255), thickness=circle_thickness)

            # Vykreslete marker pro každý roh
            cv2.drawMarker(im, start_point, (30, 144, 255), markerType=cv2.CALIB_CB_MARKER, markerSize=marker_size,
                           thickness=marker_thickness)

        cv2.drawMarker(im, end_point, (30, 144, 255), markerType=cv2.CALIB_CB_MARKER, markerSize=marker_size,
                       thickness=marker_thickness)

        cv2.circle(im, end_point, circle_radius, (30, 144, 255), thickness=circle_thickness)
        # cv2.imwrite(f"chess_.png", im)
        # Zobrazte výsledek
        # cv2.imshow('Vykreslené rohy', im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('corners', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    cv2.imshow('corners', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}

i = 1
while os.path.exists(output_file):
    output_file = f"calibration_{i}.yaml"
    i += 1

with open(output_file, "w") as f:
    # yaml.dump(data, f)
    pass

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)
