import cv2
import numpy as np
import os
import yaml
import glob

n = "canon"  # "1280x960" # "1920x1080" # "2560x1440" # "canon"
with open(f'calibration_{n}.yaml') as f:
    loaded_dict = yaml.safe_load(f)

mtx_loaded = np.array(loaded_dict.get('camera_matrix'))
dist_loaded = np.array(loaded_dict.get('dist_coeff'))

images = glob.glob(f'./{n}/*.jpg')

for frame in images:
    img = cv2.imread(frame)
    # Oprava zkreslení
    img_undistorted = cv2.undistort(img, mtx_loaded, dist_loaded, None, mtx_loaded)

    # Zobrazení originálního a opraveného obrazu
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Modified', cv2.WINDOW_NORMAL)
    cv2.imshow('Original', img)
    cv2.imshow('Modified', img_undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
