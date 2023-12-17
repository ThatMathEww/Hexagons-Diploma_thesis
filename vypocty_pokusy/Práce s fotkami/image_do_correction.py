import cv2
import numpy as np
import os
import yaml
import glob

with open('calibration.yaml') as f:
    loaded_dict = yaml.safe_load(f)

mtx_loaded = np.array(loaded_dict.get('camera_matrix'))
dist_loaded = np.array(loaded_dict.get('dist_coeff'))

images = glob.glob('./*.jpg')

for frame in images:
    img = cv2.imread(frame)
    # Oprava zkreslení
    img_undistorted = cv2.undistort(img, mtx_loaded, dist_loaded, None, mtx_loaded)

    # Zobrazení originálního a opraveného obrazu
    cv2.imshow('Original', img)
    cv2.imshow('Modified', img_undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
