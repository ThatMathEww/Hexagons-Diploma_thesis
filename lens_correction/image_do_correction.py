import cv2
import numpy as np
import os
import yaml
# import glob

photos_main_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
show_photos = False
n = "2560x1440"  # "1280x960" # "1920x1080" # "2560x1440" # "canon"
image_original_shape = (1440, 2560)
delete_all_existing_files = True

with open(f'calibration_{n}.yaml') as f:
    loaded_dict = yaml.safe_load(f)

mtx_loaded = np.array(loaded_dict.get('camera_matrix'))
dist_loaded = np.array(loaded_dict.get('dist_coeff'))

images_folders = [name for name in os.listdir(photos_main_folder) if name.startswith("H02")]

for folder in images_folders:
    input_path = os.path.join(photos_main_folder, folder, 'detail_original')
    output_path = os.path.join(photos_main_folder, folder, 'detail_modified')

    if not os.path.isdir(input_path):
        print(f"Folder {folder} does not exist.")
        continue

    if not os.path.isdir(output_path):
        print('Output folder was created.')
        os.makedirs(output_path)
    else:
        # Smazání všech souborů
        if delete_all_existing_files:
            [os.remove(f) for f in [os.path.join(output_path, f) for f in os.listdir(output_path)] if os.path.isfile(f)]
        print('Output folder already exists.')

    images = [name for name in os.listdir(input_path) if name.endswith((".JPG", ".jpg", ".png"))]

    for frame in images:
        img = cv2.imread(os.path.join(input_path, frame))
        img_undistorted = np.zeros((*image_original_shape, 3), dtype=np.uint8)

        h_big, w_big = image_original_shape
        h_small, w_small = img.shape[:2]

        start_x = (w_big - w_small) // 2
        start_y = (h_big - h_small) // 2

        img_undistorted[start_y:start_y + h_small, start_x:start_x + w_small] = img

        # Oprava zkreslení
        img_undistorted = cv2.undistort(img_undistorted, mtx_loaded, dist_loaded, None, mtx_loaded)

        img_undistorted = img_undistorted[start_y:start_y + h_small, start_x:start_x + w_small]

        if show_photos:
            # Zobrazení originálního a opraveného obrazu
            cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Modified', cv2.WINDOW_NORMAL)
            cv2.imshow('Original', img)
            cv2.imshow('Modified', img_undistorted)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if 'max' not in folder:
            img_undistorted = img_undistorted[:, :1200]

        cv2.imwrite(os.path.join(output_path, 'mod-' + frame), cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2GRAY))
    print(f"Folder {folder} is done.")
