import os
import sys
import cv2
# import time
import keyboard
import numpy as np


def bar1(_):
    global brightness
    brightness = (cv2.getTrackbarPos(names[0], "Parameters") - 50) * 5


def bar2(_):
    global contrast
    contrast = (cv2.getTrackbarPos(names[1], "Parameters") / 50)


def bar3(_):
    # global gamma, hls_image, hls_to_brg_image
    """gamma = (cv2.getTrackbarPos(names[2], "Parameters")) / 50

    hls_to_brg_image = cv2.cvtColor(image1, cv2.COLOR_BGR2HLS_FULL)

    # Úprava hodnoty barevného kanálu S (sytost) pro změnu sytosti
    hsv_image[:, :, 1] = hls_to_brg_image[:, :, 1] * gamma

    hls_to_brg_image = cv2.cvtColor(hsv_image, cv2.COLOR_HLS2BGR_FULL)"""


def bar4(_):
    global temp, hsv_to_bgr_image
    temp = (cv2.getTrackbarPos(names[3], "Parameters") - 50) * -2
    # Úprava a omezení hodnoty barevného kanálu na rozmezí 0-179

    im = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV_FULL)
    im[:, :, 0] = np.uint8(np.clip(im[:, :, 0] + temp, 0, 255))  # 179
    hsv_to_bgr_image[:, :, 0] = cv2.cvtColor(im, cv2.COLOR_HSV2BGR_FULL)[:, :, 0]


def bar5(_):
    global saturate, hsv_to_bgr_image
    saturate = (cv2.getTrackbarPos(names[4], "Parameters")) / 50

    im = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV_FULL)
    im[:, :, 1] = np.uint8(np.clip(im[:, :, 1] * saturate, 0, 255))
    hsv_to_bgr_image[:, :, 1] = cv2.cvtColor(im, cv2.COLOR_HSV2BGR_FULL)[:, :, 1]


def bar6(_):
    global red, bgr
    bgr[2] = cv2.getTrackbarPos(names[5], "Parameters") / 100
    red = red_channel * bgr[2]


def bar7(_):
    global green, bgr
    bgr[1] = cv2.getTrackbarPos(names[6], "Parameters") / 100
    green = green_channel * bgr[1]


def bar8(_):
    global blue, bgr
    bgr[0] = cv2.getTrackbarPos(names[7], "Parameters") / 100
    blue = blue_channel * bgr[0]


def bar9(_):
    global left
    left = int(cv2.getTrackbarPos(names[8], "Crop"))


def bar10(_):
    global right
    right = int(width - cv2.getTrackbarPos(names[9], "Crop"))


def bar11(_):
    global top
    top = int(cv2.getTrackbarPos(names[10], "Crop"))


def bar12(_):
    global down
    down = int(height - cv2.getTrackbarPos(names[11], "Crop"))


def bar_lz(_):
    global left_z
    left_z = int(min(cv2.getTrackbarPos(names[8], "Zoom"), right - left - 101))


def bar_tz(_):
    global top_z
    top_z = int(min(cv2.getTrackbarPos(names[10], "Zoom"), down - top - 101))


def bar13(_):
    global V, hsv_to_bgr_image
    V = cv2.getTrackbarPos("HSV - V", "Other settings") / 50

    im = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV_FULL)
    im[:, :, 2] = np.uint8(np.clip(im[:, :, 2] * V, 0, 255))
    hsv_to_bgr_image[:, :, 2] = cv2.cvtColor(im, cv2.COLOR_HSV2BGR_FULL)[:, :, 2]


def bar14(_):
    global L, lab_to_bgr_image
    L = cv2.getTrackbarPos("LAB - L", "Other settings") / 50

    im = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    im[:, :, 0] = np.uint8(np.clip(im[:, :, 0] * L, 0, 255))
    lab_to_bgr_image[:, :, 0] = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)[:, :, 0]


def bar15(_):
    global A, lab_to_bgr_image
    A = cv2.getTrackbarPos("A", "Other settings") / 50

    im = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    im[:, :, 1] = np.uint8(np.clip(im[:, :, 1] * A, 0, 255))
    lab_to_bgr_image[:, :, 1] = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)[:, :, 1]


def bar16(_):
    global B, lab_to_bgr_image
    B = cv2.getTrackbarPos("B", "Other settings") / 50

    im = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB)
    im[:, :, 2] = np.uint8(np.clip(im[:, :, 2] * B, 0, 255))
    lab_to_bgr_image[:, :, 2] = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)[:, :, 2]


def bar17(_):
    global Y, yuv_to_bgr_image
    Y = cv2.getTrackbarPos("YUV - Y", "Other settings") / 50

    im = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
    im[:, :, 0] = np.uint8(np.clip(im[:, :, 0] * Y, 0, 255))
    yuv_to_bgr_image[:, :, 0] = cv2.cvtColor(im, cv2.COLOR_YUV2BGR)[:, :, 0]


def bar18(_):
    global U, yuv_to_bgr_image
    U = cv2.getTrackbarPos("U", "Other settings") / 50

    im = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
    im[:, :, 1] = np.uint8(np.clip(im[:, :, 1] * U, 0, 255))
    yuv_to_bgr_image[:, :, 1] = cv2.cvtColor(im, cv2.COLOR_YUV2BGR)[:, :, 1]


def bar19(_):
    global V2, yuv_to_bgr_image
    V2 = cv2.getTrackbarPos("V", "Other settings") / 50

    im = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
    im[:, :, 2] = np.uint8(np.clip(im[:, :, 2] * V2, 0, 255))
    yuv_to_bgr_image[:, :, 2] = cv2.cvtColor(im, cv2.COLOR_YUV2BGR)[:, :, 2]


"""def apply_rgb_effect(image):
    image[:, :, :3] = np.dstack((blue, green, red))
    return image


# Funkce pro úpravy v prostoru HSV
def apply_hsv_effect(image, hsv):
    modified_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    modified_image[:, :, :3] = np.multiply(hsv[:, :, :3], np.array([temp, saturate, V]))
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_HSV2BGR_FULL)
    return modified_image


# Funkce pro úpravy v prostoru Lab
def apply_lab_effect(image, lab):
    modified_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    modified_image[:, :, :3] = np.multiply(lab[:, :, :3], np.array([L, A, B]))
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_LAB2BGR)
    return modified_image


# Funkce pro úpravy v prostoru YUV
def apply_yuv_effect(image, yuv):
    modified_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    modified_image[:, :, :3] = np.multiply(yuv[:, :, :3], np.array([Y, U, V2]))
    modified_image = cv2.cvtColor(modified_image, cv2.COLOR_YUV2RGB)
    return modified_image"""


def make_photo(input_image):
    input_image = input_image[top:down, left:right]

    image_temp = input_image.copy()
    cvt_im = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV_FULL)
    im = cvt_im.copy()
    im[:, :, 0] = np.uint8(np.clip(im[:, :, 0] + temp, 0, 255))
    image_temp[:, :, 0] = cv2.cvtColor(im, cv2.COLOR_HSV2BGR_FULL)[:, :, 0]
    im = cvt_im.copy()
    im[:, :, 1] = np.uint8(np.clip(im[:, :, 1] * saturate, 0, 255))
    image_temp[:, :, 1] = cv2.cvtColor(im, cv2.COLOR_HSV2BGR_FULL)[:, :, 1]
    im = cvt_im.copy()
    im[:, :, 2] = np.uint8(np.clip(im[:, :, 2] * V, 0, 255))
    image_temp[:, :, 2] = cv2.cvtColor(im, cv2.COLOR_HSV2BGR_FULL)[:, :, 2]

    image_out = np.minimum(np.uint8(np.clip(input_image.copy() * bgr, 0, 255)), image_temp)

    image_temp = input_image.copy()
    cvt_im = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
    im = cvt_im.copy()
    im[:, :, 0] = np.uint8(np.clip(im[:, :, 0] * L, 0, 255))
    image_temp[:, :, 0] = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)[:, :, 0]
    im = cvt_im.copy()
    im[:, :, 1] = np.uint8(np.clip(im[:, :, 1] * A, 0, 255))
    image_temp[:, :, 1] = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)[:, :, 1]
    im = cvt_im.copy()
    im[:, :, 2] = np.uint8(np.clip(im[:, :, 2] * B, 0, 255))
    image_temp[:, :, 2] = cv2.cvtColor(im, cv2.COLOR_LAB2BGR)[:, :, 2]

    image_out = np.minimum(image_out, image_temp)

    image_temp = input_image.copy()
    cvt_im = cv2.cvtColor(input_image, cv2.COLOR_BGR2YUV)
    im = cvt_im.copy()
    im[:, :, 0] = np.uint8(np.clip(im[:, :, 0] * Y, 0, 255))
    image_temp[:, :, 0] = cv2.cvtColor(im, cv2.COLOR_YUV2BGR)[:, :, 0]
    im = cvt_im.copy()
    im[:, :, 1] = np.uint8(np.clip(im[:, :, 1] * U, 0, 255))
    image_temp[:, :, 1] = cv2.cvtColor(im, cv2.COLOR_YUV2BGR)[:, :, 1]
    im = cvt_im.copy()
    im[:, :, 2] = np.uint8(np.clip(im[:, :, 2] * V2, 0, 255))
    image_temp[:, :, 2] = cv2.cvtColor(im, cv2.COLOR_YUV2BGR)[:, :, 2]

    image_out = np.minimum(image_out, image_temp)

    if make_gray:
        return cv2.cvtColor(cv2.convertScaleAbs(image_out, alpha=contrast, beta=brightness), cv2.COLOR_BGR2GRAY)
    else:
        return cv2.convertScaleAbs(image_out, alpha=contrast, beta=brightness)


def get_photos_from_folder(folder, img_types=(".jpg", ".jpeg", ".jpe", ".JPG", ".jp2", ".png", ".bmp", ".dib", ".webp",
                                              ".avif", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".pfm", ".sr", ".ras",
                                              ".tiff", ".tif", ".exr", ".hdr", ".pic")):
    if any(item not in (".jpg", ".jpeg", ".jpe", ".JPG", ".jp2", ".png", ".bmp", ".dib", ".webp",
                        ".avif", ".pbm", ".pgm", ".ppm", ".pxm", ".pnm", ".pfm", ".sr", ".ras",
                        ".tiff", ".tif", ".exr", ".hdr", ".pic") for item in img_types):
        sys.exit("Chyba v typu fotek.")
    else:
        folders = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and
                   f.lower().endswith(img_types) and not f.startswith("0.")]
        first_type = os.path.splitext(folders[0])[1]
        if all(os.path.splitext(f)[1] == first_type for f in folders):  # kontrola jestli josu všechny fotky stejné
            return folders
        else:
            sys.exit("Různé typy fotek.")


def create_image_window(name, image):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, int(0.1 * width), int(0.1 * height))
    cv2.imshow(name, image)


def create_param_bar():
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 700, 0)
    cv2.createTrackbar(names[0], "Parameters", int((brightness / 5 + 50)), 100, bar1)
    cv2.createTrackbar(names[1], "Parameters", int(contrast * 50), 100, bar2)
    # cv2.createTrackbar(names[2], "Parameters", 50, 100, bar3)
    cv2.createTrackbar(names[3], "Parameters", int(temp / -2 + 50), 100, bar4)
    cv2.createTrackbar(names[4], "Parameters", int(saturate * 50), 100, bar5)
    cv2.createTrackbar(names[5], "Parameters", int(bgr[2] * 100), 100, bar6)
    cv2.createTrackbar(names[6], "Parameters", int(bgr[1] * 100), 100, bar7)
    cv2.createTrackbar(names[7], "Parameters", int(bgr[0] * 100), 100, bar8)
    cv2.resizeWindow("Parameters", 350, 150)


def create_other_set_bar():
    cv2.namedWindow("Other settings")
    cv2.resizeWindow("Other settings", 700, 0)
    cv2.createTrackbar("HSV - V", "Other settings", int(V * 50), 100, bar13)
    cv2.createTrackbar("LAB - L", "Other settings", int(L * 50), 100, bar14)
    cv2.createTrackbar("A", "Other settings", int(A * 50), 100, bar15)
    cv2.createTrackbar("B", "Other settings", int(B * 50), 100, bar16)
    cv2.createTrackbar("YUV - Y", "Other settings", int(Y * 50), 100, bar17)
    cv2.createTrackbar("U", "Other settings", int(U * 50), 100, bar18)
    cv2.createTrackbar("V", "Other settings", int(V2 * 50), 100, bar19)
    cv2.resizeWindow("Other settings", 350, 150)


def create_crop_bar():
    cv2.namedWindow("Crop")
    cv2.resizeWindow("Crop", 700, 0)
    cv2.createTrackbar(names[8], "Crop", int(left), int(width / 2) - 1, bar9)
    cv2.createTrackbar(names[9], "Crop", width - right, int(width / 2) - 1, bar10)
    cv2.createTrackbar(names[10], "Crop", int(top), int(height / 2) - 1, bar11)
    cv2.createTrackbar(names[11], "Crop", height - down, int(height / 2) - 1, bar12)
    cv2.resizeWindow("Crop", 350, 85)


def create_zoom_crop_bar():
    cv2.namedWindow("Zoom")
    cv2.resizeWindow("Zoom", 700, 0)
    cv2.createTrackbar(names[8], "Zoom", int(left_z), int(width), bar_lz)
    cv2.createTrackbar(names[10], "Zoom", int(top_z), int(height), bar_tz)
    cv2.resizeWindow("Zoom", 350, 40)


def change_image(image, last_image):
    global image1, yuv_to_bgr_image, hsv_to_bgr_image, lab_to_bgr_image, names, restart, first_load_profile
    global brightness, contrast, temp, saturate, left, right, top, down, V, L, A, B, Y, V, U, V2
    global bgr, red, green, blue, red_channel, green_channel, blue_channel, width, height, profile
    global left_z, top_z

    image1 = image.copy()
    height, width = image1.shape[:2]

    names = ["Jas", "Kontrast", "Gama", "Teplota", "Sytost", "Red", "Green", "Blue", "Vlevo", "Vpravo", "Nahore",
             "Dole"]

    close = False
    while True:
        brightness = contrast = saturate = V = L = A = B = Y = U = V2 = 1
        temp = 0
        left, right, top, down = 0, width, 0, height
        left_z, top_z = 0, 0

        blue_channel, green_channel, red_channel = image1[:, :, 0], image1[:, :, 1], image1[:, :, 2]
        blue, green, red = blue_channel, green_channel, red_channel
        bgr = np.array([1, 1, 1], dtype=np.float64)

        yuv_to_bgr_image, hsv_to_bgr_image, lab_to_bgr_image = image1.copy(), image1.copy(), image1.copy()

        # Načtení profilu
        if (os.path.exists(profile + '.txt') or restart) and first_load_profile:
            if os.path.exists(profile + '.txt'):
                try:
                    loaded_profile = np.loadtxt(profile + '.txt', dtype=float)
                    brightness, contrast, temp, saturate, V, L, A, B, Y, V, U, V2 = loaded_profile[0:12]
                    bgr = loaded_profile[12:15]
                    left, right, top, down = map(int, loaded_profile[15:19])
                    del loaded_profile
                except ValueError:
                    print("Profil se nepovedlo načíst")
                    pass
            if restart:
                if os.path.exists('profile.txt'):
                    try:
                        loaded_profile = np.loadtxt('profile.txt', dtype=float)
                        brightness, contrast, temp, saturate, V, L, A, B, Y, V, U, V2 = loaded_profile[0:12]
                        bgr = loaded_profile[12:15]
                        left, right, top, down = map(int, loaded_profile[15:19])
                        del loaded_profile
                        print("\nProfil načten.")

                    except ValueError:
                        print("Profil se nepovedlo načíst")
                else:
                    print('Profil "profile.txt" nenalezen.')

        first_load_profile = True
        restart = False

        cv2.namedWindow("Original image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original image", int(0.1 * width), int(0.1 * height))

        cv2.namedWindow("Modified image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Modified image", int(0.1 * width), int(0.1 * height))

        cv2.namedWindow("Modified gray image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Modified gray image", int(0.1 * width), int(0.1 * height))

        cv2.namedWindow("Modified last gray image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Modified last gray image", int(0.1 * width), int(0.1 * height))

        cv2.namedWindow("Modified zoomed gray image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Modified zoomed gray image", int(0.1 * width), int(0.1 * width))

        create_param_bar()

        create_other_set_bar()

        create_crop_bar()

        create_zoom_crop_bar()

        while True:
            if keyboard.is_pressed('escape') or keyboard.is_pressed('enter') and keyboard.is_pressed('space') or \
                    keyboard.is_pressed('r') and keyboard.is_pressed('shift') or \
                    keyboard.is_pressed('s') and keyboard.is_pressed('shift') or \
                    keyboard.is_pressed('l') and keyboard.is_pressed('shift'):

                if keyboard.is_pressed('escape') or keyboard.is_pressed('enter') and keyboard.is_pressed('space'):
                    close = True
                    cv2.destroyAllWindows()
                    break

                elif keyboard.is_pressed('r') and keyboard.is_pressed('shift'):
                    cv2.destroyAllWindows()
                    break

                elif keyboard.is_pressed('s') and keyboard.is_pressed('shift'):
                    data = [brightness, contrast, temp, saturate, V, L, A, B, Y, V, U, V2]
                    data.extend(bgr)
                    data.extend([left, right, top, down])
                    np.savetxt(profile + '.txt', data)
                    print("\nProfil uložen.")

                elif keyboard.is_pressed('l') and keyboard.is_pressed('shift'):
                    if os.path.exists(profile + '.txt'):
                        print("\nZahajuji načítání profilu.")
                        restart, close = True, True
                        cv2.destroyAllWindows()
                        break
                    else:
                        print('Profil "profile.txt" nenalezen.')

            fin_img = cv2.convertScaleAbs(
                np.minimum(np.concatenate((blue[top:down, left:right, np.newaxis],
                                           green[top:down, left:right, np.newaxis],
                                           red[top:down, left:right, np.newaxis]), axis=2),
                           np.minimum(hsv_to_bgr_image[top:down, left:right],
                                      np.minimum(yuv_to_bgr_image[top:down, left:right],
                                                 lab_to_bgr_image[top:down, left:right]))),
                alpha=contrast, beta=brightness)

            modified_img_gray = cv2.cvtColor(fin_img, cv2.COLOR_BGR2GRAY)

            # Kontrola, zda bylo okno zavřeno
            if cv2.getWindowProperty("Original image", cv2.WND_PROP_VISIBLE) < 1 or \
                    cv2.getWindowProperty("Modified image", cv2.WND_PROP_VISIBLE) < 1 or \
                    cv2.getWindowProperty("Modified gray image", cv2.WND_PROP_VISIBLE) < 1 or \
                    cv2.getWindowProperty("Modified last gray image", cv2.WND_PROP_VISIBLE) < 1 or \
                    cv2.getWindowProperty("Modified zoomed gray image", cv2.WND_PROP_VISIBLE) < 1 or \
                    cv2.getWindowProperty("Parameters", cv2.WND_PROP_VISIBLE) < 1 or \
                    cv2.getWindowProperty("Other settings", cv2.WND_PROP_VISIBLE) < 1 or \
                    cv2.getWindowProperty("Crop", cv2.WND_PROP_VISIBLE) < 1:

                if cv2.getWindowProperty("Original image", cv2.WND_PROP_VISIBLE) < 1:
                    create_image_window("Original image", image1)
                if cv2.getWindowProperty("Modified image", cv2.WND_PROP_VISIBLE) < 1:
                    create_image_window("Modified image", modified_img_gray)
                if cv2.getWindowProperty("Modified last gray image", cv2.WND_PROP_VISIBLE) < 1:
                    create_image_window("Modified last gray image", last_image[top:down, left:right])
                if cv2.getWindowProperty("Modified zoomed gray image", cv2.WND_PROP_VISIBLE) < 1:
                    create_image_window("Modified zoomed gray image",
                                        modified_img_gray[top_z:top_z + 100, left_z:left_z + 100])
                    cv2.resizeWindow("Modified zoomed gray image", int(0.1 * width), int(0.1 * width))
                if cv2.getWindowProperty("Modified gray image", cv2.WND_PROP_VISIBLE) < 1:
                    create_image_window("Modified gray image", fin_img)
                if cv2.getWindowProperty("Parameters", cv2.WND_PROP_VISIBLE) < 1:
                    create_param_bar()
                if cv2.getWindowProperty("Other settings", cv2.WND_PROP_VISIBLE) < 1:
                    create_other_set_bar()
                if cv2.getWindowProperty("Crop", cv2.WND_PROP_VISIBLE) < 1:
                    create_crop_bar()

            cv2.imshow("Modified image", fin_img)
            cv2.imshow("Modified gray image", modified_img_gray)
            cv2.imshow("Original image", image1)
            cv2.imshow("Modified last gray image", last_image[top:down, left:right])
            cv2.imshow("Modified zoomed gray image", modified_img_gray[top_z:top_z + 100, left_z:left_z + 100])

            cv2.waitKey(1)

        if close:
            break


def main():
    global restart, profile, image_folder, left, right, top, down, make_gray
    print("\n")

    # Získání seznamu jmen názvů složek dle jmen měření
    folder_names = [name for name in [f for f in os.listdir(image_folder)] if
                    os.path.isdir(os.path.join(image_folder, name)) and (
                            name.startswith(type_) or name.startswith(","))]
    [print(f"{i: >6}:  {file}") for i, file in enumerate(folder_names)]
    print("")

    while True:
        start = input("\t\tZvolte start: ").replace(",", ".")
        try:
            start = int(abs(round(float(start))))  # pokus o převod na číslo
            if start >= 0:
                break
        except ValueError as ve:
            print(f"\n Zadejte platnou odpověď.\n\tPOPIS: {ve}")
            pass

    while True:
        end = input("\t\tZvolte konec: ").replace(",", ".")
        try:
            end = int(abs(round(float(end)))) + 1  # pokus o převod na číslo
            if start < end <= len(folder_names):
                break
        except ValueError as ve:
            print(f"\n Zadejte platnou odpověď.\n\tPOPIS: {ve}")
            pass

    # folder_names = [folder_names[i] for i in (10, 11, 12, 13, 19, 33, 37, 38)]
    # folder_names = [folder_names[i] for i in range(len(folder_names)) if i not in (10, 11, 12, 13, 19, 33, 37, 38)]
    # folder_names = [f for i, f in enumerate(folder_names) if i in (9, 13, 18,28)]
    folder_names = folder_names[start:end]  # jaké složky budu načítat (první je 0) př: "files[2:5] od 2 do 5"

    # ############################# Todo Načítání fotek ze složek #############################
    folder_names = [f for f in folder_names if os.path.isdir(os.path.join(image_folder, f, "original"))]

    print("Zvolte kterou složku chcete načíst.")
    while True:
        if len(folder_names) == 1:
            ans = 1
        else:
            ans = input(f"Zvolte: 1 - {len(folder_names)}: ")
        try:
            ans = int(abs(round(float(ans)))) - 1
            try:
                current_folder = os.path.join(image_folder, folder_names[ans], "original")
                break
            except IndexError:
                print("Zadejte platnou odpověď.")
        except ValueError:
            print("Zadejte platnou odpověď.")

    print("Zvolte kterou fotografii chcete načíst.")
    # Načtení seznamu obrázků ve složce
    image_files = get_photos_from_folder(current_folder, (".jpg", ".jpeg", ".png", ".JPG"))

    # Omezeni počtu snímků
    image_files = image_files[:]  # jaké snímky budu načítat (první je 0) př: "image_files[2:5] od 2 do 5"

    while True:
        if len(image_files) == 1:
            ans = 1
        else:
            ans = input(f"Zvolte: 1 - {len(image_files)}: ")
        try:
            current_image = int(abs(round(float(ans)))) - 1
            try:
                print("Zvolena fotografie:", image_files[current_image])
                break
            except IndexError:
                print("Zadejte platnou odpověď.")
        except ValueError:
            print("Zadejte platnou odpověď.")

    change_image(cv2.imread(os.path.join(current_folder, image_files[current_image])),
                 cv2.imread(os.path.join(current_folder, image_files[-1])))

    while restart:
        change_image(cv2.imread(os.path.join(current_folder, image_files[current_image])),
                     cv2.imread(os.path.join(current_folder, image_files[-1])))

    print("\nChcete uložit soubory?")
    while True:
        save = input("\tZadejte Y / N: ")
        if save == "Y":
            new_profile_path = None
            first_image = make_photo(cv2.imread(os.path.join(current_folder, image_files[0])))
            last_image = make_photo(cv2.imread(os.path.join(current_folder, image_files[-1])))
            create_image_window("First modified image", first_image)
            create_image_window("Last modified image", last_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("\nJe vše v pořádku?")
            while True:
                ans = input("\tZadejte Y / N: ")
                if ans == "Y":
                    break
                elif ans == "N":
                    change_image(cv2.imread(os.path.join(current_folder, image_files[current_image])),
                                 cv2.imread(os.path.join(current_folder, image_files[-1])))
                    print("\nJe vše v pořádku?")
                else:
                    print("Zadejte platnou odpověď.")

            print("\nChcete aplikovat styl na všchny složky?")
            while True:
                ans = input("\tZadejte Y / N: ")
                if ans == "N":
                    folder_names = [folder_names[current_image]]
                    break
                elif ans == "Y":
                    break
                else:
                    print("Zadejte platnou odpověď.")

            print("\n====================================================\n\nZahájení ukládání souborů.")
            # cyklus na ukládání souborů mezi složkami
            tot_folders = len(folder_names)
            width_ = None

            print("\nChcete aplikovat oříznutí fotografie na základě sředu od template?")
            while True:
                ans = input("\tZadejte Y / N: ")
                if ans == "N":
                    use_template = False
                    break
                elif ans == "Y":
                    use_template = True
                    width_ = round((right - left) / 2, -1) + 100
                    break
                else:
                    print("Zadejte platnou odpověď.")

            print("\nChcete aplikovat černobílý filtr?")
            while True:
                ans = input("\tZadejte Y / N: ")
                if ans == "N":
                    make_gray = False
                    break
                elif ans == "Y":
                    make_gray = True
                    break
                else:
                    print("Zadejte platnou odpověď.")

            print("\n")
            for i, folder in enumerate(folder_names):
                i += 1
                print(f"{i} / {tot_folders}: \t\t\t[ {folder} ]")
                # Vytvoření cesty k cílovému souboru
                input_folder = os.path.join(image_folder, folder, "original")
                output_folder = os.path.join(image_folder, folder, "modified")

                # Zkontrolování, zda cílová složka již existuje
                if not os.path.exists(output_folder):
                    # Vytvoření nové složky v případě neexistence
                    os.makedirs(output_folder)
                    print("\tSložka vytvořena.")
                else:
                    print("\tSložka již existuje.")

                new_profile_path = os.path.join(output_folder, profile + "_" + folder + '.txt')

                image_files = get_photos_from_folder(input_folder)

                if use_template:
                    template1 = cv2.imread(os.path.join(template_path, 'top_.png'), 0)
                    template2 = cv2.imread(os.path.join(template_path, 'bottom_.png'), 0)
                    im = cv2.imread(os.path.join(input_folder, image_files[template_img_index]), 0)
                    h, w = im.shape[:2]
                    # Porovnejte šablonu s druhou fotografií pomocí metody šablony
                    x_f1, y_f1 = cv2.minMaxLoc(cv2.matchTemplate(im, template1, cv2.TM_CCOEFF_NORMED))[-1]
                    _, y_f2 = cv2.minMaxLoc(cv2.matchTemplate(im, template2, cv2.TM_CCOEFF_NORMED))[-1]
                    x_f1 = round(x_f1 + template1.shape[1] // 2, -1)
                    left, right = int(max(min(x_f1 - width_, w), 0)), int(max(min(x_f1 + width_, w), 0))
                    top, down = int(max(min(round(y_f1 - 50, -1), h), 0)), int(
                        max(min(round(y_f2 + template2.shape[0] + 50, -1), h), 0))

                if left >= right or top >= down:
                    print("Chyba při automatickém označení.")
                    continue

                data = [brightness, contrast, temp, saturate, V, L, A, B, Y, V, U, V2]
                data.extend(bgr)
                data.extend([left, right, top, down])
                np.savetxt(new_profile_path, data)

                for im in image_files:
                    img_path = os.path.join(input_folder, im)
                    img_new_path = os.path.join(output_folder, "mod-" + os.path.splitext(os.path.basename(im))[0] +
                                                "." + img_format.replace(".", ""))

                    image = cv2.imread(img_path)

                    cv2.imwrite(img_new_path, make_photo(image))
                    # Změna času vytvoření a úpravy souboru
                    os.utime(img_new_path, (os.path.getctime(img_new_path), os.path.getmtime(img_path)))

                    # print("Soubor", i + 1, ":", name, "\t- uložen")

                # print(f"{i}. - {folder} : Soubory uloženy.")

            print("\n====================================================\n\nVeškeré soubory uloženy")

            if ans == "Y":
                print("\nUkončuji program.")
                sys.exit(2)

            print("\nChcete si uložit použitý profil?")

            while True:
                ans1 = input("\tZadejte Y / N: ")

                if ans1 == "Y":
                    data = [brightness, contrast, temp, saturate, V, L, A, B, Y, V, U, V2]
                    data.extend(bgr)
                    data.extend([left, right, top, down])
                    # Uložení profilu
                    if not os.path.exists(new_profile_path):
                        np.savetxt(new_profile_path, data)
                        print("\nProfil uložen.\n\nUkončuji program.")
                        sys.exit(3)
                    else:
                        print("Máte již vytvořený profil, chcete ho přepsat?")
                        while True:
                            ans2 = input("\tZadejte Y / N: ")
                            if ans2 == "Y":
                                np.savetxt(new_profile_path, data)
                                print("\nProfil uložen.\n\nUkončuji program.")
                                sys.exit(4)
                            elif ans2 == "N":
                                np.savetxt(new_profile_path.replace(".txt", "_new.txt"), data)
                                print("\nUkončuji program.")
                                sys.exit(5)
                            else:
                                print("Zadejte platnou odpověď.")
                elif ans1 == "N":
                    print("\nUkončuji program.")
                    sys.exit(6)
                else:
                    print("Zadejte platnou odpověď.")
        elif save == "N":
            print("\nUkončuji program.")
            sys.exit(7)
        else:
            print("Špatně volená možnost, zadejte znovu.")


if __name__ == '__main__':
    global image1, yuv_to_bgr_image, hsv_to_bgr_image, lab_to_bgr_image, names
    global brightness, contrast, temp, saturate, left, right, top, down, V, L, A, B, Y, U, V2
    global bgr, red, green, blue, red_channel, green_channel, blue_channel, width, height
    global left_z, top_z

    # Nastavení cesty k složce s obrázky
    image_folder = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'

    # Nastavení názvu profilu
    profile = 'profile'

    template_path = r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\data\t\t01_max'
    template_img_index = 2
    type_ = "T01"

    img_format = 'JPG'

    first_load_profile = False

    make_gray = True

    # Spuštění programu
    restart = False
    main()
