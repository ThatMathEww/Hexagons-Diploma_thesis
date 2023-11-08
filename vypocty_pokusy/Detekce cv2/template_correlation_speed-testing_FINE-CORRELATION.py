import sys
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
import time
from numba import jit


def do_norm(img1, img2):
    # 1. Standardizace hodnot v maticích (z-score normalizace)
    picture1_norm = img1 / np.sqrt(np.sum(img1 ** 2))
    picture2_norm = img2 / np.sqrt(np.sum(img2 ** 2))
    return picture1_norm, picture2_norm


def do_normalize_z(img1, img2):
    a_mean = np.mean(img1)
    a_std = np.std(img1)
    a_normalized_z = (img1 - a_mean) / a_std

    b_mean = np.mean(img2)
    b_std = np.std(img2)
    b_normalized_z = (img2 - b_mean) / b_std
    return a_normalized_z, b_normalized_z


def do_normalize_mm(img1, img2):
    # 1. Normalizace hodnot v maticích (pomocí Min-Max normalizace)
    a_normalized_mm = (img1 - np.min(img1)) / (np.max(img1) - np.min(img1))
    b_normalized_mm = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
    return a_normalized_mm, b_normalized_mm


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / np.sqrt(norm_a * norm_b)


def euclidean_distance(a, b):
    result = np.linalg.norm(a - b)  # (np.linalg.norm(a - b) / np.sqrt(np.sum(a) ** 2 + np.sum(b) ** 2)) * 100
    return result


# 3. Porovnání normalizovaných matic pomocí korelačního koeficientu
def correlation_coefficient(a, b):
    return np.corrcoef(a.flatten(), b.flatten())[0, 1]


def pearson_correlation(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    covariance = np.cov(a_flat, b_flat)[0, 1]
    std_a = np.std(a_flat)
    std_b = np.std(b_flat)
    correlation = covariance / (std_a * std_b)
    return correlation


def spearman_correlation(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    correlation, _ = spearmanr(a_flat, b_flat)
    return correlation


def method01(img1, img2):
    global min_max
    min_max = 1
    # Výpočet rozdílu mezi obrázky
    difference = cv2.absdiff(img1, img2)

    # Výpočet průměrné hodnoty rozdílu (čím nižší hodnota, tím větší podobnost)
    similarity = np.mean(difference)

    # Zobrazíme průměrnou hodnotu rozdílu
    # print(f"Podobnost obrazků: {similarity}")

    return similarity


def method02(img1, img2):
    score, diff = ssim(img1, img2, full=True)

    # print("Podobnost Score: {:.3f}%".format(score * 100))
    return score * 100


def method03(img1, img2):
    picture1_norm, picture2_norm = do_norm(img1, img2)
    similarity = np.sum(picture2_norm * picture1_norm)
    # print("Score: {:.3f}".format(similarity))
    return similarity


def method04(img1, img2):
    a_normalized_z, b_normalized_z = do_normalize_z(img1, img2)
    similarity = np.sum(a_normalized_z * b_normalized_z)
    # print("Score Z NORM: {:.3f}".format(similarity))
    return similarity


def method05(img1, img2):
    a_normalized_mm, b_normalized_mm = do_normalize_mm(img1, img2)
    similarity = np.sum(a_normalized_mm * b_normalized_mm)
    # print("Score MM NORM: {:.3f}".format(similarity / 100))
    return similarity


def method06(img1, img2):
    a_normalized_z, b_normalized_z = do_normalize_z(img1, img2)
    # Vypočítání kosínové podobnosti mezi normalizovanými maticemi a_normalized a b_normalized
    similarity_score = cosine_similarity(a_normalized_z.flatten(), a_normalized_z.flatten())
    # print("Kosínová podobnost Z:", similarity_score)
    return similarity_score


def method07(img1, img2):
    a_normalized_mm, b_normalized_mm = do_normalize_mm(img1, img2)
    similarity_score = cosine_similarity(a_normalized_mm.flatten(), a_normalized_mm.flatten())
    # print("Kosínová podobnost Min_Max:", similarity_score)
    return similarity_score


def method08(img1, img2):
    a_normalized_z, b_normalized_z = do_normalize_z(img1, img2)
    # Vypočítání euklidovské vzdálenosti mezi normalizovanými maticemi a_normalized a b_normalized
    distance_score = euclidean_distance(a_normalized_z.flatten(), b_normalized_z.flatten())

    # Invertujeme vzdálenost, aby vyjadřovala podobnost (čím menší vzdálenost, tím větší podobnost)
    similarity_score = 1 / (1 + distance_score)

    # print("Euklidovská vzdálenost Z:", distance_score)
    # print("Podobnost Z:", similarity_score)
    return similarity_score


def method09(img1, img2):
    a_normalized_mm, b_normalized_mm = do_normalize_mm(img1, img2)
    distance_score = euclidean_distance(a_normalized_mm.flatten(), b_normalized_mm.flatten())
    similarity_score = 1 / (1 + distance_score)

    # print("Euklidovská vzdálenost Min_Max:", distance_score)
    # print("Podobnost Min_Max:", similarity_score)
    return similarity_score


def method10(img1, img2):
    first_image_hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
    second_image_hist = cv2.calcHist([img2], [0], None, [256], [0, 256])

    img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_CORREL)  # HISTCMP_BHATTACHARYYA
    # print("histogram:", img_hist_diff)
    return img_hist_diff


def method11(img1, img2):
    a_normalized_z, b_normalized_z = do_normalize_z(img1, img2)
    # Vypočítání korelačního koeficientu mezi normalizovanými maticemi a_normalized a b_normalized
    correlation_score = correlation_coefficient(a_normalized_z, b_normalized_z)

    # print("Korelační koeficient Z:", correlation_score)
    return correlation_score


def method12(img1, img2):
    a_normalized_mm, b_normalized_mm = do_normalize_mm(img1, img2)
    correlation_score = correlation_coefficient(a_normalized_mm, b_normalized_mm)
    # print("Korelační koeficient MM:", correlation_score)
    return correlation_score


def method13(img1, img2):
    a_normalized_z, b_normalized_z = do_normalize_z(img1, img2)
    # Vypočítání korelačního koeficientu Pearsonovy korelace mezi normalizovanými maticemi a_normalized a b_normalized
    correlation_score = pearson_correlation(a_normalized_z, b_normalized_z)

    # ("Korelační koeficient Pearsonovy korelace Z:", correlation_score)
    return correlation_score


def method14(img1, img2):
    a_normalized_mm, b_normalized_mm = do_normalize_mm(img1, img2)
    correlation_score = pearson_correlation(a_normalized_mm, b_normalized_mm)
    # print("Korelační koeficient Pearsonovy korelace MM:", correlation_score)
    return correlation_score


def method15(img1, img2):
    a_normalized_z, b_normalized_z = do_normalize_z(img1, img2)
    # Vypočítání korelačního koeficientu Spearmanovy korelace mezi maticemi a_matrix a b_matrix
    correlation_score = spearman_correlation(a_normalized_z, b_normalized_z)
    # print("Korelační koeficient Spearmanovy korelace Z:", correlation_score)
    return correlation_score


def method16(img1, img2):
    a_normalized_mm, b_normalized_mm = do_normalize_mm(img1, img2)
    correlation_score = spearman_correlation(a_normalized_mm, b_normalized_mm)
    # print("Korelační koeficient Spearmanovy korelace MM:", correlation_score)
    return correlation_score


def method17(img1, img2):
    compare_image = CompareImage(img1, img2)
    image_difference = compare_image.compare_image()
    return image_difference


def method18(img1, img2):
    # sim = (np.linalg.norm(img1 - img2) / np.sqrt(np.sum(img1) ** 2 + np.sum(img2) ** 2))

    global min_max
    min_max = 1
    a_normalized_mm, b_normalized_mm = do_normalize_mm(img1, img2)
    sim = (np.linalg.norm(a_normalized_mm - b_normalized_mm) /
           np.sqrt(np.sum(a_normalized_mm) ** 2 + np.sum(b_normalized_mm) ** 2))
    return sim * 10


def method19(img1, img2):
    return np.corrcoef(img1.flatten(), img2.flatten())[0, 1]


def method20(img1, img2):
    global min_max
    min_max = 1
    a_normalized_mm, b_normalized_mm = do_normalize_mm(img1, img2)
    return np.sqrt(np.sum((a_normalized_mm - b_normalized_mm) ** 2)) / 100
    # np.sqrt(np.sum((img1 - img2) ** 2))/100


def method21(img1, img2):
    global min_max
    min_max = 1
    return np.sqrt(np.mean((img1 - img2) ** 2))


def method22(img1, img2):
    global min_max
    min_max = 1
    return np.linalg.norm(img1 - img2) / 100


@jit(nopython=True, cache=True)
def method_best3(img1, img2):
    min1, min2, max1, max2 = np.min(img1), np.min(img2), np.max(img1), np.max(img2)
    img1, img2 = (img1 - min1) / (max1 - min1), (img2 - min2) / (max2 - min2)
    return np.linalg.norm(img1 - img2)


@jit(nopython=True, cache=True)
def method_best2(img1, img2):
    min1 = np.min(img1)
    max_min = np.max(img1) - min1
    img1, img2 = (img1 - min1) / max_min, (img2 - min1) / max_min
    return np.linalg.norm(img1 - img2)


def method_best(img1, img2):
    min1, max1 = np.min(img1), np.max(img1)
    img1 = cv2.normalize(img1, None, min1, max1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)  # 0, 1
    img2 = cv2.normalize(img2, None, min1, max1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    return np.linalg.norm(img1 - img2)


def method_orb(img1, img2):
    # Detekce klíčových bodů a popisovačů
    orb = cv2.SIFT_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Párování klíčových bodů
    bf = cv2.BFMatcher()
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Vykreslení shod
    result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=2)

    # Zobrazení výsledku
    cv2.imshow('Výsledek', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class CompareImage(object):

    def __init__(self, image_1_path, image_2_path):
        self.minimum_commutative_image_diff = 1
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path

    def compare_image(self):
        """image_1 = cv2.imread(self.image_1_path, 0)
        image_2 = cv2.imread(self.image_2_path, 0)"""
        image_1, image_2 = self.image_1_path, self.image_2_path
        commutative_image_diff = self.get_image_difference(image_1, image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            # ########## print("Matched")
            return commutative_image_diff
        return 1000

    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = \
            cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        # commutative_image_diff = (img_hist_diff * 0.1) + (img_template_diff * 0.9)
        return commutative_image_diff


"""# Ujistěte se, že oba obrázky mají stejnou velikost
if picture1.shape != pic2.shape:
    print("Obrázky nemají stejnou velikost.")
    sys.exit()"""

"""size = 10
new_height = height + size + size
new_width = width + size + size

template = np.zeros((new_height, new_width), dtype=np.uint8)
template[size:-size, size:-size] = pic1

height2, width2 = picture2.shape
pixel_values_np = np.zeros((height2 - height, width2 - width))
for x in range(width2 - width):
    for y in range(height2 - height):

        pic2 = np.zeros((height + size + size, width + size + size), dtype=np.uint8)
        pic2[size:-size, size:-size] = picture2[y:y + height, x:x + width]

        pixel_values_np[y, x] = method03(template, pic2)"""

if __name__ == '__main__':
    min_max = 0

    # Načtěte dva obrázky (jako příklad použijeme stejný obrázek pro ukázku)
    picture1 = cv2.imread("photos/IMG_0385.JPG", cv2.IMREAD_GRAYSCALE)[400:650, 2750:3000]  # [1100:1170, 2470:2540]
    picture2 = cv2.imread("photos/IMG_0417.JPG", cv2.IMREAD_GRAYSCALE)[1075:1325, 2747:2997]
    # [1750:1820, 2500:2570]
    # picture2 = cv2.rotate(picture2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # picture2 = cv2.imread("photos/IMG_0385.JPG", cv2.IMREAD_GRAYSCALE)[405:850, 2730:3100]
    np.clip(np.array(picture2 * 1, dtype=np.uint8), 0, 255, out=picture2)

    height1, width1 = picture1.shape[:2]
    height2, width2 = picture2.shape[:2]

    points = np.array(((180, 5), (193, 35), (224, 5)), np.int32)
    mask1 = np.zeros((height1, width1), dtype=np.uint8)
    cv2.fillPoly(mask1, [points], 255)
    pic1 = picture1 & mask1

    x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(points)
    relative_points = points - np.array([x_bound, y_bound])

    pic1 = pic1[y_bound:(y_bound + h_bound), x_bound:(x_bound + w_bound)]

    pixel_values_np = np.zeros((height2 - h_bound, width2 - w_bound))

    pixel_sum = np.zeros((height2 - h_bound, width2 - w_bound))
    pixel_mean = np.zeros((height2 - h_bound, width2 - w_bound))


    def process_patch(x_, y_, pic1_, min1_, max1_, picture2_, relative_points_, h_bound_, w_bound_):
        """mask2_ = np.zeros((h_bound_, w_bound_), dtype=np.uint8)
        pic2_ = picture2_[y_:y_ + h_bound_, x_:x_ + w_bound_]
        cv2.fillPoly(mask2_, [relative_points_], 255)
        pic2_ = pic2_ & mask2_

        pic2_ = cv2.normalize(pic2_, None, min1_, max1_, cv2.NORM_MINMAX, dtype=cv2.CV_64F)

        pixel_value_ = np.linalg.norm(pic1_ - pic2_)
        pixel_sum_ = np.sum(pic2_)
        pixel_mean_ = np.mean(pic2_)"""

        pixel_value_ = np.linalg.norm(pic1_ - cv2.normalize((picture2_[y_:y_ + h_bound_, x_:x_ + w_bound_] &
                                                             cv2.fillPoly(
                                                                 np.zeros((h_bound_, w_bound_), dtype=np.uint8),
                                                                 [relative_points_], 255)),
                                                            None, min1_, max1_, cv2.NORM_MINMAX, dtype=cv2.CV_64F))
        pixel_sum_, pixel_mean_ = 1, 1

        return pixel_value_, pixel_sum_, pixel_mean_


    # ################################################################################################################ #

    start_time = time.time()

    """method18(pic1, picture2)"""

    min1_1, max1_1 = np.min(pic1), np.max(pic1)
    pic1_1 = cv2.normalize(pic1, None, min1_1, max1_1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)  # 0, 1

    for x in range(width2 - w_bound):
        for y in range(height2 - h_bound):
            """mask2 = np.zeros((h_bound, w_bound), dtype=np.uint8)
            pic2_1 = picture2[y:y + h_bound, x:x + w_bound]
            cv2.fillPoly(mask2, [relative_points], 255)
            pic2_1 = pic2_1 & mask2

            pic2_1 = cv2.normalize(pic2_1, None, min1_1, max1_1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)

            pixel_values_np[y, x] = np.linalg.norm(
                pic1_1 - pic2_1)  # TODO ##############################################
            pixel_sum[y, x] = np.sum(pic2_1)
            pixel_mean[y, x] = np.mean(pic2_1)"""

            pixel_values_np[y, x] = np.linalg.norm(pic1_1 - cv2.normalize((picture2[y:y + h_bound, x:x + w_bound] &
                                                                           cv2.fillPoly(np.zeros((h_bound, w_bound),
                                                                                                 dtype=np.uint8),
                                                                                        [relative_points], 255)),
                                                                          None, min1_1, max1_1, cv2.NORM_MINMAX,
                                                                          dtype=cv2.CV_64F))
            # TODO ##########################################

    end_time = time.time()
    print("Čas 1:", end_time - start_time)

    """first_image_hist = cv2.calcHist([pic1_1], [0], None, [256], [0, 256])
    second_image_hist = cv2.calcHist([picture2[0:0 + h_bound, 0:0 + w_bound]], [0], None, [256], [0, 256])"""

    # ################################################################################################################ #

    start_time = time.time()

    x_grid, y_grid = np.meshgrid(np.arange(width2 - w_bound), np.arange(height2 - h_bound))

    # Definujte rozměry hlavní matice a podmatic
    main_cols, main_rows = x_grid.shape


    def make_matrix(rows, cols, sub_matrix):
        # Vytvoření prázdné matice
        matrix = np.empty((cols, rows), dtype=object)
        # Naplnění matice podmaticemi
        for i in range(cols):
            for j in range(rows):
                matrix[i, j] = sub_matrix
        return matrix


    vectorized_process = np.vectorize(process_patch, excluded=['min1_', 'max1_', 'h_bound_', 'w_bound_'])

    min1_2, max1_2 = np.min(pic1), np.max(pic1)
    pic1_2 = cv2.normalize(pic1, None, min1_2, max1_2, cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    # Spojte všechny vstupy do matic a vektorů do jednoho vstupu
    inputs = (
        x_grid, y_grid,
        make_matrix(main_rows, main_cols, pic1_2),
        min1_2, max1_2,
        make_matrix(main_rows, main_cols, picture2),
        make_matrix(main_rows, main_cols, relative_points),
        h_bound,
        w_bound
    )

    # Výstupy z vektorizované funkce
    pixel_values_np_2, pixel_sum, pixel_mean = vectorized_process(*inputs)

    end_time = time.time()
    print("Čas 2:", end_time - start_time)

    # ################################################################################################################ #
    start_time = time.time()

    # pixel_values_np_3 = np.zeros((height2 - h_bound, width2 - w_bound), dtype=np.float64)

    min1_3, max1_3 = np.min(pic1), np.max(pic1)
    pic1_3 = cv2.normalize(pic1, None, min1_3, max1_3, cv2.NORM_MINMAX, dtype=cv2.CV_64F)  # 0, 1

    """a = [(cv2.normalize(pic1[y:y + h_bound, x:x + w_bound], None, min1, max1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
          - cv2.normalize(picture2[y:y + h_bound, x:x + w_bound] &
                          cv2.fillPoly(np.zeros((h_bound, w_bound), dtype=np.uint8), [relative_points], 255),
                          None, min1, max1, cv2.NORM_MINMAX, dtype=cv2.CV_64F))]"""

    pixel_values_np_3 = np.array([
        [
            np.linalg.norm(pic1_3 - cv2.normalize(picture2[y:y + h_bound, x:x + w_bound] &
                                                  cv2.fillPoly(np.zeros((h_bound, w_bound), dtype=np.uint8),
                                                               [relative_points], 255),
                                                  None, min1_3, max1_3, cv2.NORM_MINMAX, dtype=cv2.CV_64F))
            for x in range(width2 - w_bound)
        ]
        for y in range(height2 - h_bound)
    ])

    end_time = time.time()
    print("Čas 3:", end_time - start_time)

    # ################################################################################################################ #
    start_time = time.time()

    """min1_4, max1_4 = np.min(pic1), np.max(pic1)

    pixel_values_np_4 = np.array([
        [
            (cv2.matchTemplate(pic1, (picture2[y:y + h_bound, x:x + w_bound] &
                                      cv2.fillPoly(np.zeros((h_bound, w_bound), dtype=np.uint8),
                                                   [relative_points], 255)), cv2.TM_CCOEFF_NORMED)[0][0])
            for x in range(width2 - w_bound)
        ]
        for y in range(height2 - h_bound)
    ])"""

    end_time = time.time()
    print("Čas 4:", end_time - start_time)

    # ################################################################################################################ #

    print("\n")

    min_position = list(np.unravel_index(np.argmin(pixel_values_np), pixel_values_np.shape))
    max_position = list(np.unravel_index(np.argmax(pixel_values_np), pixel_values_np.shape))

    min_position_2 = list(np.unravel_index(np.argmin(pixel_values_np_2), pixel_values_np.shape))
    max_position_2 = list(np.unravel_index(np.argmax(pixel_values_np_2), pixel_values_np.shape))

    min_position_3 = list(np.unravel_index(np.argmin(pixel_values_np_3), pixel_values_np.shape))
    max_position_3 = list(np.unravel_index(np.argmax(pixel_values_np_3), pixel_values_np.shape))

    found_sum_min = pixel_sum[min_position[0], min_position[1]]
    found_sum_max = pixel_sum[max_position[0], max_position[1]]

    found_mean_min = round(pixel_mean[min_position[0], min_position[1]], 3)
    found_mean_max = round(pixel_mean[max_position[0], max_position[1]], 3)

    min_position[0], min_position[1] = min_position[1], min_position[0]
    max_position[0], max_position[1] = max_position[1], max_position[0]

    coefficient = pixel_values_np[min_position[1], min_position[0]]

    print("Koeficient:", coefficient, "\n")
    if coefficient > 200:
        if coefficient > 1000:
            print("Pozor, koeficient korelace je obrovský.")
        else:
            print("Pozor, koeficient je velký.")

    print("Správná pozice:         ", [x_bound, y_bound])
    print("Pozice nejmenší hodnoty:", min_position)
    print("Pozice nejvyšší hodnoty:", max_position)
    print("\t  Délka výpočtu:", round((end_time - start_time), 2), "sekund")

    print("\nHledaný  -  SUM:", np.sum(pic1))
    print("Nalezený -  SUM: min", found_sum_min, ", max", found_sum_max)
    print("Hledaný  - MEAN:", round(int(np.mean(pic1)), 3))
    print("Nalezený - MEAN: min", found_mean_min, ", max", found_mean_max)

    # ############################################################################################## #
    #                                           GRAF                                                 #
    plt.figure()
    plt.subplot(231)
    plt.scatter(x_bound, y_bound, color="yellowgreen", marker="+")
    x_points, y_points = zip(*points)
    plt.plot(x_points + (x_points[0],), y_points + (y_points[0],), linewidth=1, color='lime')

    plt.imshow(cv2.cvtColor(picture1, cv2.COLOR_BGR2RGB))
    rectangle = plt.Rectangle((x_bound, y_bound), w_bound, h_bound, edgecolor='green', facecolor='none')
    plt.gca().add_patch(rectangle)

    plt.subplot(232)
    plt.imshow(cv2.cvtColor(pic1.astype(np.uint8), cv2.COLOR_BGR2RGB))

    plt.subplot(233)
    plt.imshow(pixel_values_np, cmap='jet', interpolation='nearest')
    plt.colorbar()

    plt.subplot(234)
    plt.scatter(min_position[0], min_position[1], color="blue", marker="+")
    plt.scatter(max_position[0], max_position[1], color="red", marker="+")
    x_points, y_points = zip(*(relative_points + min_position))
    plt.plot(x_points + (x_points[0],), y_points + (y_points[0],), linewidth=1, color='skyblue')
    plt.imshow(cv2.cvtColor(picture2, cv2.COLOR_BGR2RGB))
    rectangle = plt.Rectangle((min_position[0], min_position[1]), w_bound, h_bound, edgecolor='dodgerblue',
                              facecolor='none')
    plt.gca().add_patch(rectangle)
    x_points, y_points = zip(*(relative_points + max_position))
    plt.plot(x_points + (x_points[0],), y_points + (y_points[0],), linewidth=1, color='coral')
    rectangle = plt.Rectangle((max_position[0], max_position[1]), w_bound, h_bound,
                              edgecolor='crimson', facecolor='none')
    plt.gca().add_patch(rectangle)

    plt.subplot(235)

    if min_max == 1:
        res_cor = max_position
    else:
        res_cor = min_position

    """plt.imshow(cv2.cvtColor(picture2[res_cor[1]:res_cor[1] + h_bound, res_cor[0]:res_cor[0] + w_bound],
                               cv2.COLOR_BGR2RGB))"""

    mask2 = np.zeros((h_bound, w_bound), dtype=np.uint8)
    pic2 = picture2[res_cor[1]:res_cor[1] + h_bound, res_cor[0]:res_cor[0] + w_bound]
    cv2.fillPoly(mask2, [relative_points], 255)
    pic2 = pic2 & mask2
    plt.imshow(cv2.cvtColor(pic2, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()
