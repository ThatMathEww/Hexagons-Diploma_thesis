def do_templ_cor(cv2, np, picture1, picture2, start1x, start1y, end1x, end1y, start2x, start2y, end2x, end2y, points):

    picture1 = picture1[start1y:end1y, start1x:end1x]
    picture2 = picture2[start2y:end2y, start2x:end2x]

    height1, width1 = picture1.shape[:2]
    height2, width2 = picture2.shape[:2]

    mask1 = np.zeros((height1, width1), dtype=np.uint8)
    cv2.fillPoly(mask1, [points], 255)
    pic1 = picture1 & mask1

    x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(points)
    relative_points = points - np.array([x_bound, y_bound])

    pic1 = pic1[y_bound:(y_bound + h_bound), x_bound:(x_bound + w_bound)]

    # ################################################################################################################ #

    min1, max1 = np.min(pic1), np.max(pic1)
    pic1 = cv2.normalize(pic1, None, min1, max1, cv2.NORM_MINMAX, dtype=cv2.CV_64F)  # 0, 1

    pixel_values_np = np.array([
        [
            np.linalg.norm(pic1 - cv2.normalize(picture2[y:y + h_bound, x:x + w_bound] &
                                                cv2.fillPoly(np.zeros((h_bound, w_bound), dtype=np.uint8),
                                                             [relative_points], 255),
                                                None, min1, max1, cv2.NORM_MINMAX, dtype=cv2.CV_64F))
            for x in range(width2 - w_bound)
        ]
        for y in range(height2 - h_bound)
    ])

    # ################################################################################################################ #

    min_position = np.int32(np.unravel_index(np.argmin(pixel_values_np), pixel_values_np.shape))
    min_position[0], min_position[1] = min_position[1], min_position[0]

    # print("Pozice nejmenší hodnoty:", min_position)

    return (x_bound, y_bound, w_bound, h_bound, points, picture1, pic1, pixel_values_np,
            min_position, relative_points, picture2)
