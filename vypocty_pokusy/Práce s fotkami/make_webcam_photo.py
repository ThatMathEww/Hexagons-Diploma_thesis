import cv2

settings = [[4032, 3040, 10], [3840, 2160, 20], [2592, 1944, 30], [2560, 1440, 30], [1920, 1080, 60], [1600, 1200, 50],
            [1280, 960, 100], [1280, 760, 100], [640, 480, 80]]


for s in settings:
    capture = cv2.VideoCapture(0)
    print(s[0], s[1], s[2])
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, s[0])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, s[1])
    capture.set(cv2.CAP_PROP_FPS, s[2])

    _, frame = capture.read()  # Načtení snímku z kamery
    _, frame = capture.read()  # Načtení snímku z kamery

    # cv2.imwrite(f"makrobloking_photos/Frame_new_{s}.jpg", frame)
    # cv2.imwrite(f'makrobloking_photos/Frame_uncompressed_{s}.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imwrite(f'makrobloking_photos/Frame_q10_{s}.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 10])
    capture.release()
