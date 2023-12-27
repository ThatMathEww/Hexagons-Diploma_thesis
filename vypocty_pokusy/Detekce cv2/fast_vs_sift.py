# Importing the libraries
import cv2
import matplotlib.pyplot as plt
import time

# Reading the image and converting into B/W
image = cv2.imread(r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
                   r'\H01_01-I-max_12s\original\IMG_1119.JPG', 0)
image2 = cv2.imread(r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\photos'
                    r'\H01_01-I-max_12s\original\IMG_1152.JPG', 0)

# Applying the function
start = time.time()
fast = cv2.FastFeatureDetector_create()
brief = cv2.xfeatures2d_BriefDescriptorExtractor.create(bytes=32, use_orientation=True)
orb = cv2.ORB_create()
brisk = cv2.BRISK_create()
akaze = cv2.AKAZE_create()
kaze = cv2.KAZE_create()
sift = cv2.SIFT_create()
"""surt = cv2.xfeatures2d_SURF.create(hessianThreshold=100, nOctaves=4, nOctaveLayers=3, extended=False, upright=False)
freak = cv2.xfeatures2d_FREAK.create()
daisy = cv2.xfeatures2d_DAISY.create()
lucid = cv2.xfeatures2d_LUCID.create()
latch = cv2.xfeatures2d_LATCH.create()
vgg = cv2.xfeatures2d_VGG.create()"""
mser = cv2.MSER_create()
gftt = cv2.GFTTDetector_create()
# harris = cv2.cornerHarris()
blop = cv2.SimpleBlobDetector_create()
star = cv2.xfeatures2d_StarDetector()

fast.setNonmaxSuppression(False)
# Drawing the keypoints
kp = fast.detect(image, None)
print("FAST", time.time() - start)
# kp_image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))

kp_brief, ds_brief = brief.compute(image, kp)
kp_sift_, ds_sift_ = sift.compute(image, kp)

start = time.time()

# Drawing the keypoints
kp_sift, ds_sift = sift.detectAndCompute(image, None)
print("SIFT", time.time() - start)
# sift_image = cv2.drawKeypoints(image, kp_sift, None, color=(0, 255, 0))

plt.figure()
plt.imshow(kp_image, cmap='gray')

plt.figure()
plt.imshow(sift_image, cmap='gray')

plt.show()
