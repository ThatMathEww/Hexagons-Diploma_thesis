import matplotlib.pyplot as plt
import numpy as np
import cv2
import easyocr

image_path = (r'C:\Users\matej\PycharmProjects\pythonProject\Python_projects\HEXAGONS\Friction_photos'
              r'\F07_005_01_o\cam_frame_ 1004.png')

im = cv2.imread(image_path, 1)[880:, 10:240]  # [950:1100, :320]

plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()

im = cv2.cvtColor(cv2.bitwise_and(im, im, mask=cv2.inRange(im, (100, 25, 25), (160, 100, 100))), cv2.COLOR_BGR2GRAY)
# (15, 25, 40), (65, 55, 55)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()

reader = easyocr.Reader(['en'])
result = reader.readtext(im)
print(result)
