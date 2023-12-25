import cv2

# Načtěte fotku
img = cv2.imread('Frame_[1920, 1080, 60].jpg', 0)
img = img[600:650, 530:580]

"""# Zobrazte výsledný obrázek
cv2.namedWindow('D', cv2.WINDOW_NORMAL)
cv2.imshow('D', img[600:650, 530:580])
cv2.waitKey(0)
cv2.destroyAllWindows()"""

# Detekce hran
edges = cv2.Canny(img, 300, 300)

# Získání kontur
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Vykreslení kontur na originální obrázek
result_img = cv2.drawContours(img.copy(), contours, -1, (255, 0, 0), 1)

# Zobrazte výsledný obrázek
cv2.namedWindow('Detekce', cv2.WINDOW_NORMAL)
cv2.imshow('Detekce', result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
