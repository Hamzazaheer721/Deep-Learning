import cv2
DATA = "./data/"
img = cv2.imread(DATA + "cones/im2.png")
resized_image = cv2.resize(img, (int(img.shape[0]*2), int(img.shape[1]*2)))
print(img.shape)
cv2.imshow("Penguin", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()