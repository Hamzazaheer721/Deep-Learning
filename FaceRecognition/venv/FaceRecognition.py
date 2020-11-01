import cv2 as cv

DATA = "./data/"

fc = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
ec = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

img = cv.imread(DATA + "humans/me.jpg")

grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# smaller the value of scaleFactor, greater is the accuracry
faces = fc.detectMultiScale(grayImg, scaleFactor=1.05, minNeighbors=5)

print(type(faces))
print(faces)

for x,y,w,h in faces:
    img = cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

resized = cv.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
cv.imshow('wow', resized)
cv.waitKey(0)
cv.destroyAllWindows()