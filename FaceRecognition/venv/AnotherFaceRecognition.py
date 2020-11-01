import cv2 as cv

DATA = "./data/"
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

video = cv.VideoCapture(0)

while True:
    check, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
    for x, y, w, h in faces:        print(x, y, w, h)
        target = gray[y:y + h, x:x + w]
        name = "face.jpg"
    # cv.imwrite(name, target) this will make image of face in video capture per frame

    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key == ord('c'):
        break
video.release()
cv.destroyAllWindows()
