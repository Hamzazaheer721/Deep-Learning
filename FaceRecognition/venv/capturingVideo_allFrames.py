import time, cv2 as cv

video = cv.VideoCapture(0)


while True:

    check, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21,21), 0)
    cv.imshow("video", frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break


video.release()
cv.destroyAllWindows()