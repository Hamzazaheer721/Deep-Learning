import time, cv2 as cv

video = cv.VideoCapture(0)
check, frame = video.read()
print(check)
print(frame)
time.sleep(3)

cv.imshow("video",frame)
cv.waitKey(0)
video.release()
cv.destroyAllWindows()