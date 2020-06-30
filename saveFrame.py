import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
startTime = int(time.time())
counter = 1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if (startTime != int(time.time())):
        cv2.imwrite("/Users/akshar/Documents/Coding/Hitchhikers/photos/frame%d.jpg" % counter, frame)
        counter+=1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
