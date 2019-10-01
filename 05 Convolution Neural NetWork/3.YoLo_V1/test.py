from cv2 import cv2 as cv2
import numpy as np
 
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    print(frame.shape)
    #设置显示的窗口大小为500,500，建议大于等于摄像头分辨率
    cv2.imshow("camera", frame)
 
    if cv2.waitKey(1) == ord('q'):
        break
 
 
cv2.destroyAllWindows()
    