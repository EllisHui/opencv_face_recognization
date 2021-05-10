'''
Haar Cascade Face detection with OpenCV  
    Based on tutorial by pythonprogramming.net
    Visit original post: https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/  
Adapted by Marcelo Rovai - MJRoBot.org @ 7Feb2018 
'''

import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
# 3,4实际就是编号，但是这个跟别的码表相同，也就是说这个数字3，4实际上都已经是固定或者是规定好的数字
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, img = cap.read()
    # img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detectMultiScaleh 函数的第1个参数为图片img
    # objects被检测物体的矩形框向量组
    # scaleFactor表示每次图像尺寸减小的比例
    # minNeighbors表示每一个目标至少要被检测到3次才算
    # 是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)
    # minSize为目标的最小尺寸
    # maxSize为目标的最大尺寸
    # 适当调整4,5,6两个参数可以用来排除检测结果中的干扰项
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
