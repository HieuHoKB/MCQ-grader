import cv2
import urllib.request as ur
import time
import numpy as np


url = "http://192.168.1.10:8080/shot.jpg"

while True:
    imgResp=ur.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    cv2.imshow('IPWebcam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

