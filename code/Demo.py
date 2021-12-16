import cv2 as cv
import tensorflow as tf
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

#Imports
new_model = tf.keras.models.load_model("C:/Users/tmcar/cs1430/ASL/our_model")
new_model.summary()

#Sets up Webcam
capture = cv.VideoCapture(0)


while True:
    isTrue,frame = capture.read()
    #Creates Green Square In the middle of the image
    frame[320,240:400,1]=255 #Bottom Line
    frame[160,240:400,1]=255 #Top Line
    frame[160:320,240,1]=255 #Left Line
    frame[160:320,400,1]=255 #Right Line
    cv.imshow('Video',frame)
    img = frame[160:320,240:400,:] #Crops frame
    cv.imshow('Hand', img)
    
    #Transfroms image into from that model accepts
    img = img.astype(np.float32)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img,(28,28))
    img = img.reshape(-1,28, 28, 1)

    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

    print(class_names[np.argmax(new_model.predict_on_batch(img))])
    sleep(.1)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()


