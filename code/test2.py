import cv2 as cv
import tensorflow as tf
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

new_model = tf.keras.models.load_model("C:/Users/tmcar/cs1430/ASL/our_model")
#new_model.summary()


capture = cv.VideoCapture(0)#

imagesize = 160
alphadict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M',
                 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y'}

while True:
    isTrue,frame = capture.read()
    frame[320,240:400,1]=255 #Bottom Line
    frame[160,240:400,1]=255 #Top Line
    frame[160:320,240,1]=255 #Left Line
    frame[160:320,400,1]=255 #Right Line
    cv.imshow('Video',frame)
    img = frame[160:320,240:400,:]
    cv.imshow('Hand', img)
    img = img.astype(np.float32)
    
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # plt.imshow(img)
    # plt.show()
    #img = np.array(img, dtype = 'float32')
    #img = img / 255
    
    cv.imshow('HandProcessed',img)
    img = cv.resize(img,(28,28))
    img = img.reshape(-1,28, 28, 1)
    #img = np.reshape(img, (28,28,1))
    #print(img.shape)
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ]

    print(class_names[np.argmax(new_model.predict_on_batch(img))])
    #print(new_model.predict_on_batch(img))
    sleep(.1)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()


