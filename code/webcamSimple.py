import cv2
import hyperparameters as hp
from run import VGGModel
from imageai import Detection
#First we need our pre trained weights downloaded
#Now we write the path to these weights
modelpath = "cs1430_env/ASL/ASLData/weights" #this will need to be edited

#TODO: Load in our model here

#============================
yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(modelpath)
yolo.loadModel()
#============================

#Here is where we access the device camera using open cv
cam = cv2.VideoCapture(0) #0=front-cam, 1=back-cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

#We want the camera to be on a loop
while True:
    ## read each frame
    ret, img = cam.imread()

    #TODO: Use our model to make predictions

    #=======================================
    ## predict yolo
    img, preds = yolo.detectCustomObjectsFromImage(input_image=img, 
    custom_objects=None, input_type="array",
    output_type="array",
    minimum_percentage_probability=70,
    display_percentage_probability=False,
    display_object_name=True)
    #=======================================

    #Show the prediction
    cv2.imshow("", img)
    
    #Quit with q or esc
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

#close camera
cam.release()
cv2.destroyAllWindows()

