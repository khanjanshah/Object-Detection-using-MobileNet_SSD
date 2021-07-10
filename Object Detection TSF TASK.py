
# Required libraries
import numpy as np # For numeric computations
import cv2 # computer vision library used for image processing
import imutils # use for image-processing for translation, resizing, rotation etc of an image.
import datetime # give current date and time.


"""Caffe:
Caffe is an open-source deep learning framework developed for Machine Learning. 
It is written in C++ and Caffe's interface is coded in Python. 
"""

"""CaffeModel:
A CAFFEMODEL file is a machine learning model created by Caffe. 
It contains an image classification or image segmentation model that has been trained using Caffe.
The weights of the layers of the neural network are stored as a . caffemodel file.
"""

"""prototxt:
The structure of the neural network is stored as a . prototxt file. 
"""

# Path to the pretrained model.
PROTOPATH = 'MobileNetSSD_deploy.prototxt'
MODEL = 'MobileNetSSD_deploy.caffemodel'

# classes and colors on which mobilenet is trained.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))



print("Initializing Model and Loading...... ")
net = cv2.dnn.readNetFromCaffe(prototxt=PROTOPATH,caffeModel=MODEL)


capt = cv2.VideoCapture('traffic-mini.mp4')

fps_start = datetime.datetime.now()
fps = 0
total_frames = 0
while capt.isOpened():

    ret, frame = capt.read()
    frame = imutils.resize(frame,width=500)
    total_frames += 1

    fps_end = datetime.datetime.now()

    total_time = fps_end - fps_start
    if total_time.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / total_time.seconds)

    fps_text = "FPS of this video is : {:.2f}".format(fps)

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    #cv2.imshow("FPS DISPLAY APP", frame)
    key = cv2.waitKey(15)
    if key == ord('q'):
        break
    #print("FPS in this video are: ", fps_text)

    if not ret:
        break
    (h, w) = frame.shape[:2]
    # Read the image and detect blob from image.
    # Blob detection is a kind of object detection with high intensity differences.

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
        (300, 300), 127.5)

    #print(image.shape[:2])

    #print("COMPUTATION OF OBJECT DETECTIONS..")
    net.setInput(blob)
    detections = net.forward()
    #print(detections)

    #print(detections.shape)

    # computing confidence and assign labels with its % conf to each object
    for i in np.arange(0,detections.shape[2]):
      confidence = detections[0,0,i,2]

      if confidence > 0.5:
        index = int(detections[0,0,i,1])
        #if CLASSES[index] != 'person':
        #    continue
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (sX,sY,eX,eY) = box.astype('int')

        label = '{}: {:.2f}%'.format(CLASSES[index],confidence*100)

        cv2.rectangle(frame,(sX,sY),(eX,eY),COLORS[index],2)
        y  = sY -15 if sY-15 > 15 else sY+15
        cv2.putText(frame,label,(sX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[index],2)

    cv2.imshow('Person Detection',frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break



capt.release()
# Detected Objects from image.
cv2.destroyAllWindows()