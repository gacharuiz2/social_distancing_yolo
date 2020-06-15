#!/usr/bin/env python
# coding: utf-8

# # **SOCIAL DISTANCING DETECTOR**

# el siguiente proyecto tiene por objetivo revisar desde diversos metodos [imagen, video, camara] el cumplimiente del distanciamiento social en el rubro de contac center.
# 
# Toda esta tecnica utiliza una licencia MIT de codigo abierto, la cual a su vez utiliza.
# 
# Esta herramienta realiza 3 cosas principamente:
# 
#     *1) Detecta a los humanos en el marco con la red neuronal convolucional yolov3.*
#     *2) Calcule la distancia entre todas las instancias de humanos detectadas en el marco.*
#     *3) Clasifique las distancias determinadas como 'Alerta' u 'Ok' para el distanciamiento social.*
#     
# requisitos de instalación de librerias:
# 
#     1) Numpy
#     2) time
#     3) OpenCV
#     4) OpenCV_Contrib
#     5) math

# ## [0] importación y ubicación de archivos

# ### *Importación de las librearias necesarias.*

# In[1]:


import numpy as np
import time
import cv2
import math


# ### *Definimos la ubicación de los archivos*

# Para poder empezar necesitamos decargar manualmente los siguientes paquetes:
#     
#     1) coco.names; este archiv contiene los nombres o labels de los objetos que la cámara podra identificar.
#     2) yolov3.weights; este archivo contiene los pesos del pre-entrenamiento del modelo yolo para data coco
#     3) yolov3.cfg; contiene todos los detalles de configuración del modelo convusional a utilizar.
# 
# Archivos disponibles en: https://pjreddie.com/darknet/yolo/

# In[2]:


labelsPath = r"D:\KONECTA\APLICACIONES\SOCIAL_DISTANCING\coco.names"
weightsPath = r"D:\KONECTA\APLICACIONES\SOCIAL_DISTANCING\yolov3.weights"
configPath = r"D:\KONECTA\APLICACIONES\SOCIAL_DISTANCING\yolov3.cfg"


# ## [1] Cámara streaming 

# In[3]:


LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    
    ret,image=cap.read()
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Frame Prediction Time : {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    ind = []
    for i in range(0,len(classIDs)):
        if(classIDs[i]==0):
            ind.append(i)
    a = []
    b = []

    if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                a.append(x)
                b.append(y)
                
    distance=[] 
    nsd = []
    for i in range(0,len(a)-1):
        for k in range(1,len(a)):
            if(k==i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)
                if(d <=100):
                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))
                print(nsd)
    color = (0, 0, 255) 
    for i in nsd:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "Alert"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    color = (0, 255, 0) 
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (i in nsd):
                break
            else:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = 'OK'
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   
    
    cv2.imshow("Social Distancing Detector", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# ## [2] video almacenado

# In[7]:


videopath=r'D:\KONECTA\APLICACIONES\SOCIAL_DISTANCING\test_video.mp4'

LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

cap = cv2.VideoCapture(videopath)
hasFrame, frame = cap.read()
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))


while cv2.waitKey(1) < 0:
    
    ret,image=cap.read()
    image=cv2.resize(image,(640,360))
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 300.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    print("Frame Prediction Time : {:.6f} seconds".format(end - start))
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
    ind = []
    for i in range(0,len(classIDs)):
        if(classIDs[i]==0):
            ind.append(i)
    a = []
    b = []

    if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                a.append(x)
                b.append(y)
                
    distance=[] 
    nsd = []
    for i in range(0,len(a)-1):
        for k in range(1,len(a)):
            if(k==i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)
                if(d <=100):
                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))
                print(nsd)
    color = (0, 0, 255) 
    for i in nsd:
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "Alert"
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    color = (0, 255, 0) 
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (i in nsd):
                break
            else:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = 'OK'
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   
    
    cv2.imshow("Social Distancing Detector", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    vid_writer.write(image)

vid_writer.release()
cap.release()
cv2.destroyAllWindows()


# ## [3] Imagen almacenada

# In[6]:


imagepath=r'D:\KONECTA\APLICACIONES\SOCIAL_DISTANCING\cola.jpg'

LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


image =cv2.imread(imagepath)
(H, W) = image.shape[:2]
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()
print("Frame Prediction Time : {:.6f} seconds".format(end - start))
boxes = []
confidences = []
classIDs = []
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.5 and classID == 0:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
ind = []
for i in range(0,len(classIDs)):
    if(classIDs[i]==0):
        ind.append(i)
a = []
b = []
color = (0,255,0) 
if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            a.append(x)
            b.append(y)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            

distance=[] 
nsd = []
for i in range(0,len(a)-1):
    for k in range(1,len(a)):
        if(k==i):
            break
        else:
            x_dist = (a[k] - a[i])
            y_dist = (b[k] - b[i])
            d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
            distance.append(d)
            if(d<=100.0):
                nsd.append(i)
                nsd.append(k)
            nsd = list(dict.fromkeys(nsd))
   
color = (0, 0, 255)
text=""
for i in nsd:
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = "Alert"
    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
           
cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
cv2.imshow("Social Distancing Detector", image)
cv2.imwrite('output.jpg', image)
cv2.waitKey()

