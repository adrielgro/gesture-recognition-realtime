#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:04:37 2017

@author: adriel
"""

import os,glob
import sys
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import tensorflow as tf

CLASSES = ["A", "B"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# cargar el modelo serializado
print("[INFO] cargando modelo...")
## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('dogs-cats-model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 2)) 

# inicializamos el streaming de video, y le permitimos al sensor de la
# camara que se prepare e incialize el contador FPS
print("[INFO] iniciando streaming de video...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# bucle sobre los fotogramas de la transmision de video
while True:
    
    # toma el marco de la secuencia de video subproceso y cambia su tamano 
    # para que tenga un ancho maximo de 400 pixeles
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    
    #Toma las dimensiones del marco y lo convierte en un blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (128, 128)), 0.007843, (128, 128), 127.5)

    image_size=128
    num_channels=3
    images = []
    # Lee la imagen usando OpenCV
    image = frame #cv2.imread(frame)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0) 
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)
    
    
    
    # pasa el blob a traves de la red y obtiene las detecciones  y predicciones
    #net.setInput(blob)
    #detections = net.forward()

    # bucle sobre las detecciones
    '''for i in np.arange(0, detections.shape[2]):
        # Extraer la confianza (es decir, la probabilidad) asociada 
        # con la prediccion
		confidence = detections[0, 0, i, 2]

		# filtra las detecciones debiles asegurando que la "confianza" 
         # sea mayor que la confianza minima
		if confidence > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# dibuja la prediccion en el frame
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)'''
            
    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")
    
    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0") 
    y_test_images = np.zeros((1, 2)) 
    
    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    #print("A: " + str(result[0][0]))
    #print("B: " + str(result[0][1])
    
    labelA = "{}: {:.2f}%".format(CLASSES[0],
				result[0][0])
    
    labelB = "{}: {:.2f}%".format(CLASSES[1],
				result[0][1])
    
    cv2.putText(frame, "A: " + labelA + " - B: " + labelB, (50, 50),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

    # mostramos los frames de salida
    cv2.imshow("Camara UABC", frame)
    key = cv2.waitKey(1) & 0xFF

    # presionar la letra q para salir del ciclo
    if key == ord("q"):
        break

    # actualizar el contador fps
    fps.update()

# detener el temporizador y mostrar la informacion FPS
fps.stop()
print("[INFO] tiempo transcurrido: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()