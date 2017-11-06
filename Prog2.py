import cv2
import glob
import random
import numpy as np
import FaceExtracter
import sys

#fishface.load("/".join([args.output_dir,"fishface.xml"]))
fishface = cv2.face.createFisherFaceRecognizer() #Initialize fisher face classifier
fishface.load("fishface.xml")

f = "OriginalImages\\" + sys.argv[1]
FaceExtracter.detectfaces(f)
f2 = "ExtractedFaces\\" + sys.argv[1]
print("Returned to prog2")

img = cv2.imread(f2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pred, conf = fishface.predict(gray)

# emotions = ["neutral", "anger", "disgust", "fear", "happy", "sad", "surprise"] #Emotion list
# choices = {"AF": "fear", "AN": "anger", "DI":"disgust", "HA":"happy", "NE":"neutral", "SA":"sad", "SU":"surprise"}
emotions = {0:"Neutral", 1:"Anger", 2:"Disgust", 3:"Fear", 4:"Happy", 5:"Sad", 6:"Surprise"}
emotion = emotions[pred]
print("Emotion :", emotion, "Confidence :", conf)
#fishface.train(prediction_data, np.asarray(prediction_labels))
#update(prediction_data, np.asarray(prediction_labels)
#fishface.save("fishface.xml"])) #Save final