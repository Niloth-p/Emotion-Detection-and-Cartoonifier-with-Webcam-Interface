import cv2
import glob
import random
import numpy as np

emotions = ["neutral", "anger", "disgust", "fear", "happy", "sad", "surprise"] #Emotion list
fishface = cv2.face.createFisherFaceRecognizer() #Initialize fisher face classifier

data = {}

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    print("Got files")
    training = files
    return training

def make_sets():
    training_data = []
    training_labels = []
    print("Making sets")
    for emotion in emotions:
        training = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        thumb = "dataset\\" + emotion + "\\Thumbs.db"
        for item in training:
            if(item!=thumb):
                print(item)
                image = cv2.imread(item) #open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                training_data.append(gray) #append image array to training data list
                training_labels.append(emotions.index(emotion))

    return training_data, training_labels
    # , prediction_data, prediction_labels

def funcy():
    training_data, training_labels = make_sets()
    
    print("training fisher face classifier")
    print("size of training set is:", len(training_labels), "images")
    fishface.train(training_data, np.asarray(training_labels))
    #fishface.save("/".join([args.output_dir,"fishface.xml"])) #Save progress

    fishface.save("fishface.xml")
   
funcy()