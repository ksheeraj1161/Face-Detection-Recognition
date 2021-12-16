import cv2
import glob as gb
import random
import numpy as np
size = 4
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
fisher_face = cv2.face.FisherFaceRecognizer_create()

person_list = ["ksheeraj", "deepak"]
def getFiles(person):
 files = gb.glob("final_dataset/%s/*" %person)
 random.shuffle(files)
 training = files[:int(len(files)*0.67)]
 prediction = files[-int(len(files)*0.33):]
 return training, prediction
def makeTrainingAndValidationSet():
 training_data = []
 training_labels = []
 prediction_data = []
 prediction_labels = []
 for person in person_list:
    training, prediction = getFiles(person)

    for item in training:
        image = cv2.imread(item)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        training_data.append(gray)
        training_labels.append(person_list.index(person))

        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prediction_data.append(gray)
            prediction_labels.append(person_list.index(person))
    return training_data, training_labels, prediction_data, prediction_labels
image=cv2.imread("ksheeraj.jpg")
training_data, training_labels, prediction_data, prediction_labels = makeTrainingAndValidationSet()
fisher_face.train(training_data, np.asarray(training_labels))
mini = cv2.resize(image,(int(image.shape[1]/size),int(image.shape[0]/size)))
gray = cv2.cvtColor(mini, cv2.COLOR_BGR2GRAY)
facefeatures=cascade_classifier.detectMultiScale(gray)
for f in facefeatures:
 
 (x,y,w,h) = [v*size for v in f]
 cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),4)
 
 subface = image[y:y+h,x:x+w]
 
 facefile = "face.jpg"
 cv2.imwrite(facefile,subface)
pred, conf = 0, 0
for (x, y, w, h) in facefeatures:
 
 gray = gray[y:y+h, x:x+w]
 try:
     out = cv2.resize(gray, (350, 350))
     pred,conf = fisher_face.predict(out)
 except:
     pass 
person = person_list[pred]
font = cv2.FONT_HERSHEY_TRIPLEX
cv2.putText(image,person,(x+w,y),font,2,(0,0,255),2)
cv2.imshow("Facial Recognition on an Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
