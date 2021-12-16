import cv2
size=4
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image=cv2.imread("C:/Users/kandr/Desktop/FALL 21/A1-Image Processing/Burglar detection/face.jpg")
print(image)
print(int(image.shape[0]),int(image.shape[1]))
grayimg=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
mini=cv2.resize(image,(int(image.shape[1]/size),int(image.shape[0]/size)))
print(mini)
faces=face_cascade.detectMultiScale(mini)
for f in faces:
 (x,y,w,h)=[v*size for v in f]
 cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),4)
 subface=image[y:y+h,x:x+w]
 
 facefile="face.jpg"
 cv2.imwrite(facefile,subface)
 
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

