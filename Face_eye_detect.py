import cv2  
import numpy

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 
  
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")  

mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")  
  
# capture frames from a camera 
cap = cv2.VideoCapture(0) 
font = cv2.FONT_HERSHEY_SIMPLEX
 
# loop runs if capturing has been initialized. 
while True:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
  
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = face_cascade.detectMultiScale(gray, 1.1, 5) 
    # faces = face_cascade.detectMultiScale(img, 
    #                              scaleFactor=1.3, 
    #                              minNeighbors=4, 
    #                              minSize=(30, 30),
    #                              flags=cv2.CASCADE_SCALE_IMAGE) 


  ###
    for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

    smile = mouth_cascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.16,
        minNeighbors=35,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE)

    for (sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        cv2.putText(img,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:  
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.putText(img,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)

    cv2.putText(img,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)      
    
  ###




    # for (x,y,w,h) in faces: 
    #     # To draw a rectangle in a face
    #     cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
    #     roi_gray = gray[y:y+h, x:x+w] 
    #     roi_color = img[y:y+h, x:x+w] 
  
    #     # Detects eyes of different sizes in the input image 
    #     eyes = eye_cascade.detectMultiScale(roi_gray)  
          
    #     #To draw a rectangle in eyes 
    #     for (ex,ey,ew,eh) in eyes: 
    #         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

    #     mouth = mouth_cascade.detectMultiScale(roi_gray)

    #     for (sx,sy,sw,sh) in mouth:
    #         cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy,sh),(255,0,0),2)


  
    # Display an image in a window 
    cv2.imshow('img',img) 
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(5)
    if k == 27: 
        break
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  