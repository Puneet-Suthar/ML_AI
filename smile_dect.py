import cv2
fd = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
sd = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Video read using web cam
vid = cv2.VideoCapture(0)
counter = 0
captured = True

# mai loop to read and show image until we break the loop
while captured :
    flag , img = vid.read()
    img_gray =   cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # th, img = cv2.threshold(img,50, 255,cv2.THRESH_BINARY )
    
    # if flag is true then only show image
    if flag:
        faces = fd.detectMultiScale(img,1.1 ,5)
        for x,y,w,h in faces:
            face = img[y:y+h, x:x+w]
            
            smiles = sd.detectMultiScale(face,1.1 ,200)
            for xs,ys,ws,hs in smiles:
                cv2.rectangle(img, pt1=(x+xs, y+ys), pt2 = (x+xs+ws, y+ys+hs), color=(0,0,255), thickness= 3)
            
            # Draw a rectange on img   
            cv2.rectangle(img, pt1=(x,y), pt2 = (x+w,y+h), color=(255,0,0), thickness= 3)
            
            # It will check if face is present in front of cam for 20 frames
            # if len(faces) == 1:
            #     counter += 1
            #     print(counter)
            #     if counter == 100 :
            #         captured = False
            # else:
            #     counter = 0
            
        cv2.imshow("webcam_image",img)    
        # cv2.imshow("webcam_image",img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    else:
        break
cv2.destroyAllWindows()
vid.release()