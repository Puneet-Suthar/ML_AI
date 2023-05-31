## FACE REG
import cv2
import pandas as pd
import face_recognition as fr

fd = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
filename = "database.csv"
try:
    face_db = pd.read_csv(filename, index_col= 0)
    data = {"NAME":face_db["name"].values.tolist(),
            "ENCODING": face_db["enc"].values.tolist()}
except Exception as e:
    print(e)
    data = {"NAME":[], "ENCODING": []}

name = data["NAME"]
enc = data["ENCODING"]
print(name)
print(enc)

# Video read using web cam
vid = cv2.VideoCapture(0)
counter = 0
captured = True

# mai loop to read and show image until we break the loop
while captured :
    flag , img = vid.read()
    cropped_face = img
    # if flag is true then only show image
    if flag:
        faces = fd.detectMultiScale(img,1.1 ,5)
        for x,y,w,h in faces:
            cropped_face = img[y:y+h, x:x+w].copy()
            
            # It will check if face is present in front of cam for 20 frames
            if len(faces) == 1:
                counter += 1
                print(counter)
                if counter == 100 :
                    name.append(input("enter your name"))
                    face_enc = fr.face_encodings(cropped_face)
                    # print(face_enc)
                    enc.append(face_enc[0].tolist())
                    data = {"name":name, "enc": enc}
                    face_db = pd.DataFrame(data)
                    face_db.to_csv(filename, sep=",")
                    captured = False
            else:
                counter = 0
            
        cv2.imshow("webcam_image",cropped_face)    
        # cv2.imshow("webcam_image",img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    else:
        break
cv2.destroyAllWindows()
vid.release()