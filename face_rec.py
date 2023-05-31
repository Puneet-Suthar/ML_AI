## FACE REC
import cv2
import pandas as pd
import face_recognition as fr
import numpy as np

fd = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
filename = "database.csv"
try:
    face_db = pd.read_csv(filename, index_col=0)
    data = {"NAME": face_db["name"].values.tolist(),
            "ENCODING": face_db["enc"].values.tolist()}
except Exception as e:
    print(e)
    data = {"NAME": [], "ENCODING": []}

name = data["NAME"]
enc = data["ENCODING"]
print(name)
print(enc)

# Video read using web cam
vid = cv2.VideoCapture(0)
counter = 0
captured = True

# main loop to read and show image until we break the loop
while captured:
    flag, img = vid.read()
    cropped_face = img
    # if flag is true then only show image
    if flag:
        faces = fd.detectMultiScale(img, 1.1, 5)
        for x, y, w, h in faces:
            cropped_face = img[y:y + h, x:x + w].copy()

            # It will check if face is present in front of cam for 20 frames
            if len(faces) == 1:
                # Here the Out of index problem is handled by using try and except method
                try:
                    fresh_face_enc = fr.face_encodings(cropped_face)[0]
                    for ind, fe in enumerate(enc):
                        matched = fr.compare_faces([fresh_face_enc], np.array(eval(fe)))[0]
                        if matched:
                            print("matched")
                            print(name[ind])
                            cv2.putText(img, name[ind], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 4)
                            cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
                        else:
                            print("not matched")
                except IndexError:
                    print("No face detected in the image.")
        cv2.imshow("webcam_image", img)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    else:
        break

cv2.destroyAllWindows()
vid.release()
