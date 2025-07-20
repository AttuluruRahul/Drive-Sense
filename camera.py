import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    # Use raw string to avoid Unicode escape error, or use forward slashes
    cap = cv2.VideoCapture(r"C:\Users\yadal\OneDrive\Desktop\Drive Sense\Demo.gif") # for camera use video = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame from the camera")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            
            # to beep when alert:
            # if(prob > 90):
            #     os.system("say beep")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
        cv2.imshow('Video', frame)  
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    startapplication()