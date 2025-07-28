from ultralytics import YOLO
import cv2
import cvzone
import math
from deepface import DeepFace


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4, 480)
frame_count = 0
model = YOLO('../Yolo-Weights/yolov8n.pt')
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            conf = math.ceil((box.conf[0] * 100))/100
            cls = int(box.cls[0])
            name = model.names[cls]
            cvzone.cornerRect(img, (x1, y1, w, h))

            cvzone.putTextRect(img, f'{conf} {name}', (x1, y1-20))
            if name == 'person' and frame_count % 15 == 0:
                try:
                    # Crop the person region
                    person_crop = img[y1:y2, x1:x2]
                    # Run DeepFace analysis
                    result = DeepFace.analyze(person_crop, actions=('age', 'gender', 'race', 'emotion'),
                                              enforce_detection=False)
                    age = result[0]['age']
                    gender = result[0]['dominant_gender']
                    race = result[0]['dominant_race']
                    emotion = result[0]['dominant_emotion']

                    print(f"Age: {age}, Gender: {gender}, Race: {race}, Emotion: {emotion}")

                    # Display on screen
                    info = f"{gender}, {age}y, {emotion}"
                    cvzone.putTextRect(img, info, (x1, y2 + 20), scale=0.7, thickness=1, offset=5)
                except Exception as e:
                    print("DeepFace error:", e)




    frame_count += 1
    cv2.imshow('img', img)
    cv2.waitKey(1)

