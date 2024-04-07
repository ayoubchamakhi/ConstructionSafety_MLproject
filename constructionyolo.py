import cv2
from ultralytics import YOLO
import cvzone
import math

#For video
#vid = cv2.VideoCapture("../Running YOLO/video/bikes.mp4")  # For Video

# define a video capture object
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model = YOLO("Yolo-Weights/best.pt")

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
         'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery',
         'mini-van', 'sedan', 'semi', 'trailer', 'truck', 'truck and trailer', 'van', 'vehicle', 'wheel loader']

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    #Predict the image
    results = model(frame, stream = True)

    #Draw the box of prediction
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255))

            #Class name
            cls = box.cls[0]

            #Add confidence interval for prediction
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)
            cvzone.putTextRect(frame, f'{classNames[int(cls)]} {conf}', (max(0, x1), max(35, y1)), scale = 0.9, thickness = 1)




    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()