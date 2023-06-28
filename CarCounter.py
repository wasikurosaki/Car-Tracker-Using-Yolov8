from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *



cap = cv2.VideoCapture("Videos/cars.mp4")
cap.set(3, 640) # width
cap.set(4,480) # height


model = YOLO("YOLO-Weights/yolov8n.pt")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


mask = cv2.imread("masks/mask.png")

tracker = Sort(max_age=20,min_hits=2,iou_threshold=0.3)



while True:
    success, img = cap.read()

    imgregion =cv2.bitwise_and(img,mask)
    results = model(imgregion,stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            w,h = x2-x1, y2-y1
            #cv2.rectangle(img, (x,y), (w,h), (255,0,255), 2)
            
            con = math.ceil((box.conf[0]*100))
            print(con)
            #cvzone.putTextRect(img,f"{con}%",(max(0,x1),max(35,y1)))


            c = box.cls[0] #ClassName

            currentClass = classNames[int(c)]

            detec = ["car","bus","truck","motorbike"]
            if (currentClass in detec )and con >50:

                cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5)
                currentarray = np.array([x1,y1,x2,y2,con])
                detections = np.vstack((detections,currentarray))

    resultstracker = tracker.update(detections)

    for i in resultstracker:
        x1,y1,x2,y2,id = i
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        print(i)

        #cvzone.cornerRect(img,(x1,y1,w,h),l=9,rt=5)
        cvzone.putTextRect(img,f"{int(id)}",(max(0,x1),max(35,y1)),scale = 0.8, thickness=1,offset=5)
    


    cv2.imshow("Image", img)
    cv2.waitKey(0)