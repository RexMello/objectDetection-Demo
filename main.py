from ultralytics import YOLO
import cv2

source = input('Enter video source (0 for webcam): ')
confidence = float(input('Enter confidence (0.1 to 0.9): '))

if source.isnumeric():
    source = int(source)

cap = cv2.VideoCapture(source)
yolo_model = YOLO('yolov8s.pt')


while True:
    ret, frame = cap.read()

    if not ret:
        print('Video source expired..')


    results = yolo_model(frame, conf=confidence)
    
    locations = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            conf = round(box.conf[0].item(), 2)
            name = r.names[box.cls[0].item()]
            x1, y1, x2, y2 = int(b[0].item()), int(b[1].item()), int(b[2].item()), int(b[3].item())
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{str(name)} {str(conf)}', (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 1)

    cv2.imshow('Output',frame)
    if cv2.waitKey(1) == ord('q'):
        break
