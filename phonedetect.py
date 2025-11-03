from ultralytics import YOLO
import cv2

# Load YOLOv8 small model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # automatically downloads weights

# Start webcam
cap = cv2.VideoCapture(0)
print("Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)
    phone_detected = False

    # Check detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == "cell phone":
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Phone Detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Print alert in console
    if phone_detected:
        print("📱 Phone detected")
    else:
        print("False")

    # Show webcam feed
    cv2.imshow("Phone Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
