from ultralytics import YOLO
import cv2
import winsound
from datetime import datetime

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # downloads automatically on first run

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Starting phone detection... Press 'q' to quit.\n")

# Variables for clean terminal output
previous_state = None  # To avoid repeating same message
log_file = "phone_log.txt"

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to read from webcam.")
        break

    # Run YOLOv8 inference
    results = model(frame, verbose=False)
    phone_detected = False

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

    # Handle detection alerts and logs
    if phone_detected and previous_state != "detected":
        print("📱 Phone detected!")
        winsound.Beep(1000, 200)
        with open(log_file, "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Phone detected\n")
        previous_state = "detected"

    elif not phone_detected and previous_state != "none":
        print("❌ No phone in frame.")
        previous_state = "none"

    # Display video feed
    cv2.imshow("Phone Detection", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n🛑 Detection stopped. Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
