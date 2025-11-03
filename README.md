# Real-Time-Phone-Detection-using-YOLOv8
This project demonstrates how to use Ultralytics YOLOv8 and OpenCV to perform real-time object detection through a webcam feed. The system identifies when a cell phone appears in the camera frame and visually highlights it with a bounding box and on-screen label.

## 🚀 Features

- Real-time object detection using a webcam.
- Detects **cell phones** using YOLOv8 pretrained on the COCO dataset.
- Displays bounding boxes and labels directly on the video feed.
- Prints detection alerts to the console.

---

## 🧰 Requirements

Before running the script, make sure you have Python installed (≥ 3.8).  
Then install the required libraries:

```bash
pip install ultralytics opencv-python
