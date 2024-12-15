import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Config value
video_path = "data_ext/44.mp4"
conf_threshold = 0.5  # Confidence threshold

# Initialize DeepSORT
tracker = DeepSort(max_age=30)

# Initialize YOLOv8 model
model = YOLO("weights/yolov8n.pt")  # Chọn mô hình YOLOv8 phù hợp

# Load class names
with open("data_ext/classes.names") as f:
    classnames = f.read().strip().split('\n')

# Random colors for each class
colors = np.random.randint(0, 255, size=(len(classnames), 3))

# Initialize VideoCapture
cap = cv2.VideoCapture(video_path)

while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model.predict(source=frame, conf=conf_threshold, save=False, save_txt=False, device="cpu")

    # Extract detections
    detections = []
    for result in results[0].boxes:
        bbox = result.xyxy[0].cpu().numpy()  # Bounding box: [x1, y1, x2, y2]
        confidence = result.conf[0].cpu().item()  # Confidence score
        class_id = int(result.cls[0].cpu().item())  # Class ID

        if confidence < conf_threshold:
            continue  # Bỏ qua đối tượng có độ tin cậy thấp

        x1, y1, x2, y2 = map(int, bbox)
        detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Update tracker with detections
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw bounding boxes and labels
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            label = "{}-{}".format(classnames[class_id], track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Resize frame for display
    scale_percent = 60  # Resize to 60% of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # Show frame
    cv2.imshow("Object Tracking", resized_frame)

    # Exit on pressing 'q' or 'ESC'
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
