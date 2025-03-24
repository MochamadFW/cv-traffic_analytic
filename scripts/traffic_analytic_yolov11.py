import cv2
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime, timedelta
from pymongo import MongoClient
from tkinter import Tk, filedialog
import os

client = MongoClient("mongodb://localhost:27017/")
db = client["traffic_analytic"]
collection = db["reports"]

model = YOLO('yolo11n.pt')

class_list = model.names

confidence_thresholds = {
    "car": 0.6175,
    "bus": 0.7,
    "truck": 0.7,
    "motorcycle": 0.4,
    "train": 0.8,
}

def select_video_source():
    root = Tk()
    root.withdraw()
    choice = input("Enter '1' to select a video file or '2' to enter a CCTV URL: ").strip()
    
    if choice == '1':
        video_path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if not video_path:
            print("No video file selected. Exiting.")
            exit()
        return video_path, "file"
    elif choice == '2':
        cctv_url = input("Enter the CCTV URL: ").strip()
        if not cctv_url:
            print("No CCTV URL provided. Exiting.")
            exit()
        return cctv_url, "cctv"
    else:
        print("Invalid choice. Exiting.")
        exit()

video_source, source_type = select_video_source()

output_dir = "data/output"
os.makedirs(output_dir, exist_ok=True)

input = cv2.VideoCapture(video_source)
if not input.isOpened():
    print("Error: Could not open video source.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(input.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(input.get(cv2.CAP_PROP_FPS))

def create_new_output_video():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"cctv_output_{timestamp}.mp4")
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

out = create_new_output_video()

class_counts = defaultdict(int)
vehicle_ids = set()
interval = timedelta(minutes=5)
start_time = datetime.now()
last_report_time = start_time

while input.isOpened():
    ret, frame = input.read()
    if not ret:
        if source_type == "file":
            print("Video ended. Exiting.")
            break
        else:
            print("Error: Failed to read frame from CCTV feed. Retrying...")
            continue

    results = model.track(frame, persist=True, classes=[1, 2, 3, 5, 6, 7])

    if results[0].boxes.data is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu()

        for box, track_id, class_idx, conf in zip(boxes, track_ids, class_indices, confidences):
            class_name = class_list[class_idx]
            confidence_threshold = confidence_thresholds.get(class_name, 0.5)

            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx_box = (x1 + x2) // 2
            cy_box = (y1 + y2) // 2

            cv2.circle(frame, (cx_box, cy_box), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if track_id not in vehicle_ids:
                vehicle_ids.add(track_id)
                class_counts[class_name] += 1

    y_offset = 30
    for class_name, count in class_counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (50, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    out.write(frame)

    cv2.imshow("YOLOv11 Traffic Tracking & Counting", frame)

    current_time = datetime.now()
    if current_time - last_report_time >= interval:
        if class_counts:
            report = {
                "timestamp": current_time,
                "total_vehicles": sum(class_counts.values()),
                "vehicle_types": dict(class_counts)
            }
            collection.insert_one(report)
            print("Report saved to MongoDB:", report)

        class_counts = defaultdict(int)
        vehicle_ids = set()

        if source_type == "cctv":
            out.release()
            out = create_new_output_video()
            print(f"New output video file created: {out}")

        last_report_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User pressed 'q'. Exiting.")
        break

if class_counts:
    report = {
        "timestamp": datetime.now(),
        "total_vehicles": sum(class_counts.values()),
        "vehicle_types": dict(class_counts)
    }
    collection.insert_one(report)
    print("Final report saved to MongoDB:", report)

input.release()
out.release()
cv2.destroyAllWindows()
client.close()