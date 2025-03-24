import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime, timedelta
from pymongo import MongoClient
from deep_sort_realtime.deepsort_tracker import DeepSort

client = MongoClient("mongodb://localhost:27017/")
db = client["traffic_analytic"]
collection = db["reports"]

model = YOLO("yolov8n.pt")

deepsort = DeepSort(
    max_age=70,  # Maximum number of missed frames before a track is deleted
    n_init=3,  # Number of frames to confirm a track
    nms_max_overlap=0.5,  # Non-maximum suppression threshold
    max_cosine_distance=0.2,  # Maximum cosine distance for matching
    nn_budget=100,  # Maximum size of the appearance descriptor gallery
    override_track_class=None,  # Optional: Override the default track class
    embedder="mobilenet",  # Use 'mobilenet' as the embedder
    half=True,  # Use half-precision for the embedder
    bgr=True,  # Input images are in BGR format
    embedder_gpu=True  # Use GPU for the embedder
)

video_path = "data/input/sample_video.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

output_path = "data/output/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

interval = timedelta(minutes=5)
start_time = datetime.now()
aggregated_data = {
    "start_time": start_time,
    "end_time": start_time + interval,
    "total_vehicles": 0,
    "vehicle_types": defaultdict(int)
}

confidence_threshold = 0.4

unique_vehicle_ids = set()

frame_skip = 4
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    results = model(frame, conf=confidence_threshold)

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)

            if class_name in ["car", "truck", "bus", "motorcycle"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_name)) 

    if len(detections) > 0:
        tracks = deepsort.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            track_id = track.track_id
            class_name = track.det_class
            bbox = track.to_tlbr()

            if track_id not in unique_vehicle_ids:
                unique_vehicle_ids.add(track_id)
                aggregated_data["total_vehicles"] += 1
                aggregated_data["vehicle_types"][class_name] += 1

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    current_time = datetime.now()
    if current_time >= aggregated_data["end_time"] or not ret:
        collection.insert_one({
            "start_time": aggregated_data["start_time"],
            "end_time": current_time if not ret else aggregated_data["end_time"],
            "total_vehicles": aggregated_data["total_vehicles"],
            "vehicle_types": dict(aggregated_data["vehicle_types"])
        })

        start_time = current_time
        aggregated_data = {
            "start_time": start_time,
            "end_time": start_time + interval,
            "total_vehicles": 0,
            "vehicle_types": defaultdict(int)
        }
        unique_vehicle_ids = set()

    out.write(frame)

    cv2.imshow("Traffic Analytics - YOLOv8 + DeepSORT", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if aggregated_data["total_vehicles"] > 0:
    collection.insert_one({
        "start_time": aggregated_data["start_time"],
        "end_time": datetime.now(),
        "total_vehicles": aggregated_data["total_vehicles"],
        "vehicle_types": dict(aggregated_data["vehicle_types"])
    })

cap.release()
out.release()
cv2.destroyAllWindows()

client.close()