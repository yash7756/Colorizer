import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import generate_detections as gdet

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize Deep SORT
max_cosine_distance = 0.3
nn_budget = None
model_filename = '/model/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Function to process a single frame and extract person coordinates
def process_frame(frame):
    results = model(frame)
    bboxes = []
    confidences = []
    for result in results.xyxy[0]:  # xyxy format for bounding boxes
        if result[-1] == 0:  # Class ID 0 corresponds to 'person' in COCO dataset
            bbox = result[:4].cpu().numpy()
            confidence = result[4].cpu().numpy()
            bboxes.append(bbox)
            confidences.append(confidence)

    features = encoder(frame, bboxes)
    detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in zip(bboxes, confidences, features)]
    
    # Update tracker
    tracker.predict()
    tracker.update(detections)
    
    person_coords = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        person_coords.append((x_center, y_center))
    
    return person_coords

# Function to process video and collect coordinates
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    footprints = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        footprints.extend(process_frame(frame))
    cap.release()
    return footprints

# Function to generate and display heatmap
def generate_heatmap(footprints, store_layout_image_path=None):
    # Convert list of coordinates to numpy array
    footprint_array = np.array(footprints)
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    if store_layout_image_path:
        store_layout = plt.imread(store_layout_image_path)
        plt.imshow(store_layout, aspect='auto', alpha=0.5)
    sns.kdeplot(footprint_array[:, 0], footprint_array[:, 1], shade=True, cmap='viridis')
    plt.title('Footprint Heatmap')
    plt.xlabel('Store Width')
    plt.ylabel('Store Height')
    plt.show()

# Example usage
video_path = 'video2.mp4'
#store_layout_image_path = 'path/to/your/store_layout.png'  # Optional, if you have a store layout image
footprints = process_video(video_path)
generate_heatmap(footprints)
