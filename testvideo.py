import torch
from ultralytics import YOLO
import cv2

# Load your model
model = YOLO('best.pt')

# Load the video
video_path = 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get video writer initialized to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Visualize the results
    result_frame = results.render()[0]

    # Write the frame into the output file
    out.write(result_frame)

    # Optionally display the frame
    cv2.imshow('Frame', result_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
