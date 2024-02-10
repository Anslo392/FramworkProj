import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Load YOLO model
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')  # You need to download the weights and cfg files
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to perform object detection and generate boxes
def generate_boxes_confidences_classids(frame, tconf=0.5):
    height, width, channels = frame.shape

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            if confidence > tconf:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')
                x = int(centerX - (bwidth / 2))
                y = int(centerY - (bheight / 2))

                boxes.append([x, y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)

    return boxes, confidences, classids

# Function to open a file dialog for video upload
def upload_video():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4")])
    return file_path

# Get the video file path from the user
video_path = upload_video()

# Example usage
cap = cv2.VideoCapture(video_path)

# Initialize a list to store player positional matrices
player_positions = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    boxes, confidences, classids = generate_boxes_confidences_classids(frame)

    # Draw rectangles on the frame
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Store player positional matrix in the list
    player_positions.append(boxes)

    # Display the frame with rectangles
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()


