import cv2
import numpy as np

# Load pre-trained license plate detection model
net = cv2.dnn.readNet("yolov3_custom_last.weights", "yolov3_custom.cfg")

# Load class names
with open("obj.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape

    # Detect objects (license plates) in the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponds to license plate
                # Calculate coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw rectangle around license plate
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display frame with license plate detection
    cv2.imshow("License Plate Detection", frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
