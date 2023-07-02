import cv2
import numpy as np

# Load the configuration and weights files of the YOLO model
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Load the list of classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Filter classes for transport and pedestrians
desired_classes = ['car', 'bus', 'truck', 'bicycle', 'motorbike', 'person']

# Get the indices of the desired classes
desired_class_ids = [classes.index(cls) for cls in desired_classes]

# Generate random colors for each desired class
colors = np.random.uniform(0, 255, size=(len(desired_class_ids), 3))

# Open the smartphone camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Pass the blob through the network and get the outputs
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []

    # Process the outputs
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections based on class and confidence
            if class_id in desired_class_ids and confidence > 0.5:
                # Scale the coordinates of the detected object
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate the coordinates of the top-left corner
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Add the coordinates, confidence, and class to the respective lists
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate redundant detections
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Display the results on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, width, height = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            # Draw bounding box and text
            color = colors[desired_class_ids.index(class_ids[i])]
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 0.5, color, 2)

    # Display the frame with object detection
    cv2.imshow("Object Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()