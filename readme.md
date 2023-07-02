# Object Detection using Smartphone Camera

This project demonstrates object detection using a smartphone camera. It utilizes the YOLO (You Only Look Once) model to detect objects in real-time through the camera feed. The objects of interest in this implementation are vehicles and pedestrians.

## Requirements

To run this project, the following dependencies need to be installed:

- Python 3.x
- OpenCV library (cv2)
- Pre-trained YOLO model files:
  - `yolov3.cfg` (configuration file)
  - `yolov3.weights` (weights file)
- `coco.names` file (contains the list of classes for the COCO dataset)

## Setup

1. Download the pre-trained YOLO model files (`yolov3.cfg` and `yolov3.weights`) from the official YOLO repository: [https://github.com/pjreddie/darknet/tree/master/cfg](https://github.com/pjreddie/darknet/tree/master/cfg).

2. Download the `coco.names` file from the same repository: [https://github.com/pjreddie/darknet/tree/master/data](https://github.com/pjreddie/darknet/tree/master/data).

3. Place the downloaded files (`yolov3.cfg`, `yolov3.weights`, and `coco.names`) in the same directory as the Python script for this project.


## Usage

1. Connect your smartphone to the computer.

2. Run the Python script using the command:
```
python object_detection.py
```

3. The script will open a window displaying the live video stream from the smartphone's camera, with objects detected and labeled in real-time. The detected objects will be limited to vehicles (cars, buses, trucks, bicycles, motorbikes) and pedestrians.

4. Press 'q' on the keyboard to quit the application.

## Notes

- Ensure that the smartphone camera is properly connected and accessible from the computer.

- Adjust the confidence threshold (currently set to 0.5) in the code to control the sensitivity of object detection. Higher values will lead to more confident detections but might miss some objects, while lower values might include false positives.

- The code can be modified to detect other classes of objects by updating the `desired_classes` list in the script and including the desired class names.

- Additional optimization and customization can be implemented based on specific requirements.