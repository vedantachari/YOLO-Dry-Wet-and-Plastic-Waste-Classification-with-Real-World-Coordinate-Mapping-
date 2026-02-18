# Dry-Wet-Plastic Waste Classifier with Real World Coordinate Mapping

A powerful, user-friendly GUI tool that bridges the gap between 2D object detection and 3D robotic control. 

**YOLOKeyCoords** allows you to load custom YOLOv8 models, perform real-time object detection, calibrate your camera, and instantly transform 2D pixel detections into 3D World Coordinates, Robot Joint Angles (IK).

## Demo
[https://github.com/vedantachari/YOLO-Dry-Wet-and-Plastic-Waste-Classification-with-Real-World-Coordinate-Mapping-/issues/1#issue-3760290705](https://github.com/user-attachments/assets/2a80da31-30dd-413a-b8a6-2d5dce5fbfb3)

<img width="989" height="596" alt="Image" src="https://github.com/user-attachments/assets/cdb2aa26-8ac7-4a86-a595-a8248764c28b" />

## Features

* **Real-time Detection:** Seamlessly load and run any custom Ultralytics YOLOv8 (`.pt`) model.
* **Integrated Camera Calibration:** * Built-in tool to capture checkerboard frames.
    * Generates and saves camera intrinsics/extrinsics to YAML.
    * Automatically sets the World Origin $(0,0,0)$ to the bottom-center of your calibration pattern.
* **Flexible Coordinate Mapping:**
    * **2D Image Space:** Pixel coordinates $(u, v)$.
    * **3D Camera Space:** Metric coordinates relative to the lens $(x, y, z)$.
    * **3D World Space:** Transformed coordinates based on calibration or homography $(X, Y, Z)$.
* **Robotics Integration:**
    * **Inverse Kinematics (IK):** Built-in solver for a 3-DOF planar robot arm.
    * **ROS Support:** Publishes detection results as `PoseStamped` messages to `/detected_poses`.
    * **JSON Export:** Save coordinate data to JSON for offline analysis.
* **Input/Output Versatility:**
    * Supports Webcams, Video Files (`.mp4`, `.avi`), and RealSense Depth cameras.
    * Adjustable robot base position offsets.

## Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/vedantachari/YOLO-Dry-Wet-and-Plastic-Waste-Classification-with-Real-World-Coordinate-Mapping-.git
    ```

2.  **Install Dependencies**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install kivy opencv-python numpy ultralytics
    ```

##  Usage Guide

### 1. Launch the App
```bash
python ui.py
```

### 2. Load Your Model
Click "Load YOLO .pt".

Select your trained YOLOv8 model file (e.g., best.pt).

### 3. Calibration (Crucial for 3D accuracy)

You have two options for mapping pixels to meters:
#### Option A: Camera Calibration
 - Click "Calibrate Camera".
 - Hold a checkerboard pattern (default 9x6 inner corners) in front of the camera.
 - Capture at least 5 frames from different angles.
 - Click "Calibrate". This saves a camera_calib.yaml file.
#### Option B: Homography
 If you have a planar workspace, you can load a pre-computed Homography matrix (.npy or text file) using "Load homography".

### 4. Run Inference
 - Select Input Source: Camera or Video File.
 - Click Start.
 - The system will draw bounding boxes and, if calibrated, coordinate axes $(X, Y, Z)$ on the feed.

 
