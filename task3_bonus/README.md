# Bonus Task: Single Object Tracking in ROS

## Overview

This project implements an object detection and tracking system using ROS (Robot Operating System) integrated with the DETR (DEtection TRansformers) model. The system processes a sequence of images from a ROS bag file, performs object detection using DETR, and visualizes the results in RViz. It supports object tracking over time and ground truth comparison using separate nodes for object detection and visualization.

## Features

 - __Object Detection__ using DETR (`detr-resnet-50`) pre-trained model.
 - __Object Tracking__ based on spatial proximity across frames.
 - __Ground Truth Comparison__ for tracking accuracy evaluation.
 - __ROS Integration__ with customized real-time RViz visualization.
 - __Decoupled Architecture__ using two separate ROS nodes:
    - `object_detection_publisher.py` for detection and tracking.
    - `bbox_visualizer.py` for visualization.
 - __Dynamic Sequence Selection__ via ROS launch arguments.

## Project Structure

```
task3_bonus/
├── data/                      # Contains ground truth and first track data
│   ├── seq2/
│   └── seq3/
│       ├── groundtruth.txt
│       ├── firsttrack.txt
│       └── nlp.txt
├── devel/                     # Catkin workspace development environment
├── rosbags/                   # ROS bag files for sequences
│   ├── seq2.bag
│   └── seq3.bag
├── results/
│   └── 4_rosbags/             # ROS bag files for results
│       ├── task3_seq2_result.bag
│       └── task3_seq3_result.bag
├── src/
│   └── task3_tracker/         # Main ROS package
│       ├── launch/
│       │   └── runtask3.launch # Launch file for the system
│       ├── src/
│       │   ├── bbox_visualizer.py            # Node for visualizing bounding boxes
│       │   ├── object_detection_publisher.py # Node for object detection and tracking
│       │   └── detr-resnet-50/               # DETR model files
│       │       ├── config.json
│       │       ├── model.safetensors
│       │       ├── preprocessor_config.json
│       │       ├── pytorch_model.bin
│       │       └── README.md
│       ├── CMakeLists.txt
│       └── package.xml
└── README.md                  # This documentation
```

## Prerequisites

 - __ROS Noetic__ (or compatible version)
 - __Python 3__
 - __PyTorch__ with GPU support (if available)
 - __transformers__ library (for DETR)
 - __OpenCV__ (cv2)
 - __cv_bridge__ for ROS-OpenCV integration
 - __vision_msgs__ for publishing `Detection2D` messages

## Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/YukiKuma111/ME5413_Homework1_Perception.git
cd task3_bonus
```

### 2. Build the Workspace

Ensure your ROS environment is properly sourced and build the workspace using `catkin_make`:

```
# source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

## Running the Program

### 0. Start Roscore

Open a terminal and run:

```
roscore
```

### 1. Launch the Tracking System

In a new terminal, run the following command:

```
roslaunch task3_tracker runtask3.launch seq_num:=2
```

 - `seq_num` can be __2__ or __3__ to select the desired sequence.

_<span style="color:red;">__Note:__</span> By default, the ROS bag starts in a paused state.
After the RViz window has opened, return to the terminal and press the __spacebar__ to start playing the ROS bag.
Otherwise, no data will stream to the system._

### 2. Visualization in RViz

 - <span style="color:green;">__Green Boxes__</span> → Tracked object bounding boxes.
 - <span style="color:blue;">__Blue Boxes__</span> → Ground truth bounding boxes (if available).
 - The bounding boxes will update as the ROS bag plays.

## ROS Nodes

1. `object_detection_publisher.py`

  - Loads the DETR model and processes incoming images.
  - Publishes detected and tracked bounding boxes.
  - Publishes the NUSNET ID to `/me5413/nusnetID`.

2. `bbox_visualizer.py`

  - Subscribes to detection and ground truth topics.
  - Overlays bounding boxes onto the image stream.
  - Publishes visualization results to RViz.

![rqt graph](./rosgraph.png)

## Published Topics

| **Topic Name**           | **Message Type**     | **Description**                                  |
|--------------------------|----------------------|--------------------------------------------------|
| `/me5413/image_raw`       | `sensor_msgs/Image`       | Raw images from the ROS bag.                     |
| `/me5413/groundtruth`     | `vision_msgs/Detection2D` | Ground truth bounding boxes.              |
| `/me5413/track`           | `vision_msgs/Detection2D` | Tracked object bounding box.              |
| `/me5413/viz_output`      | `sensor_msgs/Image`       | Image with overlaid tracking results.            |
| `/me5413/nusnetID`        | `std_msgs/String`         | NUS student net ID.

## How It Works

1. __ROS Bag Playback:__

    The selected ROS bag is played and publishes image data to `/me5413/image_raw`.

2. __Object Detection (DETR):__

    Each frame is processed using the DETR model to detect objects.

3. __Tracking Logic:__

    The system selects the bounding box closest to the previously tracked position for continuity.

4. __Ground Truth Comparison:__

    If ground truth data is available, the system overlays it for visual evaluation.

5. __RViz Visualization:__

    Results are displayed in RViz, showing both detected and ground truth bounding boxes.

## Troubleshooting

 - __No images in RViz?__
 
    Ensure you've pressed the __spacebar__ in the terminal after RViz opens to start the ROS bag playback.

 - __CUDA not detected?__
 
    Verify PyTorch is installed with GPU support by running:
    ```
    import torch
    print(torch.cuda.is_available())
    ```

 - __Model loading errors?__
 
    Ensure the `detr-resnet-50` model files (`pytorch_model.bin`) are in the correct directory:

    `task3_bonus/src/task3_tracker/src/detr-resnet-50/`

 - __ImportError: Undefined symbol `ffi_type_pointer` in `libp11-kit.so.0`__
 
    If you encounter the following error when running the program:

    ```
    ImportError: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
    ```

    This indicates a compatibility issue with the system's `libffi` library. To resolve this, preload the correct version of `libffi` using the following command:

    ```
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7
    ```

     - After running this command, re-launch your ROS setup and the issue should be resolved.
     - If you want to make this change persistent across terminal sessions, add the above line to your `~/.bashrc` file:

    ```
    echo 'export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so.7' >> ~/.bashrc
    source ~/.bashrc
    ```
    
    This ensures the library is preloaded every time you open a new terminal.

## Additional Notes

- __Data Directory:__

    The `data/` folder contains sequence-specific files like `groundtruth.txt` and `firsttrack.txt` for each sequence.

- __Modifying Sequences:__

    To add new sequences, place the corresponding `.bag` file in the `rosbags/` folder and the associated data in the `data/` directory.