# ME5413_Homework1_Perception

This repository contains solutions for __ME5413 Homework 1: Perception__, which consists of three tasks:

- __Task 1:__ Multi-object tracking using template matching, object detection with data association, and improvement strategies.
- __Task 2:__ Trajectory prediction using a constant velocity model and evaluation metrics such as ADE (Average Displacement Error) and FDE (Final Displacement Error).
- __Task 3:__ Please review [Task3_README.md](./task3_bonus/README.md)

## Note on `pytorch_model.bin` File

Due to persistent issues with Git LFS when uploading the `pytorch_model.bin` file, it has not been included directly in this repository.

Instead, you can download the `pytorch_model.bin` file from OneDrive using the link below:

https://1drv.ms/u/c/c979dcfcaa816b41/Eb1hWEtsTsNChBlHtt5rJuMB2Q_KN1VVjDJs6QcD25EMHw?e=cmA3Nm

### How to use the file

1. Download the pytorch_model.bin file from the provided OneDrive link.
2. Place the file in the following directory:
```
.
├── task1_tracking
│   └── detr-resnet-50
│       └── pytorch_model.bin
└── task3_bonus
    └── src
        └── task3_tracker
            └── src
                └── detr-resnet-50
                    └── pytorch_model.bin
```
3. Ensure the directory structure matches the project before running any code that depends on this file.

If you encounter any issues while downloading or using the file, feel free to open an issue in this repository.

## Folder Structure

```
ME5413_Homework1_Perception/
│
├── data/                         # Task 1 data
│   ├── seq1/
│   │   ├── firsttrack.txt
│   │   ├── groundtruth.txt
│   │   ├── img/                  # Image sequence
│   │   │   └── 00000001.jpg ... 00000150.jpg
│   │   └── nlp.txt
│   ├── seq2/
│   └── seq3/
│   └── seq4/
│   └── seq5/
│
├── detr-resnet-50/               # Pretrained model for object detection
│   ├── config.json
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── pytorch_model.bin
│   └── README.md
│
├── results/                      # Task 1 results
│   ├── 1_template_matching/
│   ├── 2_objectdetection_withassociation/
│   ├── 3_improved/
│   └── README.md
│
├── requirements.txt              # Python dependencies
│
├── task1.ipynb                   # Jupyter Notebook for Task 1
│
├── task2_data/                    # Task 2 data samples
│   └── sample_*.npz
│
├── task2_evaluation_data/         # Task 2 evaluation data
│   └── sample_*.npz
│
├── example.png                    # Task 2 sample image
│
└── task2.ipynb                    # Jupyter Notebook for Task 2
```

## Requirements

Before running the notebooks, install the required dependencies:

```
conda env create -f environment.yaml
```

## Task 1: Multi-Object Tracking

### Description

In Task 1, I implement and compare three tracking methods:

1. __Template Matching:__

    - Basic object tracking using template matching.
    - Results stored in `results/1_template_matching/`.

2. __Object Detection with Data Association:__

    - Leverages a pre-trained DETR (DEtection TRansformer) model for object detection.
    - Data association is done using a tracking algorithm to link detected objects across frames.
    - Results stored in `results/2_objectdetection_withassociation/`.

3. __Improved Method:__

    - Enhancements to increase tracking robustness and accuracy.
    - May include refined association strategies, motion models, or noise reduction.
    - Results stored in `results/3_improved/`.

### How to Run

1. Open `task1.ipynb` in Jupyter Notebook:

```
cd task1_tracking/
jupyter notebook task1.ipynb
```

2. Execute the cells sequentially. The notebook will:

    - Load image sequences from the `/data` directory.
    - Perform template matching, object detection, and improved tracking.
    - Visualize tracking results and save them in the `/results` folder.

3. Check the results in the corresponding folders.

### Output

 - __Tracking results__ for each sequence are saved in the `/results` directory.
 - Performance analysis and visualizations are included in the notebook.

## Task 2: Trajectory Prediction

### Description

__Task 2__ focuses on predicting object trajectories using a __Constant Velocity Model__ and __Constant Acceleration Model__ and then evaluating the predictions using two key metrics:

- __ADE (Average Displacement Error):__ Mean error over all predicted positions.
- __FDE (Final Displacement Error):__ Error at the final time step.

### How to Run

1. Open task2.ipynb in Jupyter Notebook:

```
cd task2_prediction/
jupyter notebook task2.ipynb
```

2. Execute the cells to:

    - Load the provided `.npz` files from `task2_data/`.
    - Apply the constant velocity model for trajectory prediction.
    - Evaluate results using data from `task2_evaluation_data/`.
    - Visualize predicted vs. actual trajectories.

### Outputs

- __Predicted Trajectories__ plotted alongside ground truth.
- __ADE__ and __FDE__ scores printed for each sequence.
- Visualizations saved for further analysis.