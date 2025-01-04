# Traffic Sign Detection and Tracking System

## Project Overview
This project implements a real-time traffic sign detection and tracking system using a Raspberry Pi with camera. The system detects traffic signs, tracks their position, and calculates their distance from the camera.

## Key Features
- Traffic sign detection using classical computer vision techniques
- Real-time tracking using meanShift algorithm
- Distance estimation through camera calibration
- Sign classification using SVM with SIFT features
- Optimized performance with selective frame processing

## Installation
```bash
git clone https://github.com/[username]/Vision-por-Ordenador.git
cd Vision-por-Ordenador
pip install -r requirements.txt
```

## Usage
To run the main detection system:
```bash
python detect_signs.py
```

## Training the Classifier
To train the SVM classifier:
```bash
python src/classifying/train_classifier.py
```
The classifier uses SIFT features extracted from traffic sign images for training.

## Technical Details
- Detection: Performed every 100 frames for efficiency
- Tracking: MeanShift algorithm maintains sign positions between detections
- Classification: SVM classifier with SIFT keypoint features
- Camera calibration ensures accurate distance measurements

## Project Team
- María González Gómez
- Jorge Vanco Sampedro

# Vision-por-Ordenador
## Extended Explanation

This project focuses on detecting and tracking traffic signs on a Raspberry Pi equipped with a camera. By performing camera calibration, the system determines the real-world distance to each sign. Since the recognition step is computationally costly, signs are only detected every 100 frames. In the intervening frames, they are tracked with meanShift to maintain their positions. A trained SVM classifier using SIFT features distinguishes different traffic signs after they have been identified.

### Camera Calibration
A standard calibration procedure is used, where known patterns (such as a checkerboard) help compute intrinsic camera parameters. These values allow accurate distance estimations to track how far a detected sign is from the camera.

### Project Workflow
1. Clone the repository and install the required dependencies.  
2. Run the main script to initiate the detection and tracking loop.  
3. A detection routine triggers periodically, applying classical computer vision algorithms (e.g., color thresholding, feature matching) to locate traffic signs.  
4. Between detections, meanShift tracking locks onto the previously recognized sign regions to reduce computational load.  
5. Distances to detected signs are calculated using the calibrated camera parameters.

### Classifier Training
A set of labeled traffic sign images is processed to extract SIFT keypoints. These features train an SVM model that classifies each sign type. The training script automatically loads dataset images, computes SIFT descriptors, and fits the classifier to these features. After training, the model can be integrated into the main pipeline.

### Execution
• Download or clone the repository.  
• Install all required libraries using the provided script.  
• Launch the detection program to start identifying signs and measuring distances in real time.  
• If necessary, build or retrain the SVM classifier by running the dedicated training file.

### Contributors
• María González Gómez  
• Jorge Vanco Sampedro  

