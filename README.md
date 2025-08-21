# ml-training
# Machine Learning for Sensor Fusion (6-DoF Pose Estimation)

This repository contains a training script that maps **sensor inputs** (IMU and potentiometer readings) to **6-DoF actuator poses** (`x, y, z, roll, pitch, yaw`).  
It was developed as part of a soft robotics project and demonstrates applied machine learning for regression, evaluation, and research tooling.

---

## Features
- Implements **Random Forest** (scikit-learn) and a **Neural Network** (Keras/TensorFlow).  
- Handles **multi-output regression** with six targets.  
- Reports **per-output Mean Squared Error (MSE)** and overall averages.  
- Optionally compares results against baseline estimated poses (if present in the dataset).  
- Saves trained models and evaluation plots for reuse.  

---

## Repository Structure
- `Train_Model_ML_Sensor_Fusion_Sensor_to_Pose.py` — main training script  
- `simulated_datax.csv` — sample dataset with sensor inputs and ground truth poses  
- `requirements.txt` — Python dependencies  

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python Train_Model_ML_Sensor_Fusion_Sensor_to_Pose.py
   ```


## Outputs (saved in the repo folder)

- rf_pose_model.joblib — trained Random Forest model

- nn_pose_model.keras — trained Neural Network model

- Plots: MSE comparisons, scatter plots, learning curves

