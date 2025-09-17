# Mask detection challenge
This project is part of the **Machine Learning 4** course taken in the Master's program in Applied Artificial Intelligence, Icesi University, Cali, Colombia.

#### -- Project Status: Active

## Contributing Members

|Name     |  Email   | 
|---------|-----------------|
|[Andres Cano](https://github.com/Can0land)| andres.cano.consulting@gmail.com     | 
|[Jhonattan Reales](https://github.com/JhonattanReales21) | jhonatanreales21@gmail.com      |

## Project Intro/Objective

This repository contains the full development of a deep learning model for **detecting and classifying people wearing or not wearing face masks** in images.

The project was built from scratch in PyTorch and includes:

- 📦 **Custom CNN backbone** for feature extraction.
- 🧠 **Regression head** to predict bounding boxes.
- 🧠 **Classification head** to determine whether a person is wearing a mask or not.
- 🔁 **Transfer Learning** with pretrained models (ResNet, MobileNet).
- 🔄 **Data augmentation techniques** to improve model robustness.
- 📊 **Performance evaluation** on classification (Accuracy, F1) and regression (IoU).
- 🧪 Comparison of different **hyperparameters and transformations**.

## Repository Structure
📂 data/  
┣ 📂 images/ <sub>Original images</sub>  
┗ 📄 annotations.csv <sub>Annotations (filename, bbox, label)</sub>  

📂 models/  
┣ 📄 backbone_custom.py <sub>Custom CNN</sub>  
┣ 📄 model_full.py <sub>Complete model (backbone + heads)</sub>  
┗ 📄 pretrained_backbones.py <sub>Pretrained architectures</sub>  

📂 notebooks/  
┗ 📒 exploratory_analysis.ipynb <sub>Exploratory analysis</sub>  
  
📂 utils/  
┣ 📄 dataset.py <sub>Custom Dataset for PyTorch</sub>  
┣ 📄 transforms.py <sub>Data augmentation</sub>  
┗ 📄 metrics.py <sub>Metrics (IoU, accuracy, F1)</sub>  
  
📄 train.py <sub>Main training script</sub>  
📄 evaluate.py <sub>Evaluation and visualization</sub>  
📄 requirements.txt  
📄 README.md  

## How to Run
WIP

## Evaluation
- Classification: Accuracy, Precision, Recall, F1-Score
- Regression (Bounding Box): IoU (Intersection over Union)
- Visual inspection of predictions with matplotlib

## Experiments
We evaluated:
- Hyperparameter changes: learning rate, batch size, optimizer
- Data augmentation techniques: flips, jitter, rotate, resize
- Transfer Learning: ResNet18, MobileNetV2

## Methods Used
WIP

## Technologies
WIP