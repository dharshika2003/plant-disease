# plant-disease

This repository contains the implementation of a deep learning-based plant disease detection system developed using MobileNetV2, Grad-CAM, and Otsu’s thresholding.
The project integrates classification, explainable AI, and severity quantification into a single pipeline, making it a practical tool for modern precision agriculture.

Objective: Build a lightweight CNN-based system to classify plant leaf diseases, visualize affected regions, and estimate severity.
Motivation: Farmers require not only disease identification but also extent of infection to take action (e.g., treatment vs. replacement).

Approach:
      Classification with MobileNetV2 (transfer learning).
      Explainable AI with Grad-CAM to highlight infected regions.
      Severity estimation using Otsu’s thresholding & contour detection.
      Deployment via a Gradio web app for real-time predictions.

Features:
✔Upload a plant leaf image for real-time analysis
✔Disease classification using a trained CNN (MobileNetV2)
✔Grad-CAM heatmap for visual interpretability
✔Otsu’s thresholding for severity segmentation
✔Severity percentage estimation (how much of the leaf is infected)
✔Lightweight Gradio interface (browser-based, no installation needed)

Tech Stack

Programming Language: Python 3.8+
Deep Learning Frameworks: TensorFlow 2.x, Keras
Image Processing: OpenCV, Pillow
Visualization: Matplotlib, Grad-CAM utilities
Interface: Gradio
Deployment: Local/Cloud (CPU-friendly, GPU optional for training)

System Architecture:
             ┌─────────────────────────┐
             │  Input Leaf Image (RGB) │
             └─────────────┬───────────┘
                           │
                 Preprocessing (Resize, Normalize, Augment)
                           │
             ┌─────────────▼─────────────┐
             │   MobileNetV2 (CNN)       │
             │   Transfer Learning       │
             └─────────────┬─────────────┘
                           │
                 ┌─────────┴─────────┐
                 │                   │
       ┌─────────▼─────────┐   ┌────▼──────────┐
       │ Disease Prediction │   │ Grad-CAM Map │
       └─────────┬─────────┘   └─────┬────────┘
                 │                   │
                 │         Otsu’s Thresholding
                 │                   │
       ┌─────────▼──────────┐ ┌─────▼───────────┐
       │ Prediction Label   │ │ Severity % Calc │
       └─────────┬──────────┘ └─────┬───────────┘
                 │                   │
            ┌────▼────┐        ┌─────▼─────┐
            │ Gradio  │        │ Heatmap + │
            │ Output  │        │ Severity  │
            └─────────┘        └───────────┘

Methodology

Model Training:
      Backbone: MobileNetV2 (ImageNet pre-trained)
      Added custom layers: GlobalAveragePooling → Dense(128) → Dense(n_classes, softmax)
      Optimizer: Adam
      Loss: Categorical Crossentropy
      Metrics: Accuracy

Explainability (Grad-CAM):
      Generates heatmaps showing which parts of the leaf influenced the classification.
      
Severity Estimation: 
      Otsu’s thresholding applied on Grad-CAM map
      Contour detection to isolate diseased regions
      Percentage of affected leaf calculated

Deployment:
Gradio-based web interface for real-time predictions.

recording: https://drive.google.com/drive/folders/1937DiNoyZ23w7nGjRTh3h_a1sID4sMlW?usp=sharing





