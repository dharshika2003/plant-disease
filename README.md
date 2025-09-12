# 🌿 Plant Disease Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)  
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)  
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green?logo=opencv)  
![Status](https://img.shields.io/badge/Status-Completed-success)

A **deep learning-based plant disease detection system** built with **MobileNetV2, Grad-CAM, and Otsu’s thresholding**.  
This project integrates **classification, explainable AI, and severity quantification** into a single pipeline, making it a practical tool for **precision agriculture**.

---

## 🎯 Objective
Develop a **lightweight CNN-based system** to:
- Classify plant leaf diseases  
- Visualize infected regions (**Explainable AI**)  
- Estimate **disease severity** for better agricultural decision-making  

---

## 💡 Motivation
Farmers need not only **disease identification** but also the **extent of infection** to decide:
- Whether treatment is possible  
- Or replacement is required  

---

## ⚙️ Approach
- ✅ Classification with **MobileNetV2 (Transfer Learning)**  
- ✅ Explainable AI with **Grad-CAM** (highlight infected regions)  
- ✅ Severity estimation with **Otsu’s Thresholding + Contour Detection**  
- ✅ Deployment via **Gradio Web App** for real-time predictions  

---

## 🚀 Features
✔ Upload a plant leaf image for real-time analysis  
✔ Disease classification with **MobileNetV2**  
✔ Grad-CAM heatmap for **visual interpretability**  
✔ Otsu’s thresholding for **severity segmentation**  
✔ Estimate **percentage of infected leaf area**  
✔ Lightweight **Gradio interface** (browser-based, no installation needed)  

---

## 🛠 Tech Stack
- **Programming Language:** Python 3.8+  
- **Deep Learning Frameworks:** TensorFlow 2.x, Keras  
- **Image Processing:** OpenCV, Pillow  
- **Visualization:** Matplotlib, Grad-CAM utilities  
- **Interface:** Gradio  
- **Deployment:** Local/Cloud (CPU-friendly, GPU optional for training)  

---

## 🏗 System Architecture

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


---

## 🧪 Methodology

### 🔹 Model Training
- Backbone: **MobileNetV2** (ImageNet pre-trained)  
- Added custom layers:  
  - `GlobalAveragePooling` → `Dense(128)` → `Dense(n_classes, softmax)`  
- Optimizer: **Adam**  
- Loss: **Categorical Crossentropy**  
- Metrics: **Accuracy**  

### 🔹 Explainability (Grad-CAM)
- Generates **heatmaps** showing which regions of the leaf influenced the classification.  

### 🔹 Severity Estimation
- Apply **Otsu’s thresholding** on Grad-CAM map  
- Use **contour detection** to isolate diseased regions  
- Calculate **percentage of affected leaf area**  

### 🔹 Deployment
- **Gradio-based web app** for real-time predictions  

---

## 🎥 Demo
📺 [Screen Recording](https://drive.google.com/drive/folders/1937DiNoyZ23w7nGjRTh3h_a1sID4sMlW?usp=sharing)

---

## 📌 Conclusion
This project demonstrates the power of **deep learning + explainable AI** for practical **agriculture applications**.  
It provides not only **disease identification** but also **severity estimation**, making it useful for **farmers, researchers, and agricultural decision systems**.  

