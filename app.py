import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import gradio as gr
import cv2
import os

# Load your model
model = load_model("plant_disease_model_gc.h5")
class_names = list(np.load("class_names.npy"))

# NEW: Print model summary and find correct conv layer
print("Model layers:")
for layer in model.get_layer("mobilenetv2_1.00_224").layers:
    if 'conv' in layer.name or 'out_relu' in layer.name:
        print(layer.name)
# Typically the last conv layer is either 'out_relu' or 'block_16_expand' in MobileNetV2
LAST_CONV_LAYER = "out_relu"  # CHANGED from "Conv_1"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=LAST_CONV_LAYER, pred_index=None):  # Added default
    base_model = model.get_layer("mobilenetv2_1.00_224")
    try:
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
    except:
        raise ValueError(f"Layer {last_conv_layer_name} not found in model. Available conv layers: " +
                        str([l.name for l in base_model.layers if 'conv' in l.name]))

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, features = grad_model(img_array)
        pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
        output = model.layers[-2](pooled)
        preds = model.layers[-1](output)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), int(pred_index)

def preprocess_image(image):
    # CHANGED to match training preprocessing
    img = tf.keras.applications.mobilenet_v2.preprocess_input(image.astype(np.float32))
    img = tf.image.resize(img, (224, 224))
    return np.expand_dims(img, axis=0)

def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = 255 - heatmap
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    overlayed = cv2.addWeighted(heatmap_color, alpha, original_img, 1 - alpha, 0)
    return overlayed

def estimate_severity(heatmap, original_img, threshold=0.5):  # Adjusted default threshold
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY) if len(original_img.shape) > 2 else original_img
    _, leaf_mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    leaf_mask = np.zeros_like(gray_img)
    cv2.drawContours(leaf_mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    
    total_leaf_area = np.sum(leaf_mask > 0)
    if total_leaf_area == 0:
        return 0.0

    diseased_mask = (heatmap > threshold).astype(np.uint8) * 255
    diseased_mask = cv2.bitwise_and(diseased_mask, diseased_mask, mask=leaf_mask)
    
    return (np.sum(diseased_mask > 0) / total_leaf_area) * 100

def predict_and_visualize(image):
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    confidence = preds[0][pred_class]

    heatmap, _ = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER, pred_class)  # CHANGED

    original_img = cv2.resize(image, (224, 224))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    overlayed_img = overlay_heatmap(heatmap, original_img)

    severity = estimate_severity(heatmap, original_img)
    severity_text = f"Disease Severity: {severity:.1f}%"

    cv2.putText(overlayed_img, severity_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return overlayed_img, f"Prediction: {class_names[pred_class]} ({confidence*100:.2f}%)", severity_text

# Gradio Interface
gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="numpy", label="Upload Leaf Image"),
    outputs=[
        gr.Image(type="numpy", label="Grad-CAM Visualization"),
        gr.Label(label="Predicted Class"),
        gr.Label(label="Disease Severity")
    ],
    title="🌿 Plant Disease Detection with Grad-CAM & Severity Estimation",
    description="Upload a leaf image to detect plant disease, visualize model attention, and estimate disease severity.",
    examples=[["example_healthy.jpg"], ["example_diseased.jpg"]]
).launch(debug=True, share=True)