import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

def load_test_data(data_dir='data/color', target_size=(224, 224), batch_size=32):
    """Memory-efficient test data loader"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    class_names = list(test_generator.class_indices.keys())
    print(f"✅ Generator ready with {test_generator.samples} images across {len(class_names)} classes")
    return test_generator, class_names

def generate_classification_report_image(y_true, y_pred, class_names):
    """Generate classification report as matplotlib image with 4 decimal precision"""
    # Create classification report and round values
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    
    # Round numeric columns to 4 decimal places
    numeric_cols = ['precision', 'recall', 'f1-score', 'support']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.axis('off')
    plt.title('Classification Report', pad=20, fontsize=16)
    
    # Create table
    table = plt.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     loc='center',
                     cellLoc='center')
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig('classification_report.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✅ Classification report image saved with 4-digit precision")

def generate_confusion_matrix(y_true, y_pred, class_names):
    """Generate confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix', pad=20, fontsize=16)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()
    print("✅ Confusion matrix saved")

if __name__ == "__main__":
    try:
        # 1. Load data
        test_generator, class_names = load_test_data(
            data_dir='data/color',
            target_size=(224, 224),
            batch_size=32
        )
        
        # 2. Load model and make predictions
        model = load_model("plant_disease_model.h5")
        y_true = []
        y_pred = []
        
        test_generator.reset()
        for i in range(len(test_generator)):
            x, y = test_generator[i]
            batch_pred = model.predict(x, verbose=0).argmax(axis=1)
            y_pred.extend(batch_pred)
            y_true.extend(y.argmax(axis=1))
            
            if (i+1) % 10 == 0:
                print(f"Processed {len(y_pred)}/{test_generator.samples} samples")
        
        # 3. Generate visual reports
        generate_classification_report_image(y_true, y_pred, class_names)
        generate_confusion_matrix(y_true, y_pred, class_names)
        
    except Exception as e:
        print(f"❌ Error: {e}")
       