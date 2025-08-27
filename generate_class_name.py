import os
import numpy as np

# 👇 Put the path to your training folder (where class folders exist)
train_dir = "data/color"  # change if your path is different

# Grab folder names as class names
class_names = sorted(os.listdir(train_dir))

# Save to file
np.save("class_names.npy", np.array(class_names))

print("✅ class_names.npy created with classes:", class_names)