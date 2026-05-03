# Experiment 13: Crop Disease Classification using Leaf Images
# TensorFlow-free version for Python 3.14

import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


# Step 1: Dataset paths
train_path = "cropdataset/tomato/train"
val_path = "cropdataset/tomato/val"

image_size = (64, 64)


# Step 2: Function to load images
def load_images_from_folder(folder_path):
    images = []
    labels = []

    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)

        if not os.path.isdir(class_folder):
            continue

        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)

            try:
                img = Image.open(file_path).convert("RGB")
                img = img.resize(image_size)

                img_array = np.array(img) / 255.0
                img_array = img_array.flatten()

                images.append(img_array)
                labels.append(class_name)

            except Exception as e:
                print("Skipped file:", file_path)

    return np.array(images), np.array(labels)


# Step 3: Load train and validation data
X_train, y_train = load_images_from_folder(train_path)
X_test, y_test = load_images_from_folder(val_path)

print("Training images:", X_train.shape)
print("Validation images:", X_test.shape)

# Step 4: Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("\nClasses:")
print(le.classes_)

# Step 5: Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train_encoded)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("\nAccuracy:", accuracy_score(y_test_encoded, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_
)

plt.title("Crop Disease Classification Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

# Step 9: Test one image
sample_image = X_test[0].reshape(1, -1)
sample_prediction = model.predict(sample_image)

print("\nPrediction for first validation image:")
print(le.inverse_transform(sample_prediction)[0])

print("\nExperiment 13 completed successfully.")