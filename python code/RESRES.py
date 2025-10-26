import os
import pandas as pd
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load features and labels
features_df = pd.read_excel('train_features.xlsx')
labels_df = pd.read_excel('train_labels.xlsx')

# Specify the path to your images
image_path = "/home/labadmin/R7A_group11/Poultry disease detection.v6i.multiclass/train"

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(rescale=1./255)

# Create generators for training data
train_generator = datagen.flow_from_dataframe(dataframe=labels_df,
                                              directory=image_path,
                                              x_col="filename",
                                              y_col=[' Diseased Eye', ' Pendulous Crop', ' Sick Broiler', ' Slipped Tendon', ' Stressed -beaks open-'],
                                              target_size=(224, 224),
                                              class_mode="raw",
                                              batch_size=32,
                                              subset='training')

# Load the pre-trained ResNet50 model without the top (fully connected) layers
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for your specific classification task
x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output = layers.Dense(5, activation='sigmoid')(x)

# Create the final model
model = Model(base_model.input, output)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=200)

y_true = labels_df.iloc[:, 1:].values  # Extract true labels from DataFrame

y_pred_prob = model.predict(train_generator)
y_pred = (y_pred_prob > 0.5).astype(int)

# Decode predictions back to multi-class labels
# (Note: This assumes binary classification; adjust as needed)
label_encoder = LabelEncoder()
y_true_encoded = np.argmax(y_true, axis=1)
y_pred_decoded = np.argmax(y_pred, axis=1)

# Fit the label encoder and transform the true labels
y_true_encoded = label_encoder.fit_transform(y_true_encoded)

# Decode class labels using inverse_transform
y_true_decoded = label_encoder.inverse_transform(y_true_encoded)
y_pred_decoded = label_encoder.inverse_transform(y_pred_decoded)

# Evaluate accuracy
accuracy = accuracy_score(y_true_decoded, y_pred_decoded)
print(f'Training Accuracy: {accuracy}')
