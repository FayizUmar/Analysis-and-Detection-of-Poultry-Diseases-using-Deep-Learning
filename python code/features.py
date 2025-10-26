import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# Define the path to your dataset
train_path = "/home/labadmin/R7A_group11/Poultry disease detection.v6i.multiclass/train"
valid_path = "/home/labadmin/R7A_group11/Poultry disease detection.v6i.multiclass/valid"

# Load the labels from the Excel file
train_labels = pd.read_csv(os.path.join(train_path, "_classes.csv"))

# Create an ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Create generators for training data
print(train_labels.columns)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_labels,
                                                    directory=train_path,
                                                    x_col="filename",
                                                    y_col=[' Diseased Eye', ' Pendulous Crop', ' Sick Broiler', ' Slipped Tendon', ' Stressed -beaks open-'],
                                                    target_size=(224, 224),
                                                    class_mode="raw",
                                                    batch_size=32,
                                                    subset='training')

# Load the pre-trained VGG-16 model without the top (fully connected) layers
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

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
model.fit(train_generator, epochs=200)  # Train for just one epoch to demonstrate

# Extract features from the training set
train_features = model.predict(train_generator)

# Save features and labels to Excel
features_df = pd.DataFrame(train_features, columns=[f'feature_{i}' for i in range(train_features.shape[1])])
labels_df = train_labels.copy()  # Assuming labels are already in the correct format

features_df.to_excel('train_features.xlsx', index=False)
labels_df.to_excel('train_labels.xlsx', index=False)
