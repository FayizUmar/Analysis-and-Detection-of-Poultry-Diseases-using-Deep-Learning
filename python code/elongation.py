import os
import cv2
import numpy as np
import pandas as pd  # Import pandas for Excel operations

# Define input and output folder paths
input_folder_path = '/home/labadmin/R7A_group11/segmented dataset/augmented_normalised/stunned_norm'  # Replace with your actual input path
output_excel_path = '/home/labadmin/R7A_group11/segmented dataset/excel_sheets/Stunned_excels/elongation.xlsx'  # Replace with your desired output Excel path

# Create an empty list to store the ratios
ratios = []

# Process images in the input folder
for filename in os.listdir(input_folder_path):
    image_path = os.path.join(input_folder_path, filename)

    # Load the image
    print(f"Processing image: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error loading image: {image_path}")
        continue  # Skip to the next image

    # Check if the image is grayscale
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        print(f"Unsupported image format: {image_path}")
        continue

    # Find contours in the image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the chicken)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box around the chicken
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculate the ratio of height to width
    ratio = h / w

    # Add the ratio to the list
    ratios.append({'filename': filename, 'ratio': ratio})

# Create a DataFrame from the ratios list
ratios_df = pd.DataFrame(ratios)

try:
    # Save the DataFrame to an Excel sheet
    ratios_df.to_excel(output_excel_path, index=False)
    print(f"Height-to-width ratios saved to: {output_excel_path}")
except Exception as e:
    print(f"Error saving Excel file: {e}")

