import os
import cv2
import numpy as np
import pandas as pd  # Import pandas for Excel operations

# Define input and output folder paths
input_folder_path = '/home/labadmin/R7A_group11/segmented dataset/augmented_normalised/stunned_norm'  # Replace with your actual input path
output_excel_path = '/home/labadmin/R7A_group11/segmented dataset/excel_sheets/Stunned_excels/arealinear.xlsx'  # Replace with desired output path

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
    
    # Check if the image has only one channel
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        print(f"Unsupported image format: {image_path}")
        continue

    # Use adaptive thresholding to separate the chicken from the background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the chicken)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the area and perimeter of the largest contour
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Calculate the area-to-circumference ratio
    area_circumference_ratio = area / perimeter

    # Add the ratio to the list
    ratios.append({'filename': filename, 'area_circumference_ratio': area_circumference_ratio})

# Create a DataFrame from the ratios list
ratios_df = pd.DataFrame(ratios)

try:
    # Save the DataFrame to an Excel sheet
    ratios_df.to_excel(output_excel_path, index=False)
    print(f"Area-to-circumference ratios saved to: {output_excel_path}")
except Exception as e:
    print(f"Error saving Excel file: {e}")

