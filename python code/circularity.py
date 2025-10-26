import os
import cv2
import numpy as np
import pandas as pd  
from matplotlib import pyplot as plt

# Define input and output folder paths
input_folder_path = '/home/labadmin/R7A_group11/segmented dataset/augmented_normalised/stunned_norm'  # Replace with your actual input path
output_excel_path = '/home/labadmin/R7A_group11/segmented dataset/excel_sheets/Stunned_excels/circularity.xlsx'# Replace with desired output path

# Create an empty list to store the ratios
ratios = []

# Process images in the input folder
for filename in os.listdir(input_folder_path):
    image_path = os.path.join(input_folder_path, filename)

    print(f"Processing image: {image_path}")  # Debugging statement

    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
      # Convert image to grayscale
    if len(image.shape) == 3:  # Check if the image is not already grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # If already grayscale, no need to convert

    # Convert image to grayscale
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to separate the chicken from the background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it's the chicken)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the area and perimeter of the largest contour
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Calculate the isoperimetric quotient ratio
    isoperimetric_ratio = (4 * np.pi * area) / (perimeter * perimeter)

    # Add the ratio to the list
    ratios.append({'filename': filename, 'isoperimetric_ratio': isoperimetric_ratio})

    print(f"Processed image: {image_path}, Ratio: {isoperimetric_ratio}")  # Debugging statement

# Create a DataFrame from the ratios list
ratios_df = pd.DataFrame(ratios)

# Save the DataFrame to an Excel sheet
ratios_df.to_excel(output_excel_path, index=False)

print(f"Isoperimetric ratios saved to: {output_excel_path}")
