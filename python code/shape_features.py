import os
import cv2
import numpy as np
import pandas as pd

# Define the input and output directories
input_dir = '/home/labadmin/R7A_group11/segmented dataset/augmented_normalised/stunned_norm'
output_dir = '/home/labadmin/R7A_group11/segmented dataset/excel_sheets/Stunned_excels'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create an empty list to store the shape features for all images
all_features = []

# Iterate over all the image files in the input directory
for filename in os.listdir(input_dir):
    # Construct the full path to the image file
    image_path = os.path.join(input_dir, filename)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is None:
        print(f"Error loading image: {image_path}")
        continue
    
    # Convert image to grayscale
    if len(image.shape) == 3:  # Check if the image is not already grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # If already grayscale, no need to convert
    
    # Find contours in the image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the chicken)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the shape features
    epsilon = 0.04 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    num_corners = len(approx)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * (area / perimeter**2) if perimeter != 0 else 0
    hull = cv2.convexHull(largest_contour)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) != 0 else 0
    
    # Append the shape features to the list
    all_features.append({
        'Filename': filename,
        'Number of Corners': num_corners,
        'Circularity': circularity,
        'Solidity': solidity
    })

# Create a DataFrame from the list of shape features
df = pd.DataFrame(all_features)

# Define the output Excel file path
output_excel_path = os.path.join(output_dir, 'shape_features.xlsx')

try:
    # Save the DataFrame to an Excel file
    df.to_excel(output_excel_path, index=False)
    print(f'Shape features saved to {output_excel_path}')
except Exception as e:
    print(f'Error saving Excel file: {e}')

