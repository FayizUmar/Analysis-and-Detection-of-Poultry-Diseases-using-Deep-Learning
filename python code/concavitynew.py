import cv2
import numpy as np
import os
import pandas as pd

# Function to determine if a contour is convex or concave
def determine_convexity(contour):
    # Find the convex hull of the contour
    hull = cv2.convexHull(contour)
    
    # Calculate the area of the contour
    area_contour = cv2.contourArea(contour)
    
    # Calculate the area of the convex hull
    area_hull = cv2.contourArea(hull)
    
    # Check if there is a concavity
    if area_hull > area_contour:
        return "1"  # for concave
    else:
        return "0"  # for convex

# Directory containing the images
directory = '/home/labadmin/R7A_group11/segmented dataset/augmented_normalised/stunned_norm'  # Replace with your actual input path

# Initialize an empty list to store results
results = []

# Iterate over all image files in the directory
for filename in os.listdir(directory):
    # Load the image
    image_path = os.path.join(directory, filename)
    image = cv2.imread(image_path)
    
    # Check if the image is loaded successfully
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding or other preprocessing techniques to obtain a binary image
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        concave_count = 0
        convex_count = 0

        # Iterate through contours
        for contour in contours:
            # Determine convexity
            convexity = determine_convexity(contour)
            if convexity == "1":
                concave_count += 1
            else:
                convex_count += 1
        
        # Determine overall convexity based on counts
        if concave_count > convex_count:
            overall_convexity = "concave"
        else:
            overall_convexity = "convex"
            
        # Append the result to the list
        results.append({"Image": filename, "Concavity": overall_convexity})
        
# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results)

# Specify the full path to the desired location for saving the Excel file
output_excel_path = '/home/labadmin/R7A_group11/segmented dataset/excel_sheets/Stunned_excels/concavity.xlsx'

# Save the results to an Excel file at the specified location
results_df.to_excel(output_excel_path, index=False)

print(f"Results saved to: {output_excel_path}")

