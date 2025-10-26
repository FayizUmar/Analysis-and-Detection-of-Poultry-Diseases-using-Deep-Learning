import os
import cv2
import numpy as np

# Directories for images and segmentation results (replace with your actual paths)
segmented_image_dir = "/home/labadmin/R7A_group11/segmented dataset/augmented_balanced/stunned_growth_aug"
normalized_image_dir = "/home/labadmin/R7A_group11/segmented dataset/augmented_normalised/stunned_norm"

# Ensure output directory exists
os.makedirs(normalized_image_dir, exist_ok=True)

def normalize_images(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            print("Processing:", filename)  # Print the filename

            try:
                # Load the segmented image
                segmented_image = cv2.imread(filepath)

                # Convert the image to grayscale
                gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

                # Normalize the image to the range [0, 255]
                normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                # Save the normalized image
                output_path = os.path.join(normalized_image_dir, filename)
                cv2.imwrite(output_path, normalized_image)

                print("Normalized image saved:", filename)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Call the function to normalize segmented images
normalize_images(segmented_image_dir)

print("Normalization completed. Normalized images saved in:", normalized_image_dir)
