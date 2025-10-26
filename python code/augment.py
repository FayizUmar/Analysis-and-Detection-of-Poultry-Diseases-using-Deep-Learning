import os
import cv2
import numpy as np

# Directory containing the original images
original_image_dir = "/home/labadmin/R7A_group11/segmented dataset/stunned_growth"

# Directory to save the augmented images
augmented_image_dir = "/home/labadmin/R7A_group11/segmented dataset/stunned_growth_aug"
# Ensure output directory exists
os.makedirs(augmented_image_dir, exist_ok=True)

# Function to perform image augmentation
def augment_images(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            print("Processing:", filename)  # Print the filename

            try:
                # Load the original image
                original_image = cv2.imread(filepath)

                # Rotate the image by 90 degrees clockwise
                rotated_image_90 = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
                save_image(rotated_image_90, filename, "rotated_90")

                # Rotate the image by 180 degrees
                rotated_image_180 = cv2.rotate(original_image, cv2.ROTATE_180)
                save_image(rotated_image_180, filename, "rotated_180")

                # Flip the image horizontally
                flipped_image_horizontal = cv2.flip(original_image, 1)
                save_image(flipped_image_horizontal, filename, "flipped_horizontal")

              
                # Apply Gaussian blur to the image
                blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
                save_image(blurred_image, filename, "blurred")

                # Add Gaussian noise to the image
                noisy_image = add_gaussian_noise(original_image)
                save_image(noisy_image, filename, "noisy")

                # Shift the image by 10 pixels horizontally and vertically
                shifted_image = shift_image(original_image, 10, 10)
                save_image(shifted_image, filename, "shifted")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Function to save the augmented image
# Function to save the augmented image
def save_image(image, original_filename, augmentation):
    # Construct the new filename
    new_filename = f"{original_filename.split('.')[0]}_{augmentation}.jpg"
    output_path = os.path.join(augmented_image_dir, new_filename)
    
    # Save the augmented image
    cv2.imwrite(output_path, image)
    print("Augmented image saved:", new_filename)

# Function to add Gaussian noise to the image
def add_gaussian_noise(image):
    mean = 0
    var = 10
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian.astype(np.uint8)
    return noisy_image

# Function to shift the image
def shift_image(image, dx, dy):
    rows, cols, _ = image.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_image = cv2.warpAffine(image, M, (cols, rows))
    return shifted_image

# Call the function to augment images
augment_images(original_image_dir)

print("Image augmentation completed. Augmented images saved in:", augmented_image_dir)
output_path = os.path.join(augmented_image_dir, new_filename)
