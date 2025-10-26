
import cv2
import os
import numpy as np

def thinning(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skel

# Define input and output directories
input_dir = '/home/labadmin/R7A_group11/segmented dataset/augmented_normalised/stunned_norm'
output_dir = '/home/labadmin/R7A_group11/stunned_images'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over each image file in the input directory
for filename in os.listdir(input_dir):
    # Load the image
    image_path = os.path.join(input_dir, filename)
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply GaussianBlur to reduce noise
    blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)

    # Apply adaptive thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform thinning
    thinned_image = thinning(binary_image)

    # Save the thinned image
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, thinned_image)

    print(f"Thinned image saved to {output_path}")

print("Thinning completed for all images.")


