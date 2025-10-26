import os
import imgaug.augmenters as iaa
import cv2

# Define input and output directories
input_unhealthy_dir = "/home/labadmin/R7A_group11/segmented dataset/slipped tendons"
output_unhealthy_dir = "/home/labadmin/R7A_group11/segmented dataset/balanced_slipped11"
# Create output directory if it doesn't exist
os.makedirs(output_unhealthy_dir, exist_ok=True)

# Image augmentation parameters
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip 50% of images
    #iaa.Affine(rotate=(-20, 20)),  # rotate images by -20 to +20 degrees
    iaa.GaussianBlur(sigma=(0, 1.0)),  # apply gaussian blur with a sigma of 0 to 1.0
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # add Gaussian noise with a scale of 0 to 5% of image range
])

# Iterate over unhealthy chicken images
unhealthy_image_files = os.listdir(input_unhealthy_dir)
target_count = 43  # Adjusted target count
augmented_count = 0

for filename in unhealthy_image_files:
    if augmented_count >= target_count:
        break

    image_path = os.path.join(input_unhealthy_dir, filename)
    image = cv2.imread(image_path)
    
    # Augment image
    augmented_images = [seq.augment_image(image) for _ in range(5)]  # Augment each image 5 times
    
    # Save augmented images
    for i, augmented_image in enumerate(augmented_images):
        if augmented_count >= target_count:
            break
        
        output_path = os.path.join(output_unhealthy_dir, f"{filename[:-4]}_aug_{i}.jpg")  # Save with a different filename
        cv2.imwrite(output_path, augmented_image)
        augmented_count += 1
