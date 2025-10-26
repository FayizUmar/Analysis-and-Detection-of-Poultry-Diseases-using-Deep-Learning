import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor  # Corrected import statement
# Corrected import statement

#SAMPredictor 
#from ultralytics import YOLO, SAMPredictor # Uncomment if using Ultralytics models

# Load the model and predictor (replace paths with your model locations)
model = YOLO("/home/labadmin/R7A_group11/yolov8n.pt")  # Load the YOLOv8n model
sam_checkpoint = "/home/labadmin/R7A_group11/sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)  # Load the SAM predictor

# Directories for images and segmentation results (replace with your actual paths)
image_dir = "/home/labadmin/R7A_group11/healthy chicken2"
output_dir = "/home/labadmin/R7A_group11/healthy chicken2_seg"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def process_images(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            print("starting with ",filename)  # Print the filename

            try:
                 # Load the image
                image = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

                # Assert image loading
                assert image is not None, f"Failed to load image: {filepath}"

                # Detect objects using YOLOv8n
                results = model.predict(source=filepath, conf=0.25)

                # Segment each detected object using SAM
                for result in results:
                    boxes = result.boxes
                    for box in boxes.xyxy:
                        bbox = box.tolist()

                        predictor.set_image(image)
                        input_box = np.array(bbox)
                        masks, _, _ = predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_box[None, :],
                            multimask_output=False,
                        )

                        segmentation_mask = masks[0]
                        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
                        white_background = np.ones_like(image) * 255
                        new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]

                        # Save the segmented image
                        output_path = os.path.join(output_dir, filename)
                        print("Output saved ",filename)
                        cv2.imwrite(output_path, new_image.astype(np.uint8))

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        else:
            process_images(filepath)  # Recurse for subdirectories

# Call the recursive function to start processing
process_images(image_dir)

print("Segmentation completed. Segmented images saved in:", output_dir)
