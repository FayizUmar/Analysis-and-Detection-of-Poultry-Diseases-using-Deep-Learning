import os
import shutil

def extract_matching_images(folder1, folder2, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of files in each folder
    folder1_files = os.listdir(folder1)
    folder2_files = os.listdir(folder2)

    # Iterate through files in folder1 and check if they exist in folder2
    for file1 in folder1_files:
        if file1 in folder2_files:
            # If the file exists in folder2, copy it to the output folder
            src_path = os.path.join(folder1, file1)
            dst_path = os.path.join(output_folder, file1)
            shutil.copy(src_path, dst_path)
            print(f"Copied {file1} from {folder1} to {output_folder}")

# Set folder paths
folder1_path = "/home/labadmin/R7A_group11/unhealthy chicken"
folder2_path = "/home/labadmin/R7A_group11/unhealthy_issue"
output_folder_path = "/home/labadmin/R7A_group11/unhealthyimages"

# Call the function to extract matching images
extract_matching_images(folder1_path, folder2_path, output_folder_path)
print("Finished")
