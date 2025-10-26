import urllib.request

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
file_path = "sam_vit_h_4b8939.pth"

urllib.request.urlretrieve(url, file_path)
print("File downloaded successfully.")
