from ultralytics import YOLO

# Load a model

model = YOLO("yolov8n.pt").cuda()  # load a pretrained model (recommended for training)

# Use the model
model.train(data="data.yaml", epochs=100)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("/home/labadmin/R7A_group11/Poultry disease detection.v1i.yolov8/valid/images/yolo8test.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
