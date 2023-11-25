import torch

import cv2
import torchvision
import torchvision.transforms as T
import torchvision.utils as V
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model_path ="model.pt"

device = torch.device('cpu')



num_classes = 52 # for each letter
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()


in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
# model.load_state_dict(torch.load(model_path , map_location=device))

model.to(device)


model.eval()


# Create a VideoCapture object.
cap = cv2.VideoCapture(0)

# Define a transform to convert the frame to a PyTorch tensor.
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Start the main loop.
while True:

    # Get a frame from the camera.
    ret, frame = cap.read()

    # If the frame is not empty, preprocess it and pass it to the model.
    if ret:
        frame = cv2.resize(frame, (224, 224))
        frame_tensor = transform(frame)
        frame_tensor = frame_tensor.unsqueeze(0)

        # Pass the frame tensor to the model.
        prediction = model(frame_tensor)

        # Do something with the prediction.
        print(prediction)
