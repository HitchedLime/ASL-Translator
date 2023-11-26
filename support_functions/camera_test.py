import torch

import cv2
import torchvision
import torchvision.transforms as T
import torchvision.utils as V
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


import cv2
import torch

def draw_bounding_boxes(image, predictions, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image.

    Parameters:
    - image: The image on which to draw the bounding boxes.
    - predictions: A dictionary containing the bounding boxes, labels, and scores.
    - color: The color of the bounding boxes. Default is green (0, 255, 0).
    - thickness: The thickness of the bounding box lines. Default is 2.
    """

    # Convert the bounding boxes to a list of tuples
    bounding_boxes = predictions[0]['boxes']
    scores=  predictions[0]['scores']
    labels = predictions[0]['labels']
    for box in bounding_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image






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


    ret, frame = cap.read()

    # If the frame is not empty, preprocess it and pass it to the model.
    if ret:
        #frame = cv2.resize(frame, (224, 224))
        frame_tensor = transform(frame)
        frame_tensor = frame_tensor.unsqueeze(0)


        prediction = model(frame_tensor)
        frame = draw_bounding_boxes(frame,prediction)
        cv2.imshow('Camera Feed', frame)
        # Do something with the prediction.
        print(prediction)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

