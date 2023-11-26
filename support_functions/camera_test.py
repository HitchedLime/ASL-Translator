import torch

import cv2
import torchvision
import torchvision.transforms as T
import torchvision.utils as V
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


import cv2
import torch
def decoder(number ):
     dictionary ={0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: 'A', 27: 'B', 28: 'C', 29: 'D', 30: 'E', 31: 'F', 32: 'G', 33: 'H', 34: 'I', 35: 'J', 36: 'K', 37: 'L', 38: 'M', 39: 'N', 40: 'O', 41: 'P', 42: 'Q', 43: 'R', 44: 'S', 45: 'T', 46: 'U', 47: 'V', 48: 'W', 49: 'X', 50: 'Y', 51: 'Z'}

     return  dictionary[number]


def draw_bounding_boxes(image, predictions, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image.

    Parameters:
    - image: The image on which to draw the bounding boxes.
    - predictions: A list containing  dictionary  the bounding boxes, labels, and scores in .

    """

    # Convert the bounding boxes to a list of tuples
    bounding_boxes = predictions[0]['boxes']
    scores= predictions[0]['scores']
    index_of_best = [idx for idx, val in enumerate(scores) if val > 0.5]
    labels = predictions[0]['labels']

    for box in index_of_best:
        x1, y1, x2, y2 = map(int, bounding_boxes[box])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Display the score and label text on top of the bounding box
        label = f"{decoder(labels[box].item())}: {scores[box]:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

])

# Start the main loop.
while True:


    ret, frame = cap.read()

    # If the frame is not empty, preprocess it and pass it to the model.
    if ret:
       # frame = cv2.resize(frame, (224, 224))
        frame_tensor = transform(frame)
        frame_tensor = frame_tensor.unsqueeze(0)


        prediction = model(frame_tensor)
        frame = draw_bounding_boxes(frame,prediction)
        cv2.imshow('Camera Feed', frame)
        # Do something with the prediction.

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

