import torch
import os



from torch.optim import optimizer
from torch.testing._internal.common_quantization import train_one_epoch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from Dataset_loader import AslDataLoader
from engine import train_one_epoch, evaluate

from collate_functions import collate_fn
root_train = "data/data_detect/train/_annotations.csv"
root_test = "data/data_detect/test/_annotations.csv"

dataset_train = AslDataLoader(root_train)
dataset_test =AslDataLoader(root_test)
data_loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True,collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=True,collate_fn=collate_fn)
device =torch.device('cpu')

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 52 # for each letter
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

# let's train it for 5 epochs
num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)