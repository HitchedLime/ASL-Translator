
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from Dataset_loader import AslDataLoader


from collate_functions import collate_fn
root = "data/data_detect/train/_annotations.csv"

dataset = AslDataLoader(root)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True,collate_fn=collate_fn)



# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 26 # for each letter
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# images, targets = next(iter(data_loader))
# images = list(image for image in images)
# for target in targets:
#     print(target['boxes'])

for i, data in enumerate(data_loader):

    output = model(data['images'], data['targets'])  # Returns losses and detections
