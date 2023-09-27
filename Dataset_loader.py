import os
import pandas as pd
import typing

import torch
from torchvision.io import read_image

from support_functions.drawing import draw_bounding_box_on_first_image
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class AslDataLoader(Dataset):

    def __init__(self, root: str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = pd.read_csv(root)

    def __getitem__(self, idx):
        img_path = os.path.join("data/data_detect/train", self.imgs.loc[idx]['img_name'])

        img = read_image(img_path)
        resize= torchvision.transforms.Resize(( 370, 370))
        img =resize(img)

        boxes = torch.DoubleTensor([self.imgs.loc[idx]['x1'], self.imgs.loc[idx]['y1'], self.imgs.loc[idx]['x2'], self.imgs.loc[idx]['y2']])
        # print(boxes)
        labels = [self.imgs.loc[idx]['label']]
        #image_id = idx
        target = {}
        target["boxes"]=torch.reshape(boxes,(1,4))
        target["labels"]=labels



        return img, target

    def __len__(self):
        return len(self.imgs)
