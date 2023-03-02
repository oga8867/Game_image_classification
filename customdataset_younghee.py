from torch.utils.data import Dataset
import os
import glob
from PIL import Image
import numpy as np

class my_custom(Dataset) :
    def __init__(self, path, transform=None):
        self.all_path = glob.glob(os.path.join(path, "*", "*.png"))
        self.transform = transform
        self.label_dict = {"fifa" : 0 , "lol" : 1, "maple" : 2, "overwatch" : 3, "rostark": 4}

    def __getitem__(self, item):
        image_path = self.all_path[item]
        image = Image.open(image_path).convert("RGB")

        label_temp = image_path.split("\\")[1]
        label = self.label_dict[label_temp]
        # print(type(image))


        if self.transform is not None :
            image = self.transform(image)


        # print(image_path)
        # print(type(image))
        return image, label

    def __len__(self):
        return len(self.all_path)
    

# all_path = glob.glob(os.path.join("./dataset/train/", "*", "*.png"))
# label_dict = {"fifa" : 0 , "lol" : 1, "maple" : 2, "overwatch" : 3, "rostark": 4}
# # print(all_path)
# globed_dir = glob.glob(os.path.join("./dataset/train/", "*", "*.png"))
# path = os.path.join("./dataset/train/", "*", "*.png")
# for i in range(5):
#     print(path, len(path))
# # for i in range(40):
# #     # print(globed_dir[i*100], len(globed_dir))
# print(globed_dir[790])

# print(globed_dir[801])
# print(globed_dir[1590])
# print(globed_dir[1601])

# class Test():
#     def __init__(self,a=0):
#         self.a = a
    
#     def __getitem__(self, item):
#         print(item)

# test1 = Test()
# test2 = Test()

# test1[1]
# # test1[4]
# import torch
# import torchvision
# from torchvision import transforms
# from torch.utils.data import DataLoader

# from customdataset import my_custom
# from torchvision import models

# train_transforms = transforms.Compose([
#     transforms.Resize((224,224)),
#     # transforms.RandomHorizontalFlip(),
#     # transforms.RandomRotation(25),
#     transforms.RandomEqualize(),
#     # transforms.RandomVerticalFlip(),
#     transforms.ToTensor()
# ])
# train_dataset = my_custom("./dataset/train/", transform=train_transforms)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# # print(train_dataset.all_path)
# print(train_loader)
# for i in train_loader:
#     break

# mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]
# 2 data set data loader