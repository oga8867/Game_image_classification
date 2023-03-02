import matplotlib.pyplot as plt
import cv2
import torch
from customdata import my_customdata
from torchvision import transforms
import os
import PIL
from torchvision import models
from torch.utils.data import DataLoader
import os
import copy
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
#os.environ['KMP_DUPLICATE_LIB_OK']='True'#제 컴퓨터 환경에 따른 코드입니다 무시해도 됩니다.
import numpy as np


device = torch.device("cpu")# if torch.cuda.is_available() else "cpu")

val_transforms = A.Compose([
    #A.resize(width=224, height=224, always_apply=True),
    A.SmallestMaxSize(max_size=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.255)),
    ToTensorV2()
])

#모델 불러오기
# model = models.vit_l_32()
# model = model.to(device)
# #model = models.__dict__["efficientnet_b4"](pretrained=True)
# model.head = torch.nn.Linear(in_features=1024, out_features=5)
# #model.classifier[1] = torch.nn.Linear(in_features=1792,out_features=5)
# model.load_state_dict(torch.load("./vit_l_32.pt",map_location=device))

# model = models.efficientnet_b4(pretrained=True)
# model.classifier[1] = torch.nn.Linear(in_features=1792,out_features=5)#450개로 분류하잖음
# model.to(device)


model = models.regnet_x_3_2gf(pretrained=False)
model.fc = torch.nn.Linear(in_features=1008, out_features=5)
model.to(device)


model.load_state_dict(torch.load("./regnet.pt",map_location=device))
#테스트데이터 가져오기. batch= 1인이유 = 한장의 사진만 보기 위해
test_dataset = my_customdata("./dataset/test/", transform=val_transforms)#transforms.ToTensor()) #val_transforms)#
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

#레이벨 설정
label_dict = {0 : "fifa", 1 : "lol", 2 : "maple", 3 : "overwatch", 4 : "rostark"}


#이미지텐서 확인
#print(test_dataset[3000][0].size())

# img_tensor = [t_dataset[3000][0], train_dataset[3001][0]]
# tf = transforms.ToPILImage()
# img = tf(img_tensor)
# print(img)
# img.show()
# image_tensor = PIL.Image.open(image_path).convert("RGB")


#resize
# image, label = img_tensor.to(device), label.to(device)
# img_tensor = img_tensor.resize(1,(224,224,3))


image1 = []
model.eval()
argmax = 0
label = 0
#test_loader 에서 사진 한개만 추출( 실행될때마다 랜덤)
# with torch.no_grad():
#     for i, (images, labels) in enumerate(test_loader):
#         image, label = images.to(device), labels.to(device)
#         output = model(image)
#         _, argmax = torch.max(output, 1)
#         argmax = label_dict[argmax.item()]
#         # tf = transforms.ToPILImage()
#         image1 = image.resize(3,224,224)
#         break


with torch.no_grad():
    for i ,(image,labels) in enumerate(test_loader):
        image, labels = image.to(device), labels.to(device)
        output = model(image)
        _, argmax = torch.max(output,1)
        argmax = label_dict[argmax.item()]
        image1 = image.resize(3, 224, 224)
        print(image1)
        break


# model.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for i ,(image,labels) in enumerate(test_loader):
#         image, labels = image.to(device), labels.to(device)
#         output = model(image)
#         _, argmax = torch.max(output,1)
#         total += image.size(0)
#         correct += (labels == argmax).sum().item()
#         break
#     acc = correct / total * 100
#     print("acc for {} image: {:.2f}%".format(total, acc))



# 이미지 show by PIL
# tf = transforms.ToPILImage()
# image1 = tf(image1)
# image1.show()

#이미지  show by Pyplot
image1 = image1.permute(1,2,0)



plt.imshow(image1)
plt.title(f"predicted label : {argmax}   label : {label_dict[labels.item()]}")
plt.show()