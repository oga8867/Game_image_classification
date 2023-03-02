# Classification
# 1. 데이터 분포확인
# 2. 데이터 나눈작업
#     train, eval(), test -> 8 1 1 학습데이터가 충분한경우
#     예외로 학습데이터가 부족하면 train, eval() 9 1
#     -> test는 수집필요
#      각 라벨별 만장이 필요. 최소한
#       단 데이터가 많다고 좋은건 아니다. 데이터가 다 비슷하면 오버피팅에 빠지므로 다양성이 필요하다.
#3. Custom data set 구축!! -> ㅔpytorch dataset 상속 -> class
#__init__,__getitem__,__len__
#인잇에서 이미지폴더에서 이미지 경로 가져오기 -> 그리고 list로 만듬
#또한 transform 정의
# 읽는것도 여기서함
#
#__getitem__은 이미지 경로가 담겨있는 list에서 인덱스 추출
# 이미지 풀경로 -> cv2나 pil로 이미지를 오픈한다.
# 라벨 필요합니다. 딕셔너리나 if문 사용한다. 대부분 폴더명을 기준으로 한다.
#image.png -> 이미지 파일 이름 기준으로도 한다. label.txt가 있는 경우도 있음
# csv로 주어지는 경우도 있음 filename 해당하는 라벨 지정되어 있습니다.
#대부분 폴더, 이미지명 기준

#어그멘테이션 적용, return으로 image와 label이 나옴

# __len__ 전체 데이터 길이 반환 ->list ->len()

# 학습에 필요한코드

# 테스트


import glob
import os
import random

import torchvision
import cv2
import shutil


import sys
from os import rename, listdir



#rename 한글이름일 경우 이름을 바꿔줌

import os

# 주어진 디렉토리에 있는 항목들의 이름을 담고 있는 리스트를 반환합니다.
# 리스트는 임의의 순서대로 나열됩니다.

rename_file_path = "./dataset/test/rostark"
rename_file_names = os.listdir(rename_file_path)
#file_names


i = 1
for name in rename_file_names:
    src = os.path.join(rename_file_path, name)
    dst = "game"+str(i) + '.png'
    dst = os.path.join(rename_file_path, dst)
    os.rename(src, dst)
    i += 1






# 확장자형식변환
# PATH = './faceset/ani'
# filelist = listdir(PATH)
# for name in filelist:
#     if name.find('.') < 0:
#         continue
#     replaced = name.replace("jpg","png")
#     rename(PATH+'\\'+name, PATH+'\\'+replaced)
#     print(name,' -> ',replaced)
# print('변환 완료')

#아래는 이미지 리사이즈 참고 코드
from PIL import Image
#
# img = Image.open('data/src/sample.png')
#
# img_resize = img.resize((224, 224))
# img_resize.save('data/dst/sample_pillow_resize_nearest.jpg')
#
# img_resize_lanczos = img.resize((256, 256), Image.LANCZOS)
# img_resize_lanczos.save('data/dst/sample_pillow_resize_lanczos.jpg')



# 리사이즈 참고 코드 2 , 이걸로 미리 리사이즈 해놓을 수 있음
from PIL import Image
import os.path

targerdir = "./game/rostark" #해당 폴더 설정

files = os.listdir(targerdir)

format = [".jpg",".png",".jpeg","bmp",".JPG",".PNG","JPEG","BMP"] #지원하는 파일 형태의 확장자들
for (path,dirs,files) in os.walk(targerdir):
    for file in files:
         if file.endswith(tuple(format)):
             image = Image.open(path+"\\"+file)
             print(image.filename)
             print(image.size)

             image=image.resize((int(224), int(224)))
             image.save(path+"\\"+file)
             print(image.size)

         else:
             print(path)
             print("InValid",file)
#
#
#
#

#스플릿부분
# def image_size(path):
#     #folder size
#     fifa = glob.glob((os.path.join(path,"fifa","*.png")))
#     lol = glob.glob((os.path.join(path, "lol", "*.png")))
#     maple = glob.glob((os.path.join(path, "maple", "*.png")))
#     overwatch = glob.glob((os.path.join(path, "overwatch", "*.png")))
#     rostark = glob.glob((os.path.join(path, "rostark", "*.png")))
#
# #
# #     show_flg =False
# #     if show_flg == True:
# #         print(f"african_wildcat {len(paper)}")
# #         print(f"blackfoot_cat {len(rock)}")
# #         print(f"chinese_mountain_cat {len(scissors)}")
# #     pass
# def create_train_val_split_folder(path):
#     all_categories = os.listdir(path)
#     print("all categories >>", all_categories)
#     os.makedirs("./dataset/train/",exist_ok=True)
#     os.makedirs("./dataset/val/", exist_ok=True)
#     os.makedirs("./dataset/test/", exist_ok=True)
#
#     for category in sorted(all_categories):
#         os.makedirs(f"./dataset/train/{category}",exist_ok=True)
#         all_image = os.listdir(f"./game/{category}/")
#         #print("all_image>>", all_image)
#         for image in random.sample(all_image, int(0.8*len(all_image))):
#             #shutil의 인자값 = shutil.move(기존경로, 옮길 경로)
#             shutil.move(f"./game/{category}/{image}",f"./dataset/train/{category}/")
#     for category in sorted(all_categories):
#         os.makedirs(f"./dataset/val/{category}", exist_ok=True)
#         all_image = os.listdir(f"./game/{category}/")
#         for image in random.sample(all_image, int(0.5*len(all_image))):
#             shutil.move(f"./game/{category}/{image}",f"./dataset/val/{category}/")
#     for category in sorted(all_categories):
#         os.makedirs(f"./dataset/test/{category}", exist_ok=True)
#         all_image = os.listdir(f"./game/{category}/")
#         for image in all_image:
#             shutil.move(f"./game/{category}/{image}",f"./dataset/test/{category}/")
#
# if __name__ =="__main__":
#     path = "./game"
#     image_size(path)
#     create_train_val_split_folder(path)