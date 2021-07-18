import torchvision.transforms as transforms
import cv2
from PIL import Image
from torch.utils.data import DataLoader
import os
import numpy as np


image_transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(DataLoader):
    def __init__(self, data_path, transform):

        self.image_path = data_path + 'images/'
        self.label_path = data_path + 'labels/'
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        files = os.listdir(self.label_path)
        index_files = files[index]
        image_name = index_files.split('.')[0] + '.jpg'
        img = cv2.imread(self.image_path + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        tensor_img = self.transform(pil_img)
        label_data = []
        with open(self.label_path + index_files, "r") as f:
            for line in f.readlines():
                str_data = line.strip('\n')     # 去掉列表中每一个元素的换行符
                str_data = str_data.split(" ")
                label_data.append(str_data)
        tensor_label = np.array(label_data, dtype=np.float32)   # 将图片信息存放到列表中

        return tensor_img, tensor_label


# if __name__ == '__main__':
#     train_dateset = MyDataset(data_path='./data/train/',
#                               transform=image_transform)
#
#     train_dateset.__getitem__(1)
#
#     print(123)


