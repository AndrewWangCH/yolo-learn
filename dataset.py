import torchvision.transforms as transforms
import cv2
from PIL import Image
from torch.utils.data import DataLoader
import os
import numpy as np
import config


image_transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(DataLoader):
    def __init__(self, data_path, transform):
        self.image_path = data_path + 'image/'
        self.label_path = data_path + 'label/'
        self.transform = transform
        self.img_channel = 1    # 1表示三通道
        if config.Config["img_channel"] == 1:
            self.img_channel = 0  # 0表示单通道


    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        files = os.listdir(self.label_path)
        files_num = len(files)
        if index >= files_num:
            index = files_num - 1
        index_files = files[index]
        image_name = index_files.split('.')[0] + '.png'
        img = Image.open(self.image_path + image_name)
        img = np.array(img, dtype=np.float32)
        if self.img_channel == 0:
            img = img[:, :, np.newaxis]
        img = np.transpose(img / 255.0, (2, 0, 1))
        label_data = []
        with open(self.label_path + index_files, "r") as f:
            for line in f.readlines():
                str_data = line.strip('\n')     # 去掉列表中每一个元素的换行符
                str_data = str_data.split(" ")
                label_data.append(str_data)
        tensor_label = np.array(label_data, dtype=np.float32)   # 将图片信息存放到列表中

        # return tensor_img, tensor_label
        return img, tensor_label


def yolo_dataset_collate(data):
    images = []
    bboxes = []
    for img, box in data:
        images.append(img)
        bboxes.append(box)
    images = np.array(images, dtype=np.float32)
    return images, bboxes



if __name__ == '__main__':
    train_dateset = MyDataset(data_path='./data/circle/train/',
                              transform=image_transform)

    train_dateset.__getitem__(1)

    print(123)


