import torchvision.transforms as transforms
import cv2
from PIL import Image
from torch.utils.data import DataLoader
import os


image_transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(DataLoader):
    def __init__(self, data_path, image_size, is_train, transform):

        self.image_path = data_path + 'images/'
        self.label_path = data_path + 'labels/'
        self.img_width = image_size[0]
        self.img_height = image_size[1]
        self.is_train = is_train
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
        file_path = open(self.label_path + index_files)
        label_data = file_path.read()

        return 0


if __name__ == '__main__':
    train_dateset = MyDataset(data_path='./data/my_yolo_dataset/train/',
                              image_size=[512, 512], is_train=True, transform=image_transform)

    train_dateset.__getitem__(3)

    print(123)


