import torch
import config
import yolo_v3
import yolo_v3_loss
import dataset
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    yolo_config = config.Config
    batch_size = 8
    # ------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    # ------------------------------------------------------#
    normalize = True

    train_dataset = dataset.MyDataset(data_path="C:\\Users\\wzl\\Desktop\\Flower\\data\\ElectricMester\\",
                                      transform=dataset.image_transform)

    train_loader = dataset.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    net = yolo_v3.YoloBody(yolo_config).to(device)
    net = net.train()

    # 建立yolo函数
    yolo_loss = []
    for i in range(3):
        yolo_loss.append(yolo_v3_loss.YOLOLoss(np.reshape(yolo_config["yolo"]["anchors"], [-1, 2]),
                                               yolo_config["yolo"]["classes"], (yolo_config["img_w"], yolo_config["img_h"]), normalize))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    lr = 1e-3
    optimizer = torch.optim.Adam(net.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

