import torch
import config
import yolo_v3
import yolo_v3_loss
import dataset
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_ont_epoch(net, yolo_losses, train_loader, optimizer, yolo_config):
    net.train()
    total_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        images, target = data[0], data[1]
        images = images.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        outputs = net(images)  # 3个特征图
        losses = []
        num_pos_all = 0
        for i in range(yolo_config['anchors_group']):
            loss_item, num_pos = yolo_losses(outputs, target)
            losses.append(loss_item)
            num_pos_all += num_pos

        loss = sum(losses) / num_pos_all
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss


if __name__ == "__main__":
    yolo_config = config.Config
    batch_size = 4
    # ------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    # ------------------------------------------------------#
    normalize = True

    train_dataset = dataset.MyDataset(data_path="./data/number-test/train/",
                                      transform=dataset.image_transform)

    train_loader = dataset.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # net = torch.load('./model/yolov3.pth').to(device)
    net = yolo_v3.YoloBody(yolo_config).to(device)

    # 建立yolo函数
    yolo_loss = yolo_v3_loss.YOLOLoss(np.reshape(yolo_config["yolo"]["anchors"], [-1, 2]),
                                           yolo_config["yolo"]["classes"], (yolo_config["img_w"], yolo_config["img_h"]),
                                           yolo_config['anchors_group'], yolo_config['anchors_group'], normalize)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    for epoch in range(500):
        total_loss = fit_ont_epoch(net, yolo_loss, train_loader, optimizer, yolo_config)
        print('[%d] loss: %.3f' % (epoch, total_loss))
        torch.save(net, './model/yolov3.pth')



