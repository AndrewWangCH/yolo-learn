# 将backbone输出的特征层进行处理得到 “yolov3层”

import torch.nn as nn
from collections import OrderedDict
from backbone import darknet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    ])
    return m


class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        # ---------------------------------------------------#
        #   输入图片是416*416
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256,52,51
        #   512,26,26
        #   1024,13,13
        # ---------------------------------------------------#
        img_channel = config["img_channel"]
        num_class = config["yolo"]["classes"]
        num_anchor = len(config["yolo"]["anchors"][0])   # 默认每组的anchor个数一样

        self.backbone = darknet.DarkNet(img_channel=img_channel).to(device)

        final_out_filter = (num_class + 1 + 4) * num_anchor     # num_anchor-->代表一组有几个anchors
        self.last_layer0 = make_last_layers([512, 1024], 1024, final_out_filter)
        self.last_layer1 = make_last_layers([256, 512], 512, final_out_filter)    # 此时只有一个先验框，且为大目标
        self.last_layer2 = make_last_layers([128, 256], 256, final_out_filter)

    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
            return layer_in

        #with torch.no_grad():
        out0, out1, out2 = self.backbone(x)
        out0 = _branch(self.last_layer0, out0)
        # out1 = _branch(self.last_layer1, out1)
        # out2 = _branch(self.last_layer2, out2)
        # out0->final_out_filter, 16, 16, 75
        # out1->final_out_filter, 32, 32, 75
        # out2->final_out_filter, 64, 64, 75
        # return out0, out1, out2
        return out0


# if __name__ == '__main__':
#     net = YoloBody(num_class=20)
#     a = torch.rand(1, 3, 416, 416)
#     _a, _b, _c = net(a)
#
#     print(123)






