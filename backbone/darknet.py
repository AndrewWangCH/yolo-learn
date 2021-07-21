from collections import OrderedDict
import torch
import torch.nn as nn

# yolo3 ssp net

# ---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
# ---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        out += residual

        return out

class MyConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, **kwargs):
        super(MyConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out

class ConvSet(nn.Module):
    def __init__(self, in_channels, filters):
        super(ConvSet, self).__init__()
        self.conv1 = MyConv(in_channel=in_channels, out_channel=filters[0], kernel_size=1, padding=0, stride=1)
        self.conv2 = MyConv(in_channel=filters[0], out_channel=filters[1], kernel_size=3, padding=1, stride=1)
        self.conv3 = MyConv(in_channel=filters[1], out_channel=filters[2], kernel_size=1, padding=0, stride=1)
        self.conv4 = MyConv(in_channel=filters[2], out_channel=filters[3], kernel_size=3, padding=1, stride=1)
        self.conv5 = MyConv(in_channel=filters[3], out_channel=filters[4], kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out

# N = (W - F +2P) / S + 1
class DarkNet(nn.Module):  # input n channel 512 512
    def __init__(self, img_channel=1):
        super(DarkNet, self).__init__()

        self.conv1 = MyConv(in_channel=img_channel, out_channel=32, kernel_size=3, padding=1, stride=1)  # 32 512 512

        self.layer1 = self._make_layer(in_channel=32, out_channel=64, layer=1)
        self.layer2 = self._make_layer(in_channel=64, out_channel=128, layer=2)
        self.layer3 = self._make_layer(in_channel=128, out_channel=256, layer=8)
        self.layer4 = self._make_layer(in_channel=256, out_channel=512, layer=8)
        self.layer5 = self._make_layer(in_channel=512, out_channel=1024, layer=4)

        self.conv2 = MyConv(in_channel=1024, out_channel=512, kernel_size=1, stride=1, padding=0)   # 512 16 16
        self.conv3 = MyConv(in_channel=512, out_channel=1024, kernel_size=3, stride=1, padding=1)   # 1024 16 16
        self.conv4 = MyConv(in_channel=1024, out_channel=512, kernel_size=1, stride=1, padding=0)  # 512 16 16

        # SSP结构
        self.pool1 = nn.MaxPool2d(stride=1, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(stride=1, kernel_size=9, padding=4)
        self.pool3 = nn.MaxPool2d(stride=1, kernel_size=13, padding=6)
        # end

        # input 2048 16 16
        self.conv5 = MyConv(in_channel=2048, out_channel=512, kernel_size=1, padding=0, stride=1)
        self.conv6 = MyConv(in_channel=512, out_channel=1024, kernel_size=3, padding=1, stride=1)
        self.conv7 = MyConv(in_channel=1024, out_channel=512, kernel_size=1, padding=0, stride=1)

        # 第一个输出特征图
        self.output1_conv1 = MyConv(in_channel=512, out_channel=1024, kernel_size=3, stride=1, padding=1)
        self.output1_conv2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0)

        self.conv8 = MyConv(in_channel=512, out_channel=256, kernel_size=1, stride=1, padding=0)    # 256 16 16
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.convset1 = ConvSet(in_channels=768, filters=[256, 512, 256, 512, 256])

        # 第二个输出特征图
        self.output2_conv1 = MyConv(in_channel=256, out_channel=512, kernel_size=3, padding=1, stride=1)    # 512 32 32
        self.output2_conv2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, padding=0, stride=1)

        self.conv9 = MyConv(in_channel=768, out_channel=128, kernel_size=3, stride=1, padding=1)
        self.convset2 = ConvSet(in_channels=384, filters=[128, 256, 128, 256, 128])

        # 第三个输出特征图
        self.output3_conv1 = MyConv(in_channel=128, out_channel=256, kernel_size=3, stride=1, padding=1)
        self.output3_conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, padding=0, stride=1)


    # ---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    # ---------------------------------------------------------------------#
    def _make_layer(self, in_channel, out_channel, layer):
        layers = []

        layers.append(("my_conv", MyConv(in_channel=in_channel, out_channel=out_channel, kernel_size=3, padding=1, stride=2)))

        for i in range(0, layer):
            layers.append(("residual_{}".format(i), BasicBlock(in_channels=out_channel, out_channels=in_channel)))
        return nn.Sequential(OrderedDict(layers))


    def forward(self, x):
        conv1 = self.conv1(x)
        layer1 = self.layer1(conv1)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        conv2 = self.conv2(layer5)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # SSP
        ssp_pool_5 = self.pool1(conv4)
        ssp_pool_9 = self.pool2(conv4)
        ssp_pool_13 = self.pool3(conv4)
        ssp_cat = torch.cat((conv4, ssp_pool_5), 1)
        ssp_cat = torch.cat((ssp_cat, ssp_pool_9), 1)
        ssp_cat = torch.cat((ssp_cat, ssp_pool_13), 1)

        conv5 = self.conv5(ssp_cat)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)

        output1 = self.output1_conv1(conv7)
        output1 = self.output1_conv2(output1)   # 第一个输出特征图

        conv8 = self.conv8(conv7)
        up = self.up(conv8)
        cat = torch.cat((up, layer4), 1)
        convset1 = self.convset1(cat)

        output2 = self.output2_conv1(convset1)
        output2 = self.output2_conv2(output2)   # 第二个输出特征图

        conv9 = self.conv9(cat)
        up2 = self.up(conv9)
        cat2 = torch.cat((up2, layer3), 1)
        convset2 = self.convset2(cat2)

        output3 = self.output3_conv1(convset2)
        output3 = self.output3_conv2(output3)   # 第三个输出特征图
        # output1->1024, 16, 16
        # output2->512, 32, 32
        # output3->256, 64, 64
        return output1, output2, output3



# if __name__ == '__main__':
#     net = DarkNet()
#     a = torch.rand(1, 3, 416, 416)
#     a, b, c = net(a)
#     print(123)










        



