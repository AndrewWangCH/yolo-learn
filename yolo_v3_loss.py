import math
import torch
import torch.nn as nn

# 均方误差，是预测值与真实值之差的平方和的平均值
def MSELoss(x, y):
    return (x - y) ** 2


def clip_by_tensor(t, t_min, t_max):
    t = t.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


# 二分类交叉熵（Binary Cross Entropy)
def BCELoss(pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


def CalculIOU(_box_a, _box_b):
    # 计算真实框的左上角和右下角
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    # 计算先验框的左上角和右下角
    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    box_a = torch.zeros_like(_box_a)
    box_b = torch.zeros_like(_box_b)
    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2  # 真实框
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2  # 先验框
    A = box_a.size(0)
    B = box_b.size(0)

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)

    inter = inter[:, :, 0] * inter[:, :, 1]  # 两个框相交的面积
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    # 求IOU
    union = area_a + area_b - inter
    return inter / union


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_class, img_size, normalize):
        """
        :param anchors: 代表config.py中的anchor即先验框
        :param num_class: 代表有几类
        :param img_size: 代表原图的大小
        """

        super(YOLOLoss, self).__init__()
        # -----------------------------------------------------------#
        #   小特征层对应大anchor
        #   中特征层对应中anchor
        #   大特征层对应小anchor
        # -----------------------------------------------------------#
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_class
        self.bbox_attrs = 5 + num_class
        # -------------------------------------#
        #   获得特征层的宽高
        #   13、26、52
        # -------------------------------------#
        self.feature_length = [img_size[0] // 32, img_size[0] // 16, img_size[0] // 8]
        self.img_size = img_size
        self.normalize = normalize

        self.ignore_threshold = 0.5  # 挑选负样本的阈值


    def forward(self, input, targets=None):
        """
        :param input: 通过yolo_v3 backbone的特征层
        :param target: 训练时——>label信息, 预测时不需要
        :return: 损失值
        """
        # ----------------------------------------------------#
        #   input的shape为  bs, 3*(5+num_classes), 13, 13
        #                   bs, 3*(5+num_classes), 26, 26
        #                   bs, 3*(5+num_classes), 52, 52
        # ----------------------------------------------------#

        # -----------------------#
        #   一共多少张图片
        # -----------------------#
        bs = input.size(0)
        # -----------------------#
        #   特征层的高
        # -----------------------#
        in_h = input.size(2)
        # -----------------------#
        #   特征层的宽
        # -----------------------#
        in_w = input.size(3)
        # -----------------------------------------------------------------------#
        #   计算步长即感受野
        #   每一个特征点对应原来的图片上多少个像素点
        #   如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        #   如果特征层为26x26的话，一个特征点就对应原来的图片上的16个像素点
        #   如果特征层为52x52的话，一个特征点就对应原来的图片上的8个像素点
        #   stride_h = stride_w = 32、16、8
        # -----------------------------------------------------------------------#
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        # -------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        # -------------------------------------------------#
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        # -----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 13, 13, 5 + num_classes
        #   batch_size, 3, 26, 26, 5 + num_classes
        #   batch_size, 3, 52, 52, 5 + num_classes
        #   prediction 是预测结果进行展开
        # -----------------------------------------------#
        prediction = input.view(bs, int(self.num_anchors), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,
                                                                                                    2).contiguous()

        # -----------待拟合的数据------------------------#
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])
        # ---------------------------------------------#

        # ---------------------------------------------------------------#
        #   找到哪些先验框内部包含物体
        #   利用真实框和先验框计算交并比
        #   mask        batch_size, 3, in_h, in_w   有目标的特征点
        #   noobj_mask  batch_size, 3, in_h, in_w   无目标的特征点
        #   tx          batch_size, 3, in_h, in_w   中心x偏移情况
        #   ty          batch_size, 3, in_h, in_w   中心y偏移情况
        #   tw          batch_size, 3, in_h, in_w   宽高调整参数的真实值
        #   th          batch_size, 3, in_h, in_w   宽高调整参数的真实值
        #   tconf       batch_size, 3, in_h, in_w   置信度真实值
        #   tcls        batch_size, 3, in_h, in_w, num_classes  种类真实值
        # ----------------------------------------------------------------#
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(targets,
                                                                scaled_anchors, in_w, in_h, self.ignore_threshold)

        box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y

        # 计算中心偏移情况的loss，使用BCELoss效果好一些
        loss_x = torch.sum(BCELoss(x, tx) * box_loss_scale * mask)
        loss_y = torch.sum(BCELoss(y, ty) * box_loss_scale * mask)
        # 计算宽高调整值的loss
        loss_w = torch.sum(MSELoss(w, tw) * 0.5 * box_loss_scale * mask)
        loss_h = torch.sum(MSELoss(h, th) * 0.5 * box_loss_scale * mask)
        # 计算置信度的loss
        loss_conf = torch.sum(BCELoss(conf, mask) * mask) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask)

        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]))

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        if self.normalize:
            num_pos = torch.sum(mask)
            num_pos = torch.max(num_pos, torch.ones_like(num_pos))  # 保证num_pos大于0
        else:
            num_pos = bs/3

        print('loss: %.3f  loss_x: %.3f  loss_y: %.3f  loss_w: %.3f  loss_h: %.3f  loss_conf: %.3f  loss_cls: %.3f' % \
              (loss.item(), loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item()))

        return loss, num_pos


    def get_target(self, target, anchors, in_w, in_h, ignore_threshold):
        """
        :param target: label中的信息（即y）
        :param anchors: 相对于特征层的真实框
        :param in_w: 特征层宽
        :param in_h: 特征层高
        :return: 解码后的label
        """
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(target)
        # -------------------------------------------------------#
        #   获得当前特征层先验框所属的编号，方便后面对先验框筛选
        # -------------------------------------------------------#
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        subtract_index = [0, 3, 6][self.feature_length.index(in_w)]
        # -------------------------------------------------------#
        #   创建全是0或者全是1的阵列
        # -------------------------------------------------------#
        mask = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False).cuda()
        noobj_mask = torch.ones(bs, int(self.num_anchors), in_h, in_w, requires_grad=False).cuda()
        tx = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False).cuda()
        ty = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False).cuda()
        tw = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False).cuda()
        th = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False).cuda()
        tconf = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False).cuda()
        tcls = torch.zeros(bs, int(self.num_anchors), in_h, in_w, self.num_classes, requires_grad=False).cuda()

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False).cuda()
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors), in_h, in_w, requires_grad=False).cuda()

        for b in range(bs):  # 遍历一个batch_size
            if len(target[b]) == 0:  # 图中没有目标
                continue  # 创建的阵列与此情况符合

            # -------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            # -------------------------------------------------------#
            gxs = target[b][:, 1:2] * in_w
            gys = target[b][:, 2:3] * in_h

            # -------------------------------------------------------#
            #   计算出正样本相对于特征层的宽高
            # -------------------------------------------------------#
            gws = target[b][:, 3:4] * in_w
            ghs = target[b][:, 4:5] * in_h

            # -------------------------------------------------------#
            #   计算出正样本属于特征层的哪个特征点(torch.floor-->向下取整)
            # -------------------------------------------------------#
            gis = torch.floor(gxs)
            gjs = torch.floor(gys)

            # -------------------------------------------------------#
            #   将真实框转换一个形式
            #   num_true_box, 4
            # -------------------------------------------------------#
            gt_box = torch.cuda.FloatTensor(torch.cat([torch.zeros_like(gws), torch.zeros_like(ghs), gws, ghs], 1))

            # -------------------------------------------------------#
            #   将先验框转换一个形式
            # -------------------------------------------------------#
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((self.num_anchors, 2)), torch.FloatTensor(anchors)), 1))
            anchor_shapes = anchor_shapes.cuda()

            # -------------------------------------------------------#
            #   计算交并比
            # -------------------------------------------------------#
            anch_ious = CalculIOU(gt_box, anchor_shapes)  # iou = 两者交集 / （两个面积相加 - 两者的交集）

            # -------------------------------------------------------#
            #   计算重合度最大的先验框是哪个
            # -------------------------------------------------------#
            best_ns = torch.argmax(anch_ious, dim=-1)
            for i, best_n in enumerate(best_ns):
                if best_n not in anchor_index:
                    continue
                if anch_ious[i][int(best_n.item())] < ignore_threshold:     # 如果最大的IOU小于阈值则认为没有物体
                    continue
                # -------------------------------------------------------------#
                #   取出各类坐标：
                #   gi和gj代表的是真实框对应的特征点的x轴y轴坐标
                #   gx和gy代表真实框的x轴和y轴坐标
                #   gw和gh代表真实框的宽和高
                # -------------------------------------------------------------#
                gi = gis[i].long()
                gj = gjs[i].long()
                gx = gxs[i]
                gy = gys[i]
                gw = gws[i]
                gh = ghs[i]

                if (gj < in_h) and (gi < in_w):
                    best_n = best_n - subtract_index
                    # ----------------------------------------#
                    #   noobj_mask代表无目标的特征点
                    # ----------------------------------------#
                    noobj_mask[b, best_n, gj, gi] = 0
                    # ----------------------------------------#
                    #   mask代表有目标的特征点
                    # ----------------------------------------#
                    mask[b, best_n, gj, gi] = 1
                    # ----------------------------------------#
                    #   tx、ty代表中心调整参数的真实值(偏移量)
                    # ----------------------------------------#
                    tx[b, best_n, gj, gi] = gx - gi.float()
                    ty[b, best_n, gj, gi] = gy - gj.float()
                    # ----------------------------------------#
                    #   tw、th代表宽高调整参数的真实值(计算方法与Faster RCNN 一样)
                    # ----------------------------------------#
                    tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n + subtract_index][0])
                    th[b, best_n, gj, gi] = math.log(gh / anchors[best_n + subtract_index][1])
                    # ----------------------------------------#
                    #   用于获得xywh的比例
                    #   大目标loss权重小，小目标loss权重大
                    # ----------------------------------------#
                    box_loss_scale_x[b, best_n, gj, gi] = target[b][i, 2]  # 存放真实框的宽，高
                    box_loss_scale_y[b, best_n, gj, gi] = target[b][i, 3]
                    # ----------------------------------------#
                    #   tconf代表物体置信度
                    # ----------------------------------------#
                    tconf[b, best_n, gj, gi] = 1
                    # ----------------------------------------#
                    #   tcls代表种类置信度
                    # ----------------------------------------#
                    tcls[b, best_n, gj, gi, int(target[b][i, 0])] = 1
                else:
                    print("图像尺寸错误！！！")
                    continue
        return mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y




