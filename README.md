# yolo-learn
use pytorch learning yolo


backbone: 此文件夹存放主干网络
train.py: 训练模型
predict.py: 预测模型
config.py: 存放训练配置，如anchor, image_size
dataset.py: 读取数据
yolo_v3: 主干网络之后的yolov3特有的卷积操作
yolo_v3_loss: 将label信息进行解码，并计算损失
