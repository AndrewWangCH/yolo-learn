import os
import cv2

# 验证训练集的bbox是否ok
def test_data(path, img_format):
    file = os.listdir(path + "label")
    for i in range(len(file)):
        label_path = file[i]
        img_name = label_path.split(".")[0]
        img_name += img_format
        img = cv2.imread(path+"image\\"+img_name)

        with open(path + "label\\" + label_path, "r") as f:
            for line in f.readlines():
                str_data = line.strip('\n')  # 去掉列表中每一个元素的换行符
                str_data = str_data.split(" ")
                center_x = float(str_data[1]) * 416
                center_y = float(str_data[2]) * 416
                w = float(str_data[3]) * 416
                h = float(str_data[4]) * 416

                tl = (int(center_x - w/2), int(center_y - h/2))
                br = (int(center_x + w/2), int(center_y + h/2))

                cv2.rectangle(img, tl, br, (255, 0, 255), 3, 8)
                cv2.imshow("w", img)
                cv2.waitKey(0)


if __name__ == "__main__":
    path = 'D:\\GitHub\\yolo-learn\\data\\number-test\\train\\'
    img_format = '.png'
    test_data(path, img_format)