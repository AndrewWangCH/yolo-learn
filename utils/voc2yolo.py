from lxml import etree
import os


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

# 将voc数据集（xml） 转换成yolo所需要的txt格式
if __name__ == '__main__':
    xml_dir = 'C:\\Users\\wzl\\Desktop\\yolo_img\\xml\\'
    xml_file = os.listdir(xml_dir)
    save_txt_path = "C:\\Users\\wzl\\Desktop\\yolo_img\\txt\\"
    class_dict = {'0': 0}  # 数据集类别

    for i in range(len(xml_file)):
        xml_path = xml_dir + xml_file[i]
        # read xml
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

        name = xml_file[i][:-4]
        # write object info into txt
        with open(os.path.join(save_txt_path, name + ".txt"), "w") as f:
            assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
            for index, obj in enumerate(data["object"]):
                # 获取每个object的box信息
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                class_index = class_dict[class_name]  # 目标id从0开始

                # 将box信息转换到yolo格式
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 绝对坐标转相对坐标，保存6位小数
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

    print('ok')

