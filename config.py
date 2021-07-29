Config = \
{
    "yolo": {
        "anchors": [[[60, 105]],
                    ],
        "classes": 2,
    },
    "anchors_group": 1,      # 有几组anchor  需要配合yolo层协调
    "anchors_group_num": 1,  # 每组有几个anchor， 每个组需要相同的anchor
    "img_h": 416,
    "img_w": 416,
    "img_channel": 1,
}


# 416*416 比较通用的anchor
# "anchors": [[[116, 90], [156, 198], [373, 326]],
#             [[30, 61], [62, 45], [59, 119]],
#             [[10, 13], [16, 30], [33, 23]]],
