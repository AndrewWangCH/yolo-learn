#  参考网址：https://blog.csdn.net/u012505617/article/details/110486468?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161880004016780262562062%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=161880004016780262562062&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_v2~rank_v29-1-110486468.first_rank_v2_pc_rank_v29&utm_term=pytorch++%E8%BD%AC%E6%8D%A2%E6%88%90++tensorRT

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pth2onnx():
    model = torch.load('../model/yolov3_cpu.pth')#.to(device)  # 这里保存的是完整模型
    # model = torch.jit.load('./model/model_final.pth').to(device)
    # model.train()
    model.eval()
    example = torch.rand(1, 1, 416, 416)#.to(device)  # 生成一个随机输入维度的输入

    # input_names = ['data1', 'data2']
    # output_names = ['seg', 'cls']

    input_names = ['input']
    output_names = ['output']   # ['out0', 'out1', 'out2']

    torch.onnx.export(model, example, '../model/yolov3_cpu.onnx',
                      export_params=True,
                      verbose=True,
                      input_names=input_names,
                      output_names=output_names
                      )


# gpu-->cpu
def model2model():
    model = torch.load('../model/yolov3.pth', map_location=lambda storage, loc: storage)
    torch.save(model, '../model/yolov3_cpu.pth') #转cpu

if __name__ == '__main__':
    # model2model()
    pth2onnx()

