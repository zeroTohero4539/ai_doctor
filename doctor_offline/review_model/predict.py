import os
import torch
from review_model_config import *
from bert_chinese_encode import get_bert_encode_for_single


# 加载模型
rnn.load_state_dict(torch.load(review_model_path))

def _test(text_tensor):
    """
    description: 模型测试函数
    :param text_tensor: 测试文本张量
    :return: output，模型预测结果
    """
    # 初始化隐藏层张量
    hidden = rnn.initHidden()
    # 遍历每一个字的张量
    for i in range(text_tensor.size()[1]):
        # 将每个字输入到 RNN 网络中得到预测结果
        output, hidden = rnn(text_tensor[0][i].unsqueeze(0), hidden)

    # 返回 RNN 得到的预测结果(1, 2)
    return output


def predict(input_text):
    """
    description: 模型预测函数
    :param input_text: 需要预测的文本
    :return:
    """
    # 预测的时候不需要梯度更新
    with torch.no_grad():
        # 将需要预测的文本输入到 bert 模型中得到文本张量
        text_tensor = get_bert_encode_for_single(input_text)
        # 调用 _test() 函数得到预测结果
        output = _test(text_tensor)
        # 从预测结果中得到最大值对应的索引（0 或者 1）
        _, topi = output.topk(1, 1)
        # 返回结果数值
        return topi.item()


def batch_predict(input_path, output_path):
    """
    description: 批量审核命名实体
    :param input_path: 需要审核的实体所在路径
    :param output_path: 审核完成的实体存储路径
    """
    # 读取 input_path 路径下的所有 csv 文件
    csv_list = os.listdir(input_path)
    # 遍历每一个 csv 文件
    for csv in csv_list:
        # 以只读的方式打开需要审核的实体文件
        with open(os.path.join(input_path, csv), 'r', encoding='utf-8') as fr:
            # 以写的方式打开存储路径的同名实体文件
            with open(os.path.join(output_path, csv), 'w', encoding='utf-8') as fw:
                for input_text in fr.readlines():
                    print(csv, input_text)
                    # 使用模型进行预测
                    res = predict(input_text)
                    # 如果预测结果为 1，说明审核成功，把文本写入到文件中
                    if res == 1:
                        fw.write(input_text)
                    else:
                        pass


if __name__ == '__main__':
    input_line = '点瘀样尖针性发多'
    result = predict(input_line)
    print('[点瘀样尖针性发多]的预测结果是：', result)

    batch_predict(unreviewed_path, reviewed_path)