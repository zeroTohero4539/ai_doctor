import torch
import torch.nn as nn
from review_model_config import *
from bert_chinese_encode import get_bert_encode_for_single


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        description: RNN 模型初始化函数
        :param input_size: 输入张量的维度
        :param hidden_size: 隐藏层张量的最后一个维度
        :param output_size: 输出张量的维度
        """
        super(RNN, self).__init__()
        # 将变量传入到类内部
        self.hidden_size = hidden_size
        # 构建第一个线性层，输入张量是 input_size + hidden_size，因为真正进入全连接网络的张量是 h(t-1) 和 x(t)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # 经过 tanh 函数
        self.tanh = nn.Tanh()
        # 构建第二个线性层，目的是按照我们想要的维度输出
        self.i2o = nn.Linear(hidden_size, output_size)
        # 定义最终输出的 softmax 处理层
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs, hidden):
        """
        description: RNN 的前向传播函数
        :param inputs: 输入张量 x(t)
        :param hidden: 隐藏层张量 h(t-1)
        :return:
        """
        # 首先要将当前时间步张量和上一时间步张量进行拼接
        combined = torch.cat((inputs, hidden), 1)
        # 让输入经过第一个线性层
        hidden = self.i2h(combined)
        # 经过 tanh 层
        hidden = self.tanh(hidden)
        # 让输入经过第二个线性层
        output = self.i2o(hidden)
        # 让 output 经过 softmax 层后返回
        return self.softmax(output), hidden

    def initHidden(self):
        """
        description: 初始化隐藏层张量的函数
        :return: (1, self.hidden_size) 的张量
        """
        return torch.zeros(1, self.hidden_size)


if __name__ == '__main__':
    rnn = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    inputs = get_bert_encode_for_single('您').squeeze(0)
    hidden = rnn.initHidden()
    output, hidden = rnn(inputs, hidden)
    print(output)
    print(output.shape)