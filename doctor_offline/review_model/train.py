import random
import time
import matplotlib.pyplot as plt
import torch
import pandas as pd
from collections import Counter
from review_model_config import *
from bert_chinese_encode import get_bert_encode_for_single
import warnings
warnings.filterwarnings('ignore')


def read_data(data_path):
    """
    description: 读取 data_path 文件
    :param data_path: 训练集样本文件路径
    :return: 列表化之后的训练集
    """
    # 加载训练集数据
    train_data = pd.read_csv(data_path, header=None, sep='\t')
    # 将训练集数据转成列表形式
    train_data = train_data.values.tolist()
    # 将得到的结果返回
    return train_data


def randomTrainingExample(train_data):
    """
    description: 随机选取数据函数
    :param train_data_path: 训练集数据路径
    :return: category, text, category_tensor, text_tensor
    """
    # 从训练集数据中随机产生一条样本数据
    category, text = random.choice(train_data)
    # 将随机产生的文本 text 和 category 张量化
    text_tensor = get_bert_encode_for_single(text)
    category_tensor = torch.tensor([int(category)])
    # 返回随机产生的结果
    return category, text, category_tensor, text_tensor


def train(category_tensor, text_tensor):
    """
    description: 模型训练函数
    :param category_tensor: 样本类型张量
    :param text_tensor: 文本张量
    :return:
    """
    # 初始化隐藏层张量
    hidden = rnn.initHidden()
    # 梯度清零
    rnn.zero_grad()
    # 遍历 text_tensor 中的每一个字的张量表示
    for i in range(text_tensor.size()[1]):
        # 将每个字的张量输入到 rnn 网络中 (1, 768)
        output, hidden = rnn(text_tensor[0][i].unsqueeze(0), hidden)
    # 根据损失函数计算损失值
    loss = criterion(output, category_tensor)
    # 反向传播
    loss.backward()
    # 更新参数
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    # 返回结果和损失值
    return output, loss.item()


def valid(category_tensor, text_tensor):
    """
    description: 验证函数
    :param category_tensor: 验证标签张量
    :param text_tensor: 验证文本张量
    :return: output, loss
    """
    # 初始化隐藏层张量
    hidden = rnn.initHidden()
    # 验证模式不需要自动更新梯度
    with torch.no_grad():
        # 遍历每一个字的张量
        for i in range(text_tensor.size()[1]):
            # 将字张量输入到 RNN 网络中，得到预测结果
            output, hidden = rnn(text_tensor[0][i].unsqueeze(0), hidden)
        # 获得损失值
        loss = criterion(output, category_tensor)
    # 返回预测结果和损失
    return output, loss.item()


def time_since(start):
    """
    description: 辅助函数，用来计算训练用时
    :param since: 开始的时间
    :return: 指定字符串返回
    """
    # 计算当前时间
    now = time.time()
    # 计算时间戳
    s = now - start
    # 计算耗时的分钟数
    m = int(s / 60)
    # 计算没有除尽的秒数
    s -= 60 * m
    # 以指定的字符串返回
    return '%dm-%ds' % (m, s)


def main():
    """
    description: 主要训练逻辑函数
    :return:
    """
    # 调用 read_data() 读取样本数据
    data = read_data(train_data_path)
    # 训练数据
    train_data = data[: int(len(data) * 0.8)]
    # 验证数据
    valid_data = data[int(len(data) * 0.8): ]
    # 初始化打印间隔中训练和验证的损失和准确率
    train_current_loss = 0
    train_current_acc = 0
    valid_current_loss = 0
    valid_current_acc = 0

    # 定义存储打印间隔的平均损失和准确率
    all_train_losses = []
    all_train_acc = []
    all_valid_losses = []
    all_valid_acc = []

    # 获取训练开始前的时间
    start = time.time()

    # 循环遍历 n_iters 次
    for iter in range(1, n_iters + 1):
        # 调用 randomTrainingExample() 函数从训练集和验证集中随机产生数据样本
        category, text, category_tensor, text_tensor = randomTrainingExample(train_data)
        category_val, text_val, category_tensor_val, text_tensor_val = randomTrainingExample(valid_data)
        # 分别调用 train() 和 valid()，得到训练函数和验证函数的预测输出和损失
        train_output, train_loss = train(category_tensor, text_tensor)
        valid_output, valid_loss = valid(category_tensor_val, text_tensor_val)
        # 计算出打印间隔中的总损失
        train_current_loss += train_loss
        train_current_acc += (train_output.argmax(1) == category_tensor).sum().item()
        valid_current_loss += valid_loss
        valid_current_acc += (valid_output.argmax(1) == category_tensor_val).sum().item()

        # 当遇到打印日志间隔的时候
        if iter % plot_every == 0:
            # 计算平均损失并存储到各自的列表中，用于画图
            train_average_loss = train_current_loss / plot_every
            train_average_acc = train_current_acc / plot_every
            valid_average_loss = valid_current_loss / plot_every
            valid_average_acc = valid_current_acc / plot_every
            # 打印日志
            print("Iter:", iter, "|", "TimeSince:", time_since(start))
            print("Train Loss:", train_average_loss, "|", "Train Acc:", train_average_acc)
            print("Valid Loss:", valid_average_loss, "|", "Valid Acc:", valid_average_acc)
            # 存储平均损失和准确率
            all_train_losses.append(train_average_loss)
            all_train_acc.append(train_average_acc)
            all_valid_losses.append(valid_average_loss)
            all_valid_acc.append(valid_average_acc)
            # 将训练间隔的损失和准确率全部归零
            train_current_loss = 0
            train_current_acc = 0
            valid_current_loss = 0
            valid_current_acc = 0

    # 绘制损失和准确率对照曲线
    draw_loss_acc_curve(all_train_losses, all_train_acc, all_valid_losses, all_valid_acc)

    # 保存模型
    save_model(review_model_path)


def draw_loss_acc_curve(all_train_losses, all_train_acc, all_valid_losses, all_valid_acc):
    """
    description: 绘制损失和准确率对照曲线
    :param all_train_losses: 训练集损失
    :param all_train_acc: 训练集准确率
    :param all_valid_losses: 验证集损失
    :param all_valid_acc: 验证集准确率
    :return:
    """
    plt.figure(0)
    plt.plot(all_train_losses, label='Train Loss')
    plt.plot(all_valid_losses, color='red', label='Valid Loss')
    plt.legend()
    plt.grid()
    plt.savefig('./model/loss.png')

    plt.figure(1)
    plt.plot(all_train_acc, label='Train Acc')
    plt.plot(all_valid_acc, color='red', label='Valid Acc')
    plt.legend()
    plt.grid()
    plt.savefig('./model/acc.png')


def save_model(save_model_path):
    """
    description: 保存模型
    :param save_model_path: 模型保存的路径
    :return:
    """
    torch.save(rnn.state_dict(), save_model_path)


if __name__ == '__main__':
    main()
