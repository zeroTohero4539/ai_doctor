import torch.nn as nn
from RNN_MODEL import RNN

# 训练集数据的路径
train_data_path = './review_data/train_data.csv'

# RNN 模型初始化的相关参数
input_size = 768
hidden_size = 128
output_size = 2

# 命名实体审核模型训练函数的损失函数
criterion = nn.NLLLoss()

# 命名实体审核模型训练函数的学习率
learning_rate = 0.005

# 初始化 RNN 网络
rnn = RNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# 设置实体审核模型训练的迭代次数
n_iters = 50000

# 设置实体审核模型训练的打印日志间隔
plot_every = 500

# 定义命名实体审核模型的保存路径
review_model_path = './model/review_model.pth'

# 未审核的实体存放路径
unreviewed_path = '../structured/noreview'
reviewed_path = '../structured/reviewed'