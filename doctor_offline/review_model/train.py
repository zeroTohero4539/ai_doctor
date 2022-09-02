import pandas as pd
from collections import Counter
from review_model_config import *


# 加载训练集数据
train_data = pd.read_csv(train_data_path, header=None, sep='\t')
# 将训练集数据转成列表形式
train_data = train_data.values.tolist()
print(train_data[: 10])