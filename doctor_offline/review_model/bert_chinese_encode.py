import torch
from transformers import AutoModel, AutoTokenizer

# 定义预训练模型的名称
model_name = 'bert-base-chinese'
# 定义字映射器
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 初始化预训练模型
model = AutoModel.from_pretrained(model_name)

def get_bert_encode_for_single(text):
    """
    description: 通过预训练模型得到文本数字化的张量
    :param text: 文本信息
    :return:
    """
    # 将文本信息进行编码
    token_index = tokenizer.encode(text)[1: -1]
    # 将数字化的文本信息转成张量
    token_tensor = torch.tensor([token_index])

    # 使模型不自动计算梯度
    with torch.no_grad():
        # 调用模型，获得输出结果
        # output_layer, _ = model(token_tensor, return_dict=False)
        outputs = model(token_tensor)

    # return output_layer
    return outputs[0]


if __name__ == '__main__':
    output = get_bert_encode_for_single('您好，中国！')
    print(output)
    print(output.shape)