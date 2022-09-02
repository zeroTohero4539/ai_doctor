import os
from fileinput import FileInput
from neo4j import GraphDatabase
from config import *

driver = GraphDatabase.driver(**NEO4J_CONFIG)


def _load_data(data_path):
    """
    description: 加载已经审核通过的疾病文件，并处理字典形式返回
    :param data_path: 审核之后的文件路径
    :return: {疾病1: [症状1, 症状2...], 疾病2:: [症状1, 症状2...]}
    """
    # 读取得到 data_path 下的所有文件
    disease_csv_files = os.listdir(data_path)

    # 通过列表生成式得到所有疾病的名称
    disease_names = [disease_csv_file.split('.')[0] for disease_csv_file in disease_csv_files]
    # 初始化一个列表，用来每种疾病对应的症状
    symptom_list = []
    # 遍历所有的疾病文件，从中读取出症状，构建疾病和症状的字典
    for disease_csv in disease_csv_files:
        # 将疾病文件中的每种症状存储到 symptom_list 中
        symptom = list(map(lambda x: x.strip(), FileInput(os.path.join(data_path, disease_csv))))
        # 过滤掉长度异常的症状
        symptom = list(filter(lambda x: 0 < len(x)< 100, symptom))
        symptom_list.append(symptom)

    # 按照指定的格式返回
    return dict(zip(disease_names, symptom_list))


def write(data_path):
    """
    description: 将审核通过的实体写入到 neo4j 数据库中
    :param data_path: 审核通过的文件路径
    :return: None
    """
    # 通过 _load_data 函数加载数据
    disease_symptom_dict = _load_data(data_path)

    # 开启一个会话
    with driver.session() as session:
        # 遍历疾病和症状组成的字典
        for disease, symptom_list in disease_symptom_dict.items():
            cypher = "merge (a: Disease{name: %r}) return a" % disease
            session.run(cypher)
            for symptom in symptom_list:
                cypher = "merge (b: Symptom{name: %r}) return b" % symptom
                session.run(cypher)
                cypher = "match (a: Disease{name: %r}) match (b: Symptom {name: %r}) with a, b merge(a)-[r:dis_to_sym]-(b)" % (disease, symptom)
                session.run(cypher)
        cypher = "create index on: Disease(name)"
        session.run(cypher)
        cypher = "create index on: Symptom(name)"
        session.run(cypher)


if __name__ == '__main__':
    write(reviewed_data_path)