# 定义 neo4j 数据库的连接
NEO4J_CONFIG = {
    'uri': 'bolt://139.9.130.242:7687',
    'auth': ('neo4j', 'Lzcu159260'),
    'encrypted': False
}


# 完成命名实体审核之后的文件路径
reviewed_data_path = './structured/reviewed'