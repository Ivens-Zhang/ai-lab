
# coding: utf-8

# In[ ]:



#定义参数

class Config(object):
    """
    poetry_file: 诗词数据存放位置
    weight_file: 生成模型的存放位置
    max_len：根据前六个字预测第七个字
    batch_size：批量处理个数
    learning_rate：学习率
    """

    poetry_file = '../../../data/day10-nlp-data/poetry.txt'
    weight_file = 'model/poetry_model.h5'
    max_len = 6
    batch_size = 600
    learning_rate = 0.001

