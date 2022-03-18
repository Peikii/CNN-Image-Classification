"""
数据集处理工具
"""
import os
import sys
import csv
import random
import jieba
import numpy as np
import pandas as pd

# 解除csv长度限制
csv.field_size_limit(500 * 1024 * 1024)
curr_path, _ = os.path.split(os.path.abspath(__file__).replace("\\", "/"))

def native_content(content):
    """
    处理字符串编码
    :param content:
    :return:
    """
    if sys.version_info[0] > 2:
        is_py3 = True
    else:
        sys.setdefaultencoding("utf-8")
        is_py3 = False
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def get_sentence_and_lable_en(file_path):
    """
    读取句子和标签
    列名必须要是 id, label, txt
    :param file_path:
    :return: 句子分词结果, 标签
    """
    contents, labels = [], []
    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.DictReader(f)
        for line in csv_reader:
            try:
                id, lable, txt = line["id"], line["lable"], line["txt"]
                if txt:
                    # 英文分词。按空格分。
                    contents.append(native_content(txt).split(" "))
                    labels.append(native_content(lable))
            except Exception as e:
                raise e
    return contents, labels

def get_sentence_and_lable_china(file_path):
    """
    读取句子和标签.中文分词
    列名必须要是 id, label, txt

    对数字，符号做了处理；jieba需要添加对"<" ">"分词的支持。
    :param file_path:
    :return: 句子分词结果, 标签
    """
    jieba.load_userdict("/".join([curr_path, "user_dict.txt"]))
    contents, labels = [], []
    with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
        csv_reader = csv.DictReader(f)
        for line in csv_reader:
            try:
                id, lable, txt = line["id"], line["lable"], line["txt"]
                if txt:
                    contents.append(jieba.lcut(txt, cut_all=False))
                    labels.append(native_content(lable))
            except Exception as e:
                raise e
    return contents, labels


def get_seq_length(data_set_path, en_or_ch='en'):
    """
    统计句子长度，获取一个固定长度，用于padding
    :param data_set_path:
    :return:
    """
    if en_or_ch=='en':
        contents, labels = get_sentence_and_lable_en(data_set_path)
    elif en_or_ch == 'ch':
        print('中文分词')
        contents, labels = get_sentence_and_lable_china(data_set_path)

    seq_len_list = [len(stc) for stc in contents]
    df = pd.DataFrame(seq_len_list, columns=["seq_len"])
    sentence_info = {}
    sentence_info["max"] = df["seq_len"].max()
    sentence_info["min"] = df["seq_len"].min()
    sentence_info["median"] = df["seq_len"].median()
    sentence_info["mean"] = df["seq_len"].mean()
    sentence_info["5/10"] = df["seq_len"].quantile(0.5)
    sentence_info["6/10"] = df["seq_len"].quantile(0.6)
    sentence_info["7/10"] = df["seq_len"].quantile(0.7)
    sentence_info["8/10"] = df["seq_len"].quantile(0.8)
    sentence_info["9/10"] = df["seq_len"].quantile(0.9)

    return sentence_info


def split_dataset(data_set_path, train_set_path, test_set_path, ratio=0.7):
    """
    拆分数据集。（测试集和验证集公用吧。）
    csv头必须是'id', 'lable', 'txt'
    :param data_set_path: 数据集
    :param train_set_path: 训练集
    :param test_set_path: 测试集
    :return:
    """
    # pd.set_option('display.max_columns', None)  # 显示所有列
    # pd.set_option('display.max_rows', None)  # 显示所有行

    def split_class(d, ratio):
        data = np.array(d)
        random.shuffle(data)  # 随机打乱
        # 取前70%为训练集
        allurl_fea = [d for d in data]
        df1 = data[:int(ratio * len(allurl_fea))]
        df1 = pd.DataFrame(df1, columns=['id', 'lable', 'txt'])
        df2 = data[int(ratio * len(allurl_fea)):]
        df2 = pd.DataFrame(df2, columns=['id', 'lable', 'txt'])

        return df1, df2

    df = pd.read_csv(data_set_path, dtype=object)
    train_list = []
    test_list = []
    for lable in list(set(df["lable"])):
        one_class = df[df.lable==lable]
        one_train, one_test = split_class(one_class, ratio)
        train_list.append(one_train)
        test_list.append(one_test)

    trainset = pd.concat(train_list)
    testset = pd.concat(test_list)

    trainset.to_csv(train_set_path, index=False)
    testset.to_csv(test_set_path, index=False)
    # df = pd.read_csv(train_set_path)
    # logger.info("训练集数据量统计:\n{}".format(df.groupby(by='lable')['lable'].agg(['count'])))
    # df = pd.read_csv(test_set_path)
    # logger.info("测试集数据量统计:\n{}".format(df.groupby(by='lable')['lable'].agg(['count'])))

def check_repeat(data_set_path):
    """
    数据集去重
    :param data_set_path:数据集路径
    :return:
    """
    file_dir, file_name = os.path.split(data_set_path)
    file_sname, file_suffix = os.path.splitext(file_name)
    if file_dir.strip() == "":
        data_set_source_path = file_sname + '_repeat.csv'
    else:
        data_set_source_path = "/".join([file_dir, file_sname + '_repeat.csv'])

    df = pd.read_csv(data_set_path, dtype=object)
    df0 = df.drop_duplicates(subset=["lable", "txt"], keep='first', inplace=False)
    df0.to_csv(data_set_source_path, index=0)
    return data_set_source_path


def balance_class(data_set_path, sample_num):
    """
    类别均衡
    用上采样
    :param data_set_path:
    :param sample_num: 样本数
    :return:
    """
    # 处理文件名
    file_dir, file_name = os.path.split(data_set_path)
    file_sname, file_suffix = os.path.splitext(file_name)
    if file_dir.strip() == "":
        data_set_source_path = file_sname + '_over_samlping.csv'
    else:
        data_set_source_path = "/".join([file_dir, file_sname + '_over_samlping.csv'])
    # 均衡样本数
    df = pd.read_csv(data_set_path, dtype=object)
    group_by_company = df.groupby(by='lable')['lable'].agg(['count'])
    print("分类样本数:\n {}".format(group_by_company))
    if sample_num is None:
        max_sample_num = max(list(group_by_company["count"]))
    else:
        max_sample_num = sample_num
    print("最大样本数: {}".format(str(max_sample_num)))
    for lable in list(group_by_company.index):
        sample_count = group_by_company.loc[lable]['count']
        need_sample = max_sample_num - sample_count
        if need_sample == 0:
            continue
        # logger.info("'{}' need sample {}".format(lable, str(need_sample)))
        add_sample = df[df.lable==lable].sample(n=need_sample, random_state=1, replace=True, axis=0)
        df = pd.concat([df, add_sample])
    print("*"*30)
    print(df.groupby(by='lable')['lable'].agg(['count']))
    df.to_csv(data_set_source_path, index=False)
    return data_set_source_path


# if __name__ == '__main__':
    # get_seq_length("../datas/pdf_company/source_data_format_repeat_over_samlping.csv")
#     # < 尖括号要加入自定义字典，修改jieba源码 init文件
#     jieba.add_word("<SYMBOL>", freq=200000)
#     r = jieba.lcut("SpecialInstructions<SYMBOL><NUM><SYMBOL><NUM>年<NUM>月<NUM>日发货", cut_all=False)
#     print(type(r))
#     print(r)