"""
三
字典
"""
from collections import Counter
from commons.dataset_helper import get_sentence_and_lable_en, get_sentence_and_lable_china

def build_vocab(dataset_path, vocab_path, lable_path, en_or_ch='en'):
    """
    生成字典
    :param dataset_path:
    :param vocab_path:
    :param lable_path
    :return:
    """
    if en_or_ch=='en':
        sentents, lables = get_sentence_and_lable_en(dataset_path)
    elif en_or_ch == 'ch':
        print('中文分词')
        sentents, lables = get_sentence_and_lable_china(dataset_path)

    all_data = []
    for content in sentents:
        all_data.extend(content)

    # Counter 是一个计数器，这里是统计训练集中出现频率最高的 vocab_size - 1 个字
    counter = Counter(all_data)
    # [('你', 1), ('好', 1), ('周', 1), ('杰', 1), ('伦', 1)]
    # count_pairs = counter.most_common(vocab_size - 1)
    count_pairs = counter.most_common() # 返回所有
    # *count_pairs 星号是解压列表 ，变成一个个元组形式 ('你', 1), ('好', 1), ('周', 1), ('杰', 1), ('伦', 1)
    # zip函数会将所有参数 按列组合为 [('你', '好', '周', '杰', '伦'), (1, 1, 1, 1, 1)]
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(set(words)) # 加号是拼接成一个列表
    vocab_list = []
    for w in words:
        ws = w.strip()
        # if re.match('[A-Za-z0-9_]', ws): continue
        if len(ws)>0 :
            vocab_list.append(ws)

    open(vocab_path, "w", encoding='utf-8', errors='ignore').write('\n'.join(vocab_list) + '\n')

    lable_list = list(set(lables))
    open(lable_path, "w", encoding='utf-8', errors='ignore').write('\n'.join(lable_list) + '\n')

def read_vocab(vocab_path):
    """
    读取词汇表
    :param vocab_dir:
    :return:
    """
    with open(vocab_path, "r", encoding='utf-8', errors='ignore') as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id

def read_category(lable_path):
    """
    读取分类目录
    :param lable_path
    :return:
    """
    with open(lable_path, "r", encoding='utf-8', errors='ignore') as fp:
        categories = [_.strip() for _ in fp.readlines()]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id