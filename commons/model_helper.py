"""
训练
"""
import os
# import re
import time
import jieba
import numpy as np
import tensorflow as tf
from sklearn import metrics
from datetime import timedelta
from commons.dataset_helper import get_sentence_and_lable_en, get_seq_length, get_sentence_and_lable_china
from commons.vocab_helper import build_vocab, read_vocab, read_category
from commons.model_restore import Model_Init

from models.crnn_model import ModelConfig, CRNNModel

def get_time_dif(start_time):
    """
    获取已使用时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
    # return time_dif

def process_file(data_set_path, word_to_id, cat_to_id, max_length=600, en_or_ch='en'):
    """
    句子id化
    :param data_set_path:
    :param word_to_id:
    :param cat_to_id:
    :param max_length:
    :return: 输出是二维的句子id化矩阵，以及标签one hot向量
    """
    if en_or_ch=='en':
        print('英文分词')
        contents, labels = get_sentence_and_lable_en(data_set_path)
    elif en_or_ch == 'ch':
        print('中文分词')
        contents, labels = get_sentence_and_lable_china(data_set_path)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = tf.contrib.keras.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = tf.contrib.keras.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    """
    生成批次数据
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    # 生成一个打乱后的下标数组,indices类型是ndarray
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def feed_data(model, x_batch, y_batch, keep_prob):
    """
    构造模型输入变量
    :param model:
    :param x_batch:
    :param y_batch:
    :param keep_prob:
    :return:
    """
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict

def evaluate(sess, model, x_, y_):
    """
    评估在某一数据上的准确率和损失
    :param sess:
    :param model:
    :param x_:
    :param y_:
    :return:
    """
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(model, x_batch, y_batch, 1.0)
        y_pred_class,loss, acc = sess.run([model.y_pred_cls,model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return y_pred_class,total_loss / data_len, total_acc / data_len

def train(data_set_path, train_set_path, test_set_path, vocab_dict_path, lable_path, tensorboard_dir, model_dir, seq_len_param, logger, en_or_ch='en'):
    """
    训练
    :param data_set_path: 原始数据集
    :param train_set_path: 训练集
    :param test_set_path: 验证集
    :param vocab_dict_path: 字典
    :param lable_path: 类别
    :param tensorboard_dir: 监控
    :param model_dir: 模型目录
    :return:
    """
    model_save_path = os.path.join(model_dir, 'best_validation')  # 最佳验证结果保存路径
    config = ModelConfig()
    # 标签，标签id
    categories, cat_to_id = read_category(lable_path)
    # 所有字 和字id
    words, word_to_id = read_vocab(vocab_dict_path)
    # 统计句子长度
    seq_len_info = get_seq_length(data_set_path, en_or_ch)
    logger.info('句子长度统计如下:')
    logger.info(seq_len_info)

    config.num_epochs = 20000
    config.batch_size = 64
    config.seq_length = int(seq_len_info[seq_len_param])
    config.embedding_dim = 64
    config.num_class = len(categories)
    config.vocab_size = len(words)
    config.dropout_keep_prob = 0.8
    config.learning_rate = 1e-3
    logger.info("模型参数:")
    logger.info(config.__dict__.items())

    model = CRNNModel(config)

    logger.info("配置 TensorBoard 和 Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    # 收集数据变化,用来显示标量信息
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
    merged_summary = tf.summary.merge_all()
    # 指定一个文件用来保存图。可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver(max_to_keep=5)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    logger.info("加载训练/测试数据...")
    start_time = time.time()
    x_train, y_train = process_file(train_set_path, word_to_id, cat_to_id, config.seq_length, en_or_ch)
    x_val, y_val = process_file(test_set_path, word_to_id, cat_to_id, config.seq_length, en_or_ch)
    time_dif = get_time_dif(start_time)
    logger.info("加载训练/测试数据使用时间:{}".format(str(time_dif)) )

    # 创建session
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # 显示计算图了
        writer.add_graph(session.graph)

        logger.info("开始训练模型...")
        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = 2000  # 如果超过1000轮未提升，提前结束训练

        flag = False
        save_per_batch = 10  # 每多少轮存入tensorboard
        print_per_batch = 10  # 每多少轮输出一次结果
        for epoch in range(config.num_epochs):
            logger.info("Epoch:{}".format(str(epoch + 1)))
            # 获取每批次样本
            batch_train = batch_iter(x_train, y_train, config.batch_size)
            for x_batch, y_batch in batch_train:
                feed_dict = feed_data(model, x_batch, y_batch, config.dropout_keep_prob)

                # 每save_per_batch批次将训练结果写入tensorboard scalar
                if total_batch % save_per_batch == 0:
                    s = session.run(merged_summary, feed_dict=feed_dict) #调用sess.run运行图，生成一步的训练过程数据
                    writer.add_summary(s, total_batch) # 调用train_writer的add_summary方法将训练过程以及训练步数保存

                # 每print_per_batch批次输出训练/验证集上的性能
                if total_batch % print_per_batch == 0:
                    loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                    _, loss_val, acc_val = evaluate(session, model, x_val, y_val)  # todo

                    if acc_val > best_acc_val:
                        # 保存最好结果
                        best_acc_val = acc_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=model_save_path, global_step=total_batch)
                        improved_str = '*'
                    else:
                        improved_str = ''

                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                          + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                    logger.info(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

                _ = session.run(model.optim, feed_dict=feed_dict)  # 运行优化
                # session.run(model.print_list, feed_dict=feed_dict)
                # h只有最后一层的输出是一个ndarray，batchsize * seqlen * 最后一层lstm隐层维度
                # c是将多层lstm隐状态组装成元组，每层的状态是一个ndarray  lstm层数 * batchsize * 隐层维度
                # h, c = session.run([model.rnn_outputs_h, model.rnn_outputs_c], feed_dict=feed_dict)
                # print(h.shape, c)

                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    logger.info("No optimization for a long time, auto-stopping... {}, {}".format(last_improved, best_acc_val))
                    break  # 跳出循环

        logger.info("模型训练结束")



def batch_test(test_data_set, vocab_path, lable_path, model_dir, seq_padding, en_or_ch, logger):
    """
    批量测试
    :param test_data_set: 测试数据集
    :param vocab_path: 字典
    :param lable_path: 分类
    :param model_dir: 模型
    :param seq_padding: 句子最大长度，训练日志里可以找到该参数值。
    :param en_or_ch: 中文分词 or 英文分词
    :param logger: 日志实例
    :return:
    """
    # 加载：分类、字典、模型
    categories, cat_to_id = read_category(lable_path)
    words, word_to_id = read_vocab(vocab_path)
    model = Model_Init(model_dir)
    # word_to_id, id_to_cat, model = load_model(model_dir, vocab_path, lable_path)

    x_test, y_test = process_file(test_data_set, word_to_id, cat_to_id, seq_padding, en_or_ch)

    # 分批次
    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1) # one hot id化
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    y_pred_score = np.zeros(shape=len(x_test), dtype=np.float32) # 得分统计
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)

        score, y_pred = model.predict_batch(x_test[start_id:end_id])
        y_pred_cls[start_id:end_id] = y_pred
        y_pred_score[start_id:end_id] = score

    # 评估
    logger.info("Precision, Recall and F1-Score...")
    logger.info(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories, digits=4))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    logger.info(cm)

    logger.info("score:")
    np.sort(y_pred_score)
    logger.info(y_pred_score)

def single_test(txt, model_dir, vocab_path, lable_path, stc_format, seq_padding, en_or_ch):
    # 加载项
    word_to_id, id_to_cat, model = load_model(model_dir, vocab_path, lable_path)
    # 格式化句子
    txt = stc_format(txt)
    # id化
    if en_or_ch=='en':
        print('英文分词')
        words = txt.split(" ")
    elif en_or_ch == 'ch':
        print('中文分词')
        words = jieba.lcut(txt, cut_all=False)
    stc_id = [[word_to_id[x] for x in words if x in word_to_id]]
    x_pad = tf.contrib.keras.preprocessing.sequence.pad_sequences(stc_id, seq_padding)

    score, y_pred = model.predict(x_pad)
    y_pred = id_to_cat.get(y_pred)

    return y_pred, score


def load_model(model_dir, vocab_path, lable_path):
    """
    预测时的加载项
    :param model_dir:
    :param vocab_path:
    :param lable_path:
    :return:
    """
    categories, cat_to_id = read_category(lable_path)
    id_to_cat = {cat_id: cat_name for cat_name, cat_id in cat_to_id.items()}
    words, word_to_id = read_vocab(vocab_path)
    model = Model_Init(model_dir)

    return word_to_id, id_to_cat, model

def predict(txt, stc_format, seq_padding, word_to_id, id_to_cat, en_or_ch, model):
    """
    预测一条样本
    :param txt: 句子
    :param stc_format: 句子格式化函数
    :param seq_padding: 句子长度
    :param word_to_id: 字典，词-id
    :param id_to_cat: 字典， id-lable
    :param en_or_ch: 中文 or 英文
    :param model: 模型（已加载好的）
    :return:
    """
    # 1. 格式化文本
    txt = stc_format(txt)

    # 2. 文本id化 & padding
    words = jieba.lcut(txt, cut_all=False)
    stc_id = [[word_to_id[x] for x in words if x in word_to_id]]
    x_pad = tf.contrib.keras.preprocessing.sequence.pad_sequences(stc_id, seq_padding)

    # 3. 预测
    score, y_pred = model.predict(x_pad)
    score = [np.round(s*100, 5) for s in score]
    y_pred = id_to_cat.get(y_pred)
    return max(score), y_pred


# def predict(txt, stc_format, model_dir, vocab_path, lable_path, seq_padding):
#     """
#     预测一条
#     :param txt:
#     :param stc_format:
#     :param model_dir:
#     :param vocab_path:
#     :param lable_path:
#     :param seq_padding:
#     :return:
#     """
#     # 加载
#     categories, cat_to_id = read_category(lable_path)
#     id_to_cat = {cat_id: cat_name for cat_name, cat_id in cat_to_id.items()}
#     words, word_to_id = read_vocab(vocab_path)
#     model = Model_Init(model_dir)
#
#     # 1. 格式化文本
#     txt = stc_format(txt)
#
#     # 2. 文本id化 & padding
#     words = jieba.lcut(txt, cut_all=False)
#     stc_id = [[word_to_id[x] for x in words if x in word_to_id]]
#     x_pad = tf.contrib.keras.preprocessing.sequence.pad_sequences(stc_id, seq_padding)
#
#     # 3. 预测
#     score, y_pred = model.predict(x_pad)
#     score = [np.round(s*100, 5) for s in score]
#     y_pred = id_to_cat.get(y_pred)
#     return max(score), y_pred