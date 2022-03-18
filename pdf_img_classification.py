"""
图像分类
"""
import os
import time
import pickle
import traceback
import random
import shutil
import numpy as np
import tensorflow as tf
from datetime import timedelta
from PIL import Image
from sklearn import metrics
from commons.command_helper import get_params
from models.cnn_model import IMGCNNConfig, LeNet_5
from commons.log_helper import get_logger
from commons.file_helper import file_2_md5
from commons.model_restore import Model_Init
from commons.pdf_helper import pdf2img_fitz


def plk_dump(data, save_path):
    """

    :param data:
    :param save_path:
    :return:
    """
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
    except Exception as e:
        raise e

def check_repeat_by_pdf_md5(root_dir):
    """
    pdf去重复
    :param root_dir:
    :return:
    """
    md5_arr = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = "/".join([root, file])
                md5_code = file_2_md5(file_path)
                if md5_code in md5_arr:
                    print("删除:", file_path)
                    os.remove(file_path)
                    continue
                else:
                    print('新增:', file_path)
                    md5_arr.append(md5_code)
                    if not os.path.exists("/".join([root, md5_code+'.pdf'])):
                        os.rename(file_path, "/".join([root, md5_code+'.pdf']))
                    pass

def generate_img_dataset(pdf_root, img_root):
    """
    生成图片数据集
    目录结构和pdf目录结构一致
    :param pdf_root:
    :param img_root:
    :return:
    """
    logger.info("start generating dataset ...")
    try:
        for lable in os.listdir(pdf_root):
            lable_path = "/".join([pdf_root, lable])
            for pdf_name in os.listdir(lable_path):
                pdf_path = "/".join([lable_path, pdf_name])
                # padding img
                try:
                    img_padded = pdf2img_fitz(pdf_path, page_num=None, size=(1024, 1024), is_padding=False)
                except:
                    logger.error(pdf_path)
                    logger.error(traceback.format_exc())
                save_path = "/".join([img_root, lable])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # image_num 表示切分pdf之后每页的标记
                image_num = 0

                for img in img_padded:
                    img.save("/".join([save_path, pdf_name.split(".")[0] + "_page_" + str(image_num) + ".png"]))
                    image_num += 1
                pass
    except Exception as e:
        raise e


# ######################################################################################################################

def read_img_folder(img_root):
    """
    读取图像数据和标签数据
    :param img_root:
    :param lables:
    :return:
    """
    try:
        data_list = [] # 图像路径
        lable_list = [] # 图像标签
        lables = os.listdir(img_root) # 标签
        for lable in lables:
            lable_path = '/'.join([img_root, lable])
            for file_name in os.listdir(lable_path):
                if not file_name.endswith(".png"):
                    logger.warning("skip err file. {}-{}".format(lable, file_name))
                    continue
                img_path = "/".join([lable_path, file_name])
                data_list.append(img_path)
                lable_list.append(lable.strip())
                pass
        return np.array(data_list, dtype=np.str), np.array(lable_list, dtype=np.str), np.array(lables, dtype=np.str)
    except Exception as e:
        raise e

def balance_class(data_list, lable_list, max_sample_num=None):
    """
    类别均衡
    用上采样
    :param data_list: ndarray
    :param lable_list: ndarray
    :param max_sample_num: 采样数
    :return:
    """
    try:
        class_count = {} # 类别样本数
        class_datas = {} # 类别样本
        for lable in list(set(lable_list)):
            lable_index = np.where(lable_list == lable)[0] # where结果为元组，取第一个维度的ndarray
            one_class = data_list[lable_index]

            class_count[lable] = len(lable_index)
            class_datas[lable] = one_class

        logger.info("采样前每个类别的样本数: {}".format(class_count))
        if max_sample_num is None:
            max_sample_num = max(class_count.values())
        else:
            max_sample_num = max_sample_num
        logger.info("最大样本数: {}".format(str(max_sample_num)))

        for lable, datas in class_datas.items():
            sample_count = class_count.get(lable) # 该类别样本数
            need_sample = max_sample_num - sample_count # 需要增加的样本数
            if need_sample <= 0:
                # 当设置的采样数小于最样本数的时候会出现
                indexs = [random.randint(0, sample_count-1) for i in range(max_sample_num)]
                sample_datas = datas[indexs]
                class_datas[lable] = sample_datas
            else:
                indexs = [random.randint(0, sample_count-1) for i in range(need_sample)]
                sample_datas = datas[indexs]
                class_datas[lable] = np.hstack((class_datas.get(lable), sample_datas))
                # class_datas[lable] = list(class_datas.get(lable)).extend(list(sample_datas))
        class_count = {lable: len(datas) for lable, datas in class_datas.items()}
        logger.info("采样后每个类别样本数: {}".format(class_count))

        lable_list, data_list = [], []
        for lable, datas in class_datas.items():
            lable_list.extend([lable] * len(datas))
            data_list.extend(list(datas))

        lable_list, data_list = np.array(lable_list, dtype=np.str), np.array(data_list, dtype=np.str)

        return data_list, lable_list
    except Exception as e:
        raise e


def split_dataset(data_list, lable_list, ratio=0.9):
    """
    拆分数据集
    :param data_list: ndarray
    :param lable_list: ndarray
    :param ratio:
    :return:
    """
    try:
        def split_class(data, ratio):
            """
            按比例划分数据
            :param data: ndarray
            :param ratio: 比例
            :return:
            """
            # random.shuffle(data)  # 随机打乱
            data_len = data.shape[0]
            data_1 = data[:int(ratio * data_len)]
            data_2 = data[int(ratio * data_len):]
            return data_1, data_2

        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for lable in list(set(lable_list)):
            # ndarray tolist当数据量大资源不够时很耗时，慎用
            lable_index = np.where(lable_list == lable)[0]
            # logger.info("lable={}:{}".format(str(lable), lable_index))
            one_class = data_list[lable_index]
            one_train, one_test = split_class(one_class, ratio)


            # 数据集1
            train_x.extend(list(one_train))
            train_y.extend([lable]*one_train.shape[0])
            # 数据集2
            test_x.extend(list(one_test))
            test_y.extend([lable]*one_test.shape[0])

        train_x, train_y = np.array(train_x, dtype=np.str), np.array(train_y, dtype=np.str)
        test_x, test_y = np.array(test_x, dtype=np.str), np.array(test_y, dtype=np.str)
        return train_x, train_y, test_x, test_y
    except Exception as e:
        raise e

########################################################################################################################################

def generate_all(img_root, dataset_root):
    """
    生成数据：
    标签
    总数据集
    训练集
    验证集
    测试集
    :param img_root:
    :param dataset_root:
    :return:
    """
    try:
        if not os.path.exists(dataset_root):
            os.makedirs(dataset_root)

        # 读取图像路径，图像标签
        data_list, lable_list, lables = read_img_folder(img_root)
        plk_dump(data_list, "/".join([dataset_root, "data_set_source_x.pkl"]))
        plk_dump(lable_list, "/".join([dataset_root, "data_set_source_y.pkl"]))
        plk_dump(lables, "/".join([dataset_root, "lables.pkl"]))

        # 类别均衡
        data_list, lable_list = balance_class(data_list, lable_list, max_sample_num=None)
        plk_dump(data_list, "/".join([dataset_root, "data_set_x.pkl"]))
        plk_dump(lable_list, "/".join([dataset_root, "data_set_y.pkl"]))

        # 生成训练集、验证集、测试集 ok
        train_x, train_y, other_x, other_y = split_dataset(data_list, lable_list, ratio=0.7)
        validation_x, validation_y, test_x, test_y = split_dataset(other_x, other_y, ratio=0.5)
        plk_dump(train_x, "/".join([dataset_root, "train_x.pkl"]))
        plk_dump(train_y, "/".join([dataset_root, "train_y.pkl"]))
        plk_dump(validation_x, "/".join([dataset_root, "validation_x.pkl"]))
        plk_dump(validation_y, "/".join([dataset_root, "validation_y.pkl"]))
        plk_dump(test_x, "/".join([dataset_root, "test_x.pkl"]))
        plk_dump(test_y, "/".join([dataset_root, "test_y.pkl"]))

        logger.info("标签: {}".format(lables))
        logger.info("总数据量: {}".format(len(lable_list)))
        for lable_id in list(set(lable_list)):
            data_count = len(np.where(np.array(lable_list)==lable_id)[0])
            logger.info("   lable={} 数据量: {}".format(str(lable_id), str(data_count)))

        logger.info("训练集数据量: {}".format(len(train_y)))
        for lable_id in list(set(train_y)):
            data_count = len(np.where(np.array(train_y)==lable_id)[0])
            logger.info("   lable={} 数据量: {}".format(str(lable_id), str(data_count)))

        logger.info("验证集数据量: {}".format(len(validation_y)))
        for lable_id in list(set(validation_y)):
            data_count = len(np.where(np.array(validation_y)==lable_id)[0])
            logger.info("   lable={} 数据量: {}".format(str(lable_id), str(data_count)))

        logger.info("测试集数据量: {}".format(len(test_y)))
        for lable_id in list(set(test_y)):
            data_count = len(np.where(np.array(test_y)==lable_id)[0])
            logger.info("   lable={} 数据量: {}".format(str(lable_id), str(data_count)))

    except Exception as e:
        raise e

########################################################################################################################

def read_train_and_val_set(dataset_root):
    """

    :param dataset_root:
    :return:
    """
    try:
        with open("/".join([dataset_root, "train_x.pkl"]), 'rb') as f:
            train_x = pickle.load(f)
        with open("/".join([dataset_root, "train_y.pkl"]), 'rb') as f:
            train_y = pickle.load(f)
        with open("/".join([dataset_root, "validation_x.pkl"]), 'rb') as f:
            val_x = pickle.load(f)
        with open("/".join([dataset_root, "validation_y.pkl"]), 'rb') as f:
            val_y = pickle.load(f)
        with open("/".join([dataset_root, "lables.pkl"]), 'rb') as f:
            lables = pickle.load(f)

        return train_x, train_y, val_x, val_y, lables
    except Exception as e:
        raise e

def batch_iter(x, y, batch_size=64):
    """
    生成批次数据
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    data_len = x.shape[0]
    num_batch = int((data_len - 1) / batch_size) + 1

    # 生成一个打乱后的下标数组, indices类型是ndarray
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)

        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]



def feed_data(model, x_batch, y_batch, keep_prob, lables):
    """
    构造模型输入变量
    :param model:
    :param x_batch:
    :param y_batch:
    :param keep_prob:
    :param lables:
    :return:
    """
    try:
        # y输入
        lables = list(lables)
        # x输入
        x_batch_in = []
        for img_path in list(x_batch):
            # 读取灰度值
            image = Image.open(img_path)
            image = np.array(image, dtype=np.float32) # h*w*c
            image = image / 255.0

            x_batch_in.append(image)
        x_batch_in = np.array(x_batch_in, dtype=np.float32)

        # y输入
        y_batch_in = []
        for lable in list(y_batch):
            y_batch_in.append(lables.index(lable))
        y_batch_in = tf.contrib.keras.utils.to_categorical(y_batch_in, num_classes=len(lables))  # 将标签转换为one-hot表示

        feed_dict = {
            model.input_x: x_batch_in,
            model.input_y: y_batch_in,
            model.keep_prob: keep_prob
        }
        return feed_dict
    except Exception as e:
        raise e

def evaluate(sess, model, x_, y_, lables):
    """
    评估在某一数据上的准确率和损失
    :param sess:
    :param model:
    :param x_:
    :param y_:
    :return:
    """
    data_len = x_.shape[0]
    batch_eval = batch_iter(x_, y_, 64)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = x_batch.shape[0]
        feed_dict = feed_data(model, x_batch, y_batch, 1.0, lables)
        val_input_time = time.time()
        y_pred_class, loss, acc = sess.run([model.y_pred, model.loss, model.acc], feed_dict=feed_dict)
        logger.info("[val set] batch data loss and acc time . {}".format(get_time_dif(val_input_time)))
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return y_pred_class, total_loss / data_len, total_acc / data_len

def get_time_dif(start_time):
    """
    获取已使用时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(model_dir, tensorboard_dir, dataset_root):
    """

    :param model_dir:
    :param tensorboard_dir:
    :param dataset_root:
    :return:
    """

    # 加载算法
    config = IMGCNNConfig()
    model = LeNet_5(config)

    # 配置 Tensorboard
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar("loss", model.loss) # 收集数据变化,用来显示标量信息
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all() # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。如果没有特殊要求，一般用这一句就可一显示训练时的各种信息了。
    writer = tf.summary.FileWriter(tensorboard_dir) # 指定一个文件用来保存图。可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中

    # 配置 Saver
    model_save_path = os.path.join(model_dir, 'best_validation')  # 最佳验证结果保存路径
    saver = tf.train.Saver(max_to_keep=2)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 加载数据
    train_x, train_y, val_x, val_y, lables = read_train_and_val_set(dataset_root)
    logger.info('lables: {}'.format(lables))

    # 创建session
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph) # 显示计算图了

        start_time = time.time()
        total_batch = 0  # 总批次
        best_acc_val = 0.0  # 最佳验证集准确率
        last_improved = 0  # 记录上一次提升批次
        require_improvement = config.require_improvement  # 如果超过1000轮未提升，提前结束训练

        flag = False
        save_per_batch = config.save_per_batch  # 每多少轮存入tensorboard
        print_per_batch = config.print_per_batch  # 每多少轮输出一次结果
        for epoch in range(config.num_epochs):
            if flag:
                break
            logger.info("Epoch:{}".format(str(epoch + 1)))
            batch_train = batch_iter(train_x, train_y, config.batch_size) # 获取一个batch size
            for x_batch, y_batch in batch_train:
                feed_dict = feed_data(model, x_batch, y_batch, config.dropout_keep_prob, lables)

                # 每save_per_batch批次将训练结果写入tensorboard scalar
                if total_batch % save_per_batch == 0:
                    s = session.run(merged_summary, feed_dict=feed_dict) #调用sess.run运行图，生成一步的训练过程数据
                    writer.add_summary(s, total_batch) # 调用train_writer的add_summary方法将训练过程以及训练步数保存

                # 每print_per_batch批次输出训练/验证集上的性能
                if total_batch != 0 and total_batch % print_per_batch == 0:
                    input_time = time.time()
                    loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                    logger.info("[train set] batch data loss and acc time . {}".format(get_time_dif(input_time)))
                    _, loss_val, acc_val = evaluate(session, model, val_x, val_y, lables)  # todo

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

                start_optim = time.time()
                _ = session.run(model.optim, feed_dict=feed_dict)  # 运行优化
                logger.info("Iter:{}, optim time. {}".format(total_batch, get_time_dif(start_optim)))

                total_batch += 1

                if total_batch - last_improved > require_improvement:
                    # 验证集正确率长期不提升，提前结束训练
                    logger.info("No optimization for a long time, auto-stopping... {}, {}".format(last_improved, best_acc_val))
                    flag = True
                    break  # 跳出循环

        logger.info("模型训练结束")



def batch_test(dataset_root):
    """

    """
    try:
        # 加载
        model = Model_Init(model_dir)
        with open("/".join([dataset_root, "test_x.pkl"]), 'rb') as f:
            test_x = pickle.load(f)
        with open("/".join([dataset_root, "test_y.pkl"]), 'rb') as f:
            test_y = pickle.load(f)
        with open("/".join([dataset_root, "lables.pkl"]), 'rb') as f:
            lables = pickle.load(f)
        lables = list(lables)

        # 预测
        batch_eval = batch_iter(test_x, test_y, 32)
        y_test_cls = []
        y_pred_cls = []
        y_pred_score = []
        for x_batch, y_batch in batch_eval:
            # x输入
            x_batch_in = []
            for img_path in list(x_batch):
                # 读取灰度值
                image = Image.open(img_path)
                image = np.array(image, dtype=np.float32)  # h*w*c
                image = image / 255.0

                x_batch_in.append(image)
            x_batch_in = np.array(x_batch_in, dtype=np.float32)
            # y输入
            y_batch_in = []
            for lable in list(y_batch):
                y_batch_in.append(lables.index(lable))

            score, y_pred = model.predict_img_batch(x_batch_in)

            y_test_cls.extend(y_batch_in)
            y_pred_cls.extend(list(y_pred))
            y_pred_score.extend(list(score))

        # 评估
        logger.info("Precision, Recall and F1-Score...")
        logger.info(metrics.classification_report(y_test_cls, y_pred_cls, target_names=lables, digits=4))

        # 混淆矩阵
        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        logger.info(cm)

        logger.info("score:")
        np.sort(y_pred_score)
        logger.info(y_pred_score)
    except Exception as e:
        raise e

def single_test(model_dir, dataset_root, img_root, result_root):
    # img_root = r"\\cnshf-fs02\TFS_Automation_CN\YF_bak\dataset\B2B_NEW_IMG"
    # result_root = r"\\cnshf-fs02\TFS_Automation_CN\YF_bak\dataset\B2B_NEW_TEST_PREDICT"
    try:
        # 加载
        model = Model_Init(model_dir)
        with open("/".join([dataset_root, "lables.pkl"]), 'rb') as f:
            lables = pickle.load(f)
        lables = list(lables)

        total_num = 0
        right_num = 0
        # 预测文件
        lable_names = os.listdir(img_root)
        for lable_name in lable_names:
            lable_dir = "/".join([img_root, lable_name])
            file_names = os.listdir(lable_dir)
            for file_name in file_names:
                file_path = "/".join([lable_dir, file_name])

                image = Image.open(file_path)
                image = np.array(image, dtype=np.float32)  # h*w*c
                image = image / 255.0
                image = np.array([image], dtype=np.float32)

                score, y_pred = model.predict_img(image)
                y_pred = lables[y_pred]

                total_num = total_num + 1
                t_or_f = ''
                if y_pred == lable_name:
                    logger.info('{}. 预测正确'.format(str(total_num)))
                    right_num = right_num+1
                    t_or_f='正确'
                else:
                    logger.info('{}. 预测错误'.format(str(total_num)))
                    t_or_f = '错误'


                file_name_copy = "_".join([str(total_num), t_or_f, lable_name, str(score), '.png'])

                if not os.path.exists("/".join([result_root, y_pred])):
                    os.makedirs("/".join([result_root, y_pred]))

                shutil.copy(file_path, "/".join([result_root, y_pred, file_name_copy]))

        logger.info("总数：{}， 正确个数：{}".format(str(total_num), str(right_num)))







    except Exception as e:
        raise e

    pass



if __name__ == '__main__':

    args = get_params()
    root_dir = args.pdf_dir
    train_folder_name = args.train_folder_name
    print("--pdf_dir {}".format(root_dir))
    print("--train_folder_name {}".format(train_folder_name))

    logger = get_logger("train_{}".format(train_folder_name))
    try:
        # 当前目录
        # curr_path, _ = os.path.split(os.path.abspath(__file__).replace("\\", "/"))
        pdf_root = r"D:\PDF_Classification\Data\Newdata"
        img_root = r"D:\PDF_Classification\Data\Newimage"
        prd_root = r"D:\PDF_Classification\Data\Newpredict"


        dataset_root = "\\".join(['datas', train_folder_name, "dataset"])
        model_dir = "\\".join(['datas', train_folder_name, "ckpt"])
        tensorboard_dir = "\\".join(['datas', train_folder_name, "tensorboard"])
        if not os.path.exists(dataset_root):
            os.makedirs(dataset_root)

        logger.info("dataset path '{}'.".format(dataset_root))
        logger.info("model path   '{}'.".format(model_dir))
        logger.info("tensorboard path '{}'.".format(tensorboard_dir))
        # 0、pdf去重复
        # check_repeat_by_pdf_md5(pdf_root)
        # 1、生成图像数据集
        #generate_img_dataset(pdf_root, img_root)
        # 2、生成pkl数据集
        generate_all(img_root, dataset_root)
        # 3、训练
        logger.info("开始训练...")
        train(model_dir, tensorboard_dir, dataset_root)
        # 4、测试
        batch_test(dataset_root)

        single_test(model_dir, dataset_root, img_root, prd_root)
        logger.info("finished")

    except:
        logger.error(traceback.format_exc())

