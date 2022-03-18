"""
发票按公司格式分类
"""
import os
import re
import csv
import traceback
# import shutil
from commons.command_helper import get_params
from commons.log_helper import get_logger
from commons.file_helper import file_2_md5
from commons.pdf_helper import pdf2txt_ocr, pdf2txt_pdfplumber
from commons.dataset_helper import check_repeat, balance_class, split_dataset
from commons.vocab_helper import build_vocab
from commons.model_helper import train, batch_test

# 设置可读字符长度
csv.field_size_limit(500 * 1024 * 1024)

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
                file_path = "\\".join([root, file])
                md5_code = file_2_md5(file_path)
                if md5_code in md5_arr:
                    print("删除:", file_path)
                    os.remove(file_path)
                    continue
                else:
                    print('新增:', file_path)
                    md5_arr.append(md5_code)
                    if not os.path.exists("\\".join([root, md5_code+'.pdf'])):
                        os.rename(file_path, "\\".join([root, md5_code+'.pdf']))
                    pass




def pdf2txt_batch(dir, page_index=None):
    """
    批量读取PDF，转txt
    :param dir: 目录
    :return:
    """
    try:
        contents = []
        file_arr = os.listdir(dir)
        for file_name in file_arr:
            file_path = "/".join([dir, file_name])
            # 是pdf的文件
            if os.path.isfile(file_path) and file_name.lower().endswith(".pdf"):
                try:
                    # pdfplumber
                    txt = pdf2txt_pdfplumber(file_path, page_num=page_index)
                except:
                    logger.error(traceback.format_exc())
                    txt = None
                if txt is None:
                    txt = pdf2txt_ocr(file_path, page_num=page_index, split_img=False)  # .lower()
                contents.append(txt)
        return contents
    except Exception as e:
        raise e

def generate_dataset(root_dir, data_set_path, page_index=None):
    """
    生成csv数据集
    :param root_dir: 根目录
    :param data_set_path: 数据集保存路径
    :return:
    """
    logger.info("start generate dataset ...")
    with open(data_set_path, 'w', newline='', encoding='utf-8') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["id", "lable", "txt"])
        id = 1
        for company_name in os.listdir(root_dir):
            if company_name == "other":
                logger.info("skip company dir: '{}'".format(company_name))
                continue
            logger.info("company_name: {}".format(company_name))
            # 公司路径
            company_path = "/".join([root_dir, company_name])

            try:
                ivc_batch_txt = pdf2txt_batch(company_path, page_index)
                for ivc_txt in ivc_batch_txt:
                    csv_writer.writerow([id, company_name, ivc_txt])
                    id = id + 1
            except:
                logger.error(traceback.format_exc())


def format_sentence(txt):
    # 合并空格
    txt = re.sub(r'\s+', ' ', txt.strip())
    # 替换符号
    # txt = re.sub(r'[^\d\sa-zA-Z]', '<SYMBOL>', txt.strip())
    # 替换数字
    txt = re.sub(r'\d+', '<NUM>', txt.strip())
    # 合并<NUM> 合并<SYMBOL>
    txt = re.sub(r'(<NUM>\s*){1,}', '<NUM> ', txt.strip())
    # txt = re.sub(r'(<SYMBOL>\s*){1,}', '<SYMBOL> ', txt.strip())
    return txt

def format_data(data_set_path):
    """
    格式化句子
    处理空格，字符，数字，以及去重
    :param data_set_path:
    :return:
    """
    logger.info("start format data set...")
    file_dir, file_name = os.path.split(data_set_path)
    file_sname, file_suffix = os.path.splitext(file_name)
    if file_dir.strip() == "":
        data_set_format_path = file_sname + '_format.csv'
    else:
        data_set_format_path = "/".join([file_dir, file_sname + '_format.csv'])

    with open(data_set_path, 'r', encoding='utf-8') as fr:
        with open(data_set_format_path, 'w', newline='', encoding='utf-8') as fw:
            try:
                csv_reader = csv.DictReader(_.replace('\x00', '') for _ in fr)
                csv_writer = csv.writer(fw)
                csv_writer.writerow(["id", "lable", "txt"])
                for line in csv_reader:
                    if len(line.values()) != 3:
                        logger.warning("error line: "+str(line))
                        continue
                    if "id" not in line.keys() or "lable" not in line.keys() or "txt" not in line.keys():
                        logger.warning("key error: "+str(line))
                        continue
                    id, lable, txt = line["id"], line["lable"], line["txt"]
                    txt = format_sentence(txt)
                    csv_writer.writerow([id, lable, txt])
            except Exception as e:
                raise e
    return data_set_format_path



if __name__ == '__main__':
    # 执行命令: python pdf_txt_classification.py --train_folder_name 训练目录 --pdf_dir \\cnshf-fs02\TFS_Automation_CN\YF_bak\dataset\online_pdf

    # 获取参数
    args = get_params()
    root_dir = args.pdf_dir
    train_folder_name = args.train_folder_name
    print("--pdf_dir {}".format(root_dir))
    print("--train_folder_name {}".format(train_folder_name))


    logger = get_logger("train_{}".format(train_folder_name))
    data_set_dir = "\\".join(['datas', train_folder_name, "dataset"])
    tensorboard_dir = "\\".join(['datas', train_folder_name, "tensorboard"])
    model_dir = "\\".join(['datas', train_folder_name, "ckpt"])
    if not os.path.exists(data_set_dir):
        os.makedirs(data_set_dir)


    data_set_path = data_set_dir + '/source_data.csv'  # 数据集保存路径
    train_set_path = data_set_dir + '/train.csv'
    test_set_path = data_set_dir + '/test.csv'

    vocab_dict_path = data_set_dir + "/vocab.csv"
    lable_path = data_set_dir + "/lable.csv"

    # pdf去重
    check_repeat_by_pdf_md5(root_dir)
    # step 1. 生成数据集
    generate_dataset(root_dir, data_set_path, "0-4")
    # step 2. 数据集处理
    data_set_format_path = format_data(data_set_path)
    data_set_drop_repeat_path = check_repeat(data_set_format_path)
    data_set_balance_class_path = balance_class(data_set_drop_repeat_path, None)
    # step 3. 生成字典
    build_vocab(data_set_balance_class_path, vocab_dict_path, lable_path, 'en')
    # step 4. 拆分数据集
    split_dataset(data_set_balance_class_path, train_set_path, test_set_path)
    # step 5.训练 词向量
    train(data_set_balance_class_path, train_set_path, test_set_path, vocab_dict_path, lable_path, tensorboard_dir, model_dir, '8/10', logger, 'en')
    # ste 6.测试
    batch_test(test_set_path, vocab_dict_path, lable_path, model_dir, 'en', logger)
    logger.info("finished")



