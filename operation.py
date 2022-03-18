"""
图像分类
"""
import os
import pickle
import traceback
import numpy as np
from PIL import Image
from commons.log_helper import get_logger
from commons.model_restore import Model_Init
from commons.pdf_helper import pdf2img_fitz
import json
import shutil
import re

def copy_data(pdf_root, query_root):
    with open(query_root, "r") as f:
        list = json.loads(f.read())
        for i in list:
            id = i[0]
            path = r"\\"+"cnshf-appfs-t1\TFS_Automation_CN\ABBYY_logistics\\"+i[1]
            dir_name = str(id)
            destination = "\\".join([pdf_root, dir_name])
            try:
                if not os.path.exists(destination):
                    os.makedirs(destination)
                    print(destination + " have been created")
                shutil.copy(str(path), destination)
                print(i[1] + " successfully added")
            except IOError as e:
                print("Unable to copy file. %s" % e)
                exit(1)
            except Exception as e:
                print("Unexpected error coppying files", e)
                exit(1)


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
        for id in os.listdir(pdf_root):
            id_path = "/".join([pdf_root, id])
            for pdf_name in os.listdir(id_path):
                pdf_path = "/".join([id_path, pdf_name])
                # padding img
                try:
                    img_padded = pdf2img_fitz(pdf_path, page_num=None, size=(1024, 1024), is_padding=False)
                except:
                    logger.error(pdf_path)
                    logger.error(traceback.format_exc())
                save_path = "/".join([img_root, id])
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



def single_test(dataset_root, img_root, result_root):
    # img_root = r"\\cnshf-fs02\TFS_Automation_CN\YF_bak\dataset\B2B_NEW_IMG"
    # result_root = r"\\cnshf-fs02\TFS_Automation_CN\YF_bak\dataset\B2B_NEW_TEST_PREDICT"
    try:
        # 加载
        with open("/".join([dataset_root, "lables.pkl"]), 'rb') as f:
            lables = pickle.load(f)
        lables = list(lables)

        total_num = 0
        right_num = 0
        # 预测文件
        id_names = os.listdir(img_root)
        for id_name in id_names:
            id_dir = "/".join([img_root, id_name])
            file_names = os.listdir(id_dir)
            for file_name in file_names:
                file_path = "/".join([id_dir, file_name])
                page_num = re.match(r'(.*)_page_(.*)\.png', file_name).group(2)

                image = Image.open(file_path)
                image = np.array(image, dtype=np.float32)  # h*w*c
                image = image / 255.0
                image = np.array([image], dtype=np.float32)

                score, y_pred = model.predict_img(image)
                y_pred = lables[y_pred]
                total_num = total_num + 1

                if score > 0.99:
                    logger.info('{}. 高置信度捕捉 '.format(str(total_num)) + 'score: ' + str(score))
                    right_num = right_num + 1

                    file_name_copy = "_".join([str(total_num), y_pred, "ID", id_name, "page", page_num, str(score), '.png'])

                    if not os.path.exists("/".join([result_root, y_pred])):
                        os.makedirs("/".join([result_root, y_pred]))

                    shutil.copy(file_path, "/".join([result_root, y_pred, file_name_copy]))

                else:
                    logger.info('{}. 低置信度剔除 '.format(str(total_num) + 'score: ' + str(score) + file_path))

        logger.info("总数：{}， 捕捉个数：{}".format(str(total_num), str(right_num)))



    except Exception as e:
        raise e

    pass

if __name__ == '__main__':
    pdf_root = r"D:\PDF_Classification\Data\testdata"
    img_root = r"D:\PDF_Classification\Data\testimage"
    prd_root = r"D:\PDF_Classification\Data\testpredict"
    query_root = r"D:\PDF_Classification\Data\Query_result"

    model_dir = r"D:\PDF_Classification\WorkPlace\datas\2021_05_20\ckpt"
    dataset_root = r"D:\PDF_Classification\WorkPlace\datas\2021_05_20\dataset"

    logger = get_logger("operation_{}".format("2015_05_20"))

    try:
        # copy_data(pdf_root, query_root)
        model = Model_Init(model_dir)

        # 1.拆分pdf
        generate_img_dataset(pdf_root, img_root)

        # 2.开始测试
        logger.info("模型开始执行...")
        single_test(dataset_root, img_root, prd_root)

    except:
        logger.error(traceback.format_exc())
