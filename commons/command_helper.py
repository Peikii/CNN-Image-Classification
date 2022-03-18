import argparse
import time

def get_params():
    try:
        parser = argparse.ArgumentParser()

        parser.add_argument("--pdf_dir", type=str, default=r"D:\PDF_Classification\Data\Newdata", help='数据集路径。')
        parser.add_argument("--train_folder_name", type=str, default=time.strftime("%Y_%m_%d", time.localtime()), help='训练目录，默认是年_月_日')

        args = parser.parse_args()

        return args
    except Exception as e:
        raise e