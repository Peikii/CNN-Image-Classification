
import hashlib

def file_2_md5(file_path):
    """
    文件转MD5
    :param file_path:
    :return:
    """
    try:
        with open(file_path, 'rb') as f:
            bi_str = f.read()
        return hashlib.md5(bi_str).hexdigest()
    except Exception as e:
        raise e

def str_2_md5(value):
    try:
        return hashlib.md5(value).hexdigest()
    except Exception as e:
        raise e