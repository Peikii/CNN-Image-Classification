"""
日志工具类
"""
import os
import logging

# 当前目录
curr_path, _ = os.path.split(os.path.abspath(__file__).replace("\\", "/"))
# 日志保存路径
log_path = "/".join([curr_path, "..", "log"])
if not os.path.exists(log_path):
    print("Create log path")
    os.makedirs(log_path)
# 当前时间
# import time
# curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
# print(curr_path)
# print(log_path)
# print(curr_time)

def get_logger(logger_name="log"):
    """

    :param logger_name:
    :return:
    """
    logger = logging.getLogger(logger_name)
    logger.disabled = False
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    handler_f = logging.FileHandler(log_path+"/{}.log".format(logger_name), encoding="utf-8")
    handler_f.setLevel(logging.INFO)
    handler_f.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    handler_s = logging.StreamHandler()  # 控制台输出
    handler_s.setLevel(logging.INFO)
    # 为logger对象添加句柄
    logger.addHandler(handler_f)
    logger.addHandler(handler_s)

    return logger


