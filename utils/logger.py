import os
import sys
import logging
from time import strftime


# 设置日志格式#和时间格式
# FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'
DATEFMT = '%Y-%m-%d %H:%M:%S'

class MyLog(object):
    def __init__(self, log_path, type_='train'):
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
        self.log_filename = os.path.join(log_path, 'train.log' if type_=='train' else 'test.log')
        # self.log_filename = '{0}{1}.log'.format(log_path, strftime("%Y-%m-%d"))

        # 输出到文件
        self.logger.addHandler(self.get_file_handler(self.log_filename))
        
        # 输出到控制台
        # self.logger.addHandler(self.get_console_handler())
        
        # 设置日志的默认级别
        # 打印DEBUG级别以及以上的日志
        # 级别排序为：CRITICAL > ERROR > WARNING > INFO > DEBUG
        self.logger.setLevel(logging.INFO)

    # 输出到文件handler的函数定义
    def get_file_handler(self, filename):
        filehandler = logging.FileHandler(filename, encoding="utf-8")
        filehandler.setFormatter(self.formatter)
        return filehandler

    # 输出到控制台handler的函数定义
    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        return console_handler

def setup_logger(log_path=None, type_='train'):
    # 输出日志路径
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        return MyLog(log_path, type_).logger
    log_path = os.path.abspath('.') + '/logs/'
    return  MyLog(log_path, type_).logger