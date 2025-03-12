import os
import logging


def get_next_output_folder(base_path="outputs"):
    """获取 outputs 文件夹下的下一个数字文件夹路径"""
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # 获取所有以数字命名的文件夹
    existing_folders = [
        int(folder) for folder in os.listdir(base_path) if folder.isdigit()
    ]
    next_number = max(existing_folders, default=0) + 1  # 计算下一个文件夹编号
    new_folder_path = os.path.join(base_path, str(next_number))

    os.makedirs(new_folder_path)  # 创建新文件夹
    return new_folder_path

def setup_logging(log_file):
    """设置日志记录，写入文件 (DEBUG 及以上) 并输出到终端 (INFO 及以上)"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 全局日志级别设为 DEBUG

    # 创建文件 Handler（记录 DEBUG 及以上的日志）
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 创建控制台 Handler（只显示 INFO 及以上的日志）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # 添加 handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


output_folder = get_next_output_folder()
LOG_FILE = os.path.join(output_folder, "_app.log")
setup_logging(LOG_FILE)
logger = logging.getLogger(__name__)
logger.info(f"日志文件位置: {LOG_FILE}")