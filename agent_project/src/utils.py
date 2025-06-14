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
