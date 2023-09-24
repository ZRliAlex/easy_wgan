import os
import hashlib


def calculate_hash(file_path, block_size=65536):
    """计算文件的哈希值"""
    file_hash = hashlib.md5()
    with open(file_path, 'rb') as file:
        for block in iter(lambda: file.read(block_size), b''):
            file_hash.update(block)
    return file_hash.hexdigest()


def remove_duplicate_images(folder_path):
    """从文件夹中删除重复的图片"""
    hash_map = {}

    for root, _, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            # 仅处理图片文件（可以根据需要添加更多的文件扩展名）
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                file_hash = calculate_hash(file_path)
                if file_hash in hash_map:
                    # 如果哈希已经存在，说明是重复图片，删除它
                    print(f"Removing duplicate image: {file_path}")
                    os.remove(file_path)
                else:
                    # 否则，将哈希添加到哈希映射中
                    hash_map[file_hash] = file_path


if __name__ == "__main__":
    folder_path = "./img/train"  # 替换成包含图片的文件夹的路径
    remove_duplicate_images(folder_path)
