import os
import re
import cv2
import shutil
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YOLOFormatConverter:
    def __init__(self, data_path, save_path):
        """
        初始化转换器
        :param data_path: 原始数据集路径
        :param save_path: 转换后数据保存路径
        """
        self.data_path = data_path
        self.save_path = save_path

        # 创建保存路径
        self._create_directories()

    def _create_directories(self):
        """
        创建保存图片和标签的文件夹
        """
        try:
            for subset in ["test", "train", "val"]:
                images_save_path = os.path.join(self.save_path, subset, "images")
                labels_save_path = os.path.join(self.save_path, subset, "labels")
                if not os.path.exists(images_save_path):
                    os.makedirs(images_save_path)
                if not os.path.exists(labels_save_path):
                    os.makedirs(labels_save_path)
        except Exception as e:
            logging.error(f"创建文件夹时出错: {e}")
            raise

    @staticmethod
    def list_path_all_files(dirname):
        """
        遍历指定目录下的所有文件并返回文件路径列表
        :param dirname: 目录路径
        :return: 文件路径列表
        """
        result = []
        for maindir, subdir, file_name_list in os.walk(dirname):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                result.append(apath)
        return result

    def convert(self):
        """
        将数据集转换为YOLO格式
        """
        try:
            # 获取所有图片文件路径
            images_files = self.list_path_all_files(self.data_path)
            logging.info(f"找到 {len(images_files)} 个文件")

            # 初始化计数器
            cnt = {"test": 1, "train": 1, "val": 1}

            # 遍历所有图片文件
            for name in tqdm(images_files, desc="转换进度"):
                if name.endswith(".jpg") or name.endswith(".png"):
                    # 确定当前文件属于哪个子集（test/train/val）
                    subset = os.path.basename(os.path.dirname(name))

                    # 读取图片
                    img = cv2.imread(name)
                    if img is None:
                        logging.warning(f"无法读取图片: {name}")
                        continue

                    # 获取图片的高度和宽度
                    height, width = img.shape[0], img.shape[1]

                    # 使用正则表达式从文件名中提取坐标信息
                    try:
                        str1 = re.findall('-\d+\&\d+_\d+\&\d+-', name)[0][1:-1]
                        str2 = re.split('\&|_', str1)
                        x0 = int(str2[0])
                        y0 = int(str2[1])
                        x1 = int(str2[2])
                        y1 = int(str2[3])
                    except Exception as e:
                        logging.error(f"解析文件名时出错: {name}, 错误: {e}")
                        continue

                    # 计算边界框的中心点坐标以及宽度和高度，并进行归一化
                    x = round((x0 + x1) / 2 / width, 6)
                    y = round((y0 + y1) / 2 / height, 6)
                    w = round((x1 - x0) / width, 6)
                    h = round((y1 - y0) / height, 6)

                    # 构建保存路径
                    images_save_path = os.path.join(self.save_path, subset, "images")
                    labels_save_path = os.path.join(self.save_path, subset, "labels")

                    # 构建标签文件名和路径
                    txtfile = os.path.join(labels_save_path, f"green_plate_{str(cnt[subset]).zfill(6)}.txt")
                    # 构建图片文件名和路径
                    imgfile = os.path.join(images_save_path,
                                           f"green_plate_{str(cnt[subset]).zfill(6)}.{os.path.basename(name).split('.')[-1]}")

                    # 写入标签文件
                    with open(txtfile, "w") as f:
                        f.write(" ".join(["0", str(x), str(y), str(w), str(h)]))

                    # 移动图片到新位置
                    shutil.move(name, imgfile)

                    # 更新计数器
                    cnt[subset] += 1

            logging.info(f"转换完成，共处理 {sum(cnt.values()) - 3} 张图片")
        except Exception as e:
            logging.error(f"转换过程中出错: {e}")
            raise

if __name__ == '__main__':
    # 原始数据集路径
    data_path = "./ccpd_green"
    # 转换后数据保存路径
    save_path = "./dataset"

    converter = YOLOFormatConverter(data_path, save_path)
    converter.convert()