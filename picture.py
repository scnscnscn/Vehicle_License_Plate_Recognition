import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt
import matplotlib

# 解决 Matplotlib 中文乱码
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
matplotlib.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 初始化 PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 车牌检测模型
path = 'best.pt'
img_path = "test.jpg"
model = YOLO(path, task='detect')

# 读取原始图像
original_image = cv2.imread(img_path)

# 运行 YOLO 车牌检测
results = model(img_path)

# 透视变换
def four_point_transform(image, pts):
    rect = np.array(pts, dtype="float32")

    width = int(np.linalg.norm(rect[1] - rect[0]))
    height = int(np.linalg.norm(rect[3] - rect[0]))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped

# 遍历检测到的车牌
for result in results[0].boxes.xyxy:
    x1, y1, x2, y2 = map(int, result[:4])

    # 增加 padding
    padding = 10
    x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
    x2, y2 = min(x2 + padding, original_image.shape[1] - 1), min(y2 + padding, original_image.shape[0] - 1)

    cropped_image = original_image[y1:y2, x1:x2]

    # 透视变换
    pts = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype="float32")

    warped_image = four_point_transform(original_image, pts)

    # OCR 识别车牌
    result = ocr.ocr(warped_image, cls=True)
    plate_text = "".join([word[1][0] for line in result for word in line])

    # 显示结果
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title("检测到的车牌")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title(f"识别结果: {plate_text}")

    plt.show()
