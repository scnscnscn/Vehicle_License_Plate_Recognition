import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import logging

# 初始化 OCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 加载 YOLO 车牌检测模型
model = YOLO('best.pt', task='detect')


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


# 处理中文文字显示
def add_chinese_text(img, text, position, textColor=(255, 255, 255), textSize=30):
    """ 在 OpenCV 画面上添加中文文字 """
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
        except IOError:
            fontStyle = ImageFont.truetype("arial.ttf", textSize)
        draw.text(position, text, fill=textColor, font=fontStyle)
        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"文字渲染失败: {str(e)}")
        return img


# 选择视频或摄像头
video_path = "test.mp4"  # 摄像头: 0，或者改成 "video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 无法打开摄像头或视频")
    exit()

cv2.namedWindow("车牌识别", cv2.WINDOW_NORMAL)
cv2.resizeWindow("车牌识别", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))  # 增加分辨率，提高检测率

    # 运行 YOLO 进行车牌检测
    results = model(frame, imgsz=640, conf=0.25, iou=0.45)

    for result in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, result[:4])

        # 增加 padding，防止边缘部分丢失
        padding = 10
        x1, y1 = max(x1 - padding, 0), max(y1 - padding, 0)
        x2, y2 = min(x2 + padding, frame.shape[1] - 1), min(y2 + padding, frame.shape[0] - 1)

        cropped_plate = frame[y1:y2, x1:x2]

        # 透视变换
        pts = np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype="float32")

        warped_plate = four_point_transform(frame, pts)

        # OCR 识别车牌
        result = ocr.ocr(warped_plate, cls=True)

        # 处理识别结果，防止 NoneType 错误
        plate_text = ""
        if result and isinstance(result, list):
            try:
                plate_text = "".join([word[1][0] for line in result for word in line if word])
            except Exception as e:
                print(f"⚠️ 车牌识别异常: {e}")
                plate_text = "识别失败"
        else:
            plate_text = "未检测到车牌"

        # 在原图上绘制检测到的车牌框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 使用 add_chinese_text 添加中文车牌信息
        frame = add_chinese_text(frame, f"车牌: {plate_text}", (x1, y1 - 30), textColor=(0, 255, 0), textSize=30)

    cv2.imshow("车牌识别", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
