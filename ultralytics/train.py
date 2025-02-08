import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r"./ultralytics/cfg/models/11/yolo11.yaml")
    model.train(data=r"data.yaml",
                imgsz=640,
                epochs=50,
                batch=4,
                workers=0,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=True,
                project='runs/train',
                name='exp',
                single_cls=True,
                cache=False,
                )
