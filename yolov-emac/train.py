import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO(model=r'E:\yolov12-main\yolov12-main\ultralytics\cfg\models\v12\yolov12.yaml')
    #model = YOLO(model=r'E:\yolov12-main\yolov12-main\ultralytics\cfg\models\11\yolo11m.yaml')
    # model = YOLO(model=r'C:\YOLO-EMAC\yolov-emac\runs\train\exp\weights\best.pt')
    # model = YOLO(model=r'C:\YOLO-EMAC\yolov-emac\my.yaml')
    # model = YOLO(model=r'C:\YOLO-EMAC\yolov-emac\runs\train\YOLOv12_EMAC_SimAM\weights\last.pt')
    model = YOLO('yolov12n.pt')
    # model.load(r'C:\YOLO-EMAC\yolov-emac\yolov12n.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    #model.load('E:\yolov12-main\yolov12-main\\runs\\train\exp6\weights\\best.pt')
    #     model.train(data=r'C:\YOLO-EMAC\datasets\DeepPCB.v5i.yolov12\data.yaml',

    model.train(data=r'C:\YOLO-EMAC\datasets\HRIPCB_UPDATE\data.yaml',
                imgsz=640,
                epochs=150,
                batch=32,
                workers=4,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='YOLO_baseline',
                single_cls=False,
                amp=True,
                cache=True ,
                )
