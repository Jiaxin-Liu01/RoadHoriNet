import os
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO("yolov11n-pose.yaml")

    dataset_dir = '../datasets/BoxCars116k'
    # 开始训练
    # results = model.train(data=os.path.join(dataset_dir, 'my_data.yaml'), epochs=50, imgsz=128)
    # results = model.train(data=os.path.join(dataset_dir, 'my_data_nohm.yaml'), epochs=100, imgsz=128,batch=128,optimizer='AdamW',weight_decay=0.0005,patience=10,workers=16)
    results = model.train(data=os.path.join(dataset_dir, 'my_data_nohm.yaml'), epochs=50, imgsz=128,batch=64,lr0=0.001,lrf=0.001,optimizer='AdamW',weight_decay=0.0001,workers=16)
