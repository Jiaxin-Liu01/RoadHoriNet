import torch

from horizenLine.datasets.dimondMap import HeatmapBoxCarsDataset

import os
import yaml
import cv2
import numpy as np
import torchvision
from torch.utils import data

def heatmap_to_yolo_format(heatmaps, img_shape, scale, max_labels_per_heatmap=2):
    h, w = img_shape
    yolo_labels = []
    threshold = 0.5
    class_id = 0
    yolo_labels.append(f"{class_id} {w/w/2} {h/h/2} {w/w} {h/h}")
    for i in range(heatmaps.shape[0]):
        class_heatmap = heatmaps[i]
        y_indices, x_indices = np.where(class_heatmap > threshold)

        # 如果超过阈值的点少于2个，直接使用这些点
        if len(y_indices) == 0:
            x_center=x_indices/w
            y_center=y_indices/w
            continue

        x_center = np.mean(x_indices) / w
        y_center = np.mean(y_indices) / h
        vis = 2

        yolo_labels.append(f"{x_center} {y_center} {vis}")
    label = ' '.join(yolo_labels)
    return label


def load_data(dataset_dir, batch_size, input_size, heatmap_out, scales, peak_original,
              crop_delta, perspective_sigma, num_workers, processing_tag):
    train_dataset = HeatmapBoxCarsDataset(dataset_dir, 'train', processing_tag=processing_tag,
                                          img_size=input_size, heatmap_size=heatmap_out, scales=scales,
                                          peak_original=peak_original, crop_delta=crop_delta,
                                          perspective_sigma=perspective_sigma)

    train_data = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataset = HeatmapBoxCarsDataset(dataset_dir, 'val', processing_tag=processing_tag,
                                        img_size=input_size, heatmap_size=heatmap_out, scales=scales,
                                        peak_original=peak_original, crop_delta=crop_delta,
                                        perspective_sigma=perspective_sigma)
    val_data = data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


    class_id = 0
    vis=2
    train_path = os.path.join(dataset_dir, 'train_nohm')
    os.makedirs(os.path.join(train_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_path, 'labels'), exist_ok=True)
    for i, (imgs, vp_diamonded) in enumerate(train_data):
        for j in range(len(imgs)):
            yolo_labels = []
            img_path = os.path.join(train_path, 'images', f'image_{i * batch_size + j}.jpg')
            torchvision.utils.save_image(imgs[j], img_path)
            vp1_diamonded = vp_diamonded[0][j].cpu().numpy().tolist()
            vp2_diamonded = vp_diamonded[1][j].cpu().numpy().tolist()
            yolo_labels.append(f"{class_id} {0.5} {0.5} {1.0} {1.0} ")
            yolo_labels.append(
                f"{vp1_diamonded[0]} {vp1_diamonded[1]} {vis} {vp2_diamonded[0]} {vp2_diamonded[1]} {vis}")
            label = ' '.join(yolo_labels)
            # 保存YOLO格式的标签到文件中
            label_path = os.path.join(train_path, 'labels', f'image_{i * batch_size + j}.txt')
            with open(label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(label)
    val_path = os.path.join(dataset_dir, 'val_nohm')
    os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)
    for i, (imgs, vp_diamonded) in enumerate(val_data):
        for j in range(len(imgs)):
            # print(len(imgs))
            yolo_labels = []
            img_path = os.path.join(val_path, 'images', f'image_{i * batch_size + j}.jpg')
            torchvision.utils.save_image(imgs[j], img_path)
            vp1_diamonded = vp_diamonded[0][j].cpu().numpy().tolist()
            vp2_diamonded = vp_diamonded[1][j].cpu().numpy().tolist()
            yolo_labels.append(f"{class_id} {0.5} {0.5} {1.0} {1.0} ")
            yolo_labels.append(f"{vp1_diamonded[0]} {vp1_diamonded[1]} {vis} {vp2_diamonded[0]} {vp2_diamonded[1]} {vis}")
            label = ' '.join(yolo_labels)
            # 保存YOLO格式的标签到文件中
            label_path = os.path.join(val_path, 'labels', f'image_{i * batch_size + j}.txt')
            with open(label_path, 'w') as f:
                for label in yolo_labels:
                    f.write(label)
    # 创建 YAML 配置文件
    yaml_data = {
        'train': os.path.join(dataset_dir, 'train_nohm', 'images'),
        'val': os.path.join(dataset_dir, 'val_nohm', 'images'),
        'kpt_shape': [2, 3],
        'names': '0'  # 类别名称，可以根据实际情况修改
    }
    # 保存为my_data.yaml文件

    yaml_path=os.path.join(dataset_dir,'my_data_nohm.yaml')
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as outfile:
        yaml.dump(yaml_data, outfile)
    return


if __name__ == '__main__':
    # ------------------------------ 训练参数 ------------------------------ #
    # 模型参数
    # 图片输入大小
    input_size = 128
    # 输出热图大小
    heatmap_out = 64
    # 多尺度
    # scales = [0.03, 0.1, 0.3, 1.0]
    scales = [1]
    # 数据增广参数1
    perspective_sigma = 25.0
    # 数据增广参数1
    crop_delta = 10
    # 是否在原始空间构建峰值
    peak_original = False
    # stacks
    num_stacks = 4
    # channels
    channels = 256

    # 数据集参数
    # 数据集文件夹路径
    dataset_dir = '../datasets/BoxCars116k'
    # 训练数据可视化
    vis_train_sample_flag = False
    # 设置num worker
    num_workers = 4
    # 0:原方法 2:修改2D框裁剪的范围
    processing_tag = 2

    Freeze_batch_size = 8
    _ = load_data(dataset_dir, Freeze_batch_size, input_size, heatmap_out, scales, peak_original, crop_delta,
                  perspective_sigma, num_workers, processing_tag)
