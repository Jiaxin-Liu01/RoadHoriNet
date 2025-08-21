# 0.地平线检测介绍
高速公路广泛部署的监控相机因视角受限，难以实现大范围连续感知。跨相机鸟瞰图视角（Bird’s Eye View, BEV）的道路几何对齐可以提升场景一致性和完整性，受限于图像不一致与结构错位影响，导致几何对齐困难。地平线作为全局几何先验可统一视角差异，然而其隐式表示特性导致检测过程易受干扰，检测难度大。为此，提出地平线检测网络RoadHoriNet，面向复杂交通场景提升地平线提取的精度与鲁棒性。本方法通过透视变换与包围框裁剪进行数据增强，引入钻石空间缓解消失点学习不稳定问题，并结合感受野注意力卷积与动态上采样提升特征表达与重建精度。同时，设计几何一致性损失加强地平线检测的方向与位置约束。最终，检测结果作为几何先验用于相机姿态归一化与坐标统一，实现高精度的跨视角BEV图像对齐。在公开数据集BrnoCompSpeed上进行的实验证明，RoadHoriNet在像素误差（5.166%）和角度误差（0.0325°）方面优于现有方法，检测精度达94.834%；在跨相机BEV视角的道路几何对齐任务中，整体对齐精度高达99.129%，验证了该方法在交通环境中的实用性与推广潜力。
# 1.前期准备
# 1.1 库  
本网络以yolo11为基础进行改进，因此相关库和yolo11一致。
# 1.1 数据集
本项目的训练数据集为公开数据集BoxCars116k，包含4个pkl文件和18个场景的json文件。  
请将pkl文件放置于`horizenLine/datasets/BoxCars116k`文件夹，json文件夹放置于`horizenLine/datasets`，形成`horizenLine/datasets/json_data/...`
# 1.2 数据预处理
（1）运行`RoadHoriNet/horizenLine/datasets/dimondMap.py`，将消失点标签进行钻石空间参数化   
（2）运行`horizenLine/train/createYaml.py` 准备待送入网络的单张车辆图像和对应消失点标签，生成yaml文件
# 2.运行
# 2.1 训练
运行`horizenLine/train/trainYolo11.py`，生成权重文件，保存于`horizenLine/train/runs/pose`目录下。
## 2.2 预测
修改`horizenLine/predict/predictYolo11.py`的相关路径：    
    `hl_py_path：指向训练得到的的权重`  
    `file_path：指向待预测地平线的图像`  
    `json_path：指向BoxCares的json文件`  
运行该py文件，结果生成在horizenLine/predict/runs文件夹。
