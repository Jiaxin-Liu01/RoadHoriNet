# 0.介绍
本网络以yolo11为基础进行改进，因此相关库和yolo11一致。
# 1.前期准备
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
