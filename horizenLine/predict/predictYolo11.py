import json
import math
import os
from time import time

import cv2
import numpy as np
import torch
from horizenLine.utils.diamond_space import process_heatmaps, heatmap_to_vp, original_coords_from_diamond
from ultralytics import YOLO
from horizenLine.utils.common_utils import calculate_line_kb, draw_horizon_line, calculate_hl_error

# 一次就处理一个
if __name__ == '__main__':
    start_time=time()
    scales = [1.0]
    input_size = 128
    eval_record = True

    # hl_py_path = r'E:\liujiaxin\yolo11\horizenLine\train\runs\pose\yolo11\weights\best.pt'
    hl_py_path = r'E:\liujiaxin\yolo11\horizenLine\train\runs\pose\yolo11_Dysample_pl\weights\best.pt'
    # -----ljx for moredatasets
    # file_path = r'E:\liujiaxin\yolo11\horizenLine\datasets\BoxCars116k\predictDataset\l2.png'
    file_path = r'E:\liujiaxin\yolo11\horizenLine\datasets\BoxCars116k\predictDataset\l2.png'
    json_path = 'E:\liujiaxin\yolo11\horizenLine\datasets\json_data\session2_left\system_dubska_optimal_calib.json'
    # hl_py_path = r'E:\liujiaxin\yolo11\horizenLine\train\runs\pose\train2\weights\best.pt'

    # 计算真实的k和b
    gt_k, gt_b = None, None
    if eval_record:
        with open(json_path, "r") as anno_file:
            data = json.load(anno_file)
            gt_vp1 = np.array(data["camera_calibration"]["vp1"])
            gt_vp2 = np.array(data["camera_calibration"]["vp2"])
            gt_k, gt_b = calculate_line_kb(gt_vp1, gt_vp2)
            print("真实的k：", gt_k, " b:", gt_b)
            print("真实的vp:", gt_vp1, "  ", gt_vp2)

    # gt_k, gt_b=None, None
    # gt_vp1,gt_vp2=(0,401),(1920,437)
    # gt_k, gt_b = calculate_line_kb(gt_vp1, gt_vp2)
    print("真实的k：", gt_k, " b:", gt_b)
    print("真实的vp:", gt_vp1, "  ", gt_vp2)

    # 处理图片
    if (file_path.split('.')[-1] == 'jpg') or (file_path.split('.')[-1] == 'png') or (
            file_path.split('.')[-1] == 'bmp'):
        image = cv2.imread(file_path, 1)
        # image = image.astype(np.float32)
        h, w = image.shape[:2]
        print("image的高：", h, " 宽：", w)

        # 为了防止地平线在图像外进行的图像顶部扩充
        top_gray = np.full((h // 2, w, 3), 127, dtype=np.uint8)  # 填充灰色像素
        hl_canvas = np.concatenate([top_gray, image], axis=0)  # 把灰色图像放在image上面
        hl_canvas_copy = hl_canvas.copy()

        ks, bs = [], []
        vp1sx, vp1sy, vp2sx, vp2sy = [], [], [], []

        # -----------------------------目标检测-----------------------------#
        detect_model = YOLO("yolo11n.yaml").load("yolo11n.pt")
        result = detect_model.predict(file_path, conf=0.5, save_txt=True)  # return a list of Results objects
        txt_save_dir = result[0].save_dir

        print(f"The txt files are saved in: {txt_save_dir}")
        boxes = result[0].boxes  # 因为检测的单张图像，所以result[0]就行。result是列表
        start_time = time()
        for box in boxes:
            # box形式:tensor([[1,2,3,4]],device='cuda:0')
            # 原点在左上角，box是左上，右下
            image_copy = image.copy()
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = int(max(0, x_min)), int(max(0, y_min)), int(min(1920, x_max)), int(
                min(1080, y_max))
            box_center = np.array([x_min + x_max, y_min + y_max]) / 2
            box_scale = np.array([x_max - x_min, y_max - y_min]) / 2
            box_img = image_copy[y_min:y_max, x_min:x_max, :]
            try:
                warped_box_img = cv2.resize(box_img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
            except Exception as e:
                print(f"Error resizing image: {e}")  # 打印异常信息
                continue
            # cv2.imshow("box",warped_box_img)
            # cv2.waitKey(0)
            warped_box_img_data = np.expand_dims(np.transpose(np.array(warped_box_img, dtype="float32"), (2, 0, 1)), 0)

            with torch.no_grad():
                warped_box_img = torch.from_numpy(np.asarray(warped_box_img_data)).type(torch.FloatTensor)

                warped_box_img = warped_box_img.cuda()
                # -----------------------------低平线检测-----------------------------#
                model = YOLO(hl_py_path)
                total_params = sum(p.numel() for p in model.parameters())
                output_list = model(warped_box_img, visualize=True)
                # 获取 Results 对象中的 keypoints
                # 迭代每个图像的结果
                for idx, output in enumerate(output_list):
                    # 获取当前图像的 keypoints 数据
                    keypoints_data = output.keypoints  # 每个 output 对象包含当前图像的检测结果

                    if keypoints_data is not None:
                        keypoints_array = keypoints_data.xy  # 关键点坐标(基于钻石空间)

                        # 输出第一个和第二个关键点()
                        if keypoints_array.shape[1] >= 2:  # 检查是否有至少两个关键点
                            # 0-1范围
                            vp1_box = keypoints_array[0][0].cpu().numpy()  # 获取第一个检测对象的第一个关键点
                            vp2_box = keypoints_array[0][1].cpu().numpy()  # 获取第一个检测对象的第二个关键点
                            # print("网络输出的参数化的vp1:",vp1_box,",vp2:",vp2_box) # vp1:[74.82 58.95] ,vp2: [0 0]

                            # -----------消失点恢复（坐标转换）--------------#
                            # 热力图-》相对box
                            # vp1=heatmap_to_orig(vp1_b0x,64,scale=1.0)
                            #  去除热力图版本
                            # 恢复为-1~1区间
                            vp1_box /= input_size
                            vp2_box /= input_size
                            vp1 = vp1_box * 2 - 1
                            vp2 = vp2_box * 2 - 1
                            # 从钻石空间恢复出来
                            vp1 = original_coords_from_diamond(vp1, 1)
                            vp2 = original_coords_from_diamond(vp2, 1)

                            # 相对box-》相对于原始图像
                            vp1 = vp1 * box_scale + box_center
                            vp2 = vp2 * box_scale + box_center

                            vp1 = tuple(np.array(vp1, dtype=np.int32))
                            vp2 = tuple(np.array(vp2, dtype=np.int32))
                            box_center = tuple(np.array(box_center, dtype=np.int64))

                            vp1sx.append(vp1[0])
                            vp1sy.append(vp1[1])
                            vp2sx.append(vp2[0])
                            vp2sy.append(vp2[1])

                        else:
                            print(f"图像 {idx} 没有足够的关键点")
                    else:
                        print(f"图像 {idx} 没有检测到关键点")

        # # 计算单张图像的k和b
        frame_pred_vp1, frame_pred_vp2 = (np.nanmedian(vp1sx), np.nanmedian(vp1sy)), (
        np.nanmedian(vp2sx), np.nanmedian(vp2sy))
        print("frame_pred_vp1:{}, frame_pred_vp2:{}".format(frame_pred_vp1, frame_pred_vp2))
        frame_pred_k, frame_pred_b = calculate_line_kb(frame_pred_vp1, frame_pred_vp2)
        print("frame_pred_k:{}, frame_pred_b:{}".format(frame_pred_k, frame_pred_b))

        hl_canvas = draw_horizon_line(hl_canvas, h, w, frame_pred_k, frame_pred_b, eval_record, gt_k, gt_b)
        max_dis, max_dis_ratio = calculate_hl_error(h, w, frame_pred_k, frame_pred_b, gt_k, gt_b)
        theta = abs(math.atan(frame_pred_k) - math.atan(gt_k))

        end_time = time()  # 记录结束时间
        print("误差：max_dis:", max_dis, " max_dis_ratio:", abs(max_dis_ratio * 100), "%,theta的差:", theta)
        output_path = os.path.join(txt_save_dir, "hl_canvas.jpg")
        output_txt_path = os.path.join(txt_save_dir, "result.txt")
        content = (
            f"误差：max_dis:{max_dis}  max_dis_ratio:{max_dis_ratio * 100}%\n"
            f"---------------------------\n"
            f"使用的权重：{hl_py_path}\n"
            f"使用的图片：{file_path}\n"
            f"---------------------------\n"
            f"原始k：{gt_k}, 原始b：{gt_b}\n"
            f"预测k：{frame_pred_k}, 预测b：{frame_pred_b},theta:{theta}\n"
            f"---------------------------\n"
            f"预测的vp:{frame_pred_vp1}，  {frame_pred_vp2}\n"
            f"真实的vp: {gt_vp1},  {gt_vp2}\n"
            f"---------------------------\n"
            f"ms：{(end_time-start_time)/len(boxes)}\n"
            f"模型的总参数量: {total_params}"
        )

        # 使用 with 打开文件，并写入内容
        with open(output_txt_path, 'w', encoding='utf-8') as file:
            file.write(content)

        # 保存图像
        cv2.imwrite(output_path, hl_canvas)

        print("误差：max_dis:", max_dis, " max_dis_ratio:", abs(max_dis_ratio * 100), "%,theta的差:", theta,"平均time:",(end_time-start_time)/len(boxes),f"模型的总参数量: {total_params}")
    # 处理文件夹
    # else:
