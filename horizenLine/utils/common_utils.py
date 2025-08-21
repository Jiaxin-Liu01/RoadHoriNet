import torch
import json
import os
import cv2 as cv
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class NpEncoder(json.JSONEncoder):
    """继承json.JSONEncoder，自定义序列化方法(可打包numpy格式数据)"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def vis_train_samples(imgs, heatmaps, scales):
    if not os.path.exists("batch_samples"):
        os.makedirs("batch_samples")

    imgs = imgs.cpu().detach().numpy()
    heatmaps = heatmaps.cpu().detach().numpy()
    for num in range(len(imgs)):
        plt.figure()
        raw_img = imgs[num]
        raw_img = np.uint8(255 * raw_img).transpose(1, 2, 0)
        pil_raw_img = Image.fromarray(raw_img)
        # img
        plt.subplot2grid((2, 6), (0, 0), colspan=2, rowspan=2)  # 划分网格绘制，适用于一个图占用多行多列的情况
        plt.title("trans_img", y=-0.1)  # 显示标题在图下方
        plt.imshow(pil_raw_img)
        pil_raw_img.save("batch_samples/batch_sample_%s.png" % str(num))
        # heatmap
        raw_heatmap = heatmaps[num]
        raw_heatmap = raw_heatmap.transpose(1, 2, 0)
        for vp_idx in range(2):
            for scale_idx, scale in enumerate(scales):
                idx = len(scales) * vp_idx + scale_idx
                cv_heatmap = cv.applyColorMap(
                    np.uint8(255 * raw_heatmap[:, :, idx] / np.max(raw_heatmap[:, :, idx])),
                    cv.COLORMAP_PARULA)
                cv.imwrite("batch_samples/batch_sample_{}_heatmap_vp{}_s{}.png".format(num, vp_idx + 1, scale),
                           cv_heatmap)
                pil_heatmap = Image.fromarray(cv.cvtColor(cv_heatmap, cv.COLOR_BGR2RGB))
                # 偏移后的x序号:[0, 1], y序号:[2, 3, 4, 5]
                # idx: 0~7, scale_idx: 0~3
                plt.subplot2grid((2, 6), (idx // len(scales), scale_idx + 2), colspan=1, rowspan=1)
                plt.title("vp:{} scale:{}".format(vp_idx + 1, scale), y=-0.3)
                plt.imshow(pil_heatmap)
    plt.show()


def read_class(file):
    """ 读入目标检测类型文件，返回类别数组 """
    with open(file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def calculate_line_kb(point_1, point_2):
    difference1 = point_2[1] - point_1[1]
    difference2 = point_2[0] - point_1[0]
    k = difference1 / difference2
    b1 = point_1[1] - k * point_1[0]
    b2 = point_2[1] - k * point_2[0]
    b = np.nanmedian(np.array([b1, b2]))
    return k, b


def draw_horizon_line(hl_canvas, h, w, frame_pred_k, frame_pred_b, eval_record=False, gt_k=None, gt_b=None):
    # 绘图
    try:
        pred_hl_start_point = (0, int(frame_pred_b) + h // 2)
        pred_hl_end_point = (w, int(frame_pred_k * w + frame_pred_b) + h // 2)
        hl_canvas = cv.line(hl_canvas, pred_hl_start_point, pred_hl_end_point, (0, 0, 255), 2)  # 预测红色
    except:
        ...
    if eval_record and gt_k and gt_b:
        gt_hl_start_point = (0, int(gt_b) + h // 2)
        gt_hl_end_point = (w, int(gt_k * w + gt_b) + h // 2)
        hl_canvas = cv.line(hl_canvas, gt_hl_start_point, gt_hl_end_point, (0, 255, 0), 2)  # 真实绿色
    return hl_canvas
# def draw_horizon_line(hl_canvas, h, w, frame_pred_k, frame_pred_b, eval_record=False, gt_k=None, gt_b=None):
#     # 绘图
#     try:
#         pred_hl_start_point = (0, gt_b)
#         pred_hl_end_point = (w, gt_k * w + gt_b)
#
#         gt_hl_start_point = (0, frame_pred_b)
#         gt_hl_end_point = (w, frame_pred_k * w + frame_pred_b)
#
#         max_distance = max(abs(pred_hl_start_point[1] - gt_hl_start_point[1]),
#                            abs(pred_hl_end_point[1] - gt_hl_end_point[1]))
#         ratio = max_distance / h - 0.3 * h
#         frame_pred_b+=ratio
#         pred_hl_sp = (0, int(frame_pred_b) + h // 2)
#         pred_hl_ep = (w, int(frame_pred_k * w + frame_pred_b) + h // 2)
#
#         gt_hl_sp = (0, int(gt_b) + h // 2)
#         gt_hl_ep = (w, int(gt_k * w + gt_b) + h // 2)
#
#         hl_canvas = cv.line(hl_canvas, pred_hl_sp, pred_hl_ep, (0, 0, 255), 2)  # 预测红色
#         hl_canvas = cv.line(hl_canvas, gt_hl_sp, gt_hl_ep, (0, 255, 0), 2)  # 真实绿色
#     except:
#         ...
#     return hl_canvas


def calculate_hl_error(h, w, frame_pred_k, frame_pred_b, gt_k, gt_b):
    # 计算每张图片的地平线误差
    pred_hl_start_point = (0, gt_b)
    pred_hl_end_point = (w, gt_k * w + gt_b)

    gt_hl_start_point = (0, frame_pred_b)
    gt_hl_end_point = (w, frame_pred_k * w + frame_pred_b)
    # h维度上的w=0的差和w=w的差
    max_distance = max(abs(pred_hl_start_point[1]-gt_hl_start_point[1]), abs(pred_hl_end_point[1]-gt_hl_end_point[1]))
    rate1=(max_distance*0.1+abs(gt_k-frame_pred_k)*1000*0.9)/h
    rate2 = (abs(gt_k - frame_pred_k) * 1000) / h
    return max_distance, max_distance / h# 真实误差，相对误差
    # return max_distance, rate2

def calculate_hl_auc_results(frame_count, max_dis_ratioes):
    auc_x_axes_list = np.arange(0.1, 1.1, 0.1)
    auc_results_list = np.zeros((10,))
    for i in range(0, 10):
        for j in max_dis_ratioes:
            if j <= (i + 1) * 0.1:
                auc_results_list[i] += 1
    # 根据dict绘制曲线图
    auc_figure = plt.figure(dpi=200)

    plt.bar(auc_x_axes_list, auc_results_list, width=0.06)
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0, len(max_dis_ratioes), frame_count//10))
    plt.xlabel("error ratio/%")
    plt.ylabel("frame num")

    # 将两个list结果合并为dict
    auc_results_dict = dict(zip([str(x) for x in np.array(auc_x_axes_list, dtype=np.float32)], np.array(auc_results_list, dtype=np.int32)))
    print(auc_results_dict)

    return auc_results_dict



