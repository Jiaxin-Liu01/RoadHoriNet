# heatmap数据集
import os
import pickle
import cv2 as cv
import numpy as np
from torch.utils import data

from horizenLine.utils.diamond_space import vp_to_heatmap, heatmap_to_orig,diamond_coords_from_original


class GenerateHeatmap:
    def __init__(self, output_res, scales):
        self.output_res = output_res
        self.scales = scales
        self.sigma = self.output_res / 64
        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

    def __call__(self, vps):
        hms = np.zeros(shape=(self.output_res, self.output_res, len(vps) * len(self.scales)), dtype=np.float32)
        for vp_idx, vp in enumerate(vps):
            for scale_idx, scale in enumerate(self.scales):
                idx = len(self.scales) * vp_idx + scale_idx

                vp_heatmap = vp_to_heatmap(vp, self.output_res, scale=scale)

                # vp_heatmap = (vp_diamond + 0.5) * self.output_res
                # vp_heatmap = ((self.R @ vp_diamond.T)) * (np.sqrt(2) / 2 * self.output_res) + self.output_res / 2
                # self.R = np.array([[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]])

                x, y = int(vp_heatmap[0]), int(vp_heatmap[1])
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue

                ul = int(y - 3 * self.sigma - 1), int(x - 3 * self.sigma - 1)
                br = int(y + 3 * self.sigma + 2), int(x + 3 * self.sigma + 2)
                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]
                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)

                hms[aa:bb, cc:dd, idx] = np.maximum(hms[aa:bb, cc:dd, idx], self.g[a:b, c:d])
        return hms


class HeatmapBoxCarsDataset(data.Dataset):
    def __init__(self, path, split, processing_tag=0, img_size=128, heatmap_size=64,
                 scales=(0.03, 0.1, 0.3, 1), peak_original=False, perspective_sigma=25.0, crop_delta=10):
        """
        :param path: 'datasets/BoxCars116k'
        :param split: train/val
        :param img_size: 128
        :param heatmap_size:64
        :param scales: 增广参数
        :param peak_original:
        :param perspective_sigma: 透视变换角度
        :param crop_delta: 随即裁剪像素阈值，裁剪范围±10
        将关键点的位置转换为热力图
        """
        """
        # atlas.pkl
        要解码图像（以RGB通道顺序），请使用以下语句。
        atlas = load_cache(path_to_atlas_file)
        image = cv2.cvtColor(cv2.imdecode(atlas[vehicle_id][instance_id], 1), cv2.COLOR_BGR2RGB)

        # dataset.pkl
        cameras: information about used cameras (vanishing points, principal point)
        samples: list of vehicles (index correspons to vehicle id).
        """
        with open(os.path.join(path, 'dataset.pkl'), 'rb') as f:
            self.data = pickle.load(f, encoding="latin-1", fix_imports=True)

        with open(os.path.join(path, 'atlas.pkl'), 'rb') as f:
            self.atlas = pickle.load(f, encoding="latin-1", fix_imports=True)

        self.split = split
        self.img_dir = os.path.join(path, 'images')

        self.img_size = img_size
        self.heatmap_size = heatmap_size

        self.scales = scales

        self.processing_tag = processing_tag

        # if peak_original:
        #     self.orig_coord_heatmaps = []
        #     for scale in scales:
        #         orig_coord_heatmap = heatmap_to_orig(heatmap_size, scale=scale)
        #         # make nans inf to calc inf distance
        #         orig_coord_heatmap[np.isnan(orig_coord_heatmap)] = np.inf
        #         self.orig_coord_heatmaps.append(orig_coord_heatmap)
        # else:
        #     self.generate_heatmaps = GenerateHeatmap(self.heatmap_size, self.scales)

        self.perspective_sigma = perspective_sigma
        self.crop_delta = crop_delta
        self.aug = perspective_sigma > 0 or crop_delta > 0

        self.instance_list = []

        # generate split every tenth sample is validation - remove useless samples from atlas
        # 8:1:1的sample划分给train、val、test，并且把atlas对应处理。单个sample中的所有instances都给了instance_list
        for s_idx, sample in enumerate(self.data['samples']):
            if s_idx % 10 == 0:
                if self.split != 'val':
                    self.atlas[s_idx] = None
                else:
                    for i_idx, instance in enumerate(sample['instances']):
                        self.instance_list.append((s_idx, i_idx))
            elif s_idx % 10 == 1:
                if self.split != 'test':
                    self.atlas[s_idx] = None
                else:
                    for i_idx, instance in enumerate(sample['instances']):
                        self.instance_list.append((s_idx, i_idx))
            else:
                if self.split != 'train':
                    self.atlas[s_idx] = None
                else:
                    for i_idx, instance in enumerate(sample['instances']):
                        self.instance_list.append((s_idx, i_idx))

        self.idxs = np.arange(len(self.instance_list))
        if self.split == 'train':# 意义不大但是不影响
            np.random.shuffle(self.idxs)

    def __getitem__(self, idx):
        try:
            actual_idxs = self.idxs[idx]
            img, vp_diamonded = self.get_single_item(actual_idxs)
            img = cv.cvtColor(img.astype(np.float32), cv.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            raise e
        return np.array(img).transpose([2, 0, 1]), vp_diamonded

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.instance_list)

    # for generate_item-->get_single_item-->__getitem__
    def equal_ratio_with_gray(self, image, img_size):
        """
        for generate_item
        把image的长边缩放至img_size大小，短边按比例缩放，最终输出img_size*img_size的图
        无像素部分用灰色像素填充
        :param image: 输入图片
        :param img_size: 要求输出图片大小
        :return:
        """
        h, w, c = image.shape
        if h < w:
            image_ratio = img_size / w
            # 在4x4像素的邻域上进行二元插值
            warp_image = cv.resize(image, (img_size, int(h * image_ratio)), cv.INTER_CUBIC)
            gray_image = np.zeros((warp_image.shape[1] - warp_image.shape[0], img_size, 3), dtype=np.uint8)
            gray_image[gray_image == 0] = 127
            image_final = np.concatenate([warp_image, gray_image], axis=0)
        else:
            image_ratio = img_size / h
            # 在4x4像素的邻域上进行二元插值
            warp_image = cv.resize(image, (int(w * image_ratio), img_size), cv.INTER_CUBIC)
            gray_image = np.zeros((img_size, warp_image.shape[0] - warp_image.shape[1], 3), dtype=np.uint8)
            gray_image[gray_image == 0] = 127
            image_final = np.concatenate([warp_image, gray_image], axis=1)

        return image_final

    # for __getitem__
    def get_single_item(self, index):
        s_idx, i_idx = self.instance_list[index]

        sample = self.data['samples'][s_idx]
        instance = sample['instances'][i_idx]
        camera = self.data['cameras'][sample['camera']]

        # 获取3D框
        bbox = instance['3DBB'] - instance['3DBB_offset']

        vp1 = camera['vp1'] - instance['3DBB_offset']
        vp2 = camera['vp2'] - instance['3DBB_offset']
        # vp3 = camera['vp3'] - instance['3DBB_offset']

        img = cv.imdecode(self.atlas[s_idx][i_idx], 1)

        return self.generate_item(img, bbox, vp1, vp2)

    # for generate_item-->get_single_item-->__getitem__
    def random_perspective_transform(self, img, bbox, vp1, vp2, force_no_perspective=False):

        rect_src = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]],
                            dtype=np.float32)
        if force_no_perspective:
            rect_dst = rect_src
        else:
            rect_dst = rect_src + self.perspective_sigma * np.random.randn(*rect_src.shape)
        rect_dst = 2.0 * rect_dst.astype(np.float32)  # + self.perspective_sigma * 4

        if np.random.rand() > 0.5:
            rect_src = np.array([[img.shape[1], 0], [0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]]],
                                dtype=np.float32)

        rect_dst[:, 0] -= np.min(rect_dst[:, 0]) - self.crop_delta
        rect_dst[:, 1] -= np.min(rect_dst[:, 1]) - self.crop_delta

        M = cv.getPerspectiveTransform(rect_src[:, :], rect_dst[:, :])

        bbox_warped = cv.perspectiveTransform(bbox[:, np.newaxis, :], M)

        max_x = min(int(np.max(bbox_warped[:, 0, 0])) + self.crop_delta, 900)
        max_y = min(int(np.max(bbox_warped[:, 0, 1])) + self.crop_delta, 900)

        img_warped = cv.warpPerspective(img, M, (max_x, max_y), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        vp1_warped = cv.perspectiveTransform(vp1[np.newaxis, np.newaxis, :], M)
        vp2_warped = cv.perspectiveTransform(vp2[np.newaxis, np.newaxis, :], M)

        # print("Original Bounding Box Coordinates:")
        # print(bbox)
        # print("Transformed Bounding Box Coordinates:")
        # print(bbox_warped[:, 0, :])
        return img_warped, bbox_warped[:, 0, :], vp1_warped[0, 0], vp2_warped[0, 0]

    # for init
    def generate_heatmaps(self, vps):
        heatmaps = np.empty([self.heatmap_size, self.heatmap_size, len(vps) * len(self.orig_coord_heatmaps)])
        for i, vp in enumerate(vps):
            for j, orig_coord_heatmap in enumerate(self.orig_coord_heatmaps):
                # exp(- 1/2 * (d/sigma)**2)
                d_sqr = np.sum((orig_coord_heatmap - vp[np.newaxis, np.newaxis, :]) ** 2, axis=-1)

                vp_heatmap_int = np.round(vp_to_heatmap(vp, self.heatmap_size, scale=self.scales[j])).astype(np.int)
                i_min, i_max = max(vp_heatmap_int[0] - 3, 0), min(vp_heatmap_int[0] + 4, self.heatmap_size)
                j_min, j_max = max(vp_heatmap_int[1] - 3, 0), min(vp_heatmap_int[1] + 4, self.heatmap_size)

                sigma_dist_sqr = np.nanpercentile(d_sqr[i_min: i_max, j_min: j_max], 20)
                neg_half_times_inv_sigma_sqr = 1.0 / sigma_dist_sqr
                h = np.exp(- d_sqr * neg_half_times_inv_sigma_sqr)

                heatmaps[:, :, i * len(self.orig_coord_heatmaps) + j] = h / (np.sum(h) + 1e-8)

        return heatmaps

    # for get_single_item-->__getitem__
    def generate_item(self, img, bbox, vp1, vp2):
        """
        :param img: 车辆图片
        :param bbox: 3D框
        :param vp1: 消失点1
        :param vp2: 消失点2
        :return:
        """
        tries = 0

        while True:

            if tries < 4 and self.split == 'train' and self.aug:
                # 返回b图(透视变换图)，透视变换后的3D框，以及透视变化后的消失点1，2
                warped_img, warped_bbox, warped_vp1, warped_vp2 = self.random_perspective_transform(img, bbox, vp1, vp2)
                # 从3D框切2D框
                x_min = int(
                    max(np.floor(np.min(warped_bbox[:, 0])) + np.random.randint(-self.crop_delta, self.crop_delta), 0))
                x_max = int(
                    min(np.ceil(np.max(warped_bbox[:, 0])) + np.random.randint(-self.crop_delta, self.crop_delta),
                        warped_img.shape[1]))
                y_min = int(
                    max(np.floor(np.min(warped_bbox[:, 1])) + np.random.randint(-self.crop_delta, self.crop_delta), 0))
                y_max = int(
                    min(np.ceil(np.max(warped_bbox[:, 1])) + np.random.randint(-self.crop_delta, self.crop_delta),
                        warped_img.shape[0]))

            else:
                # warped_img, warped_bbox, warped_vp1, warped_vp2 = self.random_perspective_transform(img, bbox, vp1, vp2, force_no_perspective=True)
                warped_img, warped_bbox, warped_vp1, warped_vp2 = img, bbox, vp1, vp2
                # 从3D框切2D框 np.floor向下取整 np.ceil向上取整
                x_min = int(max(np.floor(np.min(warped_bbox[:, 0])), 0))
                x_max = int(min(np.ceil(np.max(warped_bbox[:, 0])), warped_img.shape[1]))
                y_min = int(max(np.floor(np.min(warped_bbox[:, 1])), 0))
                y_max = int(min(np.ceil(np.max(warped_bbox[:, 1])), warped_img.shape[0]))
                warped_img = warped_img[max(y_min, 0): y_max, max(x_min, 0): x_max, :]
                break

            if y_min + 25 < min(y_max, warped_img.shape[0]) and x_min + 25 < min(x_max, warped_img.shape[1]):
                if self.processing_tag == 2:
                    # 数据预处理2、修改裁剪的范围，要求裁剪后图片长宽1:1，然后resize到128*128
                    # 数据预处理2.1、固定长边，将短边延长至和长边一致
                    if (x_max - x_min) > (y_max - y_min):
                        y_max = y_min + (x_max - x_min)
                    else:
                        x_max = x_min + (y_max - y_min)

                    # # 数据预处理2.2、固定短边，将长边缩小至和短边一致
                    # if (x_max-x_min) > (y_max-y_min):
                    #     x_max = x_min + (y_max - y_min)
                    # else:
                    #     y_max = y_min + (x_max - x_min)

                    # print(x_max-x_min, y_max-y_min)

                # 数据预处理2.0、不做处理，按x,y长宽裁剪
                warped_img = warped_img[max(y_min, 0): y_max, max(x_min, 0): x_max, :]
                break

            tries += 1

        # # # 这里使用数据增强1，或数据增强2。数据增强1配套原始实验使用；数据增强2配套直接resize使用
        if self.processing_tag == 1:
            # 数据增强1、返回带灰度条的车辆图片
            warped_img = self.equal_ratio_with_gray(warped_img, self.img_size)

        # 直接resize的方式获取128*128的图像(最开始)
        warped_img = cv.resize(warped_img, (self.img_size, self.img_size), interpolation=cv.INTER_CUBIC)

        # 将vp转换为新的坐标系，其中img的左上角是（-1, -1），右下角是（1, 1）。(图像原点从左上转换至中间)
        warped_vp1 -= np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
        warped_vp2 -= np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
        warped_vp1[0] /= (x_max - x_min) / 2.0
        warped_vp2[0] /= (x_max - x_min) / 2.0
        warped_vp1[1] /= (y_max - y_min) / 2.0
        warped_vp2[1] /= (y_max - y_min) / 2.0

        # print("vp1: ", warped_vp1)
        # print("vp2: ", warped_vp2)
        #
        # print("vp1 diamond:",  diamond_coords_from_original(warped_vp1, 1.0))
        # print("vp2 diamond:",  diamond_coords_from_original(warped_vp2, 1.0))

        # 生成热图。self.generate_heatmaps是由Generate类，call函数传入([warped_vp1, warped_vp2])生成的，返给heatmaps
        # ---------zhaochunhui----
        # heatmaps = self.generate_heatmaps([warped_vp1, warped_vp2]) #zhaochunhui
        # out_img = warped_img / 255
        # out_heatmaps = heatmaps
        warped_vp1_diamonded = diamond_coords_from_original(warped_vp1, 1.0)
        warped_vp2_diamonded = diamond_coords_from_original(warped_vp2, 1.0)
        # 归一化到 [0, 1] 范围
        warped_vp1_diamond = (warped_vp1_diamonded + 1) / 2.0
        warped_vp2_diamond = (warped_vp2_diamonded + 1) / 2.0
        out_img = warped_img / 255
        out_diamond=[warped_vp1_diamond,warped_vp2_diamond]

        # out_img = transforms.ToTensor()(out_img)
        # out_img = torch.from_numpy(out_img).float()
        # out_heatmap = torch.from_numpy(heatmap).float()



        return out_img, out_diamond


if __name__ == '__main__':
    # 热图数据集
    path = 'BoxCars116k/'

    scales = [0.03, 0.1, 0.3, 1.0]
    heatmap_out = 64
    peak_original = False
    orig_coord_heatmaps = []
    pre_processing_tag = 0

    for scale in scales:
        orig_coord_heatmap = heatmap_to_orig(heatmap_out, scale=scale)
        # make nans inf to calc inf distance
        # np.nan 返回判断是否是nan的bool类型
        # np.inf 返回判断是否是inf的bool类型
        orig_coord_heatmap[np.isnan(orig_coord_heatmap)] = 0
        orig_coord_heatmap[np.isinf(orig_coord_heatmap)] = 0
        orig_coord_heatmaps.append(orig_coord_heatmap)

    d = HeatmapBoxCarsDataset(path, 'val_nohm', processing_tag=pre_processing_tag, img_size=128, heatmap_size=heatmap_out,
                              scales=scales,
                              peak_original=peak_original, perspective_sigma=25.0, crop_delta=10)
    print("Dataset loaded with size: ", len(d.instance_list))

    # cum_heatmap = np.zeros([heatmap_out, heatmap_out, 2*len(scales)])
    # cum_heatmap_aug = np.zeros([heatmap_out, heatmap_out, 2*len(scales)])
    # cum_heatmap_noaug = np.zeros([heatmap_out, heatmap_out, 2*len(scales)])
    index=[1856]
    for idx in index:

        print("idx: ", idx)
        img, heatmap = d.get_single_item(idx)
        cv.imshow("vis/img_{}.png".format(idx), img)
        # 检查图像类型是否为 float64
        if img.dtype != np.uint8:
            # 将图像转换为 uint8，通常图像数据是浮点类型时在 [0, 1] 或者 [0, 255] 范围内
            img = np.uint8(255 * (img - np.min(img)) / (np.max(img) - np.min(img)))
        # 现在可以安全地转换颜色空间
        if img.shape[-1] == 3:  # 确保图像是三通道
            img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        cv.imwrite("vis/img_{}.png".format(idx), img_bgr)
        print(img.shape)

        # 2(两个消失点)*4(四个scale)
        # for vp_idx in range(2):
        #     for scale_idx, scale in enumerate(scales):
        #         idx = len(scales) * vp_idx + scale_idx
        #         cv.imshow("vis/heatmap_{}_vp{}_s{}.png".format(idx, vp_idx + 1, scale),
        #                   cv.applyColorMap(np.uint8(255 * heatmap[:, :, idx] / np.max(heatmap[:, :, idx])),
        #                                    cv.COLORMAP_PARULA))
        #         cv.waitKey(0)



 #  path = 'BoxCars116k/'
 #
 #    scales = [0.03, 0.1, 0.3, 1.0]
 #    heatmap_out = 64
 #    # scales = [0.1, 1.0]
 #    peak_original = False
 #
 #    orig_coord_heatmaps = []
 #    for scale in scales:
 #        orig_coord_heatmap = heatmap_to_orig(heatmap_out, scale=scale)
 #        # make nans inf to calc inf distance
 #        # np.nan 返回判断是否是nan的bool类型
 #        # np.inf 返回判断是否是inf的bool类型
 #        orig_coord_heatmap[np.isnan(orig_coord_heatmap)] = 0
 #        orig_coord_heatmap[np.isinf(orig_coord_heatmap)] = 0
 #        orig_coord_heatmaps.append(orig_coord_heatmap)
 #
 #    d = HeatmapBoxCarsDataset(path, 'val', img_size=128, heatmap_size=heatmap_out, scales=scales,
 #                              peak_original=peak_original, perspective_sigma=0.0, crop_delta=0)
 #    print("Dataset loaded with size: ", len(d.instance_list))
 #
 #    # cum_heatmap = np.zeros([heatmap_out, heatmap_out, 2*len(scales)])
 #    # cum_heatmap_aug = np.zeros([heatmap_out, heatmap_out, 2*len(scales)])
 #    # cum_heatmap_noaug = np.zeros([heatmap_out, heatmap_out, 2*len(scales)])
 #
 #    for i in [1856, 3815, 3611]:
 #        print("idx: ", i)
 #
 #        img, heatmap = d.get_single_item(i)
 #
 #        # cv2.imwrite("vis/img_{}.png".format(i), img)
 #        cv2.imshow("vis/img_{}.png".format(i), img)
 #
 #        # 3(三张图片)*2(两个消失点)*4(四个scale)
 #        for vp_idx in range(2):
 #            for scale_idx, scale in enumerate(scales):
 #                idx = len(scales) * vp_idx + scale_idx
 #                # cv2.imwrite("vis/heatmap_{}_vp{}_s{}.png".format(i, vp_idx + 1, scale), cv2.applyColorMap(np.uint8(255 * heatmap[:, :, idx] / np.max(heatmap[:, :, idx])), cv2.COLORMAP_PARULA))
 #                cv2.imshow("vis/heatmap_{}_vp{}_s{}.png".format(i, vp_idx + 1, scale),
 #                           cv2.applyColorMap(np.uint8(255 * heatmap[:, :, idx] / np.max(heatmap[:, :, idx])),
 #                                             cv2.COLORMAP_PARULA))
 #                cv2.waitKey(0)
# import pprint
#
# if isinstance(d.data, dict):
#     print(list(d.data.items())[:10])  # 打印前10个键值对
# elif isinstance(d.data, list):
#     print(d.data[:10])  # 打印前10个元素
#
# print("-------------")
# if isinstance(d.atlas, dict):
#     print(list(d.atlas.items())[:10])  # 打印前10个键值对
# elif isinstance(d.atlas, list):
#     print(d.atlas[:10])
# [
#     [array([[255],
#        [216],
#        [255],
#        ...,
#        [ 15],
#        [255],
#        [217]], dtype=uint8),
#     array([[255],
#        [216],
#        [255],
#        ...,
#        [  3],
#        [255],
#        [217]], dtype=uint8),
#     array([[255],
#        [216],
#        [255],
#        ...,
#        [ 99],
#        [255],
#        [217]], dtype=uint8)],
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None,
#     None]