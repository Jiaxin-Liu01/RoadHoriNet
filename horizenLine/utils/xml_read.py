# -*- coding: utf-8 -*-
import re

import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import parse

def read_calib_xml(path):
    # 打开XML文件并解析为ElementTree对象
    tree = ET.parse(path)
    # 获取根节点
    root = tree.getroot()
    # 获取标定信息
    root = list(root)
    f = float('%.11f' % float(root[0].text))
    fi = float('%.17f' % float(root[1].text))
    theta = float('%.15f' % float(root[2].text))
    h = float('%.8f' % float(root[3].text))
    vps_list = root[4].text.replace("\n", "").split(" ")
    VPS = list()
    for digit in vps_list:
        if len(digit) != 0:
            VPS.append(float('%.15f' % float(digit)))

    vps = [(VPS[0], VPS[1]), (VPS[2], VPS[3])]

    return f, fi, theta, h, vps


def read_vehicle_xml(path):
    image_size, vehicle_infos = None, list()
    domTree = parse(path)
    # 文档根元素
    rootNode = domTree.documentElement
    object_nodes = rootNode.getElementsByTagName("object")
    size_nodes = rootNode.getElementsByTagName("size")

    for size_node in size_nodes:
        width = float(size_node.getElementsByTagName("width")[0].childNodes[0].data)
        height = float(size_node.getElementsByTagName("height")[0].childNodes[0].data)
        depth = float(size_node.getElementsByTagName("depth")[0].childNodes[0].data)
        image_size = np.array([width, height, depth])

    # vehicle
    for object_node in object_nodes:
        vehicle_info = dict()
        vehicle_info["type"] = str(object_node.getElementsByTagName("type")[0].childNodes[0].data)
        vehicle_info["bbox2d"] = np.array(object_node.getElementsByTagName("bbox2d")[0].childNodes[0].data.split(" "), int)
        vehicle_info["vertex2d"] = np.array([eval(t) for t in re.findall(r'\(\d+,\s*\d+\)', object_node.getElementsByTagName("vertex2d")[0].childNodes[0].data)])
        vehicle_info["vehicle_size"] = np.array(object_node.getElementsByTagName("veh_size")[0].childNodes[0].data.split(" "), float)
        vehicle_info["perspective"] = str(object_node.getElementsByTagName("perspective")[0].childNodes[0].data)
        vehicle_info["base_point"] = np.array(object_node.getElementsByTagName("base_point")[0].childNodes[0].data.split(" "), int)
        vehicle_info["vertex3d"] = [eval(t) for t in re.findall(r'\(-?[\d\.]+,\s*-?[\d\.]+,\s*-?[\d\.]+\)', object_node.getElementsByTagName("vertex3d")[0].childNodes[0].data)]
        vehicle_info["vehicle_loc_2d"] = np.array(object_node.getElementsByTagName("veh_loc_2d")[0].childNodes[0].data.split(" "), int)

        vehicle_infos.append(vehicle_info)

    return image_size, vehicle_infos


if __name__ == '__main__':
    # 相机标定xml文件路径
    # calib_path = "../videos/vehicle_3D_eval_images/DATA2021/Calib/session0_center_calibParams.xml"
    # print(read_calib_xml(path=calib_path))
    # 车辆信息xml文件路径
    vehicle_path = "../videos/vehicle_3D_eval_images/DATA2021/Annotations/session0_center/session0_center_000000.xml"
    print(read_vehicle_xml(path=vehicle_path))