import math
import numpy as np
import cv2 as cv


class Point:
    """
    2D坐标点
    """

    def __init__(self, point):
        x, y = point
        self.X = x
        self.Y = y


class Line:
    def __init__(self, point1, point2):
        """
        初始化包含两个端点
        :param point1:
        :param point2:
        """
        self.Point1 = point1
        self.Point2 = point2


def GetAngle(line1, line2):
    """
    计算两条线段之间的夹角
    :param line1:
    :param line2:
    :return:
    """
    dx1 = line1.Point1.X - line1.Point2.X
    dy1 = line1.Point1.Y - line1.Point2.Y
    dx2 = line2.Point1.X - line2.Point2.X
    dy2 = line2.Point1.Y - line2.Point2.Y
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        insideAngle = abs(angle1 - angle2)
    else:
        insideAngle = abs(angle1) + abs(angle2)
        if insideAngle > 180:
            insideAngle = 360 - insideAngle
    insideAngle = insideAngle % 180
    return insideAngle


def line_intersection(points):
    # 计算直线交点
    point1, point2, point3, point4 = points
    k1 = (point1[1] - point2[1]) / (point1[0] - point2[0])  # 直线1
    b1 = point1[1] - k1 * point1[0]
    k2 = (point3[1] - point4[1]) / (point3[0] - point4[0])  # 直线2
    b2 = point3[1] - k2 * point3[0]

    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1

    return x, y


def find_center(points):
    """
    返回众多预测消失点的几何中心点
    :param points:
    :return:
    """
    n = len(points)
    x_center = sum(p[0] for p in points) / n
    y_center = sum(p[1] for p in points) / n
    return (x_center, y_center)


def geometric_center_point(points):
    # 返回n个点的几何中心点
    return points.mean(0)


def distance_between_two_points(points):
    """
    返回的是两个坐标距离的平方
    :param points:
    :return:
    """
    x1, y1 = points[0]
    x2, y2 = points[1]

    return pow((x1 - x2), 2) + pow((y1 - y2), 2)


def pixel2world(H, point, z):
    """
    2D点转为3D点
    :return:
    """
    u, v = point
    b1 = u * (H[2][2] * z + H[2][3]) - (H[0][2] * z + H[0][3])
    b2 = v * (H[2][2] * z + H[2][3]) - (H[1][2] * z + H[1][3])
    x = (b1 * (H[1][1] - H[2][1] * v) - b2 * (H[0][1] - H[2][1] * u)) / (
            (H[0][0] - H[2][0] * u) * (H[1][1] - H[2][1] * v) - (H[0][1] - H[2][1] * u) * (H[1][0] - H[2][0] * v))
    y = (-b1 * (H[1][0] - H[2][0] * v) + b2 * (H[0][0] - H[2][0] * u)) / (
            (H[0][0] - H[2][0] * u) * (H[1][1] - H[2][1] * v) - (H[0][1] - H[2][1] * u) * (H[1][0] - H[2][0] * v))

    return x, y, z


def world2pixel(H, point):
    """
    3D点转为2D点
    :return:
    """
    x, y, z = point
    u = (H[0][0] * x + H[0][1] * y + H[0][2] * z + H[0][3]) / (H[2][0] * x + H[2][1] * y + H[2][2] * z + H[2][3])
    v = (H[1][0] * x + H[1][1] * y + H[1][2] * z + H[1][3]) / (H[2][0] * x + H[2][1] * y + H[2][2] * z + H[2][3])

    return u, v


def world2image(view_tag, H, point_3D1, l, w, h):
    x_3D1, y_3D1, z_3D1 = point_3D1
    # 空间点世界坐标
    if view_tag == "left":
        point_3D0 = (x_3D1 + w, y_3D1, z_3D1)
        point_3D2 = (x_3D1, y_3D1 + l, z_3D1)
        point_3D3 = (x_3D1 + w, y_3D1 + l, z_3D1)
        point_3D4 = (x_3D1 + w, y_3D1, z_3D1 + h)
        point_3D5 = (x_3D1, y_3D1, z_3D1 + h)
        point_3D6 = (x_3D1, y_3D1 + l, z_3D1 + h)
        point_3D7 = (x_3D1 + w, y_3D1 + l, z_3D1 + h)
    else:
        point_3D0 = (x_3D1 - w, y_3D1, z_3D1)
        point_3D2 = (x_3D1, y_3D1 + l, z_3D1)
        point_3D3 = (x_3D1 - w, y_3D1 + l, z_3D1)
        point_3D4 = (x_3D1 - w, y_3D1, z_3D1 + h)
        point_3D5 = (x_3D1, y_3D1, z_3D1 + h)
        point_3D6 = (x_3D1, y_3D1 + l, z_3D1 + h)
        point_3D7 = (x_3D1 - w, y_3D1 + l, z_3D1 + h)

    # 八点的三维坐标转换回图像坐标
    point_2D0 = tuple(world2pixel(H, point_3D0))
    point_2D1 = tuple(world2pixel(H, point_3D1))
    point_2D2 = tuple(world2pixel(H, point_3D2))
    point_2D3 = tuple(world2pixel(H, point_3D3))
    point_2D4 = tuple(world2pixel(H, point_3D4))
    point_2D5 = tuple(world2pixel(H, point_3D5))
    point_2D6 = tuple(world2pixel(H, point_3D6))
    point_2D7 = tuple(world2pixel(H, point_3D7))

    return point_2D0, point_2D1, point_2D2, point_2D3, point_2D4, point_2D5, point_2D6, point_2D7


def cos_calculation_function(points):
    """
    计算vp约束中的余弦值
    :param points: point1, point2, vp
    :return:
    """
    point1, point2, point3 = points
    distance_1A2 = distance_between_two_points([point1, point2])
    distance_1Avp = distance_between_two_points([point1, point3])
    distance_2Avp = distance_between_two_points([point2, point3])
    cos_value = (distance_1A2 + distance_1Avp - distance_2Avp) / (2 * math.sqrt(distance_1A2 * distance_1Avp))

    return cos_value


def get_3D_points(H, points, car_h):
    point0, point1 = points[:2], points[2:]
    # 返回右下角点对应的三维空间坐标,基准点1
    x_tl, y_tl, z_tl = pixel2world(H, point0, car_h)
    x_lr, y_lr, z_lr = pixel2world(H, point1, 0)

    return (x_tl, y_tl, z_tl), (x_lr, y_lr, z_lr)


def visualization_3D_frames(image, point0, point1, point2, point3, point4, point5, point6, point7):
    image = cv.line(image, (int(point0[0]), int(point0[1])), (int(point1[0]), int(point1[1])), (0, 0, 0), 2, 2)
    image = cv.line(image, (int(point0[0]), int(point0[1])), (int(point3[0]), int(point3[1])), (255, 0, 0), 2, 2)
    image = cv.line(image, (int(point0[0]), int(point0[1])), (int(point4[0]), int(point4[1])), (0, 255, 0), 2, 2)
    image = cv.line(image, (int(point1[0]), int(point1[1])), (int(point5[0]), int(point5[1])), (0, 0, 255), 2, 2)
    image = cv.line(image, (int(point2[0]), int(point2[1])), (int(point1[0]), int(point1[1])), (255, 255, 0), 2, 2)
    image = cv.line(image, (int(point2[0]), int(point2[1])), (int(point3[0]), int(point3[1])), (0, 255, 255), 2, 2)
    image = cv.line(image, (int(point2[0]), int(point2[1])), (int(point6[0]), int(point6[1])), (255, 0, 255), 2, 2)
    image = cv.line(image, (int(point3[0]), int(point3[1])), (int(point7[0]), int(point7[1])), (255, 255, 255), 2, 2)
    image = cv.line(image, (int(point4[0]), int(point4[1])), (int(point5[0]), int(point5[1])), (0, 0, 0), 2, 2)
    image = cv.line(image, (int(point4[0]), int(point4[1])), (int(point7[0]), int(point7[1])), (255, 0, 0), 2, 2)
    image = cv.line(image, (int(point6[0]), int(point6[1])), (int(point5[0]), int(point5[1])), (0, 255, 0), 2, 2)
    image = cv.line(image, (int(point6[0]), int(point6[1])), (int(point7[0]), int(point7[1])), (0, 0, 255), 2, 2)
    cv.imshow("image", image)
    cv.waitKey(0)

    return image


def geometric_center_point_total(p3D_0_pixel, p3D_1_pixel, p3D_2_pixel, p3D_3_pixel, p3D_4_pixel, p3D_5_pixel,
                                 p3D_6_pixel, p3D_7_pixel):
    """
    返回几何中心点
    :param p3D_0_pixel:
    :param p3D_1_pixel:
    :param p3D_2_pixel:
    :param p3D_3_pixel:
    :param p3D_4_pixel:
    :param p3D_5_pixel:
    :param p3D_6_pixel:
    :param p3D_7_pixel:
    :return:
    """
    point1 = line_intersection([p3D_0_pixel, p3D_3_pixel, p3D_1_pixel, p3D_2_pixel])
    point2 = line_intersection([p3D_1_pixel, p3D_2_pixel, p3D_5_pixel, p3D_6_pixel])
    point3 = line_intersection([p3D_5_pixel, p3D_6_pixel, p3D_4_pixel, p3D_7_pixel])
    point4 = line_intersection([p3D_4_pixel, p3D_7_pixel, p3D_0_pixel, p3D_3_pixel])
    point5 = line_intersection([p3D_1_pixel, p3D_2_pixel, p3D_4_pixel, p3D_7_pixel])
    point6 = line_intersection([p3D_0_pixel, p3D_3_pixel, p3D_5_pixel, p3D_6_pixel])
    point7 = line_intersection([p3D_3_pixel, p3D_2_pixel, p3D_0_pixel, p3D_1_pixel])
    point8 = line_intersection([p3D_0_pixel, p3D_1_pixel, p3D_4_pixel, p3D_5_pixel])
    point9 = line_intersection([p3D_4_pixel, p3D_5_pixel, p3D_7_pixel, p3D_6_pixel])
    point10 = line_intersection([p3D_7_pixel, p3D_6_pixel, p3D_3_pixel, p3D_2_pixel])
    point11 = line_intersection([p3D_4_pixel, p3D_5_pixel, p3D_3_pixel, p3D_2_pixel])
    point12 = line_intersection([p3D_0_pixel, p3D_1_pixel, p3D_7_pixel, p3D_6_pixel])

    center1 = geometric_center_point(np.array([point1, point2, point3, point4, point5, point6]))
    center2 = geometric_center_point(np.array([point7, point8, point9, point10, point11, point12]))

    return center1, center2


def calib_param_to_matrix_basic(focal, fi, theta, h, pcx, pcy):
    """
    将标定参数转换为变换矩阵(世界坐标y轴沿道路方向)
    :param focal: 焦距
    :param fi: 俯仰角(rad)
    :param theta: 旋转角(rad)
    :param h: 相机高度(mm)
    :param pcx: 主点u
    :param pcy: 主点v
    :return: world -> image 变换矩阵
    """
    K = np.array([focal, 0, pcx, 0, focal, pcy, 0, 0, 1]).reshape(3, 3).astype(np.float)
    Rx = np.array([1, 0, 0, 0, -math.sin(fi), -math.cos(fi), 0, math.cos(fi), -math.sin(fi)]).reshape(3, 3).astype(
        np.float)
    Rz = np.array([math.cos(theta), -math.sin(theta), 0, math.sin(theta), math.cos(theta), 0, 0, 0, 1]).reshape(3,
                                                                                                                3).astype(
        np.float)
    R = np.dot(Rx, Rz)
    T = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -h]).reshape(3, 4).astype(np.float)
    trans = np.dot(R, T)
    H = np.dot(K, trans)
    return H, R, T, K


def cons_function(initial_value, H, box2D, perspective, vps):
    """
    车辆三维尺寸求解优化函数
    :param initial_value: 车辆初始三维
    :param H: 构造的标定矩阵
    :param box2D: 车辆二维框，四个点
    :param base_point: 基点，二维框标号为1的点
    :param perspective: 视角
    :param vps: 两个预测消失点
    :return:
    session0_center精度记录
    d1+d2:87.735%
    d1+d3:88.966%
    d1+d2+d3:88.457%
    """
    # -------------------------- 初始计算参数 -------------------------- #
    l, w, h = initial_value  # 初始值
    vp1, vp2 = vps  # 消失点
    point0_2D, point1_2D, point2_2D, point3_2D = box2D  # 车辆2D框
    t_d1, t_d2, t_d3 = np.zeros([1, ]), np.zeros([8, ]), np.zeros([1, ])  # 初始真实值
    d1, d2, d3 = np.zeros([1, ]), np.zeros([8, ]), np.zeros([1, ])  # 初始优化值

    # 首先根据已知二维框的基点，求解该基点对应的三维坐标；
    # 然后根据车辆初始长宽高，分别计算其他点的三维坐标；
    # 再将这八个三维点坐标利用标定矩阵H换算至映射到图像的坐标
    # 三维框顶点在图像上的坐标

    # 1 根据已知二维框的基点，求解该基点及确定的点对应的三维坐标
    point1_3D = pixel2world(H, point1_2D, 0)  # 三维框1坐标

    # 2、3 根据基点的三维坐标及车辆长宽高计算其他点的三维坐标
    point0_3D, point1_3D, point2_3D, point3_3D, point4_3D, point5_3D, point6_3D, point7_3D = \
        world2image(perspective, H, point1_3D, l, w, h)
    # image = visualization_3D_frames(image, point0_3D, point1_3D, point2_3D, point3_3D, point4_3D, point5_3D, point6_3D, point7_3D)

    # -------------------------- 对角线约束 -------------------------- #
    d_2D = distance_between_two_points([point3_2D, point1_2D])
    d_3D = distance_between_two_points([point3_3D, point1_3D])
    d1 = np.array([(math.sqrt(d_2D) - math.sqrt(d_3D))])

    # # -------------------------- 余弦约束 -------------------------- #
    # # 沿道路方向余弦约束
    # line_0A3 = Line(Point(point0_3D), Point(point3_3D))
    # line_0Avp1 = Line(Point(point0_3D), Point(vp1))
    # cos0_3 = np.array([math.cos(GetAngle(line_0A3, line_0Avp1) * math.pi / 180)])
    # line_1A2 = Line(Point(point1_3D), Point(point2_3D))
    # line_1Avp1 = Line(Point(point1_3D), Point(vp1))
    # cos1_2 = np.array([math.cos(GetAngle(line_1A2, line_1Avp1) * math.pi / 180)])
    # line_5A6 = Line(Point(point5_3D), Point(point6_3D))
    # line_5Avp1 = Line(Point(point5_3D), Point(vp1))
    # cos5_6 = np.array([math.cos(GetAngle(line_5A6, line_5Avp1) * math.pi / 180)])
    # line_4A7 = Line(Point(point4_3D), Point(point7_3D))
    # line_4Avp1 = Line(Point(point4_3D), Point(vp1))
    # cos4_7 = np.array([math.cos(GetAngle(line_4A7, line_4Avp1) * math.pi / 180)])
    # # 垂直道路方向余弦约束
    # line_0A1 = Line(Point(point0_3D), Point(point1_3D))
    # line_0Avp2 = Line(Point(point0_3D), Point(vp2))
    # cos0_1 = np.array([math.cos(GetAngle(line_0A1, line_0Avp2) * math.pi / 180)])
    # line_4A5 = Line(Point(point4_3D), Point(point5_3D))
    # line_4Avp2 = Line(Point(point4_3D), Point(vp2))
    # cos4_5 = np.array([math.cos(GetAngle(line_4A5, line_4Avp2) * math.pi / 180)])
    # line_6A7 = Line(Point(point6_3D), Point(point7_3D))
    # line_7Avp2 = Line(Point(point7_3D), Point(vp2))
    # cos6_7 = np.array([math.cos(GetAngle(line_6A7, line_7Avp2) * math.pi / 180)])
    # line_2A3 = Line(Point(point3_3D), Point(point2_3D))
    # line_3Avp2 = Line(Point(point3_3D), Point(vp2))
    # cos2_3 = np.array([math.cos(GetAngle(line_2A3, line_3Avp2) * math.pi / 180)])
    # d2 = np.array([1-abs(cos0_3[0]), 1-abs(cos1_2[0]), 1-abs(cos4_7[0]), 1-abs(cos5_6[0]),
    #                1-abs(cos0_1[0]), 1-abs(cos4_5[0]), 1-abs(cos6_7[0]), 1-abs(cos2_3[0])])  # array值

    # -------------------------- 地平线约束 -------------------------- #
    # 计算几何中心点
    point_center1, point_center2 = geometric_center_point_total(point0_3D, point1_3D, point2_3D, point3_3D,
                                                                point4_3D, point5_3D, point6_3D, point7_3D)
    line_true = Line(Point(vp1), Point(vp2))  # 真实地平线
    line_test = Line(Point(point_center1), Point(point_center2))  # 两个中心点构成的直线
    res = GetAngle(line_true, line_test)  # 单位是度
    d3 = np.array([math.cos(res * math.pi / 180)])

    return 0.5 * (np.concatenate((d1, d2, d3)) - np.concatenate((t_d1, t_d2, t_d3)))
