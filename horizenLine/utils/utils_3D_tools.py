import math
import numpy as np


def getFocal(vp1, vp2, pp):
    return math.sqrt(- np.dot(vp1[0:2]-pp[0:2], vp2[0:2]-pp[0:2]))


def computeCameraCalibration(_vp1, _vp2, _pp):
    vp1 = np.concatenate((_vp1, [1]))
    vp2 = np.concatenate((_vp2, [1]))
    pp = np.concatenate((_pp, [1]))
    focal = getFocal(vp1, vp2, pp)
    vp1W = np.concatenate((_vp1, [focal]))  # vp1世界坐标
    vp2W = np.concatenate((_vp2, [focal]))  # vp2世界坐标
    ppW = np.concatenate((_pp, [0]))  # 主点世界坐标
    vp3W = np.cross(vp1W - ppW, vp2W - ppW)  # vp3世界坐标
    vp3 = np.concatenate((vp3W[0:2] / vp3W[2] * focal + ppW[0:2], [1]))
    vp3Direction = np.concatenate((vp3[0:2], [focal])) - ppW
    roadPlane = np.concatenate((vp3Direction / np.linalg.norm(vp3Direction), [10]))
    return vp1, vp2, vp3, pp, roadPlane, focal