import cv2
import numpy as np

def guided_filter(guide, src, radius, eps):
    """
    使用 opencv 的 boxFilter 实现引导滤波
    guide: 引导图像 (高清原图), shape [H, W, C], 范围 [0, 1]
    src:   待滤波图像 (RainNet上采样后的模糊图), shape [H, W, C], 范围 [0, 1]
    radius: 滤波半径 (控制平滑区域大小)
    eps:   正则化参数 (控制边缘保持程度)
    """
    # 确保输入是 float32
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    
    # 均值滤波函数
    def box_filter(x, r):
        return cv2.boxFilter(x, -1, (r, r))

    # 1. 计算各种均值
    mean_I = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    mean_Ip = box_filter(guide * src, radius)
    mean_II = box_filter(guide * guide, radius)

    # 2. 计算协方差
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    # 3. 计算线性系数 a, b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 4. 计算 a, b 的均值
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)

    # 5. 生成最终输出
    q = mean_a * guide + mean_b
    return q