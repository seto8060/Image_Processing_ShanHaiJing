# 安装: pip install mediapipe opencv-python
import cv2
import numpy as np
import mediapipe as mp

mp_selfie_segmentation = mp.solutions.selfie_segmentation
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_seg:
    image = cv2.imread('6.png')
    results = selfie_seg.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 得到二值化掩模（Mask），白色为人，黑色为背景
    mask = results.segmentation_mask > 0.5
    cv2.imwrite('mask.png', (mask * 255.0).astype(np.uint8))