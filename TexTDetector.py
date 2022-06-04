import cv2
from object_detection import non_max_suppression
import numpy as np
import time
import global_vars as gv

class TexTDetector:
	# model_path: 模型路径
    def __init__(self, model_path):
    	# 输入网络的图片尺寸（必须为8的倍数）
        self.rw = 320
        self.rh = 320
        # 最小置信度
        self.min_confidence = 0.5

        self.net = cv2.dnn.readNet(model_path)
        self.layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
    def detect_text(self, image):
        (imgh, imgw) = image.shape[:2]
        # 记录原图像大小与缩小后的比值，得到预测结果后需要乘上这个缩放因子
        rW = imgw / float(self.rw)
        rH = imgh / float(self.rh)
        # 放缩到网络可以接受的尺寸
        image = cv2.resize(image, (self.rw, self.rh))
        # 图像预处理
        blob = cv2.dnn.blobFromImage(image, 1.0, (self.rw, self.rh), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        # 前向传播，获得置信度和几何数据
        (scores, geometry) = self.net.forward(self.layerNames)

        (rows, cols) = scores.shape[2:4]
        rects = []
        confidences = []

        # 对得到的框进行筛选，先用最小置信度进行粗筛选
        for i in range(rows):
            scoresData = scores[0, 0, i]
            xData0 = geometry[0, 0, i]
            xData1 = geometry[0, 1, i]
            xData2 = geometry[0, 2, i]
            xData3 = geometry[0, 3, i]
            anglesData = geometry[0, 4, i]
            for j in range(cols):
                if scoresData[j] < self.min_confidence:
                    continue

                (offsetX, offsetY) = (j * 4.0, i * 4.0)
                angle = anglesData[j]
                cos = np.cos(angle)
                sin = np.sin(angle)

                h = xData0[j] + xData2[j]
                w = xData1[j] + xData3[j]

                endX = int(offsetX + (cos * xData1[j]) + (sin * xData2[j]))
                endY = int(offsetY - (sin * xData1[j]) + (cos * xData2[j]))
                startX = int(endX - w)
                startY = int(endY - h)

                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[j])

        # 再使用非极大值抑制算法细筛选
        rects = np.array(rects)
        pick = non_max_suppression(rects, probs=confidences)
        boxes = rects[pick]

        # 得到最终的文本框
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] * rW).astype(np.int32)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * rH).astype(np.int32)

        return boxes
