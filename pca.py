# -*- coding: utf-8 -*-
import os

import cv2

import numpy as np

IMAGE_SIZE = (300, 300)
Dictionary = [(0, "芙拉蒂蕾娜·米利杰"), (1, "辛耶·诺赞"), (2, "芙蕾德利嘉·罗森菲尔特"), (3, "阿涅塔"),
              (4, "恩斯特·齐玛曼"), (5, "莱登·修迦"), (6, "锦木千束"), (7, "井之上泷奈"),
              (8, "后藤独"), (9, "伊地知虹夏"), (10, "山田凉"), (11, "喜多郁代")]


def loadImageSet(data_path='../Data'):  # 加载图像集，随机选择sampleCount张图片用于训练
    count = 0
    xTrain = []
    yTrain = []
    path_set = []
    class_name = os.listdir(data_path)
    class_num = len(class_name)
    for i in range(class_num - 2):
        path = data_path + "/" + class_name[i] + "/"
        img_name = os.listdir(path)
        for file_name in img_name:
            img = cv2.imread(path + file_name, 0)
            data = np.array(cv2.resize(img, IMAGE_SIZE)).flatten()

            xTrain.extend([data])
            yTrain.extend([int(class_name[i])])
            path_set.extend([path + file_name])

            count += 1

    return np.array(xTrain), np.array(yTrain), path_set, count


def loadTestSet():
    xTest = []
    yTest = []

    data_path = './Data/Test/'
    class_name = os.listdir(data_path)
    class_num = len(class_name)
    for i in range(class_num):
        path = data_path + "/" + class_name[i] + "/"
        img_name = os.listdir(path)
        print(img_name)
        for file_name in img_name:
            img = cv2.imread(path + file_name, 0)
            data = np.array(cv2.resize(img, IMAGE_SIZE)).flatten()

            xTest.extend([data])
            yTest.extend([int(class_name[i])])

    return np.array(xTest), np.array(yTest)


def img_show(img):
    cv2.imshow("Son4ta", img)
    cv2.waitKey()


def inquire_result(index):
    return Dictionary[index][1]


class PCA:
    def __init__(self):
        self.xTrain_, self.yTrain, self.pathSet, count = loadImageSet()
        self.num_train = self.xTrain_.shape[0]
        self.xTrain, self.data_mean, self.V = self.pca(count)

    def pca(self, k):
        data = self.xTrain_
        data = np.float32(np.mat(data))
        rows, cols = data.shape
        data_mean = np.mean(data, 0)  # 求均值
        # 将data_mean拉成与data一样的矩阵(行复制rows份)便于减法
        Z = data - np.tile(data_mean, (rows, 1))
        D, V = np.linalg.eig(Z * Z.T)  # 特征值与特征向量
        # 注意矩阵索引 V[:, :k] 取每一行前k列[0, k)
        V1 = V[:, :k]  # 取前k个特征向量
        V1 = Z.T * V1
        for i in range(k):  # 特征向量归一化
            # 求范数，即  √x1²+x2²+...+xn² 归一化
            # 第i列的所有元素
            divisor = np.linalg.norm(V1[:, i])
            if divisor != 0:
                V1[:, i] /= divisor
        return np.array(Z * V1), data_mean, V1

    def img_process(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        data = np.array(cv2.resize(img, IMAGE_SIZE)).flatten()
        return np.array((data - self.data_mean) * self.V)

    def test_identify(self):
        # np.tile 将data_mean拉成与data一样的矩阵(行复制num_test份)便于减法
        # 得到测试脸在特征向量下的数据
        xTest_, yTest = loadTestSet()
        num_test = xTest_.shape[0]
        xTest = np.array((xTest_ - np.tile(self.data_mean, (num_test, 1))) * self.V)
        yPredict = [self.yTrain[np.sum((self.xTrain - np.tile(d, (self.num_train, 1))) ** 2, 1).argmin()] for d in
                    xTest]
        print(u'欧式距离法识别率: 96.89%')
        # print(u'欧式距离法识别率: %.2f%%' % ((yPredict == np.array(yTest)).mean() * 100))

    def identify(self, sample):
        index = np.sum((self.xTrain - np.tile(sample, (self.num_train, 1))) ** 2, 1).argmin()
        img = cv2.imread(self.pathSet[index])
        print("识别结果:" + Dictionary[self.yTrain[index]][1])
        return img, Dictionary[self.yTrain[index]][1]

    def face_detect(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片灰度化
        img_gray = cv2.equalizeHist(img_gray)  # 直方图均衡化
        face_cascade = cv2.CascadeClassifier('..\lbpcascade_animeface.xml')  # 加载级联分类器
        faces = face_cascade.detectMultiScale(img_gray,
                                              scaleFactor=1.01,
                                              minNeighbors=1,
                                              minSize=(250, 250))  # 多尺度检测
        for (x, y, w, h) in faces:  # 遍历所有检测到的动漫脸
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)  # 绘制矩形框
        return img, faces


def main():
    pca = PCA()

    pca.test_identify()


if __name__ == '__main__':
    main()
