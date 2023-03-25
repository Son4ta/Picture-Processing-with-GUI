import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    img = cv2.imread("rice.png", 0)
    img_color = cv2.imread("rice.png")
    # 计算灰度平均值
    # initThreshold = np.mean(img)
    # 阈值迭代
    # thresh = OTSU(img)
    # dst = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    # dst = histogram_equalization(img.copy())
    show_img_plt([img, rice(img.copy(), img_color)], ["copy()", "OTSU Thresh", "Remove the background"])
    return


def rice(img, img_color):
    black = np.zeros(img_color.shape, np.uint8)
    kernels = np.ones((5, 5), np.uint8)
    # 腐蚀
    img_erode = cv2.erode(img, kernels, iterations=5)
    # 膨胀
    img_dilation = cv2.dilate(img_erode, kernels, iterations=5)
    # 阴影修正
    img = img - img_dilation

    thresh = OTSU(img)
    dst = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

    contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(black, contours, -1, (255, 255, 255), 1)

    max_perimeter = 0
    max_area = 0
    print("米粒数:" + str(len(contours)))
    max_index = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > max_area:
            max_index = i
        max_area = max(cv2.contourArea(contours[i]), max_area)
        max_perimeter = max(cv2.arcLength(contours[i], closed=True), max_perimeter)
    print("最大面积:" + str(max_area))
    print("最大周长:" + str(max_perimeter))
    cv2.drawContours(black, contours[max_index], -1, (0, 0, 255), 1)
    return black


def OTSU(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    maxCore = 0
    threshold = 0
    # img = histogram_equalization(img)
    H, W = img.shape
    histogram = np.zeros(256)
    for i in range(H):
        for j in range(W):
            histogram[img[i, j]] += 1
    averageAll = np.sum(np.dot(histogram, np.array([n for n in range(256)])))
    averageAll = averageAll / np.sum(np.array([n for n in range(256)]))
    # 找令类间方差最大的k值
    for i in range(1, 255):
        # C1均值
        mean1 = np.sum(np.dot(histogram[0:i], np.array([n for n in range(i)])))
        mean1 = mean1 / np.sum(histogram[0:i])
        # C2均值
        mean2 = np.sum(np.dot(histogram[i:256], np.array([n for n in range(i, 256)])))
        mean2 = mean2 / np.sum(histogram[i:256])
        # 公式计算
        score = sum(histogram[0:i]) * ((averageAll - mean1) ** 2) + sum(histogram[i:256]) * ((averageAll - mean2) ** 2)
        if maxCore < score:  # 记录最大值
            maxCore = score
            threshold = i
        dst = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
    return dst


def histogram_equalization(img):
    H, W = img.shape
    pix_count = H * W
    histogram = np.zeros(256)
    for i in range(H):
        for j in range(W):
            histogram[img[i, j]] += 1
    histogram = histogram / pix_count
    for i in range(1, 256):
        histogram[i] += histogram[i - 1]
    histogram = np.around(histogram * 255)
    for i in range(H):
        for j in range(W):
            img[i, j] = histogram[img[i, j]]
    return img


def Iterate_Thresh(img, max_iter_times=20, target=1):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    init_threshold = np.mean(img)
    mask1, mask2 = (img > init_threshold), (img <= init_threshold)
    T1 = np.sum(mask1 * img) / np.sum(mask1)
    T2 = np.sum(mask2 * img) / np.sum(mask2)
    T = (T1 + T2) / 2
    # 终止条件
    if abs(T - init_threshold) < target or max_iter_times == 0:
        return cv2.threshold(img, T, 255, cv2.THRESH_BINARY)[1]
    return Iterate_Thresh(img, T, max_iter_times - 1)


def show_img_plt(img_array, title=[]):
    size = len(img_array)
    plt.figure()
    for i in range(1, size + 1):
        plt.subplot(1, size, i)
        if len(img_array[i - 1].shape) == 2:
            plt.imshow(img_array[i - 1], cmap='gray')
        else:
            img_array[i - 1] = cv2.cvtColor(img_array[i - 1].copy(), cv2.COLOR_BGR2RGB)
            plt.imshow(img_array[i - 1])
        if len(title):
            plt.title(title[i - 1])
    plt.show()


def img_show(img):
    cv2.imshow("Son4ta", img)
    cv2.waitKey()


# 初始种子选择
def originalSeed(gray, th):
    ret, thresh = cv2.cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)  # 二值图，种子区域(不同划分可获得不同种子)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元

    thresh_copy = thresh.copy()  # 复制thresh_A到thresh_copy
    thresh_B = np.zeros(gray.shape, np.uint8)  # thresh_B大小与A相同，像素值为0

    seeds = []  # 为了记录种子坐标

    # 循环，直到thresh_copy中的像素值全部为0
    while thresh_copy.any():

        Xa_copy, Ya_copy = np.where(thresh_copy > 0)  # thresh_A_copy中值为255的像素的坐标
        thresh_B[Xa_copy[0], Ya_copy[0]] = 255  # 选取第一个点，并将thresh_B中对应像素值改为255

        # 连通分量算法，先对thresh_B进行膨胀，再和thresh执行and操作（取交集）
        for i in range(200):
            dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
            thresh_B = cv2.bitwise_and(thresh, dilation_B)

        # 取thresh_B值为255的像素坐标，并将thresh_copy中对应坐标像素值变为0
        Xb, Yb = np.where(thresh_B > 0)
        thresh_copy[Xb, Yb] = 0

        # 循环，在thresh_B中只有一个像素点时停止
        while str(thresh_B.tolist()).count("255") > 1:
            thresh_B = cv2.erode(thresh_B, kernel, iterations=1)  # 腐蚀操作

        X_seed, Y_seed = np.where(thresh_B > 0)  # 取处种子坐标
        if X_seed.size > 0 and Y_seed.size > 0:
            seeds.append((X_seed[0], Y_seed[0]))  # 将种子坐标写入seeds
        thresh_B[Xb, Yb] = 0  # 将thresh_B像素值置零
    return seeds


# 区域生长
def regionGrow(img, thresh=3, p=8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seeds = originalSeed(gray, th=253)
    seedMark = np.zeros(gray.shape)
    # 八邻域
    if p == 8:
        connection = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    elif p == 4:
        connection = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # seeds内无元素时候生长停止
    while len(seeds) != 0:
        # 栈顶元素出栈
        pt = seeds.pop(0)
        for i in range(p):
            tmpX = pt[0] + connection[i][0]
            tmpY = pt[1] + connection[i][1]

            # 检测边界点
            if tmpX < 0 or tmpY < 0 or tmpX >= gray.shape[0] or tmpY >= gray.shape[1]:
                continue

            if abs(int(gray[tmpX, tmpY]) - int(gray[pt])) < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = 255
                seeds.append((tmpX, tmpY))
    return seedMark


if __name__ == '__main__':
    main()
