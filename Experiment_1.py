import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def main():
    img = cv2.imread("wit2.jpg")
    img_1 = cv2.imread("AI.jpg")
    show_img_plt([cv2.resize(img, (200, 200)), scale(img.copy(), (200, 200))],
                 ["CV 200x200", "MY_scale 200x200"])
    return 0


def scale(img, target_size):
    H, W, C = img.shape
    th, tw = target_size
    result = np.zeros((th, tw, C), np.uint8)
    for i in range(th):
        for j in range(tw):
            for k in range(C):
                # 找到在原图中对应的点的(X, Y)坐标 坐标变换如下
                # x' = x * c1
                # y' = y * c2
                corr_x = int(i * (H / th))
                corr_y = int(j * (W / tw))
                # 防止越界
                if corr_x + 1 >= H:
                    corr_x = H - 2
                if corr_y + 1 >= W:
                    corr_y = W - 2
                # 左上角的点
                point1 = [corr_x, corr_y]
                point2 = (point1[0], point1[1] + 1)
                point3 = (point1[0] + 1, point1[1])
                point4 = (point1[0] + 1, point1[1] + 1)
                # 双线性插值
                fr1 = (point2[1] - corr_y) * img[point1[0], point1[1], k] + (corr_y - point1[1]) * img[
                    point2[0], point2[1], k]
                fr2 = (point2[1] - corr_y) * img[point3[0], point3[1], k] + (corr_y - point1[1]) * img[
                    point4[0], point4[1], k]
                result[i, j, k] = (point3[0] - corr_x) * fr1 + (corr_x - point1[0]) * fr2

    return result


def histogram_equalization(img):
    H, W, C = img.shape
    pix_count = H * W
    BGR = np.zeros((3, 256))
    for i in range(H):
        for j in range(W):
            for c in range(C):
                BGR[c, img[i, j, c]] += 1
    BGR = BGR / pix_count
    for i in range(3):
        for j in range(1, 256):
            BGR[i, j] += BGR[i, j - 1]
    BGR = np.around(BGR * 255)
    for i in range(H):
        for j in range(W):
            for c in range(C):
                img[i, j, c] = BGR[c, img[i, j, c]]
    return img


def img_overlay(img1, img2, r):
    H = max(img1.shape[0], img2.shape[0])
    W = max(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (W, H))
    img1 = np.array(img1)
    img2 = np.array(img2)

    return img1 * r + img2 * (1 - r)


def img_horizontal_reverse(img):
    temp = img.copy()
    wide = len(img[0])
    high = len(img)
    for i in range(len(img)):
        for j in range(wide):
            temp[i][wide - 1 - j] = img[i][j]
    return temp


def img_color_reverse(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            for c in range(0, 3):
                img[i][j][c] = 255 - img[i][j][c]
    return img


def RGB_single(img, channel):
    for i in range(0, 3):
        if i == channel:
            continue
        img[:, :, i] = 0
    return img


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


if __name__ == '__main__':
    main()
