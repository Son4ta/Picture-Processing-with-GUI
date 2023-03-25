import cv2
import matplotlib
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QProgressBar, \
    QInputDialog, QMessageBox

from mainForm import Ui_MainWindow
from view import Draw

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

import Experiment_1 as tool_1
import Experiment_2 as tool_2
import Experiment_3 as tool_3
from pca import PCA
from pylab import *

# 图像上显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']


def draw_img_plt(img_array, title=[]):
    size = len(img_array)
    figure = EmbedFigure()
    for i in range(1, size + 1):
        figure.axes1 = figure.fig.add_subplot(1, size, i)
        if len(img_array[i - 1].shape) == 2:
            figure.axes1.imshow(img_array[i - 1], cmap='gray')
        else:
            img_array[i - 1] = cv2.cvtColor(img_array[i - 1].copy(), cv2.COLOR_BGR2RGB)
            figure.axes1.imshow(img_array[i - 1])
        if len(title):
            figure.axes1.set_title(title[i - 1])
    # width, height = self.result_img_graphicsView.width(), self.result_img_graphicsView.height()
    # figure.resize(width, height)
    return figure


class win(QMainWindow, Ui_MainWindow):
    def __init__(self):
        # 初始化
        super().__init__()
        self.old_hook = sys.excepthook
        sys.excepthook = self.catch_exceptions

        self.setWindowTitle('图像处理')
        self.setupUi(self)
        self.progressBar = ProgressBar()
        self.statusBar.addPermanentWidget(self.progressBar)
        # action
        self.open.triggered.connect(self.open_img)
        self.save.triggered.connect(self.save_img)
        self.reverse_color.triggered.connect(self.Reverse_color)
        self.hist.triggered.connect(self.Hist)
        self.OTSU.triggered.connect(self.Otsu)
        self.spin.triggered.connect(self.Spin)
        self.gray.triggered.connect(self.Gray)
        self.iterate.triggered.connect(self.Iterate)
        self.threshold.triggered.connect(self.Threshold)
        self.line.triggered.connect(self.Line)
        self.circle.triggered.connect(self.Circle)
        self.rect.triggered.connect(self.Rect)
        self.text.triggered.connect(self.Text)
        self.laplace.triggered.connect(self.Laplace)
        self.sobel.triggered.connect(self.Sobel)
        self.median.triggered.connect(self.Median)
        self.middle.triggered.connect(self.Middle)
        self.maxf.triggered.connect(self.Maxf)
        self.minf.triggered.connect(self.Minf)
        self.sharpen.triggered.connect(self.Sharpen)
        self.dilate.triggered.connect(self.Dilate)
        self.erode.triggered.connect(self.Erode)
        self.opening.triggered.connect(self.Opening)
        self.closing.triggered.connect(self.Closing)
        self.region_grow.triggered.connect(self.RegionGrow)

        # 全局变量
        self.pca = PCA()
        self.origin = np.array([])
        self.mode_img = False
        self.result = np.array([])
        self.mode_result = False

    # 多图像展示

    # 直方图绘制
    def draw_plt(self, img):
        if not len(img):
            return False
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.flatten()
        figure = EmbedFigure(width=5, height=4, dpi=100)
        figure.axes1 = figure.fig.add_subplot(111)
        figure.axes1.hist(img, bins=256, range=(0, 255), histtype='stepfilled', align='left')
        figure.axes1.set_title("hist")

        width, height = self.result_img_graphicsView.width(), self.result_img_graphicsView.height()
        figure.resize(width, height)
        return figure

    # 按钮槽函数
    def PCA_clicked(self):
        self.start("PCA")
        if len(self.origin):
            result_image, name = self.pca.identify(self.pca.img_process(self.origin.copy()))
            self.result_img_graphicsView.set_img(draw_img_plt([result_image], [name]))
            self.end("识别结果: " + name)
        else:
            self.end("空目标，PCA无效")

    def multiple_PCA_clicked(self):
        self.start("多目标PCA")
        if len(self.origin):
            img, faces = self.pca.face_detect(self.origin)
            if len(faces):
                result_images = []
                result_names = []
                self.origin = img
                self.origin_flash()
                for (x, y, w, h) in faces:
                    result_img, name = self.pca.identify(self.pca.img_process(img[y:y + w, x:x + h]))
                    result_images.extend([result_img])
                    result_names.extend([name])
                self.result_img_graphicsView.set_img(draw_img_plt(result_images, result_names))
                self.end()
            else:
                self.end("未检测到目标")
        else:
            self.end("空目标，PCA无效")

    def origin_switch_clicked(self):
        if len(self.origin) == 0:
            return
        if self.mode_img:
            self.origin_img_graphicsView.set_img(self.origin)
        else:
            self.origin_img_graphicsView.set_img(self.draw_plt(self.origin))
        self.mode_img = not self.mode_img

    def result_switch_clicked(self):
        if len(self.result) == 0:
            return
        if self.mode_result:
            self.result_flash()
        else:
            self.result_img_graphicsView.set_img(self.draw_plt(self.result))
            print("直方图！")
        self.mode_result = not self.mode_result

    def result_reset_clicked(self):
        if len(self.result) == 0:
            return
        self.result = self.origin
        self.result_flash()

    # 刷新图像
    def result_flash(self):
        self.result_img_graphicsView.set_img(self.result)

    def origin_flash(self):
        self.origin_img_graphicsView.set_img(self.origin)

    # 文件操作action
    def save_img(self):
        result = self.result
        if len(result):
            filepath, _ = QFileDialog.getSaveFileName(self, "保存处理结果", "/", '*.png *.jpg *.bmp')
            if filepath:
                print(filepath)
                cv2.imwrite(filepath, result)

    def open_img(self):
        self.start("打开")
        fileName, _ = QFileDialog.getOpenFileName(
            self, '打开图像', '', '*.png *.jpg *.bmp')
        if fileName:
            self.origin = self.result = np.array(cv2.imdecode(np.fromfile(fileName, dtype=np.uint8), -1))
            self.origin_img_graphicsView.set_img(self.origin)
            self.result_img_graphicsView.set_img(self.origin)
        self.end()
        return

    # 调用action
    def Line(self):
        self.start("绘制直线")
        self.result_img_graphicsView.scene.paint_graph_init(Draw.line)
        self.end()

    def Circle(self):
        self.start("绘制圆")
        self.result_img_graphicsView.scene.paint_graph_init(Draw.circle)
        self.end()

    def Rect(self):
        self.start("绘制矩形")
        self.result_img_graphicsView.scene.paint_graph_init(Draw.rect)
        self.end()

    def Text(self):
        self.start("绘制文字")
        self.result_img_graphicsView.scene.paint_graph_init(Draw.text)
        self.end()

    def processing(self, method):
        self.start("处理中")
        # 图像不为空
        if len(self.result):
            self.result = method(self.result.copy())
            self.result_flash()
        self.end()

    def Reverse_color(self):
        self.processing(tool_1.img_color_reverse)

    def Hist(self):
        self.processing(tool_1.histogram_equalization)

    def Otsu(self):
        self.processing(tool_3.OTSU)
        QMessageBox(QMessageBox.Warning, '危险！', '您刚刚进行了阈值操作，这意味着现在这是一张灰度图，可能无法进行RGB有关的处理').exec_()

    def Spin(self):
        self.processing(tool_1.img_horizontal_reverse)

    def Gray(self):
        if len(self.result):
            self.result = cv2.cvtColor(self.result.copy(), cv2.COLOR_BGR2GRAY)
            self.result_flash()

    def Iterate(self):
        self.processing(tool_3.Iterate_Thresh)
        QMessageBox(QMessageBox.Warning, '危险！', '您刚刚进行了阈值操作，这意味着现在这是一张灰度图，可能无法进行RGB有关的处理').exec_()

    def Threshold(self):
        self.start()
        if len(self.result):
            self.origin_switch_clicked()
            threshold, ok = QInputDialog.getInt(self, '阈值', '输入阈值', 127, 0, 255, 1)
            if ok and threshold:
                self.result = cv2.threshold(cv2.cvtColor(self.result.copy(), cv2.COLOR_BGR2GRAY),
                                            threshold, 255, cv2.THRESH_BINARY)[1]
                self.result_flash()
            self.origin_switch_clicked()
            QMessageBox(QMessageBox.Warning, '危险！', '您刚刚进行了阈值操作，这意味着现在这是一张灰度图，可能无法进行RGB有关的处理').exec_()
        self.end()

    def Laplace(self):
        self.processing(tool_2.Laplace)

    def Sobel(self):
        self.processing(tool_2.Sobel)

    def Sharpen(self):
        self.processing(tool_2.sharpen)

    def filter(self, method):
        self.start("滤波处理中")
        if len(self.result):
            size, ok = QInputDialog.getInt(self, 'kernel', '输入滤波核大小', 3, 3, 11, 2)
            if ok and size >= 3 and size % 2:
                self.result = method(self.result.copy(), size)
                self.result_flash()
            else:
                QMessageBox(QMessageBox.Warning, '意外输入', '核不合法！核应该 >=3 并 是奇数').exec_()
        self.end()

    def Median(self):
        self.filter(tool_2.median_filter)

    def Middle(self):
        self.filter(tool_2.middle_filter)

    def Maxf(self):
        self.filter(tool_2.max_filter)

    def Minf(self):
        self.filter(tool_2.min_filter)

    def Dilate(self):
        self.start()
        if len(self.result):
            size, ok = QInputDialog.getInt(self, 'kernel', '输入滤波核大小', 3, 3, 11, 2)
            if ok and size >= 3 and size % 2:
                kernel = np.ones((size, size), dtype=np.uint8)
                self.result = cv2.dilate(self.result.copy(), kernel, iterations=1)
                self.result_flash()
            else:
                QMessageBox(QMessageBox.Warning, '意外输入', '核不合法！核应该是奇数').exec_()
        self.end()

    def Erode(self):
        self.start()
        if len(self.result):
            size, ok = QInputDialog.getInt(self, 'kernel', '输入滤波核大小', 3, 3, 11, 2)
            if ok and size >= 3 and size % 2:
                kernel = np.ones((size, size), dtype=np.uint8)
                self.result = cv2.erode(self.result.copy(), kernel, iterations=1)
                self.result_flash()
            else:
                QMessageBox(QMessageBox.Warning, '意外输入', '核不合法！核应该是奇数').exec_()
        self.end()

    def Opening(self):
        self.start()
        if len(self.result):
            size, ok = QInputDialog.getInt(self, 'kernel', '输入滤波核大小', 3, 3, 11, 2)
            if ok and size >= 3 and size % 2:
                kernel = np.ones((size, size), dtype=np.uint8)
                self.result = cv2.morphologyEx(self.result.copy(), cv2.MORPH_OPEN, kernel, iterations=1)
                self.result_flash()
            else:
                QMessageBox(QMessageBox.Warning, '意外输入', '核不合法！核应该是奇数').exec_()
        self.end()

    def Closing(self):
        self.start()
        if len(self.result):
            size, ok = QInputDialog.getInt(self, 'kernel', '输入滤波核大小', 3, 3, 11, 2)
            if ok and size >= 3 and size % 2:
                kernel = np.ones((size, size), dtype=np.uint8)
                self.result = cv2.morphologyEx(self.result.copy(), cv2.MORPH_CLOSE, kernel, iterations=1)
                self.result_flash()
            else:
                QMessageBox(QMessageBox.Warning, '意外输入', '核不合法！核应该是奇数').exec_()
        self.end()

    def RegionGrow(self):
        self.processing(tool_3.regionGrow)

    # 功能函数
    def catch_exceptions(self, ty, value, traceback):
        """
            捕获异常，并弹窗显示
        :param ty: 异常的类型
        :param value: 异常的对象
        :param traceback: 异常的traceback
        """
        # traceback_format = traceback.format_exception(ty, value, traceback)
        # traceback_string = "".join(traceback_format)
        # QMessageBox(QMessageBox.Critical, "An exception was raised", "{}".format(ty.err)).exec_()
        QMessageBox(QMessageBox.Critical, '错误', '抱歉，我们出现了错误，这可能是不当的操作顺序造成的\n'
                    + '已恢复操作前状态，有需要请联系开发人员').exec_()
        self.end("Error")
        self.old_hook(ty, value, traceback)

    def start(self, msg="执行中"):
        self.progressBar.reset()
        self.statusBar.showMessage(msg)
        self.progressBar.busy()

    def end(self, msg="Done"):
        self.progressBar.done()
        self.statusBar.showMessage(msg)


# 重写一个matplotlib图像绘制类
class EmbedFigure(FigureCanvasQTAgg):
    def __init__(self, width=5, height=4, dpi=200):
        # 1、创建一个绘制窗口Figure对象
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 2、在父类中激活Figure窗口,同时继承父类属性
        super(EmbedFigure, self).__init__(self.fig)


class ProgressBar(QProgressBar):
    def __init__(self, parent=None, step=8):
        super().__init__(parent)
        self.step = step
        self.setRange(0, step)  # 设置进度条的范围

    def done(self):
        self.setMaximum(self.step)
        self.setValue(self.step)

    def busy(self):
        self.setMaximum(0)
        self.setMinimum(0)


if __name__ == '__main__':
    a = QtWidgets.QApplication(sys.argv)
    window = win()
    window.show()
    sys.exit(a.exec_())
