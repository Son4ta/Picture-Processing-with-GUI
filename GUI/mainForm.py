# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1115, 669)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(7)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(13, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setContentsMargins(0, 0, 0, -1)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.origin_img_graphicsView = View(self.centralwidget)
        self.origin_img_graphicsView.setObjectName("origin_img_graphicsView")
        self.verticalLayout_8.addWidget(self.origin_img_graphicsView)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.origin_switch_pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.origin_switch_pushButton.setObjectName("origin_switch_pushButton")
        self.horizontalLayout_3.addWidget(self.origin_switch_pushButton)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_2.addWidget(self.pushButton_4)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout_8.addLayout(self.horizontalLayout_3)
        self.horizontalLayout.addLayout(self.verticalLayout_8)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.result_img_graphicsView = View(self.centralwidget)
        self.result_img_graphicsView.setObjectName("result_img_graphicsView")
        self.verticalLayout_9.addWidget(self.result_img_graphicsView)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_4.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_4.addWidget(self.pushButton)
        self.verticalLayout_9.addLayout(self.horizontalLayout_4)
        self.horizontalLayout.addLayout(self.verticalLayout_9)
        spacerItem2 = QtWidgets.QSpacerItem(13, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1115, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_P = QtWidgets.QMenu(self.menubar)
        self.menu_P.setObjectName("menu_P")
        self.menu_G = QtWidgets.QMenu(self.menubar)
        self.menu_G.setObjectName("menu_G")
        self.menu_T = QtWidgets.QMenu(self.menubar)
        self.menu_T.setObjectName("menu_T")
        self.menu_B = QtWidgets.QMenu(self.menubar)
        self.menu_B.setObjectName("menu_B")
        self.menu_D = QtWidgets.QMenu(self.menubar)
        self.menu_D.setObjectName("menu_D")
        self.menu_L = QtWidgets.QMenu(self.menubar)
        self.menu_L.setObjectName("menu_L")
        MainWindow.setMenuBar(self.menubar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.toolBar_2 = QtWidgets.QToolBar(MainWindow)
        self.toolBar_2.setObjectName("toolBar_2")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar_2)
        self.open = QtWidgets.QAction(MainWindow)
        self.open.setObjectName("open")
        self.save = QtWidgets.QAction(MainWindow)
        self.save.setObjectName("save")
        self.spin = QtWidgets.QAction(MainWindow)
        self.spin.setObjectName("spin")
        self.reverse = QtWidgets.QAction(MainWindow)
        self.reverse.setObjectName("reverse")
        self.reverse_color = QtWidgets.QAction(MainWindow)
        self.reverse_color.setObjectName("reverse_color")
        self.gray = QtWidgets.QAction(MainWindow)
        self.gray.setObjectName("gray")
        self.cut = QtWidgets.QAction(MainWindow)
        self.cut.setObjectName("cut")
        self.hist = QtWidgets.QAction(MainWindow)
        self.hist.setObjectName("hist")
        self.dilate = QtWidgets.QAction(MainWindow)
        self.dilate.setObjectName("dilate")
        self.erode = QtWidgets.QAction(MainWindow)
        self.erode.setObjectName("erode")
        self.opening = QtWidgets.QAction(MainWindow)
        self.opening.setObjectName("opening")
        self.closing = QtWidgets.QAction(MainWindow)
        self.closing.setObjectName("closing")
        self.threshold = QtWidgets.QAction(MainWindow)
        self.threshold.setObjectName("threshold")
        self.iterate = QtWidgets.QAction(MainWindow)
        self.iterate.setObjectName("iterate")
        self.OTSU = QtWidgets.QAction(MainWindow)
        self.OTSU.setObjectName("OTSU")
        self.region_grow = QtWidgets.QAction(MainWindow)
        self.region_grow.setObjectName("region_grow")
        self.laplace = QtWidgets.QAction(MainWindow)
        self.laplace.setObjectName("laplace")
        self.sobel = QtWidgets.QAction(MainWindow)
        self.sobel.setObjectName("sobel")
        self.rect = QtWidgets.QAction(MainWindow)
        self.rect.setObjectName("rect")
        self.line = QtWidgets.QAction(MainWindow)
        self.line.setObjectName("line")
        self.circle = QtWidgets.QAction(MainWindow)
        self.circle.setObjectName("circle")
        self.text = QtWidgets.QAction(MainWindow)
        self.text.setObjectName("text")
        self.middle = QtWidgets.QAction(MainWindow)
        self.middle.setObjectName("middle")
        self.median = QtWidgets.QAction(MainWindow)
        self.median.setObjectName("median")
        self.maxf = QtWidgets.QAction(MainWindow)
        self.maxf.setObjectName("maxf")
        self.minf = QtWidgets.QAction(MainWindow)
        self.minf.setObjectName("minf")
        self.sharpen = QtWidgets.QAction(MainWindow)
        self.sharpen.setObjectName("sharpen")
        self.menu.addAction(self.open)
        self.menu.addAction(self.save)
        self.menu_P.addAction(self.spin)
        self.menu_P.addAction(self.reverse)
        self.menu_P.addAction(self.reverse_color)
        self.menu_P.addAction(self.gray)
        self.menu_P.addAction(self.cut)
        self.menu_P.addAction(self.hist)
        self.menu_G.addAction(self.dilate)
        self.menu_G.addAction(self.erode)
        self.menu_G.addAction(self.opening)
        self.menu_G.addAction(self.closing)
        self.menu_T.addAction(self.threshold)
        self.menu_T.addAction(self.iterate)
        self.menu_T.addAction(self.OTSU)
        self.menu_T.addAction(self.region_grow)
        self.menu_B.addAction(self.laplace)
        self.menu_B.addAction(self.sobel)
        self.menu_D.addAction(self.rect)
        self.menu_D.addAction(self.line)
        self.menu_D.addAction(self.circle)
        self.menu_D.addAction(self.text)
        self.menu_L.addAction(self.middle)
        self.menu_L.addAction(self.median)
        self.menu_L.addAction(self.maxf)
        self.menu_L.addAction(self.minf)
        self.menu_L.addAction(self.sharpen)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_P.menuAction())
        self.menubar.addAction(self.menu_G.menuAction())
        self.menubar.addAction(self.menu_T.menuAction())
        self.menubar.addAction(self.menu_B.menuAction())
        self.menubar.addAction(self.menu_L.menuAction())
        self.menubar.addAction(self.menu_D.menuAction())
        self.toolBar.addAction(self.open)
        self.toolBar.addAction(self.save)
        self.toolBar.addSeparator()

        self.retranslateUi(MainWindow)
        self.pushButton_2.clicked.connect(MainWindow.result_switch_clicked)
        self.origin_switch_pushButton.clicked.connect(MainWindow.origin_switch_clicked)
        self.pushButton.clicked.connect(MainWindow.result_reset_clicked)
        self.pushButton_3.clicked.connect(MainWindow.PCA_clicked)
        self.pushButton_4.clicked.connect(MainWindow.multiple_PCA_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.origin_switch_pushButton.setText(_translate("MainWindow", "直方图/原图"))
        self.pushButton_3.setText(_translate("MainWindow", "PCA"))
        self.pushButton_4.setText(_translate("MainWindow", "多目标PCA"))
        self.pushButton_2.setText(_translate("MainWindow", "直方图/原图"))
        self.pushButton.setText(_translate("MainWindow", "重置(Reset)"))
        self.menu.setTitle(_translate("MainWindow", "文件(&F)"))
        self.menu_P.setTitle(_translate("MainWindow", "基本操作(&P)"))
        self.menu_G.setTitle(_translate("MainWindow", "形态学(&G)"))
        self.menu_T.setTitle(_translate("MainWindow", "阈值(&T)"))
        self.menu_B.setTitle(_translate("MainWindow", "边缘检测(&B)"))
        self.menu_D.setTitle(_translate("MainWindow", "绘制(&D)"))
        self.menu_L.setTitle(_translate("MainWindow", "滤波(&L)"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.toolBar_2.setWindowTitle(_translate("MainWindow", "toolBar_2"))
        self.open.setText(_translate("MainWindow", "打开"))
        self.save.setText(_translate("MainWindow", "保存"))
        self.spin.setText(_translate("MainWindow", "旋转"))
        self.reverse.setText(_translate("MainWindow", "翻转"))
        self.reverse_color.setText(_translate("MainWindow", "反色"))
        self.gray.setText(_translate("MainWindow", "灰度化"))
        self.cut.setText(_translate("MainWindow", "裁剪"))
        self.hist.setText(_translate("MainWindow", "直方图均衡"))
        self.dilate.setText(_translate("MainWindow", "膨胀"))
        self.erode.setText(_translate("MainWindow", "腐蚀"))
        self.opening.setText(_translate("MainWindow", "开运算"))
        self.closing.setText(_translate("MainWindow", "闭运算"))
        self.threshold.setText(_translate("MainWindow", "给定阈值分割"))
        self.iterate.setText(_translate("MainWindow", "全局自动阈值分割"))
        self.OTSU.setText(_translate("MainWindow", "OTSU"))
        self.region_grow.setText(_translate("MainWindow", "区域生长"))
        self.laplace.setText(_translate("MainWindow", "Laplace"))
        self.sobel.setText(_translate("MainWindow", "Sobel"))
        self.rect.setText(_translate("MainWindow", "矩形"))
        self.line.setText(_translate("MainWindow", "直线"))
        self.circle.setText(_translate("MainWindow", "圆"))
        self.text.setText(_translate("MainWindow", "文字"))
        self.middle.setText(_translate("MainWindow", "中值滤波"))
        self.median.setText(_translate("MainWindow", "均值滤波"))
        self.maxf.setText(_translate("MainWindow", "最大值滤波"))
        self.minf.setText(_translate("MainWindow", "最小值滤波"))
        self.sharpen.setText(_translate("MainWindow", "锐化"))

from view import View
