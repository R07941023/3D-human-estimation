# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/Sport_project_v1.4.6/UI/3D_reconstruction/calibration_setting.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(595, 435)
        MainWindow.setMaximumSize(QtCore.QSize(99999, 99999))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setMinimumSize(QtCore.QSize(136, 0))
        self.pushButton_4.setMaximumSize(QtCore.QSize(300, 16777215))
        self.pushButton_4.setSizeIncrement(QtCore.QSize(300, 0))
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_2.addWidget(self.pushButton_4, 0, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setMaximumSize(QtCore.QSize(421, 16777215))
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 0, 1, 1, 2)
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 1, 0, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.gridLayout_2.addWidget(self.comboBox_2, 1, 1, 1, 2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setMinimumSize(QtCore.QSize(133, 0))
        self.label_3.setMaximumSize(QtCore.QSize(133, 16777215))
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setMinimumSize(QtCore.QSize(110, 0))
        self.lineEdit_3.setMaximumSize(QtCore.QSize(110, 16777215))
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 0, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(38, 22, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 0, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 0, 3, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_6.setMaximumSize(QtCore.QSize(110, 16777215))
        self.lineEdit_6.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout.addWidget(self.lineEdit_6, 0, 4, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setMinimumSize(QtCore.QSize(133, 0))
        self.label_4.setMaximumSize(QtCore.QSize(133, 16777215))
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 0, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setMinimumSize(QtCore.QSize(110, 0))
        self.lineEdit_4.setMaximumSize(QtCore.QSize(110, 16777215))
        self.lineEdit_4.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 1, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 2, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout.addWidget(self.checkBox, 1, 3, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 2, 0, 1, 8)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setMinimumSize(QtCore.QSize(0, 61))
        self.groupBox_2.setMaximumSize(QtCore.QSize(16777215, 70))
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton_6 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_6.setGeometry(QtCore.QRect(13, 33, 85, 23))
        self.radioButton_6.setObjectName("radioButton_6")
        self.radioButton_7 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_7.setGeometry(QtCore.QRect(104, 33, 101, 23))
        self.radioButton_7.setChecked(True)
        self.radioButton_7.setObjectName("radioButton_7")
        self.gridLayout_2.addWidget(self.groupBox_2, 3, 0, 1, 3)
        self.label_18 = QtWidgets.QLabel(self.centralwidget)
        self.label_18.setMinimumSize(QtCore.QSize(130, 0))
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.gridLayout_2.addWidget(self.label_18, 4, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.gridLayout_2.addWidget(self.label_19, 4, 1, 1, 1)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_13.setEnabled(True)
        self.lineEdit_13.setMaximumSize(QtCore.QSize(110, 16777215))
        self.lineEdit_13.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.gridLayout_2.addWidget(self.lineEdit_13, 4, 2, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.gridLayout_2.addWidget(self.label_20, 4, 3, 1, 1)
        self.lineEdit_14 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_14.setEnabled(True)
        self.lineEdit_14.setMaximumSize(QtCore.QSize(110, 16777215))
        self.lineEdit_14.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.gridLayout_2.addWidget(self.lineEdit_14, 4, 4, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.centralwidget)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.gridLayout_2.addWidget(self.label_21, 4, 5, 1, 1)
        self.lineEdit_15 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_15.setEnabled(True)
        self.lineEdit_15.setMaximumSize(QtCore.QSize(110, 16777215))
        self.lineEdit_15.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.gridLayout_2.addWidget(self.lineEdit_15, 4, 6, 1, 2)
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setMinimumSize(QtCore.QSize(130, 0))
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.gridLayout_2.addWidget(self.label_22, 5, 0, 1, 1)
        self.lineEdit_16 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_16.setEnabled(True)
        self.lineEdit_16.setMaximumSize(QtCore.QSize(110, 16777215))
        self.lineEdit_16.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.gridLayout_2.addWidget(self.lineEdit_16, 5, 1, 1, 2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 6, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setMinimumSize(QtCore.QSize(0, 0))
        self.lineEdit.setMaximumSize(QtCore.QSize(110, 16777215))
        self.lineEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_2.addWidget(self.lineEdit, 6, 1, 1, 2)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 7, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_2.addWidget(self.lineEdit_2, 7, 1, 1, 2)
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setMinimumSize(QtCore.QSize(120, 0))
        self.pushButton_6.setMaximumSize(QtCore.QSize(300, 16777215))
        self.pushButton_6.setSizeIncrement(QtCore.QSize(300, 0))
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout_2.addWidget(self.pushButton_6, 8, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setMaximumSize(QtCore.QSize(421, 16777215))
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 8, 1, 1, 2)
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setChecked(False)
        self.checkBox_3.setObjectName("checkBox_3")
        self.gridLayout_2.addWidget(self.checkBox_3, 9, 0, 1, 1)
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setMaximumSize(QtCore.QSize(90, 16777215))
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout_2.addWidget(self.pushButton_7, 9, 6, 1, 1)
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setMaximumSize(QtCore.QSize(90, 16777215))
        self.pushButton_8.setObjectName("pushButton_8")
        self.gridLayout_2.addWidget(self.pushButton_8, 9, 7, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 595, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton_7.clicked['bool'].connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_4.setText(_translate("MainWindow", "checkerboard path"))
        self.label_8.setText(_translate("MainWindow", "????"))
        self.label_17.setText(_translate("MainWindow", "Camera name:"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Camera_1"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Camera_2"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "Camera_3"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "Camera_4"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "Camera_5"))
        self.label_3.setText(_translate("MainWindow", "corners width:"))
        self.lineEdit_3.setText(_translate("MainWindow", "8"))
        self.label_9.setText(_translate("MainWindow", "corners height:"))
        self.lineEdit_6.setText(_translate("MainWindow", "8"))
        self.label_4.setText(_translate("MainWindow", "block length [m]:"))
        self.lineEdit_4.setText(_translate("MainWindow", "0.05"))
        self.checkBox.setText(_translate("MainWindow", "SubPix filter"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Core:"))
        self.radioButton_6.setText(_translate("MainWindow", "single"))
        self.radioButton_7.setText(_translate("MainWindow", "Multi (max)"))
        self.label_18.setText(_translate("MainWindow", "video range"))
        self.label_19.setText(_translate("MainWindow", "min:"))
        self.lineEdit_13.setText(_translate("MainWindow", "1"))
        self.label_20.setText(_translate("MainWindow", "shift:"))
        self.lineEdit_14.setText(_translate("MainWindow", "1"))
        self.label_21.setText(_translate("MainWindow", "max:"))
        self.lineEdit_15.setText(_translate("MainWindow", "default"))
        self.label_22.setText(_translate("MainWindow", "kmeans:"))
        self.lineEdit_16.setText(_translate("MainWindow", "30"))
        self.label.setText(_translate("MainWindow", "cube length [m]:"))
        self.lineEdit.setText(_translate("MainWindow", "1"))
        self.label_2.setText(_translate("MainWindow", "display ratio:"))
        self.lineEdit_2.setText(_translate("MainWindow", "1"))
        self.pushButton_6.setText(_translate("MainWindow", "cube path"))
        self.label_10.setText(_translate("MainWindow", "????"))
        self.checkBox_3.setText(_translate("MainWindow", "freeze"))
        self.pushButton_7.setText(_translate("MainWindow", "Close"))
        self.pushButton_8.setText(_translate("MainWindow", "Apply"))
