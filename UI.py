#!/usr/bin/env python
# encoding: utf-8

# library
from importlib import reload
from PyQt5 import QtCore, QtGui, QtWidgets
import os
import numpy as np
import cv2
import pygame
import sys, shutil
import multiprocessing as mp
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import qdarkstyle
import time
import hashlib
import datetime
import torch
import alphapose_import
# UI
from main_windows import Ui_MainWindow as UI_main
from calibration_setting import Ui_MainWindow as UI_calibration_setting
from alphapose_setting import Ui_Form as UI_alphapose_setting
from threeD_plt_setting import Ui_MainWindow as UI_threeD_plt_setting
from postprocessing_setting import Ui_MainWindow as UI_postprocessing_setting
from post_plt_setting import Ui_MainWindow as UI_post_plt_setting
from transformation_setting import Ui_MainWindow as UI_transformation_setting
from anomaly_setting import Ui_MainWindow as UI_anomaly_setting
from frame_check import Ui_MainWindow as UI_frame_check
from analysis_plt_setting import Ui_MainWindow as UI_analysis_plt_setting
from log import Ui_MainWindow as UI_log
import pandas as pd
import pickle
# myfunction
# sys.path.append('../../')
import fn, checkerboard_calibration, threeD_pose_human
# import PyQt5_stylesheets

# pyinstaller need it
# import torch
# def script_method(fn, _rcb=None):
#     return fn
# def script(obj, optimize=True, _frames_up=0, _rcb=None):
#     return obj
# import torch.jit
# torch.jit.script_method = script_method
# torch.jit.script = script
# os.environ["PYTORCH_JIT"] = "0"


def CV2QImage(cv_image):
    # width = cv_image.shape[1]
    # height = cv_image.shape[0]
    # pixmap = QPixmap(width, height)
    # qimg = pixmap.toImage()
    # for row in range(0, height):
    #     for col in range(0, width):
    #         r = cv_image[(row, col, 2)]
    #         g = cv_image[(row, col, 1)]
    #         b = cv_image[(row, col, 0)]
    #         pix = qRgb(r, g, b)
    #         qimg.setPixel(col, row, pix)
    # totalBytes = cv_image.nbytes
    # bytesPerLine = int(totalBytes / cv_image.shape[0])
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    height, width, byteValue = cv_image.shape
    bytesPerLine = byteValue * width
    qimg = QtGui.QImage(cv_image.data, cv_image.shape[1], cv_image.shape[0], bytesPerLine, QtGui.QImage.Format_RGB888)
    return qimg

class DialogUI(QWidget):

    def __init__(self = None, parent = None):
        super(DialogUI, self).__init__(parent)
        self.setWindowTitle('Login')
        flo = QFormLayout()
        e1 = QLineEdit()
        BtnOk = QPushButton('Login')
        BtnCancel = QPushButton('Exit')
        BtnCancel.clicked.connect(self.close)
        e1.setEchoMode(QLineEdit.Password)
        e1.textChanged.connect(self.textchanged)
        flo.addRow('Password\xef\xbc\x9a', e1)
        flo.addRow(BtnOk, BtnCancel)
        self.setLayout(flo)
        self.center()

    def textchanged(self, text):
        code = self.encryption(text)
        now_time = datetime.datetime.today()
        if code == 'a124d06e3b9d5b95c6fd37a0ea1b322f':
            activate_time = datetime.datetime(2020, 8, 31, 23, 59, 59)
            vali_time = activate_time - now_time
            if int(vali_time.days) < 0:
                print('The key Key has expired, please contact the NTU-YR-Lab425.')
            else:
                print('The key is still available for ' + str(vali_time.days) + ' days.')
                self.close()
                MyMainWindow.show()

    def encryption(self, code):
        code_method = hashlib.md5()
        code_method.update(code.encode('utf-8', **None))
        return code_method.hexdigest()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

class EmittingStr(QtCore.QObject):  # display terminal
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

class Thread_calibration(QThread):

    def __int__(self=None):
        super(Thread_calibration, self).__init__()

    def update_parameter(self, job_set):
        self.job_set = job_set

    def kernel(self):
        MyMainWindow.tabWidget.setCurrentIndex(0)
        MyMainWindow.block_project_run(block=True)
        print('3D calibration....')
        n = 1
        for job in self.job_set:
            time.sleep(0.1)
            MyMainWindow.calibration_printer = True
            # MyMainWindow.progressBar_3.setProperty('value', (n - 1) * 100 / len(self.job_set))
            # 1 freeze
            if job[1].lower() == 'false':
                # 0 camera / 2 checkerboard path
                new_file = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_' + job[0] + '_checkerboard' + os.path.splitext(job[2])[1]
                if job[12].lower() == 'false':
                    shutil.copyfile(job[2], new_file)
                    # 3 block width / 4 block height / 5. block length
                    block_length = float(job[5])
                    block_range = [int(job[3]), int(job[3]), int(job[4]), int(job[4])]
                    # 8 i_frame / 9 d_frame / 10 f_frame
                    i_frame, d_frame = int(job[8]), int(job[9])
                    if job[10].lower() == 'default':
                        f_frame = 'default'
                    else:
                        f_frame = int(job[10])
                    # 7 Subpix filter
                    Subpix = True if job[7].lower() == 'true' else False
                    # 6 core
                    if job[6] == 'single':
                        checkerboard_calibration.calibration(new_file).scan_checkboard(block_length, i_frame, d_frame, f_frame, block_range, Subpix)
                    elif job[6] == 'Multi (max)':
                        checkerboard_calibration.calibration(new_file).muticore_scan_checkboard(block_range, block_length, i_frame, d_frame, f_frame, Subpix)
                # 9 kmeans
                k = int(job[11])
                checkerboard_calibration.calibration(new_file).undistortion(k)
            n += 1
        time.sleep(0.1)
        MyMainWindow.calibration_printer = False
        # MyMainWindow.progressBar_3.setProperty('value', 100)
        MyMainWindow.block_project_run(block=False)
        print('Complete!')

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class Thread_transformation(QThread):

    def __int__(self=None):
        super(Thread_transformation, self).__init__()

    def update_parameter(self, job_set):
        self.job_set = job_set

    def kernel(self):
        print('Transformation....')
        MyMainWindow.tabWidget.setCurrentIndex(1)
        MyMainWindow.block_project_run(block=True)
        n = 1
        camera_int, camera_dist, dict_path, dict_transformation_path, camera_candidate, camera_candidate_transformation, camera_cube_img, camera_cube_obj, score_threshold = [], [], [], [], [], [], [], [], []
        camera_rvec, camera_tvec = [], []
        for job in self.job_set:
            time.sleep(0.1)
            # MyMainWindow.progressBar_5.setProperty('value', (n - 1) * 100 / len(self.job_set))
            # 1 freeze
            if job[1].lower() == 'false':
                field_title = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_' + job[0]
                camera_candidate.append(field_title+'_cube'+os.path.splitext(job[2])[1])
                camera_candidate_transformation.append(field_title + '_transformation' + os.path.splitext(job[2])[1])
                camera_int.append(field_title + '_checkerboard_mtx.npy')
                camera_dist.append(field_title + '_checkerboard_dist.npy')
                camera_cube_img.append(field_title + '_cube_imgpoints.npy')
                camera_cube_obj.append(field_title + '_cube_objpoints.npy')
                dict_path.append(field_title + '_cube_block.csv')
                dict_transformation_path.append(field_title + '_transformation_block.csv')
                print(job[0]+'mtx')
                print(np.load(field_title + '_checkerboard_mtx.npy'))
                print(job[0]+'dist')
                print(np.load(field_title + '_checkerboard_dist.npy'))

        detail_csv = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_' + job[0] + '_checkerboard/calibration.csv'
        csv = pd.read_csv(detail_csv).values
        imgsize = [int(csv[0][1]), int(csv[1][1])]
        camera_position_rvec, camera_position_tvec = [], []
        print('camera relative orientation.....')
        for i in range(1, len(camera_candidate)):
            output_path = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_relational_' + str(i-1) + '_'
            checkerboard_calibration.calibration(output_path).relative_orientation_stereo(imgsize=imgsize, mtx0_path=camera_int[i-1], mtx1_path=camera_int[i], dist0_path=camera_dist[i-1], dist1_path=camera_dist[i], imgpoints0_path=camera_cube_img[i-1], imgpoints1_path=camera_cube_img[i], objpoints0_path=camera_cube_obj[i-1], objpoints1_path=camera_cube_obj[i])
            camera_rvec.append(output_path+'R.npy')
            camera_tvec.append(output_path+'T.npy')

            # bad code
            output_path = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_camera_' + str(i - 1) + '_'
            checkerboard_calibration.calibration(output_path).relative_orientation_stereo(imgsize=imgsize, mtx0_path=camera_int[0], mtx1_path=camera_int[i], dist0_path=camera_dist[0], dist1_path=camera_dist[i], imgpoints0_path=camera_cube_img[0], imgpoints1_path=camera_cube_img[i], objpoints0_path=camera_cube_obj[0], objpoints1_path=camera_cube_obj[i])
            print('C'+str(0)+'+C'+str(i)+'  Rotate...')
            print(np.load(output_path + 'R.npy'))
            print('C'+str(0)+'+C'+str(i)+'  Translation...')
            print(np.load(output_path + 'T.npy'))
            camera_position_rvec.append(output_path + 'R.npy')
            camera_position_tvec.append(output_path + 'T.npy')

        time.sleep(0.1)
        # # MyMainWindow.progressBar_5.setProperty('value', 50)
        # print('2D plot....')
        MyMainWindow.transformation_printer = True
        frame, data = threeD_pose_human.threeD().coord(output_3D=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_cube_', camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, dict_path=dict_path, imgsize=imgsize, limit_i_frame=1, limit_f_frame=2, UI=True)
        frame, data_transformation = threeD_pose_human.threeD().coord(output_3D=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_transformation_', camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, dict_path=dict_transformation_path, imgsize=imgsize, limit_i_frame=1, limit_f_frame=2, UI=True)
        fn.tool().plot_reproject(imgsize=imgsize, pose_path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_cube_threeDdata.csv', inf_path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_cube_threeDdata_inf.csv', camera_path=camera_candidate, camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, fps=1, type='cube_projection', mininame='_reprojection')
        fn.tool().plot_reproject(imgsize=imgsize, pose_path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_transformation_threeDdata.csv', inf_path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_transformation_threeDdata_inf.csv', camera_path=camera_candidate_transformation, camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, fps=1, type='cube_projection', mininame='_reprojection')
        print('3D plot....')
        leni = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6])
        lenf = np.array([1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7])
        xlim, ylim, zlim = ['', '', ''], ['', '', ''], ['', '', '']
        view = [90, 0]
        fn.tool().Plot3D(frame, data, xlim, ylim, zlim, view, path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_3D_cube.mp4', fps=3, type='cube', UI=True)
        time.sleep(0.1)
        # # MyMainWindow.progressBar_5.setProperty('value', 70)
        # print('3D transformation....')
        _, H = threeD_pose_human.threeD().get_transformation(data_transformation[0].copy(), cube_length=int(job[3]), output_path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_transformation')
        print('origin cube')
        print(data_transformation)
        data_transformation = fn.camera_orientation().threeD_transformation(dataset=data_transformation, H=H, padding=1e10)
        print('transformation cube')
        print(data_transformation)
        fn.tool().Plot3D(frame, data_transformation, xlim, ylim, zlim, view, path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_3D_cube_transformation.mp4', fps=3, type='cube', UI=True)
        camera_transformation = fn.camera_orientation().camera_coords_transformation(rvec=camera_position_rvec, tvec=camera_position_tvec, H=H, padding=1e10)
        # draw field
        camera_center, camera_length, camera_name, line_camera = [], [], [], []
        for c in range(camera_transformation.shape[1]):
            camera_name.append('C ' + str(c+1))
            camera_center.append(camera_transformation[0][c])
            camera_length.append(0.1)
            d_XY = np.sqrt(camera_transformation[0][c][0] ** 2 + camera_transformation[0][c][1] ** 2)
            d_XYZ = np.sqrt(np.sum(np.square(camera_transformation[0][c])))
            print('XY distance = ', d_XY)
            print('XYZ distance = ', d_XYZ)
            # line_camera.append(['C ' + str(c), 'auxiliary'])
        fn.tool().Plot_baseball_field(camera_center, camera_length, camera_name, line_camera, xlim=[-80, 80], ylim=[-95, 65], zlim=None, view=[180, 270], img_path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_field_XZ.png', UI=True)
        fn.tool().Plot_baseball_field(camera_center, camera_length, camera_name, line_camera, xlim=[-80, 80], ylim=[-95, 65], zlim=None, view=[180, 0], img_path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_field_YZ.png', UI=True)
        fn.tool().Plot_baseball_field(camera_center, camera_length, camera_name, line_camera, xlim=[-80, 80], ylim=[-95, 65], zlim=None, view=[-90, 0], img_path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_field_XY.png', UI=True)
        fn.tool().camera_pos2angle(camera_center)
        time.sleep(0.1)
        # MyMainWindow.progressBar_5.setProperty('value', 90)
        frame, segment_length = fn.joint_analysis().segment_analysis(path=MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_transforamtion_segment_length.csv', frame=frame, data=data, leni=leni, lenf=lenf)
        loss = np.abs(segment_length - int(job[3])) / int(job[3]) * 100
        print('cube_length [m] = ', segment_length.reshape(-1))
        print('cube length average loss [m] = ', np.average(loss)*int(job[3])/100)
        print('cube length loss [%] = ', loss.reshape(-1))
        print('cube length average loss [%] = ', np.average(loss))
        MyMainWindow.block_project_run(block=False)
        print('Complete....')
        time.sleep(0.1)
        MyMainWindow.transformation_printer = False
        # MyMainWindow.progressBar_5.setProperty('value', 100)

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

def run_alphapose(info, multi_gpu=True, simu_model=2, t_try_gpu=8):
    if info[8] != 'True':
        MyMainWindow.printer_img_vis = False
    cmd = 'python3 video_demo.py'
    cmd += ' --video ' + info[1]
    cmd += ' --outdir ' + os.path.dirname(info[1]) + '/'
    cmd += ' --save_video'  # if info[11] == 'True':
    cmd += ' --sp'
    cmd += ' --detbatch ' + info[3]
    cmd += ' --mode ' + info[6]
    cmd += ' --os ' + info[7]
    cmd += ' --track 9 10 --shift 0'
    cmd += ' --yolo_batchsize ' + info[2]
    # crop
    cam = cv2.VideoCapture(info[1])
    min_x = str(0) if info[12] == 'default' else info[12]
    min_y = str(0) if info[13] == 'default' else info[13]
    max_x = str(int(cam.get(3))) if info[14] == 'default' else info[14]
    max_y = str(int(cam.get(4))) if info[15] == 'default' else info[15]
    cmd += ' --local_i ' + min_x + ' ' + min_y
    cmd += ' --local_f ' + max_x + ' ' + max_y
    # GPU
    if multi_gpu is True:
        camera_number = int(info[0][7:])
        use_rate = fn.tool().gpu_info()
        if camera_number <= len(use_rate):
            start_flag = True
            os.environ["CUDA_VISIBLE_DEVICES"] = str(camera_number-1)
        else:
            start_flag = False
        while start_flag is False:
            time.sleep(t_try_gpu*(camera_number-len(use_rate)))
            use_rate = fn.tool().gpu_info()
            for i in range(len(use_rate)):
                if use_rate[i] <= 1/simu_model:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
                    start_flag = True
                    break
    # print(cmd)
    os.system(cmd)

class Thread_multi_alphapose(QThread):

    def __int__(self=None):
        super(Thread_alphapose, self).__init__()

    def update_parameter(self, data):
        self.data = data

    def delete_printer_log(self, path):
        try:
            shutil.rmtree(path)
        except:
            pass
        os.mkdir(path)

    def processing(self):
        absolute_path = MyMainWindow.comboBox_2.itemText(MyMainWindow.comboBox_2.currentIndex())
        project_folder = fn.tool().build_project_folder(absolute_path)
        project_name = os.path.splitext(os.path.basename(absolute_path))[0]
        data = self.data.copy()
        new_data = []
        for i in range(data.shape[0]):
            tmp_data = []
            second_name = os.path.splitext(data[i][1])[1]
            shutil.copyfile(data[i][1], project_folder + project_name + '_' + data[i][0] + second_name)
            new_name = project_folder + project_name + '_' + data[i][0] + second_name
            for j in range(len(data[i])):
                if j == 1:
                    tmp_data.append(new_name)
                else:
                    tmp_data.append(data[i][j])
            new_data.append(tmp_data)
        return new_data

    def kernel(self):
        # check GPU
        MyMainWindow.tabWidget.setCurrentIndex(2)
        MyMainWindow.block_project_run(block=True)
        print('Copy the video to work path...')
        data = self.processing()
        self.step_sleep = 0.015
        print('Synchronize...')
        time.sleep(self.step_sleep)
        # MyMainWindow.display_synchronize = True
        for info in data:
            if int(info[4]) != 1 or info[5] != 'default':
                if info[5] != 'default':
                    pass
                frame_f = ''
                printer_path = os.path.dirname(info[1]) + '/print_log/'
                try:
                    os.mkdir(printer_path)
                except:
                    pass
                MyMainWindow.printer_data_path = printer_path
                fn.tool().video_shift(path=info[1], frame_i=int(info[4]), frame_f=frame_f, new_FPS='', show='txt')
                new_name = os.path.splitext(info[1])[0] + '_shift' + os.path.splitext(info[1])[1]
                os.remove(info[1])
                os.rename(new_name, info[1])
            # # rotate
            # if info[0] == 'Camera_2':
            #     fn.tool().video_rotate(path=info[1], angle=180, UI=True)  # counterclockwise
            #     rotate_name = os.path.splitext(info[1])[0] + '_rotate' + os.path.splitext(info[1])[1]
            #     os.remove(info[1])
            #     os.rename(rotate_name, info[1])

        new_data = data

        # MyMainWindow.display_synchronize = False
        print('Complete....', end='\n')
        time.sleep(self.step_sleep)
        print('AlphaPose...')
        MyMainWindow.label_7.setText('remaining time :  multi-gpu processing....')
        pool = mp.Pool()
        pool.map(run_alphapose, new_data)
        time.sleep(self.step_sleep)
        MyMainWindow.block_project_run(block=False)
        MyMainWindow.label_7.setText('remaining time :  Complete....')
        print('Complete....')

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class Thread_alphapose(QThread):

    def __int__(self=None):
        super(Thread_alphapose, self).__init__()

    def update_parameter(self, data):
        self.data = data

    def delete_printer_log(self, path):
        try:
            shutil.rmtree(path)
        except:
            pass
        os.mkdir(path)

    def processing(self):
        absolute_path = MyMainWindow.comboBox_2.itemText(MyMainWindow.comboBox_2.currentIndex())
        project_folder = fn.tool().build_project_folder(absolute_path)
        project_name = os.path.splitext(os.path.basename(absolute_path))[0]
        data = self.data.copy()
        new_data = []
        for i in range(data.shape[0]):
            tmp_data = []
            second_name = os.path.splitext(data[i][1])[1]
            shutil.copyfile(data[i][1], project_folder + project_name + '_' + data[i][0] + second_name)
            new_name = project_folder + project_name + '_' + data[i][0] + second_name
            for j in range(len(data[i])):
                if j == 1:
                    tmp_data.append(new_name)
                else:
                    tmp_data.append(data[i][j])
            new_data.append(tmp_data)
        return new_data

    def kernel(self):
        MyMainWindow.tabWidget.setCurrentIndex(2)
        MyMainWindow.block_project_run(block=True)
        print('Copy the video to work path...')
        data = self.processing()
        self.step_sleep = 0.015
        print('Synchronize...')
        time.sleep(self.step_sleep)
        MyMainWindow.display_synchronize = True
        for info in data:
            if int(info[4]) != 1 or info[5] != 'default':
                if info[5] != 'default':
                    pass
                frame_f = ''
                printer_path = os.path.dirname(info[1]) + '/print_log/'
                try:
                    os.mkdir(printer_path)
                except:
                    pass
                MyMainWindow.printer_data_path = printer_path
                fn.tool().video_shift(path=info[1], frame_i=int(info[4]), frame_f=frame_f, new_FPS='', show='txt')
                new_name = os.path.splitext(info[1])[0] + '_shift' + os.path.splitext(info[1])[1]
                os.remove(info[1])
                os.rename(new_name, info[1])
        new_data = data
        MyMainWindow.display_synchronize = False
        print('Complete....', end='\n')
        time.sleep(self.step_sleep)
        print('AlphaPose...')
        time.sleep(self.step_sleep)
        for info in new_data:
            printer_path = os.path.dirname(info[1]) + '/print_log/'
            img_path = os.path.dirname(info[1]) + '/AlphaPose_' + os.path.splitext(os.path.basename(info[1]))[0] + '/'
            MyMainWindow.total_frame = cv2.VideoCapture(info[1]).get(7)
            try:
                os.mkdir(printer_path)
            except:
                pass
            try:
                os.mkdir(img_path)
            except:
                pass
            MyMainWindow.printer_data_path = printer_path
            MyMainWindow.printer_img_path = img_path
            MyMainWindow.label_7.setText('remaining time :' + ' (' + os.path.basename(info[1]) + ')')
            MyMainWindow.display_alphapose = True
            run_alphapose(info, multi_gpu=False)
        MyMainWindow.display_alphapose = False
        MyMainWindow.block_project_run(block=False)
        print('Complete....')
        time.sleep(self.step_sleep)
        # self.delete_printer_log(printer_path)

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class Thread_reconstruction(QThread):

    def __int__(self=None):
        super(Thread_reconstruction, self).__init__()

    def update_parameter(self, job_set):
        self.job_set = job_set

    def kernel(self):
        MyMainWindow.tabWidget.setCurrentIndex(3)
        time.sleep(0.1)
        # MyMainWindow.pushButton_29.setDisabled(True)  # reconstruction
        MyMainWindow.pushButton_19.setDisabled(True)  # post
        # MyMainWindow.block_project_run(block=True)
        print('3D relational orientation....')
        MyMainWindow.label_31.setText('3D relational orientation....')
        time.sleep(0.1)
        MyMainWindow.progressBar_4.setProperty('value', 0)
        camera_int, camera_dist, dict_path, camera_candidate, camera_cube_img, camera_cube_obj, score_threshold = [], [], [], [], [], [], []
        camera_rvec, camera_tvec = [], []
        for job in self.job_set:
            if job[1].lower() == 'false':
                field_title = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_' + job[0]
                project_title = MyMainWindow.output_path + 'AlphaPose_' + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + '_' + job[0]
                camera_candidate.append(fn.tool().find_file_from_str(MyMainWindow.output_path, os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + '_' + job[0]))
                camera_int.append(field_title + '_checkerboard_mtx.npy')
                camera_dist.append(field_title + '_checkerboard_dist.npy')
                camera_cube_img.append(field_title + '_cube_imgpoints.npy')
                camera_cube_obj.append(field_title + '_cube_objpoints.npy')
                dict_path.append(project_title + '.csv')
                score_threshold.append(float(job[2]))
        detail_csv = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_' + job[0] + '_checkerboard/calibration.csv'
        csv = pd.read_csv(detail_csv).values
        imgsize = [int(csv[0][1]), int(csv[1][1])]
        for i in range(1, len(camera_candidate)):
            output_path = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_relational_' + str(i-1) + '_'
            checkerboard_calibration.calibration(output_path).relative_orientation_stereo(imgsize=imgsize, mtx0_path=camera_int[i-1], mtx1_path=camera_int[i], dist0_path=camera_dist[i-1], dist1_path=camera_dist[i], imgpoints0_path=camera_cube_img[i-1], imgpoints1_path=camera_cube_img[i], objpoints0_path=camera_cube_obj[i-1], objpoints1_path=camera_cube_obj[i])
            camera_rvec.append(output_path+'R.npy')
            camera_tvec.append(output_path+'T.npy')
        time.sleep(0.1)
        MyMainWindow.progressBar_4.setProperty('value', 30)
        print('3D reconstruction....')
        MyMainWindow.label_31.setText('3D reconstruction....')
        frame_i = int(MyMainWindow.lineEdit_4.text())
        if MyMainWindow.lineEdit_5.text().lower() == 'default':
            frame_f = 1e10
        else:
            frame_f = int(MyMainWindow.lineEdit_5.text())
        frame, data = threeD_pose_human.threeD().coord(output_3D=MyMainWindow.output_path+os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + '_', camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, dict_path=dict_path, imgsize=imgsize, limit_i_frame=frame_i, limit_f_frame=frame_f, score_threshold=score_threshold, UI=True)
        print('total frame = ', data.shape[0])
        # transformation
        if MyMainWindow.checkBox_4.isChecked():
            transformation_file = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_transformation.npy'
            if os.path.isfile(transformation_file):
                print('Transformation....')
                H = np.load(transformation_file)
                data = fn.camera_orientation().threeD_transformation(dataset=data, H=H)
                fn.tool().data2csv3d(mypath=MyMainWindow.output_path+os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata, frame=frame, data=data)
            else:
                MyMainWindow.warning_msg('The transformation is empty!')
        fn.joint_analysis().velocity_analysis(frame=frame, data=data, fps=int(MyMainWindow.lineEdit_6.text()), output_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_velocity)
        fn.joint_analysis().segment_analysis(path=MyMainWindow.output_path+os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_segment_length, frame=frame, data=data)
        time.sleep(0.1)
        MyMainWindow.progressBar_4.setProperty('value', 70)
        print('2D plot....')
        MyMainWindow.label_31.setText('2D plot....')
        fps = int(UI_threeD_plt_setting.lineEdit_9.text())
        if MyMainWindow.checkBox_3.isChecked():
            MyMainWindow.reconstruction_printer = True
            if MyMainWindow.checkBox_4.isChecked():
                transformation_file = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_transformation.npy'
                if os.path.isfile(transformation_file):
                    print('de-transformation....')
                    H = np.load(transformation_file)
                    H_inv = np.linalg.inv(H)
                    inv_data = fn.camera_orientation().threeD_transformation(dataset=data.copy(), H=H_inv)
                    tmp_name = MyMainWindow.output_path + 'tmp.csv'
                    fn.tool().data2csv3d(mypath=tmp_name, frame=frame, data=inv_data)
                    fn.tool().plot_reproject(imgsize=imgsize, pose_path=tmp_name, inf_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + '_threeDdata_inf.csv', camera_path=camera_candidate, camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, fps=fps, mininame='_reprojection', type='normal', COM=True, UI=True)
            else:
                fn.tool().plot_reproject(imgsize=imgsize, pose_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.hreeDdata, inf_path=MyMainWindow.output_path +os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + '_threeDdata_inf.csv', camera_path=camera_candidate, camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, fps=fps, type='normal', mininame='_reprojection', COM=True, UI=True)
        time.sleep(0.1)
        MyMainWindow.progressBar_4.setProperty('value', 80)
        print('3D plot....')
        MyMainWindow.label_31.setText('3D plot....')
        if MyMainWindow.checkBox_2.isChecked():
            # default_d = 0.5
            # MyMainWindow.reconstruction_printer = True
            # view = [int(UI_threeD_plt_setting.lineEdit_8.text()), int(UI_threeD_plt_setting.lineEdit_7.text())]
            # lim, lim_detail = fn.tool().get_default_lim(data)
            # if UI_threeD_plt_setting.lineEdit.text() == 'default' or UI_threeD_plt_setting.lineEdit_2.text() == 'default':
            #     xlim = [lim[0][0], lim[0][1], default_d]
            # else:
            #     xlim = [float(UI_threeD_plt_setting.lineEdit.text()), float(UI_threeD_plt_setting.lineEdit_2.text()),
            #             default_d]
            # if UI_threeD_plt_setting.lineEdit_3.text() == 'default' or UI_threeD_plt_setting.lineEdit_4.text() == 'default':
            #     ylim = [lim[1][0], lim[1][1], default_d]
            # else:
            #     ylim = [float(UI_threeD_plt_setting.lineEdit_3.text()), float(UI_threeD_plt_setting.lineEdit_4.text()),
            #             default_d]
            # if UI_threeD_plt_setting.lineEdit_6.text() == 'default' or UI_threeD_plt_setting.lineEdit_5.text() == 'default':
            #     zlim = [lim[2][0] - (lim[2][0] - lim_detail[2][0]), lim[2][1] - (lim[2][0] - lim_detail[2][0]),
            #             default_d]
            # else:
            #     zlim = [float(UI_threeD_plt_setting.lineEdit_6.text()), float(UI_threeD_plt_setting.lineEdit_5.text()),
            #             default_d]
            # print('The default 3D range = ', xlim, ylim, zlim)
            # COM_data = fn.coord2vicon().get_COM(data)
            # vicon_axis = fn.coord2vicon().get_vicon_axis(data)
            # fn.tool().Plot3D(frame, data, xlim, ylim, zlim, view, path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_video, fps=fps, type='alphapose', COM=COM_data, vicon_axis=vicon_axis, UI=True)
            Thread_mayavi().ugly_code(data_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata, movie_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_video, record=True)
        time.sleep(0.1)
        MyMainWindow.progressBar_4.setProperty('value', 100)
        MyMainWindow.reconstruction_printer = False
        # MyMainWindow.pushButton_29.setDisabled(False)  # reconstruction
        MyMainWindow.pushButton_19.setDisabled(False)  # post
        # MyMainWindow.block_project_run(block=False)
        MyMainWindow.label_31.setText('Complete!')
        print('Complete!')

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class Thread_post(QThread):

    def __int__(self=None):
        super(Thread_post, self).__init__()

    def update_parameter(self, job_set, record_video=True):
        self.job_set = job_set
        self.record_video = record_video
        self.point_name = MyMainWindow.alposepose_point_name

    def kernel(self):
        MyMainWindow.tabWidget.setCurrentIndex(4)
        MyMainWindow.block_project_run(block=True)
        print('post processing....')
        row_data_path = MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata
        frame, row_data = fn.tool().Loadcsv_3d(row_data_path)
        output_data = row_data.copy()
        n = 1
        for job in self.job_set:
            if job[0].lower() == 'false':
                time.sleep(0.1)
                # 0: 'freeze',
                MyMainWindow.progressBar_2.setProperty('value', (n - 1) * 70 / len(self.job_set))
                MyMainWindow.label_13.setText(' job' + str(n))
                # 1:'previous result'
                if job[1].lower() == 'true':
                    now_data = output_data.copy()
                elif job[1].lower() == 'false':
                    now_data = row_data.copy()
                # 2:'fit point',
                fit_point = np.arange(0, len(self.point_name)).tolist()
                if job[2] != 'all':
                    fit_point = [np.argwhere(np.array(self.point_name) == job[2]).reshape(-1)[0]]
                # 3:'fit axis'
                fit_axis = [0, 1, 2]
                if job[3] == 'X':
                    fit_axis = [0]
                elif job[3] == 'Y':
                    fit_axis = [1]
                elif job[3] == 'Z':
                    fit_axis = [2]
                # 4:'anormaly',
                preprocessing_model = None
                if job[4].lower() == 'all channel':
                    preprocessing_model = 'all'
                elif job[4].lower() == 'single channel':
                    preprocessing_model = 'single'
                # 5:'anormaly ratio'
                anormaly_ratio = float(job[5])
                # 6:'FFT',
                if job[6].lower() == 'true':
                    # 7:'FFT noise ratio',
                    fft_ratio = [float(job[7]), float(job[7])]
                    # 8:'FFT segmentation',
                    saddle_dmin = None
                    if job[8].lower() == 'true':
                        # 9:'FFT sport ratio',
                        fft_ratio = [float(job[7]), float(job[9])]
                        # 11:'FFT segmentation min distance',
                        saddle_dmin = float(job[11])
                    # 10:'FFT segmentation type
                    segmentation_method = job[10]
                    output_data = fn.post_processing('').fft_model(frame=frame, input_data=now_data.copy(), output_data=output_data.copy(), preprocessing_model=preprocessing_model, anormaly_ratio=anormaly_ratio, fft_ratio=fft_ratio, saddle_dmin=saddle_dmin, axis_set=fit_axis, fit_point=fit_point, segmentation_method=segmentation_method, UI=True)
                # 12:'poly',
                elif job[12].lower() == 'true':
                    # 13:'poly order',
                    poly_order = [int(job[13])]
                    # 14:'poly fit min',
                    # 15:'poly fit max',
                    fit_area = [[int(job[14]), int(job[15])]]
                    fit_data = [[int(job[14]), int(job[15])]]
                    mask_area = []
                    # 16:'poly crop',
                    if job[16].lower() == 'true':
                        # 17:'poly crop min',
                        # 18:'poly crop max',
                        fit_data = [[int(job[17]), int(job[18])]]
                    # 19:'poly mask',
                    if job[19].lower() == 'true':
                        # 20:'poly mask min',
                        # 21:'poly mask max']
                        mask_area = [[int(job[20]), int(job[21])]]
                    output_data = fn.post_processing('').interpolation_model(data=now_data.copy(), order=poly_order, output_data=output_data.copy(), fit_point=fit_point, fit_axis=fit_axis, fit_area=fit_area, fit_data=fit_data, mask_area=mask_area, preprocessing_model=preprocessing_model, anormaly_ratio=anormaly_ratio, UI=True)
            n += 1
        time.sleep(0.1)
        MyMainWindow.progressBar_2.setProperty('value', 100)
        fn.tool().data2csv3d(MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_post, frame, output_data, '')
        fn.joint_analysis().segment_analysis(path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_post_segment_length, frame=frame, data=output_data)
        fn.joint_analysis().velocity_analysis(frame=frame, data=output_data, fps=int(MyMainWindow.lineEdit_6.text()), output_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_post_velocity)
        MyMainWindow.run_post_plt()
        fn.tool().plot_coord(frame, [output_data], index=[10], UI=True)
        if self.record_video:
            print('2D plot....')
            MyMainWindow.post_printer = True
            MyMainWindow.label_13.setText('2D plot....')
            fps = int(UI_threeD_plt_setting.lineEdit_9.text())
            if MyMainWindow.checkBox_6.isChecked():
                reconstruciton_job_set = MyMainWindow.read_reconstruction_table()
                camera_int, camera_dist, dict_path, camera_candidate, camera_cube_img, camera_cube_obj, score_threshold = [], [], [], [], [], [], []
                camera_rvec, camera_tvec = [], []
                for job in reconstruciton_job_set:
                    field_title = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_' + job[0]
                    camera_candidate.append(fn.tool().find_file_from_str(MyMainWindow.output_path, os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + '_' + job[0]))
                    camera_int.append(field_title + '_checkerboard_mtx.npy')
                    camera_dist.append(field_title + '_checkerboard_dist.npy')
                detail_csv = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_' + job[0] + '_checkerboard/calibration.csv'
                csv = pd.read_csv(detail_csv).values
                imgsize = [int(csv[0][1]), int(csv[1][1])]
                for i in range(1, len(camera_candidate)):
                    output_path = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_relational_' + str(i - 1) + '_'
                    camera_rvec.append(output_path + 'R.npy')
                    camera_tvec.append(output_path + 'T.npy')
                if MyMainWindow.checkBox_4.isChecked():
                    transformation_file = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_transformation.npy'
                    if os.path.isfile(transformation_file):
                        print('de-transformation....')
                        H = np.load(transformation_file)
                        H_inv = np.linalg.inv(H)
                        inv_data = fn.camera_orientation().threeD_transformation(dataset=output_data.copy(), H=H_inv)
                        tmp_name = MyMainWindow.output_path + 'tmp.csv'
                        fn.tool().data2csv3d(mypath=tmp_name, frame=frame, data=inv_data)
                        fn.tool().plot_reproject(imgsize=imgsize, pose_path=tmp_name, inf_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + '_threeDdata_inf.csv', camera_path=camera_candidate, camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, fps=fps, type='normal', mininame='_reprojection_post', COM=True, UI=True)
                else:
                    fn.tool().plot_reproject(imgsize=imgsize, pose_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_post, inf_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + '_threeDdata_inf.csv', camera_path=camera_candidate, camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, fps=fps, type='normal', mininame='_reprojection_post', UI=True)
            time.sleep(0.1)
            # MyMainWindow.progressBar_2.setProperty('value', 80)
            print('3D plot....')
            MyMainWindow.post_printer = True
            MyMainWindow.label_13.setText('3D plot....')
            if MyMainWindow.checkBox_7.isChecked():
                # default_d = 0.5
                # view = [int(UI_threeD_plt_setting.lineEdit_8.text()), int(UI_threeD_plt_setting.lineEdit_7.text())]
                # lim, lim_detail = fn.tool().get_default_lim(output_data)
                # if UI_threeD_plt_setting.lineEdit.text() == 'default' or UI_threeD_plt_setting.lineEdit_2.text() == 'default':
                #     xlim = [lim[0][0], lim[0][1], default_d]
                # else:
                #     xlim = [float(UI_threeD_plt_setting.lineEdit.text()),
                #             float(UI_threeD_plt_setting.lineEdit_2.text()), default_d]
                # if UI_threeD_plt_setting.lineEdit_3.text() == 'default' or UI_threeD_plt_setting.lineEdit_4.text() == 'default':
                #     ylim = [lim[1][0], lim[1][1], default_d]
                # else:
                #     ylim = [float(UI_threeD_plt_setting.lineEdit_3.text()),
                #             float(UI_threeD_plt_setting.lineEdit_4.text()), default_d]
                # if UI_threeD_plt_setting.lineEdit_6.text() == 'default' or UI_threeD_plt_setting.lineEdit_5.text() == 'default':
                #     zlim = [lim[2][0] - (lim[2][0] - lim_detail[2][0]), lim[2][1] - (lim[2][0] - lim_detail[2][0]),
                #             default_d]
                # else:
                #     zlim = [float(UI_threeD_plt_setting.lineEdit_6.text()),
                #             float(UI_threeD_plt_setting.lineEdit_5.text()), default_d]
                # print('The default 3D range = ', xlim, ylim, zlim)
                # COM_data = fn.coord2vicon().get_COM(output_data)
                # vicon_axis = fn.coord2vicon().get_vicon_axis(output_data)
                # fn.tool().Plot3D(frame, output_data, xlim, ylim, zlim, view, path=MyMainWindow.output_path +
                #                                                                   os.path.splitext(os.path.basename(
                #                                                                       MyMainWindow.project_name))[
                #                                                                       0] + MyMainWindow.threeDdata_post_video,
                #                  fps=fps, type='alphapose', COM=COM_data, vicon_axis=vicon_axis, UI=True)
                Thread_mayavi().ugly_code(data_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_post, movie_path=MyMainWindow.output_path + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + MyMainWindow.threeDdata_post_video, record=True)
        time.sleep(0.3)
        # MyMainWindow.progressBar_2.setProperty('value', 100)
        MyMainWindow.label_13.setText('Finish')
        MyMainWindow.post_printer = False
        MyMainWindow.block_project_run(block=False)
        print('Complete!')

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class Thread_run_all_project(QThread):

    def __int__(self=None):
        super(Thread_all_alphapose, self).__init__()

    def update_parameter(self, job_set):
        self.job_set = job_set  # ['alphapose', 'reconstruction', 'post']

    def kernel(self):
        count = MyMainWindow.comboBox_2.count()
        init = MyMainWindow.comboBox_2.currentIndex()
        for i in range(init, count):
            for job in self.job_set:
                flag = True
                while flag:
                    if MyMainWindow.pushButton_31.isEnabled() and job == 'alphapose':
                        MyMainWindow.comboBox_2.setCurrentIndex(i)
                        time.sleep(1)
                        MyMainWindow.run_multi_alphapose()
                        flag = False
                    elif MyMainWindow.pushButton_29.isEnabled() and job == 'reconstruction':
                        MyMainWindow.comboBox_2.setCurrentIndex(i)
                        time.sleep(1)
                        # MyMainWindow.load_reconstruction_table()
                        # time.sleep(1)
                        MyMainWindow.run_reconstruction()
                        flag = False
                    elif MyMainWindow.pushButton_19.isEnabled() and job == 'post':
                        MyMainWindow.comboBox_2.setCurrentIndex(i)
                        time.sleep(1)
                        MyMainWindow.run_post()
                        flag = False
                    time.sleep(1)

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class Thread_modify_model(QThread):

    def __int__(self=None):
        super(Thread_modify_model, self).__init__()

    def update_parameter(self, path):
        self.path = path

    def kernel(self):
        MyMainWindow.analysis_modify_mode = False
        frame_set = []
        MyMainWindow.label_3.setText('Loding...')
        MyMainWindow.pushButton_13.setDisabled(1)
        cam = cv2.VideoCapture(self.path)
        count = 0
        while (cam.isOpened()):
            ret, frame = cam.read()
            if ret == True:
                frame_set.append(frame)
                # out.write( frame )
                # cv2.imshow( 'frame', frame )
            else:
                break
            if cv2.waitKey( 1 ) == 27:
                break
            count += 1
        MyMainWindow.pushButton_13.setDisabled(0)
        img = cv2.resize(frame_set[0], (MyMainWindow.label_3.width() - 10, MyMainWindow.label_3.height() - 10), interpolation=cv2.INTER_CUBIC)
        disp_img = CV2QImage(img)
        MyMainWindow.label_3.setPixmap(QtGui.QPixmap.fromImage(disp_img))
        MyMainWindow.label_6.setText('frame = 1')
        MyMainWindow.analysis_modify_mode = True
        MyMainWindow.analysis_frame_set = frame_set

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class Thread_analysis_mediaplayer(QThread):

    def __int__(self=None):
        super(Thread_analysis_mediaplayer, self).__init__()

    def update_parameter(self, path):
        MyMainWindow.video_frame = 1
        self.path = path

    def kernel(self):
        # load the video
        MyMainWindow.analysis_modify_mode = False
        frame_set = []
        MyMainWindow.label_3.setText('Loding...')
        MyMainWindow.pushButton_13.setDisabled(1)
        cam = cv2.VideoCapture(self.path)
        while (cam.isOpened()):
            ret, frame = cam.read()
            if ret == True:
                frame_set.append(frame)
            else:
                break
            if cv2.waitKey(1) == 27:
                break
        MyMainWindow.pushButton_13.setDisabled(0)
        frame_set = np.array(frame_set)
        MyMainWindow.analysis_modify_mode = True
        MyMainWindow.analysis_frame_set = frame_set
        # play
        fps = 100
        MyMainWindow.horizontalSlider.setMaximum(frame_set.shape[0]-1)
        while True:
            time.sleep(1/fps)
            # MyMainWindow.label_3.width(), MyMainWindow.label_3.height()]
            dw = frame_set[MyMainWindow.video_frame-1].shape[1] - MyMainWindow.label_3.width()
            dh = frame_set[MyMainWindow.video_frame-1].shape[0] - MyMainWindow.label_3.height()
            if dw >= dh:
                display_w = MyMainWindow.label_3.width()
                ratio = MyMainWindow.label_3.width() / frame_set[MyMainWindow.video_frame-1].shape[1]
                display_h = frame_set[MyMainWindow.video_frame-1].shape[0] * ratio
            else:
                display_h = MyMainWindow.label_3.height()
                ratio = MyMainWindow.label_3.height() / frame_set[MyMainWindow.video_frame-1].shape[0]
                display_w = frame_set[MyMainWindow.video_frame-1].shape[1] * ratio
            display_w, display_h = int(display_w), int(display_h)
            img = cv2.resize(frame_set[MyMainWindow.video_frame-1], (display_w - 10, display_h - 10), interpolation=cv2.INTER_CUBIC)
            disp_img = CV2QImage(img)
            MyMainWindow.label_3.setPixmap(QtGui.QPixmap.fromImage(disp_img))
            MyMainWindow.label_6.setText('frame = '+str(MyMainWindow.video_frame))
            # if MyMainWindow.video_frame % (fps/2) == 0:
            MyMainWindow.horizontalSlider.setValue(MyMainWindow.video_frame-1)
            if MyMainWindow.analysis_exit_video:
                break
            if MyMainWindow.pushButton_8.text() == ' Pause':
                MyMainWindow.video_frame += 1
            if MyMainWindow.video_frame == frame_set.shape[0] and MyMainWindow.pushButton_8.text() == ' Pause':
                MyMainWindow.video_frame = 1

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class Thread_sync_mediaplayer(QThread):

    def __int__(self=None):
        super(Thread_sync_mediaplayer, self).__init__()

    def update_parameter(self, path):
        MyMainWindow.sync_frame = 1
        self.path = path

    def kernel(self):
        # load the video
        MyMainWindow.pushButton_42.setDisabled(1)
        # MyMainWindow.label_26.setText('Loding...')
        # MyMainWindow.label_27.setText('Loding...')
        # MyMainWindow.label_28.setText('Loding...')
        # MyMainWindow.label_29.setText('Loding...')
        frame_set = []
        for i in range(len(self.path)):
            temp_set = []
            cam = cv2.VideoCapture(self.path[i])
            while (cam.isOpened()):
                ret, frame = cam.read()
                if ret == True:
                    temp_set.append(frame)
                else:
                    break
                if cv2.waitKey(1) == 27:
                    break
            frame_set.append(np.array(temp_set))
        MyMainWindow.pushButton_42.setDisabled(0)
        MyMainWindow.sync_frame_set = frame_set
        # play
        fps = 50
        MyMainWindow.horizontalSlider_2.setMaximum(frame_set[0].shape[0]-1)
        while True:
            time.sleep(1/fps)
            ######## first
            dw = frame_set[0][MyMainWindow.sync_frame-1].shape[1] - MyMainWindow.label_26.width()
            dh = frame_set[0][MyMainWindow.sync_frame-1].shape[0] - MyMainWindow.label_26.height()
            if dw >= dh:
                display_w = MyMainWindow.label_26.width()
                ratio = MyMainWindow.label_26.width() / frame_set[0][MyMainWindow.sync_frame-1].shape[1]
                display_h = frame_set[0][MyMainWindow.sync_frame-1].shape[0] * ratio
            else:
                display_h = MyMainWindow.label_26.height()
                ratio = MyMainWindow.label_26.height() / frame_set[0][MyMainWindow.sync_frame-1].shape[0]
                display_w = frame_set[0][MyMainWindow.sync_frame-1].shape[1] * ratio
            display_w, display_h = int(display_w), int(display_h)
            try:
                img = cv2.resize(frame_set[0][MyMainWindow.sync_frame-1], (display_w - 10, display_h - 10), interpolation=cv2.INTER_CUBIC)
                disp_img = CV2QImage(img)
                MyMainWindow.label_26.setPixmap(QtGui.QPixmap.fromImage(disp_img))
            except:
                print('hi')

            time.sleep(1 / fps)
            ######## second
            dw = frame_set[1][MyMainWindow.sync_frame-1].shape[1] - MyMainWindow.label_27.width()
            dh = frame_set[1][MyMainWindow.sync_frame-1].shape[0] - MyMainWindow.label_27.height()
            if dw >= dh:
                display_w = MyMainWindow.label_27.width()
                ratio = MyMainWindow.label_27.width() / frame_set[1][MyMainWindow.sync_frame-1].shape[1]
                display_h = frame_set[1][MyMainWindow.sync_frame-1].shape[0] * ratio
            else:
                display_h = MyMainWindow.label_27.height()
                ratio = MyMainWindow.label_27.height() / frame_set[1][MyMainWindow.sync_frame-1].shape[0]
                display_w = frame_set[1][MyMainWindow.sync_frame-1].shape[1] * ratio
            display_w, display_h = int(display_w), int(display_h)
            try:
                img = cv2.resize(frame_set[1][MyMainWindow.sync_frame-1], (display_w - 10, display_h - 10), interpolation=cv2.INTER_CUBIC)
                disp_img = CV2QImage(img)
                MyMainWindow.label_27.setPixmap(QtGui.QPixmap.fromImage(disp_img))
            except:
                print('hi')
            #

            time.sleep(1 / fps)
            ######## third
            dw = frame_set[2][MyMainWindow.sync_frame-1].shape[1] - MyMainWindow.label_28.width()
            dh = frame_set[2][MyMainWindow.sync_frame-1].shape[0] - MyMainWindow.label_28.height()
            if dw >= dh:
                display_w = MyMainWindow.label_28.width()
                ratio = MyMainWindow.label_28.width() / frame_set[2][MyMainWindow.sync_frame-1].shape[1]
                display_h = frame_set[2][MyMainWindow.sync_frame-1].shape[0] * ratio
            else:
                display_h = MyMainWindow.label_28.height()
                ratio = MyMainWindow.label_28.height() / frame_set[2][MyMainWindow.sync_frame-1].shape[0]
                display_w = frame_set[2][MyMainWindow.sync_frame-1].shape[1] * ratio
            display_w, display_h = int(display_w), int(display_h)
            try:
                img = cv2.resize(frame_set[2][MyMainWindow.sync_frame-1], (display_w - 10, display_h - 10), interpolation=cv2.INTER_CUBIC)
                disp_img = CV2QImage(img)
                MyMainWindow.label_28.setPixmap(QtGui.QPixmap.fromImage(disp_img))
            except:
                print('hi')

            time.sleep(1 / fps)
            ######## forth
            dw = frame_set[3][MyMainWindow.sync_frame-1].shape[1] - MyMainWindow.label_29.width()
            dh = frame_set[3][MyMainWindow.sync_frame-1].shape[0] - MyMainWindow.label_29.height()
            if dw >= dh:
                display_w = MyMainWindow.label_29.width()
                ratio = MyMainWindow.label_29.width() / frame_set[3][MyMainWindow.sync_frame-1].shape[1]
                display_h = frame_set[3][MyMainWindow.sync_frame-1].shape[0] * ratio
            else:
                display_h = MyMainWindow.label_29.height()
                ratio = MyMainWindow.label_29.height() / frame_set[3][MyMainWindow.sync_frame-1].shape[0]
                display_w = frame_set[3][MyMainWindow.sync_frame-1].shape[1] * ratio
            display_w, display_h = int(display_w), int(display_h)
            try:
                img = cv2.resize(frame_set[3][MyMainWindow.sync_frame-1], (display_w - 10, display_h - 10), interpolation=cv2.INTER_CUBIC)
                disp_img = CV2QImage(img)
                MyMainWindow.label_29.setPixmap(QtGui.QPixmap.fromImage(disp_img))
            except:
                print('hi')

            ########################
            MyMainWindow.label_25.setText('frame = '+str(MyMainWindow.sync_frame))
            if MyMainWindow.sync_frame % (fps/2) == 0:
                MyMainWindow.horizontalSlider_2.setValue(MyMainWindow.sync_frame-1)
            if MyMainWindow.analysis_exit_video:
                break
            if MyMainWindow.pushButton_44.text() == ' Pause':
                MyMainWindow.sync_frame += 1
            if MyMainWindow.sync_frame == frame_set[0].shape[0] and MyMainWindow.pushButton_44.text() == ' Pause':
                MyMainWindow.sync_frame = 1

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class Thread_mayavi(QThread):

    def __int__(self=None):
        super(Thread_mayavi, self).__init__()

    def ugly_code(self, data_path, movie_path=None, record=False, UI=False):
        self.data_path = data_path
        _, data = fn.tool().Loadcsv_3d(self.data_path)
        self.view = [int(UI_threeD_plt_setting.lineEdit_8.text()), int(UI_threeD_plt_setting.lineEdit_7.text())]
        lim, lim_detail = fn.tool().get_default_lim(data)
        if UI_threeD_plt_setting.lineEdit.text() == 'default' or UI_threeD_plt_setting.lineEdit_2.text() == 'default':
            self.xlim = [lim_detail[0][0], lim_detail[0][1]]
        else:
            self.xlim = [float(UI_threeD_plt_setting.lineEdit.text()), float(UI_threeD_plt_setting.lineEdit_2.text())]
        if UI_threeD_plt_setting.lineEdit_3.text() == 'default' or UI_threeD_plt_setting.lineEdit_4.text() == 'default':
            self.ylim = [lim_detail[1][0], lim_detail[1][1]]
        else:
            self.ylim = [float(UI_threeD_plt_setting.lineEdit_3.text()), float(UI_threeD_plt_setting.lineEdit_4.text())]
        if UI_threeD_plt_setting.lineEdit_6.text() == 'default' or UI_threeD_plt_setting.lineEdit_5.text() == 'default':
            self.zlim = [lim_detail[2][0], lim_detail[2][1]]
        else:
            self.zlim = [float(UI_threeD_plt_setting.lineEdit_6.text()), float(UI_threeD_plt_setting.lineEdit_5.text())]
        self.record = record
        self.UI = UI
        self.movie_path = movie_path
        self.ball_velocity = UI_threeD_plt_setting.lineEdit_10.text()
        self.run()

    def update_parameter(self, data_path, record=False, UI=False):
        self.data_path = data_path
        _, data = fn.tool().Loadcsv_3d(self.data_path)
        self.view = [int(UI_threeD_plt_setting.lineEdit_8.text()), int(UI_threeD_plt_setting.lineEdit_7.text())]
        lim, lim_detail = fn.tool().get_default_lim(data)
        if UI_threeD_plt_setting.lineEdit.text() == 'default' or UI_threeD_plt_setting.lineEdit_2.text() == 'default':
            self.xlim = [lim_detail[0][0], lim_detail[0][1]]
        else:
            self.xlim = [float(UI_threeD_plt_setting.lineEdit.text()), float(UI_threeD_plt_setting.lineEdit_2.text())]
        if UI_threeD_plt_setting.lineEdit_3.text() == 'default' or UI_threeD_plt_setting.lineEdit_4.text() == 'default':
            self.ylim = [lim_detail[1][0], lim_detail[1][1]]
        else:
            self.ylim = [float(UI_threeD_plt_setting.lineEdit_3.text()), float(UI_threeD_plt_setting.lineEdit_4.text())]
        if UI_threeD_plt_setting.lineEdit_6.text() == 'default' or UI_threeD_plt_setting.lineEdit_5.text() == 'default':
            self.zlim = [lim_detail[2][0], lim_detail[2][1]]
        else:
            self.zlim = [float(UI_threeD_plt_setting.lineEdit_6.text()), float(UI_threeD_plt_setting.lineEdit_5.text())]
        self.record = record
        self.UI = UI
        self.ball_velocity = UI_threeD_plt_setting.lineEdit_10.text()

    def kernel(self):
        cmd = 'python3 ' + MyMainWindow.sys_path + 'threeD_mayavi_v2.py'
        # # path
        cmd += ' --path ' + self.data_path
        # axis range
        cmd += ' --xrange ' + str(self.xlim[0]) + ' ' + str(self.xlim[1])
        cmd += ' --yrange ' + str(self.ylim[0]) + ' ' + str(self.ylim[1])
        cmd += ' --zrange ' + str(self.zlim[0]) + ' ' + str(self.zlim[1])
        # view
        cmd += ' --view ' + str(self.view[0]) + ' ' + str(self.view[1])
        # # UI
        if self.UI:
            cmd += ' --UI ' + str(self.UI)
        # # record
        if self.record:
            cmd += ' --record ' + str(self.record)
            cmd += ' --video_path ' + str(self.movie_path)
            cmd += ' --fps ' + UI_threeD_plt_setting.lineEdit_9.text()
            if self.ball_velocity:
                cmd += ' --ball_velocity ' + self.ball_velocity

        # # run
        os.system(cmd)
        # # make the movie
        # if self.record:
        #     secondary = ['.png', '.jpg']
        #     mayavi_path += 'mayavi_movies/movie001/'
        #     file_frame = fn.tool().range_latest_file(folder=mayavi_path, secondary=secondary)
        #     if file_frame.shape[0]:
        #         fn.tool().make_movie(data=file_frame, path=self.movie_path, fps=int(UI_threeD_plt_setting.lineEdit_9.text()))

    def run(self):
        if MyMainWindow.checkBox_5.isChecked():
            self.kernel()
        else:
            try:
                self.kernel()
            except:
                MyMainWindow.warning_msg('There is a bug on the calibration!')

class calibration_printer(QThread):

    def __int__(self=None):
        super(calibration_printer, self).__init__()

    def run_data(self, folder, secondary):
        try:
            filename = fn.tool().find_latest_file(folder=folder, secondary=secondary)
            img = cv2.imread(filename)
            dw = img.shape[1] - MyMainWindow.label_19.width()
            dh = img.shape[0] - MyMainWindow.label_19.height()
            if dw >= dh:
                display_w = MyMainWindow.label_19.width()
                ratio = MyMainWindow.label_19.width() / img.shape[1]
                display_h = img.shape[0] * ratio
            else:
                display_h = MyMainWindow.label_19.height()
                ratio = MyMainWindow.label_19.height() / img.shape[0]
                display_w = img.shape[1] * ratio
            display_w, display_h = int(display_w), int(display_h)
            img = cv2.resize(img, (display_w - 10, display_h - 10), interpolation=cv2.INTER_CUBIC)
            disp_img = CV2QImage(img)
            MyMainWindow.label_19.setPixmap(QtGui.QPixmap.fromImage(disp_img))
        except:
            print('Unknown error!: System can not get the image!')

class transformation_printer(QThread):

    def __int__(self=None):
        super(transformation_printer, self).__init__()

    def run_data(self, folder, secondary):
        try:
            filename = fn.tool().find_latest_file(folder=folder, secondary=secondary)
            img = cv2.imread(filename)
            dw = img.shape[1] - MyMainWindow.label_22.width()
            dh = img.shape[0] - MyMainWindow.label_22.height()
            if dw >= dh:
                display_w = MyMainWindow.label_22.width()
                ratio = MyMainWindow.label_22.width() / img.shape[1]
                display_h = img.shape[0] * ratio
            else:
                display_h = MyMainWindow.label_22.height()
                ratio = MyMainWindow.label_22.height() / img.shape[0]
                display_w = img.shape[1] * ratio
            display_w, display_h = int(display_w), int(display_h)
            img = cv2.resize(img, (display_w - 10, display_h - 10), interpolation=cv2.INTER_CUBIC)
            disp_img = CV2QImage(img)
            MyMainWindow.label_22.setPixmap(QtGui.QPixmap.fromImage(disp_img))
        except:
            print('Unknown error!: System can not get the image!')

class alphapose_printer(QThread):

    # trigger = pyqtSignal()
    def __int__(self=None):
        super(alphapose_printer, self).__init__()

    def init_printer(self, path):
        # create data
        self.path = path
        f_print = open(self.path+'init_data.txt', 'w')
        f_print.write('progress_rate, remaining_time, vidoe_name')
        f_print.write('\n')
        f_print.write(str(0) + ' ' + str(0) + ' unknown')
        f_print.close()
        cv2.imwrite(self.path+'A_init_img.png', np.zeros((300, 300)))

    def get_printer_name(self, type_file, hand_mark=''):
        all_files = os.listdir(self.path)
        my_file = []
        for f in all_files:
            fullpath = os.path.join( self.path, f )
            secondaryname = os.path.splitext( fullpath )[1]
            if secondaryname in type_file:
                if hand_mark:
                    if f[0] == hand_mark:
                        my_file.append(f)
                else:
                    my_file.append(f)
        my_file = sorted(my_file, key=lambda x: os.path.getmtime(os.path.join(self.path, x)))
        if my_file:
            return self.path+my_file[-1]
        else:
            return ''

    def create_data(self, delay_time, n_name, total_frame):
        delay_time = delay_time/10000
        self.tf = time.time()
        n_name = os.path.basename(n_name)[2:-4]
        try:
            n_name = int(n_name)
        except:
            n_name = 0
        progress = int(100*n_name/total_frame)
        remaining_time = round(delay_time * (total_frame - n_name), 2)
        (h, m, s) = fn.tool().s2hms(float(remaining_time))
        MyMainWindow.label_8.setText(str(h) + 'h ' + str(m) + 'm ' + str(s) + 's')
        self.progress = 0 if progress < 0 else progress
        self.progress = 100 if progress > 100 else progress
        MyMainWindow.progressBar.setProperty('value', self.progress)

    def run_data(self, delay_time, path):
        self.path = path
        try:
            printer_name = self.get_printer_name(type_file=['.txt'])
        except:
            printer_name = ''
        if printer_name != '':
            time.sleep(0.01)
            dataframe = pd.read_table(printer_name)
            data = dataframe.values[-1][0]
            data = data.split()
            (h, m, s) = fn.tool().s2hms(float(data[1]))
            progress = int(data[0])
            MyMainWindow.label_8.setText(str(h) + 'h ' + str(m) + 'm ' + str(s) + 's')
            self.progress = 0 if progress < 0 else progress
            self.progress = 100 if progress > 100 else progress
            MyMainWindow.progressBar.setProperty('value', self.progress)
            MyMainWindow.label_7.setText('remaining time :' + ' (' + os.path.basename(data[2]) + ')')

    def run_img(self, delay_time, path, total_frame):
        self.path = path
        try:
            printer_name = self.get_printer_name(type_file=['.png'], hand_mark='A')
        except:
            printer_name = ''
        if printer_name != '':
            time.sleep(0.02)
            img = cv2.imread(printer_name)
            img = cv2.resize(img, (MyMainWindow.label.width()-10, MyMainWindow.label.height()-10), interpolation=cv2.INTER_CUBIC)
            disp_img = CV2QImage(img)
            MyMainWindow.label.setPixmap(QtGui.QPixmap.fromImage(disp_img))
            self.create_data(delay_time, printer_name, total_frame)

class reconstruction_printer(QThread):

    def __int__(self=None):
        super(reconstruction_printer, self).__init__()

    def run_data(self, folder, secondary):
        try:
            filename = fn.tool().find_latest_file(folder=folder, secondary=secondary)
            img = cv2.imread(filename)
            dw = img.shape[1] - MyMainWindow.label_21.width()
            dh = img.shape[0] - MyMainWindow.label_21.height()
            if dw >= dh:
                display_w = MyMainWindow.label_21.width()
                ratio = MyMainWindow.label_21.width() / img.shape[1]
                display_h = img.shape[0] * ratio
            else:
                display_h = MyMainWindow.label_21.height()
                ratio = MyMainWindow.label_21.height() / img.shape[0]
                display_w = img.shape[1] * ratio
            display_w, display_h = int(display_w), int(display_h)
            img = cv2.resize(img, (display_w - 10, display_h - 10), interpolation=cv2.INTER_CUBIC)
            disp_img = CV2QImage(img)
            MyMainWindow.label_21.setPixmap(QtGui.QPixmap.fromImage(disp_img))
        except:
            print('Unknown error!: System can not get the image!')

class post_printer(QThread):

    def __int__(self=None):
        super(post_printer, self).__init__()

    def run_data(self, folder, secondary):
        try:
            filename = fn.tool().find_latest_file(folder=folder, secondary=secondary)
            img = cv2.imread(filename)
            dw = img.shape[1] - MyMainWindow.label_11.width()
            dh = img.shape[0] - MyMainWindow.label_11.height()
            if dw >= dh:
                display_w = MyMainWindow.label_11.width()
                ratio = MyMainWindow.label_11.width() / img.shape[1]
                display_h = img.shape[0] * ratio
            else:
                display_h = MyMainWindow.label_11.height()
                ratio = MyMainWindow.label_11.height() / img.shape[0]
                display_w = img.shape[1] * ratio
            display_w, display_h = int(display_w), int(display_h)
            img = cv2.resize(img, (display_w - 10, display_h - 10), interpolation=cv2.INTER_CUBIC)
            disp_img = CV2QImage(img)
            MyMainWindow.label_11.setPixmap(QtGui.QPixmap.fromImage(disp_img))
        except:
            print('Unknown error!: System can not get the image!')

class post_table_printer(QThread):

    def __int__(self=None):
        super(post_table_printer, self).__init__()

    def run_data(self):
        MyMainWindow.post_table_printer = False
        try:
            draw_post_table = True
            for table in MyMainWindow.post_table_class:
                # data loading....
                try:
                    x = table.item(table.rowCount()-1, table.columnCount()-1).text()
                except:
                    draw_post_table = False
                    break
            if draw_post_table:
                for table in MyMainWindow.post_table_class:
                    for row in range(table.rowCount()):
                        if table.item(row, 0).text().lower() == 'false' and table.item(row, 4).text().lower() == 'true':
                            color = MyMainWindow.FFT_color
                        elif table.item(row, 0).text().lower() == 'false' and table.item(row, 10).text().lower() == 'true':
                            color = MyMainWindow.Poly_color
                        else:
                            color = (255, 255, 255)
                        for column in range(table.columnCount()):
                            table.item(row, column).setBackground(QtGui.QColor(color[0], color[1], color[2]))
        except:
            print('There is a problem on post table')
        MyMainWindow.post_table_printer = True

class MyMainWindow(QtWidgets.QMainWindow, UI_main):

    def __init__(self = None, parent = None):
        super(MyMainWindow, self).__init__(parent)
        self.recent_path = []
        self.sys_path = os.getcwd() + '/'
        self.post_list_path = os.getcwd() + '/post_list.yr3d'

        self.setupUi(self)
        self.setWindowTitle('NTU YR LAB425 - Unnamed project')
        self.setWindowIcon(QIcon('icon.png'))
        self.center()
        self.delete_alphapose_table()
        self.tabWidget.setCurrentIndex(0)

        ## Mayavi Widget 1
        # container = QWidget()
        # self.widget = MayaviQWidget(container)

        # output dataname
        self.post_padding = '_post'
        self.threeDdata = '_threeDdata.csv'
        self.threeDdata_velocity = '_threeDdata_velocity.csv'
        self.threeDdata_segment_length = '_threeDdata_segment_length.csv'
        self.threeDdata_video = '_3D_pose.mp4'
        self.threeDdata_post = os.path.splitext(self.threeDdata)[0]+self.post_padding+os.path.splitext(self.threeDdata)[1]
        self.threeDdata_post_velocity = os.path.splitext(self.threeDdata_velocity)[0]+self.post_padding+os.path.splitext(self.threeDdata_velocity)[1]
        self.threeDdata_post_segment_length = os.path.splitext(self.threeDdata_segment_length)[0]+self.post_padding+os.path.splitext(self.threeDdata_segment_length)[1]
        self.threeDdata_post_video = os.path.splitext(self.threeDdata_video)[0]+self.post_padding+os.path.splitext(self.threeDdata_video)[1]

        # calibration
        self.update_camera_number()
        self.pushButton_23.clicked.connect(self.open_calibration_setting)
        self.pushButton_25.clicked.connect(self.run_calibration)
        self.pushButton_22.clicked.connect(self.update_camera_number)
        self.pushButton_24.clicked.connect(self.delete_calibration_table)
        self.pushButton_30.clicked.connect(self.new_field_project)
        self.pushButton_28.clicked.connect(self.save_field_project)
        self.pushButton_27.clicked.connect(self.open_field_project)

        # alphapose
        self.pushButton.clicked.connect(self.open_alphapose_setting)
        self.pushButton_3.clicked.connect(self.delete_alphapose_table)
        self.pushButton_2.clicked.connect(self.run_alphapose)
        self.pushButton_31.clicked.connect(self.run_multi_alphapose)

        # transformation
        self.pushButton_38.clicked.connect(self.open_transformation_setting)
        self.pushButton_37.clicked.connect(self.delete_transformation_table)
        self.pushButton_36.clicked.connect(self.run_transformation)

        # 3D reconstruction
        self.pushButton_26.clicked.connect(self.load_reconstruction_table)
        self.pushButton_29.clicked.connect(self.run_reconstruction)
        self.pushButton_46.clicked.connect(self.run_mayavi)

        # post processing
        self.post_list = self.listWidget
        self.rename_post_list = self.pushButton_50
        self.delete_post_list = self.pushButton_51
        self.add_post_list = self.pushButton_52
        self.load_post_list = self.pushButton_53
        self.save_post_list = self.pushButton_54
        self.pushButton_49.clicked.connect(self.clear_post_table)
        self.pushButton_17.clicked.connect(self.open_anomaly_setting)
        self.pushButton_20.clicked.connect(self.open_post_plt_setting)
        self.pushButton_21.clicked.connect(self.save_post_plt)
        self.pushButton_18.clicked.connect(self.delete_post_table)
        self.pushButton_40.clicked.connect(self.unify_post_table)
        self.pushButton_16.clicked.connect(self.run_post_plt)
        self.pushButton_19.clicked.connect(self.run_post)
        self.pushButton_48.clicked.connect(self.run_single_axis)
        self.pushButton_47.clicked.connect(self.run_mayavi)
        self.add_post_list.clicked.connect(self.run_add_post_list)
        self.delete_post_list.clicked.connect(self.run_delete_post_list)
        self.rename_post_list.clicked.connect(self.run_rename_post_list)
        self.save_post_list.clicked.connect(self.run_save_post_list)
        self.load_post_list.clicked.connect(self.run_load_post_list)
        self.post_list.doubleClicked.connect(self.change_post_default)

        # analysis
        self.csv_box_path = []
        self.plt_box_path = []
        self.video_box_path = np.array([])
        self.analysis_frame_set, self.sync_frame_set = [], []
        self.analysis_video_path = ''
        self.display_analysis_combox = True
        self.analysis_modify_mode = True
        self.analysis_exit_video = False
        self.video_frame, self.sync_frame = 1, 1
        self.pushButton_12.clicked.connect(self.analysis_combox)
        self.pushButton_8.clicked.connect(self.analysis_video_pause)
        self.pushButton_44.clicked.connect(self.sync_video_pause)
        self.horizontalSlider.valueChanged.connect(self.define_video_frame)
        self.horizontalSlider_2.valueChanged.connect(self.define_sync_frame)
        self.pushButton_10.clicked.connect(self.save_analysis_video)
        self.pushButton_13.clicked.connect(self.run_analysis_video)
        self.pushButton_42.clicked.connect(self.run_sync_video)
        self.tableWidget_2.itemChanged.connect(self.run_replot_pose)
        self.comboBox_5.currentTextChanged.connect(self.run_analysis_csv)
        self.pushButton_9.clicked.connect(self.save_analysis_csv)
        self.pushButton_14.clicked.connect(self.open_analysis_plt_setting)
        self.pushButton_15.clicked.connect(self.run_analysis_plt)
        self.pushButton_11.clicked.connect(self.save_analysis_plt)

        # Thread
        self.thread_state = False
        self.Thread_calibration = Thread_calibration()
        self.Thread_transformation = Thread_transformation()
        self.Thread_alphapose = Thread_alphapose()
        self.Thread_multi_alphapose = Thread_multi_alphapose()
        self.Thread_run_all_project = Thread_run_all_project()
        self.Thread_reconstruction = Thread_reconstruction()
        self.Thread_post = Thread_post()
        self.Thread_read_video = Thread_modify_model()
        self.Thread_analysis_mediaplayer = Thread_analysis_mediaplayer()
        self.Thread_sync_mediaplayer = Thread_sync_mediaplayer()
        self.Thread_mayavi = Thread_mayavi()
        # self.Thread_printer = Thread_printer()

        # QTimer
        self.calibration_printer = False
        self.transformation_printer = False
        self.reconstruction_printer = False
        self.post_printer = False
        self.post_table_printer = True
        self.display_synchronize = False
        self.display_alphapose = False
        self.interface_delay = 50
        self.timer = QTimer(self)
        self.timer.start(self.interface_delay)
        self.timer.timeout.connect(self.printer)

        # init the printer
        try:
            os.chdir(self.label_5.text())  # alphapose
        except:
            self.warning_msg('your model is missing AlphaPose!')
        self.total_frame = 1e10
        init_path = './'
        self.printer_data_path = init_path
        self.printer_img_path = init_path
        self.printer_img_vis = True
        alphapose_printer().init_printer(init_path)

        # system
        self.output_path, self.output_field_path = './', './'
        self.project_name, self.project_field_name = '', ''
        self.set_init_table()
        self.pushButton_4.clicked.connect(self.select_model_folder)
        self.actionNew.triggered.connect(self.new_project)
        self.pushButton_5.clicked.connect(self.new_project)
        self.actionOPen.triggered.connect(self.open_project)
        self.pushButton_6.clicked.connect(self.open_project)
        self.actionSave.triggered.connect(self.save_project)
        self.pushButton_7.clicked.connect(self.save_project)
        self.actionSave_as.triggered.connect(self.save_as_project)
        self.actionSave_As_field.triggered.connect(self.save_as_field)
        self.actionSave_all.triggered.connect(self.save_all_project)
        self.comboBox_2.currentTextChanged.connect(self.change_project)
        self.pushButton_34.clicked.connect(self.test_funciotn)
        # self.lineEdit.setText(_translate("MainWindow", 'project_1'))
        self.cookie = {}
        self.load_cookie()
        self.pushButton_32.clicked.connect(self.reload_UI)
        self.pushButton_33.clicked.connect(self.reload_system)
        self.pushButton_35.clicked.connect(self.open_threeD_plt_setting)
        self.pushButton_39.clicked.connect(self.open_threeD_plt_setting)
        self.pushButton_41.clicked.connect(self.all_project_alphapose)
        self.pushButton_43.clicked.connect(self.all_project_recons_post)
        self.pushButton_45.clicked.connect(self.all_project_alpha_recons_post)


    def test_funciotn(self):
        self.block_project_run(True)

    def warning_msg(self, title):
        QMessageBox.information(self, "Warning!", title, QMessageBox.Yes | QMessageBox.No)

    def success_msg(self, title):
        QMessageBox.information(self, "Success!", title, QMessageBox.Yes | QMessageBox.No)

    def run_replot_pose(self):
        if self.analysis_modify_mode and self.checkBox.isChecked():
            row = self.tableWidget_2.currentRow()
            frame = int(float(self.tableWidget_2.item(row, 0).text()))
            data = []
            for i in range(1, self.tableWidget_2.columnCount()):
                value = self.tableWidget_2.item(row, i).text()
                value = int(float(value)) if value != 'nan' else -1
                data.append(value)
            data = np.array(data).reshape(1, -1, 2)
            if data.shape[1] != 18:
                self.warning_msg('The data is not 2D pose!')
            else:
                if self.analysis_frame_set:
                    img = self.analysis_frame_set[frame].copy()
                    img = tool().plot2D_img(img, data)
                    img = cv2.resize(img, (self.label_3.width() - 10, self.label_3.height() - 10), interpolation=cv2.INTER_CUBIC)
                    disp_img = CV2QImage(img)
                    self.label_3.setPixmap(QtGui.QPixmap.fromImage(disp_img))
                    self.label_6.setText('frame = '+str(frame))

    def run_analysis_csv(self):
        self.analysis_modify_mode = False
        if self.display_analysis_combox:
            dataframe = pd.read_csv(self.csv_box_path[self.comboBox_5.currentIndex()])
            # initial
            row_count = self.tableWidget_2.rowCount()
            for i in range(row_count):
                self.tableWidget_2.removeRow(0)
            # index
            index = np.array(dataframe.keys())
            self.tableWidget_2.setColumnCount(len(index))
            for i in range(len(index)):
                self.tableWidget_2.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem())
                self.tableWidget_2.horizontalHeaderItem(i).setText(index[i])
            # data
            data = dataframe.values
            for j in range(data.shape[0]):
                row_count = self.tableWidget_2.rowCount()
                self.tableWidget_2.insertRow(row_count)
                info = np.array(data[j]).astype(np.str)
                for i in range(len(info)):
                    self.tableWidget_2.setItem(j, i, QTableWidgetItem(info[i]))
        self.analysis_modify_mode = True

    def save_analysis_video(self):
        if self.display_analysis_combox and self.video_box_path.shape[0]:
            origin_file = self.video_box_path[self.comboBox.currentIndex()]
            second_name = os.path.splitext(origin_file)[1]
            absolute_path = QFileDialog.getSaveFileName(self, 'Save', self.lineEdit.text(), "All Files (*)")[0]
            if os.path.splitext(absolute_path)[0]:
                absolute_path = os.path.splitext(absolute_path)[0] + second_name
                shutil.copyfile(origin_file, absolute_path)
                self.success_msg('Save!')
        else:
            self.warning_msg('The data is empty!')

    def save_analysis_csv(self):
        # index
        if self.csv_box_path and self.tableWidget_2.rowCount() !=1 and self.tableWidget_2.columnCount() != 1:
            dataframe = pd.read_csv(self.csv_box_path[self.comboBox_5.currentIndex()])
            index = dataframe.keys()
            # data
            i = ''
            data = []
            # print(self.tableWidget_2.rowCount(), self.tableWidget_2.columnCount())
            for i in range(self.tableWidget_2.rowCount()):
                for j in range(self.tableWidget_2.columnCount()):
                    value = self.tableWidget_2.item(i, j).text() if self.tableWidget_2.item(i, j).text() != 'nan' else ''
                    data.append(value)
            if i != '':
                data = np.array(data).reshape(i + 1, -1)
            df = pd.DataFrame(data, columns=index)
            absolute_path = QFileDialog.getSaveFileName(self, 'Save', self.lineEdit.text(), "Files (*.csv) ;; All Files (*)")[0]
            if os.path.splitext(absolute_path)[0]:
                absolute_path = os.path.splitext(absolute_path)[0] + '.csv'
                df.to_csv(absolute_path, index=False)
                self.success_msg('Save!')
        else:
            self.warning_msg('The data is empty!')

    def save_analysis_plt(self):
        state = self.run_analysis_plt
        if state:
            type = '.png'
            absolute_path = \
            QFileDialog.getSaveFileName(self, 'Save', self.lineEdit.text(), "Files (*.png) ;; All Files (*)")[0]
            if os.path.splitext(absolute_path)[0]:
                absolute_path = os.path.splitext(absolute_path)[0] + type
                cv2.imwrite(absolute_path, img)
                self.success_msg('Save!')

    def run_analysis_plt(self):
        state = False
        xmin = UI_analysis_plt_setting.lineEdit.text()
        xmax = UI_analysis_plt_setting.lineEdit_2.text()
        ymin = UI_analysis_plt_setting.lineEdit_3.text()
        ymax = UI_analysis_plt_setting.lineEdit_4.text()
        try:
            xrange = tuple(np.linspace(float(xmin), float(xmax), 10))
        except:
            xrange = ''
        try:
            yrange = tuple(np.linspace(float(ymin), float(ymax), 10))
        except:
            yrange = ''
        if self.display_analysis_combox:
            if UI_analysis_plt_setting.radioButton.isChecked() and UI_analysis_plt_setting.radioButton.isEnabled():
                state = True
                leni, lenf = [], []
                if UI_analysis_plt_setting.checkBox.isChecked():
                    leni.append(5)
                    lenf.append(7)
                if UI_analysis_plt_setting.checkBox_2.isChecked():
                    leni.append(7)
                    lenf.append(9)
                if UI_analysis_plt_setting.checkBox_3.isChecked():
                    leni.append(6)
                    lenf.append(8)
                if UI_analysis_plt_setting.checkBox_4.isChecked():
                    leni.append(8)
                    lenf.append(9)
                if UI_analysis_plt_setting.checkBox_5.isChecked():
                    leni.append(11)
                    lenf.append(13)
                if UI_analysis_plt_setting.checkBox_6.isChecked():
                    leni.append(13)
                    lenf.append(15)
                if UI_analysis_plt_setting.checkBox_7.isChecked():
                    leni.append(12)
                    lenf.append(14)
                if UI_analysis_plt_setting.checkBox_8.isChecked():
                    leni.append(14)
                    lenf.append(16)
                if leni:
                    leni, lenf = np.array(leni), np.array(lenf)
                    origin_file = self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_segment_length
                    post_file = self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post_segment_length
                    if os.path.isfile(post_file):
                        frame, data = fn.tool().Loadcsv_3d_segment_length(post_file)
                        fn.tool().plot_segment_length(path=self.output_path, data_name=['post'], frame=frame, dataset=[data], leni=leni, lenf=lenf, plt_y_range=yrange, plt_x_range=xrange, figsize=(int(self.label_4.width() / 100), int(self.label_4.height() / 100)), UI=True)
                    else:
                        frame, data = fn.tool().Loadcsv_3d_segment_length(origin_file)
                        fn.tool().plot_segment_length(path=self.output_path, data_name=['row'], frame=frame, dataset=[data], leni=leni, lenf=lenf, plt_y_range=yrange, plt_x_range=xrange, figsize=(int(self.label_4.width() / 100), int(self.label_4.height() / 100)), UI=True)
                    # UI_analysis_plt_setting
                    img = cv2.imread(self.output_path + 'tmp_img.png')
                    img = cv2.resize(img, (self.label_4.width() - 5, self.label_4.height() - 5), interpolation=cv2.INTER_CUBIC)
                    disp_img = CV2QImage(img)
                    self.label_4.setPixmap(QtGui.QPixmap.fromImage(disp_img))
                else:
                    self.warning_msg('Please check the chekbox on setting.')
            if UI_analysis_plt_setting.radioButton_2.isChecked() and UI_analysis_plt_setting.radioButton_2.isEnabled():
                state = True
                index = []
                if UI_analysis_plt_setting.checkBox_9.isChecked():
                    index.append(0)
                if UI_analysis_plt_setting.checkBox_10.isChecked():
                    index.append(1)
                if UI_analysis_plt_setting.checkBox_11.isChecked():
                    index.append(2)
                if UI_analysis_plt_setting.checkBox_12.isChecked():
                    index.append(3)
                if UI_analysis_plt_setting.checkBox_13.isChecked():
                    index.append(4)
                if UI_analysis_plt_setting.checkBox_14.isChecked():
                    index.append(5)
                if UI_analysis_plt_setting.checkBox_15.isChecked():
                    index.append(6)
                if UI_analysis_plt_setting.checkBox_16.isChecked():
                    index.append(7)
                if UI_analysis_plt_setting.checkBox_17.isChecked():
                    index.append(8)
                if UI_analysis_plt_setting.checkBox_18.isChecked():
                    index.append(9)
                if UI_analysis_plt_setting.checkBox_19.isChecked():
                    index.append(10)
                if UI_analysis_plt_setting.checkBox_20.isChecked():
                    index.append(11)
                if UI_analysis_plt_setting.checkBox_21.isChecked():
                    index.append(12)
                if UI_analysis_plt_setting.checkBox_22.isChecked():
                    index.append(13)
                if UI_analysis_plt_setting.checkBox_23.isChecked():
                    index.append(14)
                if UI_analysis_plt_setting.checkBox_24.isChecked():
                    index.append(15)
                if UI_analysis_plt_setting.checkBox_25.isChecked():
                    index.append(16)
                if UI_analysis_plt_setting.checkBox_26.isChecked():
                    index.append(17)
                if index:
                    index = np.array(index)
                    origin_file = self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_velocity
                    post_file = self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post_velocity
                    if os.path.isfile(post_file):
                        frame, data = fn.tool().Loadcsv_3d_velocity(post_file)
                        fn.tool().plot_velocity(frame, [data], data_name=['post'], plt_x_range=xrange, plt_y_range=yrange, index=index, path=self.output_path, figsize=(int(self.label_4.width()/60), int(self.label_4.height()/60)),UI=True)
                    else:
                        frame, data = fn.tool().Loadcsv_3d_velocity(origin_file)
                        fn.tool().plot_velocity(frame, [data], data_name=['row'], plt_x_range=xrange, plt_y_range=yrange, index=index, path=self.output_path, figsize=(int(self.label_4.width()/60), int(self.label_4.height()/60)),UI=True)
                    img = cv2.imread(self.output_path+'tmp_img.png')
                    img = cv2.resize(img, (self.label_4.width() - 5, self.label_4.height() - 5), interpolation=cv2.INTER_CUBIC)
                    disp_img = CV2QImage(img)
                    self.label_4.setPixmap(QtGui.QPixmap.fromImage(disp_img))
                else:
                    self.warning_msg('Please check the chekbox on setting.')
            if not state:
                self.warning_msg('The data is empty!')
        return state

    def run_post_plt(self):
        state = False
        dataset, dataname = [], []
        row_data_exist = os.path.isfile(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata)
        post_data_exist = os.path.isfile(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post)
        row_velocity_exist = os.path.isfile(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_velocity)
        post_velocity_exist = os.path.isfile(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post_velocity)

        if row_data_exist is False or row_velocity_exist is False:
            self.warning_msg('The data is empty!')
        else:
            index = []
            if UI_post_plt_setting.checkBox_9.isChecked():
                index.append(0)
            if UI_post_plt_setting.checkBox_10.isChecked():
                index.append(1)
            if UI_post_plt_setting.checkBox_11.isChecked():
                index.append(2)
            if UI_post_plt_setting.checkBox_12.isChecked():
                index.append(3)
            if UI_post_plt_setting.checkBox_13.isChecked():
                index.append(4)
            if UI_post_plt_setting.checkBox_14.isChecked():
                index.append(5)
            if UI_post_plt_setting.checkBox_15.isChecked():
                index.append(6)
            if UI_post_plt_setting.checkBox_16.isChecked():
                index.append(7)
            if UI_post_plt_setting.checkBox_17.isChecked():
                index.append(8)
            if UI_post_plt_setting.checkBox_18.isChecked():
                index.append(9)
            if UI_post_plt_setting.checkBox_19.isChecked():
                index.append(10)
            if UI_post_plt_setting.checkBox_20.isChecked():
                index.append(11)
            if UI_post_plt_setting.checkBox_21.isChecked():
                index.append(12)
            if UI_post_plt_setting.checkBox_22.isChecked():
                index.append(13)
            if UI_post_plt_setting.checkBox_23.isChecked():
                index.append(14)
            if UI_post_plt_setting.checkBox_24.isChecked():
                index.append(15)
            if UI_post_plt_setting.checkBox_25.isChecked():
                index.append(16)
            if UI_post_plt_setting.checkBox_26.isChecked():
                index.append(17)
            if index == []:
                self.warning_msg('Please check the chekbox on setting.')
            else:
                # UI point
                ui_table_index = np.argwhere(np.array(self.ui_point_name) == self.alposepose_point_name[index[0]]).reshape(-1)[0]
                self.tabWidget_2.setCurrentIndex(ui_table_index)
                UI_postprocessing_setting.comboBox.setCurrentIndex(ui_table_index+1)
                # plot setting
                xmin = UI_post_plt_setting.lineEdit.text()
                xmax = UI_post_plt_setting.lineEdit_2.text()
                ymin = UI_post_plt_setting.lineEdit_3.text()
                ymax = UI_post_plt_setting.lineEdit_4.text()
                try:
                    xrange = tuple(np.linspace(float(xmin), float(xmax), 10))
                except:
                    xrange = ''
                try:
                    yrange = tuple(np.linspace(float(ymin), float(ymax), 10))
                except:
                    yrange = ''
                if UI_post_plt_setting.radioButton_4.isChecked():
                    if row_velocity_exist:
                        frame, row_data = fn.tool().Loadcsv_3d_velocity(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_velocity)
                    if post_velocity_exist:
                        frame, post_data = fn.tool().Loadcsv_3d_velocity(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post_velocity)
                        dataset.append(post_data)
                        dataname.append('post')
                        max_index, max_velocity = fn.joint_analysis().find_max_velocity(data=post_data[:, index, 3])
                        print('The max velocity is ', max_velocity, 'km/hr, at ', max_index, ' frame')
                    fn.tool().plot_velocity(frame, dataset, plt_x_range=xrange, plt_y_range=yrange, index=index, path=self.output_path, figsize=(int(self.label_11.width() / 60), int(self.label_11.height() / 60)), data_name=dataname, UI=True)
                elif UI_post_plt_setting.radioButton.isChecked():
                    if row_data_exist:
                        frame, row_data = fn.tool().Loadcsv_3d(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata)
                        dataset.append(row_data)
                        dataname.append('row')
                    if post_data_exist:
                        frame, post_data = fn.tool().Loadcsv_3d(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post)
                        dataset.append(post_data)
                        dataname.append('post')
                    fn.tool().plot_coord(frame, dataset, plt_x_range=xrange, plt_y_range=yrange, index=index, figsize=(int(self.label_11.width() / 100), int(self.label_11.height() / 100)), plt_scale_high=1.2, data_name=dataname, path=self.output_path, axis_set=[0], UI=True)
                    # UI axis
                    self.axis_tab[ui_table_index].setCurrentIndex(0)
                    UI_postprocessing_setting.comboBox_2.setCurrentIndex(1)
                elif UI_post_plt_setting.radioButton_2.isChecked():
                    if row_data_exist:
                        frame, row_data = fn.tool().Loadcsv_3d(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata)
                        dataset.append(row_data)
                        dataname.append('row')
                    if post_data_exist:
                        frame, post_data = fn.tool().Loadcsv_3d(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post)
                        dataset.append(post_data)
                        dataname.append('post')
                    fn.tool().plot_coord(frame, dataset, plt_x_range=xrange, plt_y_range=yrange, index=index, figsize=(int(self.label_11.width() / 100), int(self.label_11.height() / 100)), plt_scale_high=1.2, data_name=dataname, path=self.output_path, axis_set=[1], UI=True)
                    # UI axis
                    self.axis_tab[ui_table_index].setCurrentIndex(1)
                    UI_postprocessing_setting.comboBox_2.setCurrentIndex(2)
                elif UI_post_plt_setting.radioButton_3.isChecked():
                    if row_data_exist:
                        frame, row_data = fn.tool().Loadcsv_3d(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata)
                        dataset.append(row_data)
                        dataname.append('row')
                    if post_data_exist:
                        frame, post_data = fn.tool().Loadcsv_3d(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post)
                        dataset.append(post_data)
                        dataname.append('post')
                    fn.tool().plot_coord(frame, dataset, plt_x_range=xrange, plt_y_range=yrange, index=index, figsize=(int(self.label_11.width() / 60), int(self.label_11.height() / 60)), plt_scale_high=1.2, data_name=dataname, path=self.output_path, axis_set=[2], UI=True)
                    # UI axis
                    self.axis_tab[ui_table_index].setCurrentIndex(2)
                    UI_postprocessing_setting.comboBox_2.setCurrentIndex(3)
                img = cv2.imread(self.output_path + 'tmp_img.png')
                img = cv2.resize(img, (self.label_11.width() - 5, self.label_11.height() - 5), interpolation=cv2.INTER_CUBIC)
                disp_img = CV2QImage(img)
                self.label_11.setPixmap(QtGui.QPixmap.fromImage(disp_img))
                state = True
        return state

    def save_post_plt(self):
        state = self.run_post_plt
        if state:
            img = cv2.imread(self.output_path + 'tmp_img.png')
            type = '.png'
            absolute_path = QFileDialog.getSaveFileName(self, 'Save', self.lineEdit.text(), "Files (*.png) ;; All Files (*)")[0]
            if os.path.splitext(absolute_path)[0]:
                absolute_path = os.path.splitext(absolute_path)[0] + type
                cv2.imwrite(absolute_path, img)
                self.success_msg('Save!')

    def set_init_table(self):
        # calibraiton
        self.calibraiton_title = ['Camera name', 'freeze', 'checkerboard path', 'block width', 'block height', 'block length', 'core', 'SubPix filter', 'frame i', 'd frame', 'frame f', 'kmeans k', 'kmeans only', 'cube path', 'cube length', 'disp ratio']
        self.tableWidget_4.setColumnCount(len(self.calibraiton_title))
        self.calibraiton_title = np.array(self.calibraiton_title)
        for i in range(len(self.calibraiton_title)):
            self.tableWidget_4.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem())
            self.tableWidget_4.horizontalHeaderItem(i).setText(self.calibraiton_title[i])

        # alphapose
        self.alphapose_title = ['Camera name', 'video path', 'CPU batchsize', 'GPU batchsize', 'start frame', 'end frame', 'Mode', 'OS', 'vis', 'vis_height', 'vis_width', 'save the detail', 'region min x', 'region min y', 'region max x', 'region max y']
        self.tableWidget.setColumnCount(len(self.alphapose_title))
        self.alphapose_title = np.array(self.alphapose_title)
        for i in range(len(self.alphapose_title)):
            self.tableWidget.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem())
            self.tableWidget.horizontalHeaderItem(i).setText(self.alphapose_title[i])

        # 3d reconstruction
        self.reconstruction_title = ['Camera name', 'freeze', 'score threshold']
        self.tableWidget_5.setColumnCount(len(self.reconstruction_title))
        self.reconstruction_title = np.array(self.reconstruction_title)
        for i in range(len(self.reconstruction_title)):
            self.tableWidget_5.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem())
            self.tableWidget_5.horizontalHeaderItem(i).setText(self.reconstruction_title[i])

        # transformation
        self.transformation_title = ['Camera name', 'freeze', 'cube path', 'cube length']
        self.tableWidget_6.setColumnCount(len(self.transformation_title))
        self.transformation_title = np.array(self.transformation_title)
        for i in range(len(self.transformation_title)):
            self.tableWidget_6.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem())
            self.tableWidget_6.horizontalHeaderItem(i).setText(self.transformation_title[i])

        # post
        self.color_right_picher = (0, 0, 255)
        self.color_left_picher = (255, 0, 0)
        self.FFT_color = (100, 150, 255)
        self.Poly_color = (255, 255, 0)
        self.alposepose_point_name = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle', 'neck']
        self.ui_point_name = ['right shoulder', 'right elbow', 'right wrist', 'left hip', 'left knee', 'left ankle', 'left shoulder', 'left elbow', 'left wrist', 'right hip', 'right knee', 'right ankle', 'neck', 'nose', 'left eye', 'right eye', 'left ear', 'right ear']
        self.post_title = ['freeze', 'previous result', 'anormaly', 'anormaly ratio', 'FFT', 'FFT noise ratio', 'FFT segmentation', 'FFT sport ratio', 'FFT segmentation type', 'FFT segmentation min distance', 'poly', 'poly order', 'poly fit min', 'poly fit max', 'poly crop', 'poly crop min', 'poly crop max', 'poly mask', 'poly mask min', 'poly mask max']
        right_picher, left_picher = ['right shoulder', 'right elbow', 'right wrist', 'left hip', 'left knee', 'left ankle'], ['left shoulder', 'left elbow', 'left wrist', 'right hip', 'right knee', 'right ankle']
        self.post_table_class, self.post_table_name, self.axis_tab = [], [], []

        right_picher_index, left_picher_index = [], []
        # find the picher index
        for joint in right_picher:
            right_picher_index.append(self.ui_point_name.index(joint))
        for joint in left_picher:
            left_picher_index.append(self.ui_point_name.index(joint))
        # point name on post table
        for i in range(len(self.ui_point_name)):
            self.tabWidget_2.setTabText(i, self.ui_point_name[i])
        for i in range(self.tabWidget_2.count()):
            self.post_table_name.append([self.tabWidget_2.tabText(i), 'X'])
            self.post_table_name.append([self.tabWidget_2.tabText(i), 'Y'])
            self.post_table_name.append([self.tabWidget_2.tabText(i), 'Z'])
        # post axis tab
        self.axis_tab.append(self.tabWidget_3)
        self.axis_tab.append(self.tabWidget_4)
        self.axis_tab.append(self.tabWidget_5)
        self.axis_tab.append(self.tabWidget_6)
        self.axis_tab.append(self.tabWidget_7)
        self.axis_tab.append(self.tabWidget_8)
        self.axis_tab.append(self.tabWidget_9)
        self.axis_tab.append(self.tabWidget_10)
        self.axis_tab.append(self.tabWidget_11)
        self.axis_tab.append(self.tabWidget_12)
        self.axis_tab.append(self.tabWidget_13)
        self.axis_tab.append(self.tabWidget_14)
        self.axis_tab.append(self.tabWidget_15)
        self.axis_tab.append(self.tabWidget_16)
        self.axis_tab.append(self.tabWidget_17)
        self.axis_tab.append(self.tabWidget_18)
        self.axis_tab.append(self.tabWidget_19)
        self.axis_tab.append(self.tabWidget_20)
        # post table
        self.post_table_class.append(self.tableWidget_7)
        self.post_table_class.append(self.tableWidget_8)
        self.post_table_class.append(self.tableWidget_9)
        self.post_table_class.append(self.tableWidget_10)
        self.post_table_class.append(self.tableWidget_11)
        self.post_table_class.append(self.tableWidget_12)
        self.post_table_class.append(self.tableWidget_13)
        self.post_table_class.append(self.tableWidget_14)
        self.post_table_class.append(self.tableWidget_15)
        self.post_table_class.append(self.tableWidget_16)
        self.post_table_class.append(self.tableWidget_17)
        self.post_table_class.append(self.tableWidget_18)
        self.post_table_class.append(self.tableWidget_19)
        self.post_table_class.append(self.tableWidget_20)
        self.post_table_class.append(self.tableWidget_21)
        self.post_table_class.append(self.tableWidget_22)
        self.post_table_class.append(self.tableWidget_23)
        self.post_table_class.append(self.tableWidget_24)
        self.post_table_class.append(self.tableWidget_25)
        self.post_table_class.append(self.tableWidget_26)
        self.post_table_class.append(self.tableWidget_27)
        self.post_table_class.append(self.tableWidget_28)
        self.post_table_class.append(self.tableWidget_29)
        self.post_table_class.append(self.tableWidget_30)
        self.post_table_class.append(self.tableWidget_31)
        self.post_table_class.append(self.tableWidget_32)
        self.post_table_class.append(self.tableWidget_33)
        self.post_table_class.append(self.tableWidget_34)
        self.post_table_class.append(self.tableWidget_35)
        self.post_table_class.append(self.tableWidget_36)
        self.post_table_class.append(self.tableWidget_37)
        self.post_table_class.append(self.tableWidget_38)
        self.post_table_class.append(self.tableWidget_39)
        self.post_table_class.append(self.tableWidget_40)
        self.post_table_class.append(self.tableWidget_41)
        self.post_table_class.append(self.tableWidget_42)
        self.post_table_class.append(self.tableWidget_43)
        self.post_table_class.append(self.tableWidget_44)
        self.post_table_class.append(self.tableWidget_45)
        self.post_table_class.append(self.tableWidget_46)
        self.post_table_class.append(self.tableWidget_47)
        self.post_table_class.append(self.tableWidget_48)
        self.post_table_class.append(self.tableWidget_49)
        self.post_table_class.append(self.tableWidget_50)
        self.post_table_class.append(self.tableWidget_51)
        self.post_table_class.append(self.tableWidget_52)
        self.post_table_class.append(self.tableWidget_53)
        self.post_table_class.append(self.tableWidget_54)
        self.post_table_class.append(self.tableWidget_55)
        self.post_table_class.append(self.tableWidget_56)
        self.post_table_class.append(self.tableWidget_57)
        self.post_table_class.append(self.tableWidget_58)
        self.post_table_class.append(self.tableWidget_59)
        self.post_table_class.append(self.tableWidget_60)
        for table in self.post_table_class:
            table.setColumnCount(len(self.post_title))
        self.post_title = np.array(self.post_title)
        FFT_method_index, poly_method_index = [4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        for i in range(len(self.post_title)):
            FFT_color = QtWidgets.QTableWidgetItem(self.post_title[i])
            FFT_color.setBackground(QtGui.QColor(self.FFT_color[0], self.FFT_color[1], self.FFT_color[2]))
            Poly_color = QtWidgets.QTableWidgetItem(self.post_title[i])
            Poly_color.setBackground(QtGui.QColor(self.Poly_color[0], self.Poly_color[1], self.Poly_color[2]))
            for table in self.post_table_class:
                table.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem())
                table.horizontalHeaderItem(i).setText(self.post_title[i])
            if i in FFT_method_index:
                for table in self.post_table_class:
                    table.setHorizontalHeaderItem(i, FFT_color)
            if i in poly_method_index:
                for table in self.post_table_class:
                    table.setHorizontalHeaderItem(i, Poly_color)
        # segment length
        self.alposepose_point_pair = ['5-7', '7-9', '6-8', '8-10', '11-13', '13-15', '12-14', '14-16']
        # tab index color
        kind_set, color_set = [], []
        kind_set.append(right_picher_index), color_set.append(self.color_right_picher)  # right picher
        kind_set.append(left_picher_index), color_set.append(self.color_left_picher)  # left picher
        for i in range(len(kind_set)):
            for j in kind_set[i]:
                self.tabWidget_2.tabBar().setTabTextColor(j, QtGui.QColor(color_set[i][0], color_set[i][1], color_set[i][2]))
        # post history
        if os.path.isfile(self.post_list_path):
            with open(self.post_list_path, 'rb') as file:
                self.post_model = pickle.load(file)
        else:
            self.post_model = {}
        self.update_post_model()

    def update_post_model(self):
        self.post_list.clear()
        for item in self.post_model:
            self.post_list.addItem(item)

    def change_post_default(self):
        mode, ok = QInputDialog.getItem(self, 'Select the mode.', 'post table', ['Single channel', 'XYZ channel', 'all'], 0, False)
        if ok:
            joint_index = self.tabWidget_2.currentIndex()
            joint = self.tabWidget_2.tabText(joint_index)
            axis = self.axis_tab[joint_index].currentIndex()
            item = self.post_list.currentItem().text()
            data = self.post_model[item]
            axis_name = ['X', 'Y', 'Z']
            if mode == 'all':
                # init the table
                for table in self.post_table_class:
                    for i in range(table.rowCount()):
                        table.removeRow(0)
                # default table
                for joint in data:  # [[], [], []]
                    for axis in range(len(data[joint])):  # [[job1, job2....]]
                        for job in data[joint][axis]:
                            info = list(job)
                            info.insert(2, joint)
                            info.insert(3, axis_name[axis])
                            self.add_post_table(info)
            elif mode == 'XYZ channel':
                # init the table
                table_set = [self.post_table_class[joint_index * 3], self.post_table_class[joint_index * 3+1], self.post_table_class[joint_index * 3+2]]
                for table in table_set:
                    for i in range(table.rowCount()):
                        table.removeRow(0)
                # default table
                for axis in range(3):
                    for job in data[joint][axis]:
                        info = list(job)
                        info.insert(2, joint)
                        info.insert(3, axis_name[axis])
                        self.add_post_table(info)
            elif mode == 'Single channel':
                # init the table
                table = self.post_table_class[joint_index*3+axis]
                for i in range(table.rowCount()):
                    table.removeRow(0)
                # default table
                for job in data[joint][axis]:
                    info = list(job)
                    info.insert(2, joint)
                    info.insert(3, axis_name[axis])
                    self.add_post_table(info)

    def run_add_post_list(self):
        new_name, _ = QInputDialog.getText(self, 'post list', 'Enter the name!')
        if new_name in list(self.post_model):
            self.warning_msg('the name is be used.')
        else:
            self.post_model[new_name] = self.read_post_table()
            self.update_post_model()

    def run_rename_post_list(self):
        item = self.post_list.currentItem()
        if item:
            new_name, _ = QInputDialog.getText(self, 'post list', 'Enter the new name!')
            origin_key = list(self.post_model)
            if new_name in origin_key:
                self.warning_msg('the name is be used.')
            else:
                new_model = {}
                item = item.text()
                for key in origin_key:
                    if key == item:
                        new_model[new_name] = self.post_model[key]
                    else:
                        new_model[key] = self.post_model[key]
                self.post_model = new_model
                self.update_post_model()

    def run_save_post_list(self):
        f = open(self.post_list_path, 'wb')
        pickle.dump(self.post_model, f)
        f.close()

    def run_load_post_list(self):
        absolute_path = QFileDialog.getOpenFileName(self, 'Load', self.lineEdit.text(), "Files (*.yr3d) ;; All Files (*)")[0]
        if os.path.splitext(absolute_path)[0]:
            self.post_list_path = absolute_path
            with open(self.post_list_path, 'rb') as file:
                self.post_model = pickle.load(file)
            self.update_post_model()
            self.success_msg('Success!')

    def run_delete_post_list(self):
        item = self.post_list.currentItem()
        if item:
            item = item.text()
            self.post_model.pop(item)
            self.update_post_model()

    def run_analysis_video(self):
        # print(self.label_6.text())
        restart_video_time = 0.5
        if self.display_analysis_combox and self.video_box_path != []:
            self.analysis_exit_video = True
            time.sleep(restart_video_time)
            self.analysis_exit_video = False
            if self.checkBox.isChecked():
                self.Thread_read_video.update_parameter(self.video_box_path[self.comboBox.currentIndex()])
                self.Thread_read_video.start()
            else:
                self.Thread_analysis_mediaplayer.update_parameter(self.video_box_path[self.comboBox.currentIndex()])
                self.Thread_analysis_mediaplayer.start()
        else:
            self.warning_msg('The data is empty!')

    def run_sync_video(self):
        restart_video_time = 1
        temp_path = []
        # if self.display_analysis_combox and self.video_box_path != []:
        self.analysis_exit_video = True
        print('stop the video threshold...')
        time.sleep(restart_video_time)
        self.analysis_exit_video = False
        print('start the new video threshold...')
        temp_path.append(self.video_box_path[self.comboBox_3.currentIndex()])
        temp_path.append(self.video_box_path[self.comboBox_4.currentIndex()])
        temp_path.append(self.video_box_path[self.comboBox_6.currentIndex()])
        temp_path.append(self.video_box_path[self.comboBox_7.currentIndex()])
        try:
            self.Thread_sync_mediaplayer.update_parameter(temp_path)
            self.Thread_sync_mediaplayer.start()
        except:
            pass
        # else:
        #     self.warning_msg('The data is empty!')

    def reset_all_table(self):
        # alphapose
        for i in range(self.tableWidget.rowCount()):
            self.tableWidget.removeRow(0)
        # reconstruction
        for i in range(self.tableWidget_5.rowCount()):
            self.tableWidget_5.removeRow(0)
        # post
        for table in self.post_table_class:
            for i in range(table.rowCount()):
                table.removeRow(0)
        # analysis
        for i in range(self.tableWidget_2.rowCount()):
            self.tableWidget_2.removeRow(0)

    def clear_post_table(self):
        # post
        for table in self.post_table_class:
            for i in range(table.rowCount()):
                table.removeRow(0)

    def reset_field_table(self):
        # calibration
        for i in range(self.tableWidget_4.rowCount()):
            self.tableWidget_4.removeRow(0)
        # transformation
        for i in range(self.tableWidget_6.rowCount()):
            self.tableWidget_6.removeRow(0)

    def append_the_recent_file(self, new_file):
        if new_file not in self.recent_path:
            self.recent_path.append(new_file)
            self.comboBox_2.addItem(new_file)
            count = self.comboBox_2.count()
            self.comboBox_2.setCurrentIndex(count-1)

    def block_project_run(self, block=True):
        if block:
            self.pushButton_25.setDisabled(True)  # calibration
            self.pushButton_36.setDisabled(True)  # transformation
            self.pushButton_31.setDisabled(True)  # run all
            self.pushButton_2.setDisabled(True)  # Alphapose
            self.pushButton_29.setDisabled(True)  # reconstruction
            self.pushButton_19.setDisabled(True)  # post
            self.pushButton_12.setDisabled(True)  # analysis
        else:
            self.pushButton_25.setDisabled(False)
            self.pushButton_36.setDisabled(False)
            self.pushButton_31.setDisabled(False)
            self.pushButton_2.setDisabled(False)
            self.pushButton_29.setDisabled(False)
            self.pushButton_19.setDisabled(False)
            self.pushButton_12.setDisabled(False)
            path = os.path.splitext(self.project_name)[0] + '/'
            for file in os.listdir(path):
                if os.path.isdir(path + file):
                    try:
                        shutil.rmtree(path + file)
                    except:
                        pass
                        # print('Failed to delete folder!')

    def block_field_run(self, block=True):
        if block:
            self.pushButton_25.setDisabled(True)
        else:
            self.pushButton_25.setDisabled(False)

    def new_project(self):
        self.save_cookie()
        self.project_name = ''
        self.setWindowTitle('NTU YR LAB425 - Unnamed project')
        for i in range(self.comboBox_2.count()):
            self.comboBox_2.removeItem(0)
        self.comboBox_2.setCurrentIndex(-1)
        self.lineEdit.setText('project_1')
        self.reset_all_table()
        self.output_path = './'
        self.recent_path = []
        self.new_field_project()
        self.checkBox.setDisabled(True)
        self.pushButton_13.setDisabled(True)
        self.pushButton_10.setDisabled(True)
        self.pushButton_9.setDisabled(True)
        self.pushButton_14.setDisabled(True)
        self.pushButton_15.setDisabled(True)
        self.pushButton_11.setDisabled(True)
        self.comboBox.setDisabled(True)
        self.pushButton_42.setDisabled(True)  # sync
        self.comboBox_3.setDisabled(True)  # sync
        self.comboBox_4.setDisabled(True)  # sync
        self.comboBox_6.setDisabled(True)  # sync
        self.comboBox_7.setDisabled(True)  # sync
        self.comboBox_5.setDisabled(True)

    def new_field_project(self):
        self.save_cookie()
        self.project_field_name = ''
        self.lineEdit_3.setText('field_1')
        self.reset_field_table()
        self.output_field_path = './'
        self.label_17.setText('None')

    def save_dataset(self):
        dataset = {}
        dataset['output_path'] = os.path.dirname(self.project_name) + '/' + os.path.splitext(os.path.basename(self.project_name))[0] + '/'
        self.output_path = dataset['output_path']
        dataset['alphapose'] = self.read_alphapose_table()
        dataset['reconstruction'] = self.read_reconstruction_table()
        dataset['post'] = self.read_post_table()
        dataset['field name'] = self.project_field_name
        dataset['UI interface'] = self.read_UI_interface(type='project')
        # save
        f = open(self.project_name, 'wb')
        pickle.dump(dataset, f)
        f.close()

    def save_field_dataset(self):
        dataset = {}
        dataset['output_path'] = os.path.dirname(self.project_field_name) + '/' + os.path.splitext(os.path.basename(self.project_field_name))[0] + '/'
        self.output_field_path = dataset['output_path']
        dataset['calibration'] = self.read_calibration_table()
        dataset['transformation'] = self.read_transformation_table()
        dataset['UI interface'] = self.read_UI_interface(type='field')
        # save
        f = open(self.project_field_name, 'wb')
        pickle.dump(dataset, f)
        f.close()

    def save_project(self):
        self.save_cookie()
        new = False
        if self.project_name:
            if os.path.splitext(os.path.basename(self.project_name))[0] != self.lineEdit.text():
                self.save_as_project()
        else:
            type = '.yr3d'
            absolute_path = QFileDialog.getSaveFileName(self, 'Save', self.lineEdit.text(), "Files (*.yr3d) ;; All Files (*)")[0]
            if os.path.splitext(absolute_path)[0]:
                absolute_path = os.path.splitext(absolute_path)[0]+type
                self.setWindowTitle('NTU YR LAB425 - ' + os.path.basename(absolute_path))
                self.lineEdit.setText(os.path.splitext(os.path.basename(absolute_path))[0])
                self.project_name = absolute_path
                if os.path.isdir(os.path.splitext(absolute_path)[0]):
                    shutil.rmtree(os.path.splitext(absolute_path)[0])
                os.mkdir(os.path.splitext(absolute_path)[0])
                self.save_cookie()
                self.success_msg('Save!')
                new = True
        if self.project_name:
            self.save_dataset()
            if new:
                self.append_the_recent_file(absolute_path)
        else:
            self.lineEdit.setText(os.path.splitext(os.path.basename(self.project_name))[0])
            self.warning_msg('Save failed!')

    def save_all_project(self):
        now_index = self.comboBox_2.currentIndex()
        self.comboBox_2.setCurrentIndex(-1)
        for index in range(self.comboBox_2.count()):
            self.comboBox_2.setCurrentIndex(index)
            self.save_project()
        self.comboBox_2.setCurrentIndex(now_index)

    def save_field_project(self):
        self.save_cookie()
        if self.project_field_name:
            if os.path.splitext(os.path.basename(self.project_field_name))[0] != self.lineEdit_3.text():
                self.save_as_field()
        else:
            type = '.yr3d'
            absolute_path = QFileDialog.getSaveFileName(self, 'Save', self.lineEdit_3.text(), "Files (*.yr3d) ;; All Files (*)")[0]
            if os.path.splitext(absolute_path)[0]:
                absolute_path = os.path.splitext(absolute_path)[0]+type
                self.lineEdit_3.setText(os.path.splitext(os.path.basename(absolute_path))[0])
                self.project_field_name = absolute_path
                self.output_field_path = os.path.splitext(absolute_path)[0] + '/'
                if os.path.isdir(os.path.splitext(absolute_path)[0]):
                    shutil.rmtree(os.path.splitext(absolute_path)[0])
                os.mkdir(os.path.splitext(absolute_path)[0])
                self.success_msg('Save!')
        if self.project_field_name:
            self.save_field_dataset()

    def save_as_project(self):
        self.save_cookie()
        self.block_project_run(block=True)
        if self.project_name is '':
            self.save_project()
        else:
            type = '.yr3d'
            absolute_path = QFileDialog.getSaveFileName(self, 'Save as', self.lineEdit.text(), "Files (*.yr3d) ;; All Files (*)")[0]
            if os.path.splitext(absolute_path)[0]:
                absolute_path = os.path.splitext(absolute_path)[0] + type
                if self.project_name != absolute_path:
                    # copy old folder to new folder
                    if os.path.isdir(os.path.splitext(absolute_path)[0]):
                        shutil.rmtree(os.path.splitext(absolute_path)[0])
                    shutil.copytree(os.path.splitext(self.project_name)[0], os.path.splitext(absolute_path)[0])
                    fn.tool().rename_in_folder(folder=os.path.splitext(absolute_path)[0], src=os.path.splitext(os.path.basename(self.project_name))[0], dst=os.path.splitext(os.path.basename(absolute_path))[0])
                    # updata the UI setting
                    self.setWindowTitle('NTU YR LAB425 - ' + os.path.basename(absolute_path))
                    self.lineEdit.setText(os.path.splitext(os.path.basename(absolute_path))[0])
                    self.project_name = absolute_path
                    self.save_dataset()
                    self.append_the_recent_file(absolute_path)
                    self.success_msg('Save!')
                    self.save_cookie()
                else:
                    self.save_project()
            else:
                self.lineEdit.setText(os.path.splitext(os.path.basename(self.project_name))[0])
                with open(self.project_name, 'rb') as file:
                    dataset = pickle.load(file)
                self.reset_all_table()
                self.load_all_table(dataset)
                self.warning_msg('Save failed!')
        self.block_project_run(block=False)

    def save_as_field(self):
        self.save_cookie()
        self.block_field_run(block=True)
        if self.project_field_name is '':
            self.save_field_project()
        else:
            type = '.yr3d'
            absolute_path = QFileDialog.getSaveFileName(self, 'Save as', self.lineEdit_3.text(), "Files (*.yr3d) ;; All Files (*)")[0]
            if os.path.splitext(absolute_path)[0]:
                absolute_path = os.path.splitext(absolute_path)[0] + type
                if self.project_field_name != absolute_path:
                    # copy old folder to new folder
                    if os.path.isdir(os.path.splitext(absolute_path)[0]):
                        shutil.rmtree(os.path.splitext(absolute_path)[0])
                    shutil.copytree(os.path.splitext(self.project_field_name)[0], os.path.splitext(absolute_path)[0])
                    fn.tool().rename_in_folder(folder=os.path.splitext(absolute_path)[0], src=os.path.splitext(os.path.basename(self.project_field_name))[0], dst=os.path.splitext(os.path.basename(absolute_path))[0])
                    # updata the UI setting
                    self.setWindowTitle('NTU YR LAB425 - ' + os.path.basename(absolute_path))
                    self.lineEdit_3.setText(os.path.splitext(os.path.basename(absolute_path))[0])
                    self.project_field_name = absolute_path
                    self.save_field_dataset()
                    self.success_msg('Save!')
                    self.save_cookie()
                else:
                    self.save_field_project()
            else:
                self.lineEdit_3.setText(os.path.splitext(os.path.basename(self.project_field_name))[0])
                with open(self.project_field_name, 'rb') as file:
                    dataset = pickle.load(file)
                self.reset_field_table()
                self.load_field_table(dataset)
                self.warning_msg('Save failed!')
        self.block_field_run(block=False)

    def open_project(self):
        # load
        self.save_cookie()
        type = '.yr3d'
        absolute_path = QFileDialog.getOpenFileName(self, 'Load', self.lineEdit.text(), "Files (*.yr3d) ;; All Files (*)")[0]
        if os.path.splitext(absolute_path)[0]:
            absolute_path = os.path.splitext(absolute_path)[0] + type
            self.setWindowTitle('NTU YR LAB425 - ' + os.path.basename(absolute_path))
            self.lineEdit.setText(os.path.splitext(os.path.basename(absolute_path))[0])
            self.project_name = absolute_path
            with open(self.project_name, 'rb') as file:
                dataset = pickle.load(file)
            self.append_the_recent_file(absolute_path)
            self.reset_all_table()
            self.load_all_table(dataset)
            self.output_path = os.path.dirname(self.project_name) + '/' + os.path.splitext(os.path.basename(self.project_name))[0] + '/'
            self.checkBox.setDisabled(True)
            self.pushButton_13.setDisabled(True)
            self.pushButton_10.setDisabled(True)
            self.pushButton_9.setDisabled(True)
            self.pushButton_14.setDisabled(True)
            self.pushButton_15.setDisabled(True)
            self.pushButton_11.setDisabled(True)
            self.comboBox.setDisabled(True)
            self.pushButton_42.setDisabled(True)  # sync
            self.comboBox_3.setDisabled(True)  # sync
            self.comboBox_4.setDisabled(True)  # sync
            self.comboBox_6.setDisabled(True)  # sync
            self.comboBox_7.setDisabled(True)  # sync
            self.comboBox_5.setDisabled(True)
            # field
            self.project_field_name = dataset['field name']
            if self.project_field_name and os.path.isfile(self.project_field_name):
                self.load_field_project()
            else:
                self.new_field_project()
            self.save_cookie()
            self.success_msg('Ok')

    def load_field_project(self):
        with open(self.project_field_name, 'rb') as file:
            dataset = pickle.load(file)
        self.reset_field_table()
        self.load_field_table(dataset)
        self.output_field_path = os.path.dirname(self.project_field_name) + '/' + os.path.splitext(os.path.basename(self.project_field_name))[0] + '/'
        self.label_17.setText(self.project_field_name)
        self.lineEdit_3.setText(os.path.splitext(os.path.basename(self.project_field_name))[0] )

    def open_field_project(self):
        # load
        type = '.yr3d'
        absolute_path = QFileDialog.getOpenFileName(self, 'Load', self.lineEdit_3.text(), "Files (*.yr3d) ;; All Files (*)")[0]
        if os.path.splitext(absolute_path)[0]:
            absolute_path = os.path.splitext(absolute_path)[0] + type
            self.project_field_name = absolute_path
            self.load_field_project()
            self.success_msg('Ok')

    def change_project(self):
        if self.comboBox_2.currentIndex() >= 0:
            absolute_path = self.comboBox_2.itemText(self.comboBox_2.currentIndex())
            self.setWindowTitle('NTU YR LAB425 - ' + os.path.basename(absolute_path))
            self.lineEdit.setText(os.path.splitext(os.path.basename(absolute_path))[0])
            self.project_name = absolute_path
            with open(self.project_name, 'rb') as file:
                dataset = pickle.load(file)
            self.append_the_recent_file(absolute_path)
            self.reset_all_table()
            self.load_all_table(dataset)
            self.output_path = os.path.dirname(self.project_name) + '/' + os.path.splitext(os.path.basename(self.project_name))[0] + '/'
            self.checkBox.setDisabled(True)
            self.pushButton_13.setDisabled(True)
            self.pushButton_10.setDisabled(True)
            self.pushButton_9.setDisabled(True)
            self.pushButton_14.setDisabled(True)
            self.pushButton_15.setDisabled(True)
            self.pushButton_11.setDisabled(True)
            self.comboBox.setDisabled(True)
            self.comboBox_5.setDisabled(True)
            self.pushButton_42.setDisabled(True)  # sync
            self.comboBox_3.setDisabled(True)  # sync
            self.comboBox_4.setDisabled(True)  # sync
            self.comboBox_6.setDisabled(True)  # sync
            self.comboBox_7.setDisabled(True)  # sync
            # field
            self.project_field_name = dataset['field name']
            if self.project_field_name and os.path.isfile(self.project_field_name):
                self.load_field_project()
            else:
                self.new_field_project()

    def load_all_table(self, dataset):
        # alphapose
        dataname = 'alphapose'
        if dataname in dataset:
            data = dataset[dataname]
            for info in data:
                self.add_alphapose_table(info)

        # reconstruction
        dataname = 'reconstruction'
        if dataname in dataset:
            data = dataset[dataname]
            for info in data:
                self.add_reconstruction_table(info)

        # post
        dataname = 'post'
        axis_name = ['X', 'Y', 'Z']
        if dataname in dataset:
            data = dataset[dataname]  # data = ['joint'] = [ [], [], []]
            for joint in data:  # [[], [], []]
                for axis in range(len(data[joint])):  # [[job1, job2....]]
                    for job in data[joint][axis]:
                        info = list(job)
                        info.insert(2, joint)
                        info.insert(3, axis_name[axis])
                        self.add_post_table(info)

        # UI interface
        dataname = 'UI interface'
        if dataname in dataset:
            data = dataset[dataname]
            # self.label_5.setText(data[0])
            # try:
            #     os.chdir(data[0])
            # except:
            #     pass
            self.lineEdit_4.setText(data[0])
            self.lineEdit_5.setText(data[1])
            self.lineEdit_6.setText(data[2])
            self.tabWidget.setCurrentIndex(data[3])

    def load_field_table(self, dataset):
        # calibration
        dataname = 'calibration'
        if dataname in dataset:
            data = dataset[dataname]
            for info in data:
                self.add_calibration_table(info)

        # transformation
        dataname = 'transformation'
        if dataname in dataset:
            data = dataset[dataname]
            for info in data:
                self.add_transformation_table(info)

        # UI interface
        dataname = 'UI interface'
        if dataname in dataset:
            data = dataset[dataname]
            self.lineEdit_2.setText(data[0])

    def load_reconstruction_table(self):
        # calibration
        calibration_camera = []
        calibration_table = self.read_calibration_table()
        for table in calibration_table:
            if table[0] not in calibration_camera:
                calibration_camera.append(table[0])
        # alphapose
        alphapose_camera = []
        alphapose_table = self.read_alphapose_table()
        for table in alphapose_table:
            if table[0] not in alphapose_camera:
                alphapose_camera.append(table[0])
        # 3d reconstruction
        for i in range(self.tableWidget_5.rowCount()):
            self.tableWidget_5.removeRow(0)
        for camera in calibration_camera:
            if camera in alphapose_camera:
                info = [camera]
                info.append('False')
                info.append('0.05')
                self.add_reconstruction_table(info)

    def reload_function(self):
        reload(fn)
        reload(checkerboard_calibration)
        reload(threeD_pose_human)

    def reload_UI(self):
        second_name = ['.ui']
        for f in os.listdir(self.sys_path):
            fullpath = os.path.join(self.sys_path, f)
            secondaryname = os.path.splitext(fullpath)[1]
            if secondaryname in second_name:
                fn.tool().ui2py(fullpath)
                # fn.tool().qrc2py('./UI/3D_reconstruction/image.qrc')
        self.reload_system()

    def reload_system(self):
        self.save_cookie()
        self.close()
        os.chdir(self.sys_path)
        os.system('bash restart.sh')
        exit()

    def save_cookie(self):
        self.cookie['recent_path'] = self.recent_path
        self.cookie['now index'] = self.comboBox_2.currentIndex()
        # save
        f = open(self.sys_path+'cookie.yr3d', 'wb')
        pickle.dump(self.cookie, f)
        f.close()

    def load_cookie(self):
        if os.path.isfile(self.sys_path+'cookie.yr3d'):
            with open(self.sys_path+'cookie.yr3d', 'rb') as file:
                log_cookies = pickle.load(file)
                for i in range(len(log_cookies['recent_path'])):
                    file_exist = os.path.isfile(log_cookies['recent_path'][i])
                    if file_exist:
                        self.append_the_recent_file(log_cookies['recent_path'][i])
                self.comboBox_2.setCurrentIndex(log_cookies['now index'])

    def open_calibration_setting(self):
        UI_calibration_setting.show()

    def open_alphapose_setting(self):
        UI_alphapose_setting.show()

    def open_threeD_plt_setting(self):
        UI_threeD_plt_setting.show()

    def open_anomaly_setting(self):
        UI_postprocessing_setting.show()

    def open_post_plt_setting(self):
        UI_post_plt_setting.show()

    def open_analysis_plt_setting(self):
        UI_analysis_plt_setting.show()

    def open_transformation_setting(self):
        UI_transformation_setting.show()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def printer(self):  # all printer
        image_secondary = ['.jpg', '.png']

        # calibration
        if self.calibration_printer:
            calibration_printer().run_data(folder=self.output_field_path, secondary=image_secondary)

        # transformation
        if self.transformation_printer:
            transformation_printer().run_data(folder=self.output_field_path, secondary=image_secondary)

        # alphapose
        if self.display_synchronize:
            alphapose_printer().run_data(self.interface_delay, self.printer_data_path)
        if self.printer_img_vis and self.display_alphapose:
            alphapose_printer().run_img(self.interface_delay, self.printer_img_path, self.total_frame)

        # reconstruction
        if self.reconstruction_printer:
            reconstruction_printer().run_data(folder=self.output_path, secondary=image_secondary)

        # post
        if self.post_printer:
            post_printer().run_data(folder=self.output_path, secondary=image_secondary)
        # post_table
        if self.post_table_printer:
            post_table_printer().run_data()

    def define_video_frame(self):
        self.video_frame = self.horizontalSlider.value()+1

    def define_sync_frame(self):
        self.sync_frame = self.horizontalSlider_2.value()+1

    def analysis_video_pause(self):
        if self.pushButton_8.text() == ' Play':
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/Sport_project_v1.4.6/UI/3D_reconstruction/images/pause.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.pushButton_8.setIcon(icon)
            self.pushButton_8.setText(' Pause')
        elif self.pushButton_8.text() == ' Pause':
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/Sport_project_v1.4.6/UI/3D_reconstruction/images/play.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.pushButton_8.setIcon(icon)
            self.pushButton_8.setText(' Play')

    def sync_video_pause(self):
        if self.pushButton_44.text() == ' Play':
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/Sport_project_v1.4.6/UI/3D_reconstruction/images/pause.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.pushButton_44.setIcon(icon)
            self.pushButton_44.setText(' Pause')
        elif self.pushButton_44.text() == ' Pause':
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/Sport_project_v1.4.6/UI/3D_reconstruction/images/play.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.pushButton_44.setIcon(icon)
            self.pushButton_44.setText(' Play')

    def analysis_combox(self):
        self.display_analysis_combox = False
        # restart all comboBox
        self.video_box_path = []
        self.csv_box_path = []
        self.plt_box_path = []
        for i in range(self.comboBox.count()):
            self.comboBox.removeItem(0)
            self.comboBox_5.removeItem(0)
            self.comboBox_3.removeItem(0)  # sync video
            self.comboBox_4.removeItem(0)  # sync video
            self.comboBox_6.removeItem(0)  # sync video
            self.comboBox_7.removeItem(0)  # sync video


        # video comboBox
        tmp_set = []
        # alphapose table
        alphapose_data = np.array(self.read_alphapose_table())
        if alphapose_data.shape[0] > 0:
            for i in range(alphapose_data.shape[0]):
                origin_video = os.path.dirname(self.project_name) + '/' + os.path.splitext(os.path.basename(self.project_name))[0] + '/' + os.path.splitext(os.path.basename(self.project_name))[0] + '_' + alphapose_data[i][0] + os.path.splitext(alphapose_data[i][1])[1]
                folder = os.path.dirname(origin_video) + '/'
                name_first = os.path.splitext(os.path.basename(origin_video))[0]
                name_second = os.path.splitext(os.path.basename(origin_video))[1]
                tmp_set.append(folder + name_first + name_second)
                tmp_set.append(folder + 'AlphaPose_' + name_first + '.avi')
                tmp_set.append(folder + 'AlphaPose_' + name_first + '_skeleton' + '.avi')
                tmp_set.append(folder + name_first + '_reprojection' + name_second)
                tmp_set.append(folder + name_first + '_reprojection_post' + name_second)
        # 3D reconsturction
        tmp_set.append(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_video)
        tmp_set.append(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post_video)
        # checkfile
        n = 0
        post_video, combo_set = [], [self.comboBox_3, self.comboBox_4, self.comboBox_6, self.comboBox_7]
        for tmp in tmp_set:
            if os.path.isfile(tmp):
                if 'post' in os.path.basename(tmp):
                    post_video.append(n)
                self.video_box_path.append(tmp)
                self.comboBox.addItem(os.path.splitext(os.path.basename(tmp))[0])
                self.comboBox_3.addItem(os.path.splitext(os.path.basename(tmp))[0])
                self.comboBox_4.addItem(os.path.splitext(os.path.basename(tmp))[0])
                self.comboBox_6.addItem(os.path.splitext(os.path.basename(tmp))[0])
                self.comboBox_7.addItem(os.path.splitext(os.path.basename(tmp))[0])
                n += 1
        for i in range(len(post_video)):
            combo_set[i].setCurrentIndex(post_video[i])


        # csv comboxBox and plt setting
        tmp_set = []
        # alphapose table
        alphapose_data = np.array(self.read_alphapose_table())
        if alphapose_data.shape[0] > 0:
            for i in range(alphapose_data.shape[0]):
                origin_video = os.path.dirname(self.project_name) + '/' + os.path.splitext(os.path.basename(self.project_name))[0] + '/' + os.path.splitext(os.path.basename(self.project_name))[0] + '_' + alphapose_data[i][0] + os.path.splitext(alphapose_data[i][1])[1]
                folder = os.path.dirname(origin_video) + '/'
                name_first = os.path.splitext(os.path.basename(origin_video))[0]
                tmp_set.append(folder + 'AlphaPose_' + name_first + '.csv')
        # 3D reconsturction
        tmp_set.append(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata)
        tmp_set.append(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_segment_length)
        tmp_set.append(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_velocity)
        tmp_set.append(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post)
        tmp_set.append(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post_velocity)
        tmp_set.append(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_post_segment_length)
        UI_analysis_plt_setting.sss.setDisabled(True)
        UI_analysis_plt_setting.sss_2.setDisabled(True)
        UI_analysis_plt_setting.radioButton.setDisabled(True)
        UI_analysis_plt_setting.radioButton_2.setDisabled(True)
        # checkfile
        for tmp in tmp_set:
            if os.path.isfile(tmp):
                self.csv_box_path.append(tmp)
                self.comboBox_5.addItem(os.path.splitext(os.path.basename(tmp))[0])
                if tmp == self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_segment_length:
                    UI_analysis_plt_setting.radioButton.setDisabled(False)
                if tmp == self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata_velocity:
                    UI_analysis_plt_setting.radioButton_2.setDisabled(False)
        self.video_box_path = np.array(self.video_box_path)
        self.display_analysis_combox = True
        self.checkBox.setDisabled(False)
        self.pushButton_13.setDisabled(False)
        self.pushButton_10.setDisabled(False)
        self.pushButton_9.setDisabled(False)
        self.pushButton_14.setDisabled(False)
        self.pushButton_15.setDisabled(False)
        self.pushButton_11.setDisabled(False)
        self.comboBox.setDisabled(False)
        self.pushButton_42.setDisabled(False)
        self.comboBox_5.setDisabled(False)  # sync video
        self.comboBox_3.setDisabled(False)  # sync video
        self.comboBox_4.setDisabled(False)  # sync video
        self.comboBox_6.setDisabled(False)  # sync video
        self.comboBox_7.setDisabled(False)  # sync video

    def select_model_folder(self):
        absolute_path = QFileDialog.getExistingDirectory(self) + '/'
        if absolute_path != '':
            self.label_5.setText(absolute_path)
            os.chdir(absolute_path)

    def select_output_folder(self):
        absolute_path = QFileDialog.getExistingDirectory(self) + '/'
        if absolute_path != '':
            self.label_7.setText(absolute_path)

    def add_alphapose_table(self, info):
        # check the same camera name
        alphapose_table = self.read_alphapose_table()
        add_type = True
        if len(alphapose_table):
            for i in range(len(alphapose_table)):
                if info[0] == alphapose_table[i][0]:
                    add_type = False
        if add_type:
            row_count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_count)
            info = np.array(info).astype(np.str)
            for i in range(len(info)):
                self.tableWidget.setItem(row_count, i, QTableWidgetItem(info[i]))

    def add_post_table(self, info):
        def run(info, index):
            # check the same camera name
            row = self.post_table_class[index].currentRow()
            if row == -1:
                row = self.post_table_class[index].rowCount()
            self.post_table_class[index].insertRow(row)
            info = np.array(info).astype(np.str)
            for i in range(len(info)):
                self.post_table_class[index].setItem(row, i, QTableWidgetItem(info[i]))

        # self.post_table_class, self.post_table_name
        # fit point
        job, temp = [], []
        if info[2] == 'all':
            # all index
            for i in range(self.tabWidget_2.count()):
                temp.append(self.tabWidget_2.tabText(i))
        else:
            temp.append(info[2])
        # fit axis
        for point in temp:
            if info[3] == 'all':
                job.append([point, 'X'])
                job.append([point, 'Y'])
                job.append([point, 'Z'])
            else:
                job.append([point, info[3]])
        # update the post table
        del info[2]
        del info[2]
        for mini_job in job:
            index = self.post_table_name.index([mini_job[0], mini_job[1]])
            run(info, index=index)

    def add_calibration_table(self, info):
        # check the same camera name
        calibration_table = self.read_calibration_table()
        add_type = True
        if len(calibration_table):
            for i in range(len(calibration_table)):
                if info[0] == calibration_table[i][0]:
                    add_type = False
        if add_type:
            row_count = self.tableWidget_4.rowCount()
            self.tableWidget_4.insertRow(row_count)
            info = np.array(info).astype(np.str)
            for i in range(len(info)):
                self.tableWidget_4.setItem(row_count, i, QTableWidgetItem(info[i]))

    def add_transformation_table(self, info):
        # check the same camera name
        transformation_table = self.read_transformation_table()
        add_type = True
        if len(transformation_table):
            for i in range(len(transformation_table)):
                if info[0] == transformation_table[i][0]:
                    add_type = False
        if add_type:
            row_count = self.tableWidget_6.rowCount()
            self.tableWidget_6.insertRow(row_count)
            info = np.array(info).astype(np.str)
            for i in range(len(info)):
                self.tableWidget_6.setItem(row_count, i, QTableWidgetItem(info[i]))

    def add_reconstruction_table(self, info):
        row_count = self.tableWidget_5.rowCount()
        self.tableWidget_5.insertRow(row_count)
        info = np.array(info).astype(np.str)
        for i in range(len(info)):
            self.tableWidget_5.setItem(row_count, i, QTableWidgetItem(info[i]))

    def read_calibration_table(self):
        i = ''
        data = []
        for i in range(self.tableWidget_4.rowCount()):
            for j in range(self.tableWidget_4.columnCount()):
                data.append(self.tableWidget_4.item(i, j).text())
        if i != '':
            data = np.array(data).reshape(i + 1, -1)
        return data

    def read_alphapose_table(self):
        i = ''
        data = []
        for i in range(self.tableWidget.rowCount()):
            for j in range(self.tableWidget.columnCount()):
                data.append(self.tableWidget.item(i, j).text())
        if i != '':
            data = np.array(data).reshape(i + 1, -1)
        return data

    def read_reconstruction_table(self):
        i = ''
        data = []
        for i in range(self.tableWidget_5.rowCount()):
            for j in range(self.tableWidget_5.columnCount()):
                data.append(self.tableWidget_5.item(i, j).text())
        if i != '':
            data = np.array(data).reshape(i + 1, -1)
        return data

    def read_transformation_table(self):
        i = ''
        data = []
        for i in range(self.tableWidget_6.rowCount()):
            for j in range(self.tableWidget_6.columnCount()):
                data.append(self.tableWidget_6.item(i, j).text())
        if i != '':
            data = np.array(data).reshape(i + 1, -1)
        return data

    def read_post_table(self):
        def run(table):
            i = ''
            data = []
            for i in range(table.rowCount()):
                for j in range(table.columnCount()):
                    data.append(table.item(i, j).text())
            if i != '':
                data = np.array(data).reshape(i + 1, -1)
            return data
        post_data = {}
        n = 0
        for joint in self.ui_point_name:
            post_data[joint] = [run(self.post_table_class[n]), run(self.post_table_class[n+1]), run(self.post_table_class[n+2])]
            n += 3
        return post_data

    def read_UI_interface(self, type='project'):
        info = []
        if type == 'project':
            # info.append(self.label_5.text())  # alphapose path
            info.append(self.lineEdit_4.text())  # 3d frame min
            info.append(self.lineEdit_5.text())  # 3d frame max
            info.append(self.lineEdit_6.text())  # 3d fps
            info.append(self.tabWidget.currentIndex())  # tab
            return info
        elif type == 'field':
            info.append(self.lineEdit_2.text())  # camera number
            return info

    def delete_calibration_table(self):
        row = self.tableWidget_4.currentRow()
        self.tableWidget_4.removeRow(row)

    def delete_alphapose_table(self):
        row = self.tableWidget.currentRow()
        self.tableWidget.removeRow(row)
        # row_count = self.tableWidget.rowCount()
        # self.tableWidget.removeRow(row_count - 1)

    def delete_transformation_table(self):
        row = self.tableWidget_6.currentRow()
        self.tableWidget_6.removeRow(row)
        # row_count = self.tableWidget.rowCount()
        # self.tableWidget.removeRow(row_count - 1)

    def delete_post_table(self):
        point_index = self.tabWidget_2.currentIndex()
        axis_index = self.axis_tab[point_index].currentIndex()
        table_index = point_index*3+axis_index
        row = self.post_table_class[table_index].currentRow()
        self.post_table_class[table_index].removeRow(row)

    def unify_post_table(self):
        axis_name = ['X', 'Y', 'Z']
        now_count = self.comboBox_2.currentIndex()
        data = self.read_post_table()
        count = self.comboBox_2.count()
        for i in range(count):
            self.comboBox_2.setCurrentIndex(i)
            # delete post table
            for table in self.post_table_class:
                for xx in range(table.rowCount()):
                    table.removeRow(0)
            # padding the new data
            for joint in data:  # [[], [], []]
                for axis in range(len(data[joint])):  # [[job1, job2....]]
                    for job in data[joint][axis]:
                        info = list(job)
                        info.insert(2, joint)
                        info.insert(3, axis_name[axis])
                        self.add_post_table(info)
            # for j in range(data.shape[0]):
            self.save_project()
        self.comboBox_2.setCurrentIndex(now_count)

    def check_thread_state(self):
        self.thread_state = False

    def update_camera_number(self):
        # int
        for i in range(UI_calibration_setting.comboBox_2.count()):
            UI_calibration_setting.comboBox_2.removeItem(0)
        for i in range(UI_alphapose_setting.comboBox.count()):
            UI_alphapose_setting.comboBox.removeItem(0)
        for i in range(UI_transformation_setting.comboBox_2.count()):
            UI_transformation_setting.comboBox_2.removeItem(0)

        # update
        camera_name = 'Camera_'
        camera_number = self.lineEdit_2.text()
        for i in range(int(camera_number)):
            UI_calibration_setting.comboBox_2.addItem(camera_name + str(i + 1))
            UI_alphapose_setting.comboBox.addItem(camera_name + str(i + 1))
            UI_transformation_setting.comboBox_2.addItem(camera_name+str(i+1))

    def run_calibration(self):
        self.reload_function()
        self.save_field_project()
        if self.project_field_name:
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            data = np.array(self.read_calibration_table()).copy()
            self.Thread_calibration.update_parameter(data)
            self.Thread_calibration.start()
            self.Thread_calibration.start()

    def run_transformation(self):
        self.reload_function()
        self.save_field_project()
        if self.project_field_name:
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            data = np.array(self.read_transformation_table()).copy()
            self.Thread_transformation.update_parameter(data)
            self.Thread_transformation.start()

    def run_alphapose(self):
        self.reload_function()
        self.save_project()
        if self.project_name:
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            data = np.array(self.read_alphapose_table()).copy()
            self.Thread_alphapose.update_parameter(data)
            self.Thread_alphapose.start()

    def run_multi_alphapose(self):
        self.reload_function()
        self.save_project()
        if self.project_name:
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            data = np.array(self.read_alphapose_table()).copy()
            self.Thread_multi_alphapose.update_parameter(data)
            self.Thread_multi_alphapose.start()

    def run_reconstruction(self):
        self.reload_function()
        self.save_project()
        if self.project_name:
            data = np.array(self.read_reconstruction_table()).copy()
            camera_n = len(data)
            for freeze in data:
                if freeze[1].lower() == 'true':
                    camera_n -= 1
            if camera_n >= 2:
                if not MyMainWindow.checkBox_5.isChecked():
                    UI_log.openUI()
                self.Thread_reconstruction.update_parameter(data)
                self.Thread_reconstruction.start()
            else:
                self.warning_msg('At least two cameras are needed to reconstruct the 3D data!')

    def run_post(self):
        axis_name = ['X', 'Y', 'Z']
        self.reload_function()
        self.save_project()
        row_data_exist = os.path.isfile(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata)
        if row_data_exist:
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            data = self.read_post_table()
            job_set = []
            for joint in data:  # [[], [], []]
                for axis in range(len(data[joint])):  # [[job1, job2....]]
                    for job in data[joint][axis]:
                        info = list(job)
                        info.insert(2, joint)
                        info.insert(3, axis_name[axis])
                        job_set.append(info)
            data = np.array(job_set).copy()
            self.Thread_post.update_parameter(data)
            self.Thread_post.start()
        else:
            self.warning_msg('The 3D row data is empty!')

    def run_single_axis(self):
        axis_name = ['X', 'Y', 'Z']
        self.reload_function()
        self.save_project()
        row_data_exist = os.path.isfile(self.output_path + os.path.splitext(os.path.basename(self.project_name))[0] + self.threeDdata)
        if row_data_exist:
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            data = self.read_post_table()
            joint_index = self.tabWidget_2.currentIndex()
            joint = self.tabWidget_2.tabText(joint_index)
            axis = self.axis_tab[joint_index].currentIndex()
            job_set = []
            for job in data[joint][axis]:
                info = list(job)
                info.insert(2, joint)
                info.insert(3, axis_name[axis])
                job_set.append(info)
            data = np.array(job_set).copy()
            self.Thread_post.update_parameter(job_set=data, record_video=False)
            self.Thread_post.start()
        else:
            self.warning_msg('The 3D row data is empty!')

    def run_mayavi(self):
        index = self.tabWidget.currentIndex()
        if index == 3:
            data_path = os.path.splitext(MyMainWindow.project_name)[0] + '/' + os.path.splitext(os.path.basename(MyMainWindow.project_name))[0] + self.threeDdata
        elif index == 4:
            data_path = os.path.splitext(MyMainWindow.project_name)[0]+'/'+os.path.splitext(os.path.basename(MyMainWindow.project_name))[0]+self.threeDdata_post
        else:
            data_path = './sdfsdfsdfsdfsdfsdfasdfasd.csv'
        if os.path.isfile(data_path):
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            self.Thread_mayavi.update_parameter(data_path=data_path, UI=True)
            self.Thread_mayavi.start()
        else:
            self.warning_msg('The 3D row data is empty!')

    def all_project_alphapose(self):
        job_set = ['alphapose']
        self.reload_function()
        self.save_project()
        if self.project_name:
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            self.Thread_run_all_project.update_parameter(job_set)
            self.Thread_run_all_project.start()

    def all_project_recons_post(self):
        # job_set = ['reconstruction', 'post']
        job_set = ['post']
        self.reload_function()
        self.save_project()
        if self.project_name:
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            self.Thread_run_all_project.update_parameter(job_set)
            self.Thread_run_all_project.start()

    def all_project_alpha_recons_post(self):
        job_set = ['alphapose', 'reconstruction', 'post']
        self.reload_function()
        self.save_project()
        if self.project_name:
            if not MyMainWindow.checkBox_5.isChecked():
                UI_log.openUI()
            self.Thread_run_all_project.update_parameter(job_set)
            self.Thread_run_all_project.start()

class UI_calibration_setting(QtWidgets.QMainWindow, UI_calibration_setting):

    def __init__(self, parent=None):
        super(UI_calibration_setting, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('UI calibration setting')
        self.setWindowIcon(QIcon('icon.png'))
        # single
        self.pushButton_8.clicked.connect(self.build_info)
        self.pushButton_4.clicked.connect(self.select_checkerboard_path)
        self.pushButton_6.clicked.connect(self.select_cube_path)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def build_info(self):
        # calibration(mypath).scan_checkboard(block_w, block_h, block_meter, i_frame, d_frame, n_base, block_range)
        info = []
        info.append(self.comboBox_2.currentText())  # camera
        info.append(self.checkBox_3.isChecked())  # freeze
        info.append(self.label_8.text())  # checkerboard path
        info.append(self.lineEdit_3.text())  # block w
        info.append(self.lineEdit_6.text())  # block h
        info.append(self.lineEdit_4.text())  # block meter
        if self.radioButton_6.isChecked():  # single core
            info.append(self.radioButton_6.text())
        elif self.radioButton_7.isChecked():  # multi core
            info.append(self.radioButton_7.text())
        info.append(self.checkBox.isChecked())  # SubPix

        info.append(self.lineEdit_13.text())  # i_frame
        info.append(self.lineEdit_14.text())  # d_frame
        info.append(self.lineEdit_15.text())  # frame max
        info.append(self.lineEdit_16.text())  # kmeans
        info.append('False')  # kmeans only
        info.append(self.label_10.text())  # cube path
        info.append(self.lineEdit.text())  # cube length
        info.append(self.lineEdit_2.text())  # disp ratio
        MyMainWindow.add_calibration_table(info)

    def select_checkerboard_path(self):
        absolute_path = QFileDialog.getOpenFileName(self)[0]
        if absolute_path != '':
            self.label_8.setText(absolute_path)

    def select_cube_path(self):
        absolute_path = QFileDialog.getOpenFileName(self)[0]
        if os.path.isfile(absolute_path):
            print(MyMainWindow.output_field_path)
            new_file = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_' + self.comboBox_2.currentText() + '_cube' + os.path.splitext(absolute_path)[1]

            shutil.copyfile(absolute_path, new_file)
            print(absolute_path, new_file)
            fn.manual_2D().Run(path=new_file, type='cube', block_meter=float(self.lineEdit.text()), ratio_h=float(self.lineEdit_2.text()), ratio_w=float(self.lineEdit_2.text()), block_w=3, block_h=3)
            self.label_10.setText(absolute_path)
        else:
            MyMainWindow.warning_msg('The wrong file!')

class UI_transformation_setting(QtWidgets.QMainWindow, UI_transformation_setting):

    def __init__(self, parent=None):
        super(UI_transformation_setting, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('UI transformation setting')
        self.setWindowIcon(QIcon('icon.png'))
        # signal
        self.pushButton_8.clicked.connect(self.build_info)
        self.pushButton_4.clicked.connect(self.select_cube_path)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def build_info(self):
        info = []
        info.append(self.comboBox_2.currentText())  # camera
        info.append(str(self.checkBox_3.isChecked()))  # freeze
        info.append(self.label_8.text())  # cube path
        info.append(self.lineEdit_3.text())  # cube length
        MyMainWindow.add_transformation_table(info)

    def select_cube_path(self):
        # path_second = ['.np']
        # absolute_path = QFileDialog.getOpenFileName(self)[0]
        # secondname = os.path.splitext(absolute_path)[1]
        # print(secondname)
        # if absolute_path != '':
        #     if secondname in path_second:
        #         self.label_8.setText(absolute_path)
        #     else:
        #         fn.manual_2D().Run(path=absolute_path, type='cube', block_meter=float(self.lineEdit.text()), ratio_h=float(self.lineEdit_3.text()), ratio_w=float(self.lineEdit_3.text()), block_w=3, block_h=3)
        #         self.label_8.setText(os.path.splitext(absolute_path)[0]+path_second[0])
        absolute_path = QFileDialog.getOpenFileName(self)[0]
        new_file = MyMainWindow.output_field_path + os.path.splitext(os.path.basename(MyMainWindow.project_field_name))[0] + '_' + self.comboBox_2.currentText() + '_transformation' + os.path.splitext(absolute_path)[1]
        shutil.copyfile(absolute_path, new_file)
        fn.manual_2D().Run(path=new_file, type='cube', block_meter=float(self.lineEdit.text()), ratio_h=float(self.lineEdit_3.text()), ratio_w=float(self.lineEdit_3.text()), block_w=3, block_h=3)
        self.label_8.setText(absolute_path)

class UI_alphapose_setting(QtWidgets.QMainWindow, UI_alphapose_setting):

    def __init__(self, parent=None):
        super(UI_alphapose_setting, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('UI alphapose setting')
        self.setWindowIcon(QIcon('icon.png'))
        # signal
        self.pushButton_6.clicked.connect(self.open_frame_check)
        # self.pushButton_3.clicked.connect(self.select_model_folder)
        self.pushButton.clicked.connect(self.select_video_path)
        self.pushButton_5.clicked.connect(self.build_info)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def select_video_path(self):
        absolute_path = QFileDialog.getOpenFileName(self)[0]
        if absolute_path != '':
            self.label_6.setText(absolute_path)

    def select_model_folder(self):
        absolute_path = QFileDialog.getExistingDirectory(self) + '/'
        if absolute_path != '':
            self.label_7.setText(absolute_path)

    def build_info(self):
        info = []
        info.append(self.comboBox.currentText())  # camera name
        info.append(self.label_6.text())
        info.append(self.lineEdit.text())
        info.append(self.lineEdit_2.text())
        info.append(self.lineEdit_5.text())
        info.append(self.lineEdit_6.text())
        if self.radioButton.isChecked():
            self.mode = str(self.radioButton.text())
        elif self.radioButton_2.isChecked():
            self.mode = str(self.radioButton_2.text())
        elif self.radioButton_3.isChecked():
            self.mode = str(self.radioButton_3.text())
        info.append(self.mode)
        if self.radioButton_4.isChecked():
            self.os = str(self.radioButton_4.text())
        elif self.radioButton_5.isChecked():
            self.os = str(self.radioButton_5.text())
        info.append(self.os)
        info.append(self.checkBox.isChecked())
        info.append(self.lineEdit_4.text())
        info.append(self.lineEdit_3.text())
        info.append(self.checkBox_2.isChecked())
        info.append(self.lineEdit_7.text())
        info.append(self.lineEdit_8.text())
        info.append(self.lineEdit_9.text())
        info.append(self.lineEdit_10.text())
        MyMainWindow.add_alphapose_table(info)

    def open_frame_check(self):
        UI_frame_check.video_setting(self.label_6.text())
        UI_frame_check.show()
        UI_frame_check.reset()

class UI_threeD_plt_setting(QtWidgets.QMainWindow, UI_threeD_plt_setting):

    def __init__(self=None, parent=None):
        super(UI_threeD_plt_setting, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('plot setting')
        self.setWindowIcon(QIcon('icon.png'))

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

class UI_postprocessing_setting(QtWidgets.QMainWindow, UI_postprocessing_setting):

    def __init__(self = None, parent = None):
        super(UI_postprocessing_setting, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('UI post processing setting')
        self.setWindowIcon(QIcon('icon.png'))
        # signal
        self.pushButton_3.clicked.connect(self.build_info)
        self.comboBox_3.currentTextChanged.connect(self.check_anomally)
        self.pushButton.clicked.connect(self.open_anomaly_setting)
        self.point_name = MyMainWindow.ui_point_name
        # comboBox
        self.init_comboBox()

    def init_comboBox(self):
        for i in range(self.comboBox.count()):
            self.comboBox.removeItem(0)
        self.comboBox.addItem('all')
        for i in range(len(self.point_name)):
            self.comboBox.addItem(self.point_name[i])

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def build_info(self):
        # freeze / previous resule / point /
        info = []
        info.append(str(self.checkBox_2.isChecked()))  # freeze
        info.append(str(self.checkBox_3.isChecked()))  # previous result
        info.append(self.comboBox.currentText())  # fit point
        info.append(self.comboBox_2.currentText())  # fit axis

        info.append(self.comboBox_3.currentText())  # anormaly
        info.append(UI_anomaly_setting.lineEdit.text())  # anormaly ratio

        info.append(str(self.radioButton.isChecked()))  # FFT
        info.append(self.lineEdit.text())  # FFT noise ratio
        info.append(str(self.checkBox.isChecked()))  # FFT segmentaiotn
        info.append(self.lineEdit_2.text())  # FFT sport ratio
        info.append(self.comboBox_4.currentText())  # FFT segmentation type
        info.append(self.lineEdit_3.text())  # FFT segmentation min distance

        info.append(str(self.radioButton_2.isChecked()))  # poly
        info.append(self.lineEdit_11.text())  # poly order
        info.append(self.lineEdit_4.text())  # poly fit min
        info.append(self.lineEdit_5.text())  # poly fit max
        info.append(str(self.checkBox_4.isChecked()))  # poly crop
        info.append(self.lineEdit_6.text())  # poly crop min
        info.append(self.lineEdit_7.text())  # poly crop max
        info.append(str(self.checkBox_5.isChecked()))  # poly mask
        info.append(self.lineEdit_8.text())  # poly mask min
        info.append(self.lineEdit_9.text())  # poly mask max
        MyMainWindow.add_post_table(info)


    def check_anomally(self):
        if self.comboBox_3.currentIndex() >= 1:
            self.pushButton.setEnabled(True)
        else:
            self.pushButton.setDisabled(True)

    def open_anomaly_setting(self):
        UI_anomaly_setting.show()

class UI_anomaly_setting(QtWidgets.QMainWindow, UI_anomaly_setting):

    def __init__(self = None, parent = None):
        super(UI_anomaly_setting, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('UI anomaly setting')
        self.setWindowIcon(QIcon('icon.png'))

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def get_ratio(self):
        return self.lineEdit.text()

class UI_frame_check(QtWidgets.QMainWindow, UI_frame_check):

    def __init__(self = None, parent = None):
        super(UI_frame_check, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('UI frame check')
        self.setWindowIcon(QIcon('icon.png'))
        self.temp_data = { }
        self.ram_radius = 1e10
        self.spinBox.valueChanged.connect(self.value_change)
        self.horizontalSlider.valueChanged.connect(self.value_change)

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def video_setting(self, path):
        self.cam = cv2.VideoCapture(path)
        total_frame = int(self.cam.get(7))
        self.spinBox.setMaximum(total_frame)
        self.horizontalSlider.setMaximum(total_frame)

    def reset(self):
        self.temp_data = {}
        self.value_change(1)

    def value_change(self, value):
        min_x = 0 if UI_alphapose_setting.lineEdit_7.text() == 'default' else UI_alphapose_setting.lineEdit_7.text()
        min_y = 0 if UI_alphapose_setting.lineEdit_8.text() == 'default' else UI_alphapose_setting.lineEdit_8.text()
        max_x = self.cam.get(3) if UI_alphapose_setting.lineEdit_9.text() == 'default' else UI_alphapose_setting.lineEdit_9.text()
        max_y = self.cam.get(4) if UI_alphapose_setting.lineEdit_10.text() == 'default' else UI_alphapose_setting.lineEdit_10.text()
        min_x, min_y, max_x, max_y = int(min_x), int(min_y), int(max_x), int(max_y)
        self.spinBox.setValue(value)
        if value not in self.temp_data:
            self.temp_data = {}
            for i in range(int(self.cam.get(7))):
                (ret, img) = self.cam.read()
                if not ret:
                    print('the video is end!!')
                    break
                img = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 0), 7, 4)
                self.temp_data[i + 1] = img.copy()
                if i == value:
                    # box
                    img = cv2.resize(img, (self.label_2.width(), self.label_2.height()), interpolation=cv2.INTER_NEAREST)
                    img = CV2QImage(img)
                    self.label_2.setPixmap(QtGui.QPixmap.fromImage(img))
                if i >= value + self.ram_radius:
                    break
        img = self.temp_data[value].copy()
        img = cv2.resize(img, (self.label_2.width()-10, self.label_2.height()-10), interpolation=cv2.INTER_NEAREST)
        img = CV2QImage(img)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(img))

class UI_log(QtWidgets.QMainWindow, UI_log):

    def __init__(self = None, parent = None):
        super(UI_log, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('UI log')
        self.setWindowIcon(QIcon('icon.png'))
        self.pushButton.clicked.connect(self.save_log)


    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def outputWritten(self, text):
        QApplication.processEvents()
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def save_log(self):
        type = '.txt'
        absolute_path = QFileDialog.getSaveFileName(self, 'Save the log', 'log', "Files (*.txt) ;; All Files (*)")[0]
        if os.path.splitext(os.path.basename(absolute_path))[0]:
            absolute_path = os.path.splitext(absolute_path)[0] + type
            StrText = self.textBrowser.toPlainText()
            qS = str(StrText)
            f = open(absolute_path, 'w')
            f.write('\n{}'.format(qS))
            self.success_msg('Save!')

    def openUI(self):
        # terminal
        if not MyMainWindow.checkBox_5.isChecked():
            sys.stdout = EmittingStr(textWritten=self.outputWritten)
            sys.stderr = EmittingStr(textWritten=self.outputWritten)
        self.label_2.setText(MyMainWindow.lineEdit.text())
        UI_log.show()

class UI_analysis_plt_setting(QtWidgets.QMainWindow, UI_analysis_plt_setting):

    def __init__(self = None, parent = None):
        super(UI_analysis_plt_setting, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('UI analysis setting')
        self.setWindowIcon(QIcon('icon.png'))

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

class UI_post_plt_setting(QtWidgets.QMainWindow, UI_post_plt_setting):

    def __init__(self = None, parent = None):
        super(UI_post_plt_setting, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.setWindowTitle('UI plot setting')
        self.setWindowIcon(QIcon('icon.png'))
        self.pushButton.clicked.connect(self.plot)


    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def plot(self):
        MyMainWindow.run_post_plt()

if __name__ == '__main__':
    # music
    pygame.mixer.init()
    track = pygame.mixer.music.load("./UI_music.mp3")
    pygame.mixer.music.play()
    # UI
    app = QApplication(sys.argv)
    password = DialogUI()
    UI_calibration_setting = UI_calibration_setting()
    UI_alphapose_setting = UI_alphapose_setting()
    UI_transformation_setting = UI_transformation_setting()
    UI_frame_check = UI_frame_check()
    UI_analysis_plt_setting = UI_analysis_plt_setting()
    UI_anomaly_setting = UI_anomaly_setting()
    UI_post_plt_setting = UI_post_plt_setting()
    UI_threeD_plt_setting = UI_threeD_plt_setting()
    MyMainWindow = MyMainWindow()
    UI_log = UI_log()
    UI_postprocessing_setting = UI_postprocessing_setting()

    # # password checker
    # password.show()
    MyMainWindow.show()
    # # set the theme
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # app.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style="style_DarkOrange"))
    sys.exit(app.exec())

