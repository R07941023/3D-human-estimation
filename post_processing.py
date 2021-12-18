import cv2
import numpy as np
import math
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from os.path import isfile, isdir, join
import csv
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import shutil
from matplotlib.ticker import MultipleLocator
from scipy.interpolate import interp1d
import time
from fn import *
from itertools import combinations

if __name__ == '__main__':
    # init
    mypath = "../dataset/20200318/threeDdata.csv"  # calibrate video path  "./camera_backup1/
    output_path = "../dataset/20200318/threeDdata_FFT.csv"
    leni = np.array([5, 7, 6, 8, 11, 13, 12, 14])  # [5, 7, 6, 8, 11, 13, 12, 14], [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6]
    lenf = np.array([7, 9, 8, 10, 13, 15, 14, 16])  # [7, 9, 8, 10, 13, 15, 14, 16], [1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7]

    # interpolation
    # order = [1, 1, 1]  # x, y, z
    # fit_radius = [15, 15, 15]
    # fit_point = [10]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    fftmodel = False
    intermodel = False
    fit_point = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    axis_set = [0, 1, 2]
    plt_max_velocity, real_fps = 85, 300
    preprocessing_model = None
    # fft
    fft_ratio = [0.96, 0.96]
    loop_n = 1
    saddle_dmin = None  # segmentation
    acc_ratio = 3  # mapping: anormaly if value > ratio
    # inter
    inter_type = 'poly'
    fit_data = [(480, 600)]  # [(0, 500), (500, 800)]
    fit_area = [(480, 650)]
    mask_area = []
    order = [30]
    figsize = (25, 5)
    test = False

    # plot 2d
    video_fps = 80
    plot2d = True
    imgsize = [720, 540]
    # plot 3d
    plot3d = False
    xlim, ylim, zlim = [-2.2, 0.8, 0.5], [33, 36, 0.5], [-1.2, 1.8, 0.5]  # [-1.0, 1.5, 0.5], [1.8, 4.3, 0.5], [-1.0, 1.5, 0.5] / ['', '', ''], ['', '', ''], ['', '', '']
    view = [0, -100]  # YZ[0, 0]  XZ[0, -90] XY[-90, 90] 1. [0, -115] / [0, -25] / [-90, 65]
    fps = 300
    # [-0.5, 2, 0.5], [3.8, 4.1, 0.5], [-1.5, 1.5, 0.5]

    # compare
    # mypath = "../20200318/human6/origin/origin.csv"  # calibrate video path  "./camera_backup1/
    # frame, data_1 = tool().Loadcsv_3d(mypath)
    # # mypath = "../20200318/human6/all/all.csv"
    # mypath = "../20200318/human6/fft/fft.csv"
    # frame, data_2 = tool().Loadcsv_3d(mypath)
    # # # mypath = "../20200318/human6/single/single.csv"
    # # mypath = "../20200318/human6/single_fft/single_fft.csv"
    # # frame, data_3 = tool().Loadcsv_3d(mypath)
    # mypath = "../20200318/human6/all_fft/all_fft.csv"  # calibrate video path  "./camera_backup1/
    # frame, data_4 = tool().Loadcsv_3d(mypath)
    # data_name = ['row', 'FFT', 'all anomaly + FFT']
    # frame, velocity_1 = joint_analysis().velocity_analysis(frame=frame, data=data_1, fps=300, output_path='../20200318/')
    # frame, velocity_2 = joint_analysis().velocity_analysis(frame=frame, data=data_2, fps=300, output_path='../20200318/')
    # # frame, velocity_3 = joint_analysis().velocity_analysis(frame=frame, data=data_3, fps=300, output_path='../20200318/')
    # frame, velocity_4 = joint_analysis().velocity_analysis(frame=frame, data=data_4, fps=300, output_path='../20200318/')
    # data_set, velocity_set = [], []
    # data_set.append(data_1)
    # data_set.append(data_2)
    # # data_set.append(data_3)
    # data_set.append(data_4)
    # velocity_set.append(velocity_1)
    # velocity_set.append(velocity_2)
    # # velocity_set.append(velocity_3)
    # velocity_set.append(velocity_4)
    # tool().plot_coord(frame, data_set, index=[14], figsize=(24, 3), data_name=data_name)
    # tool().plot_velocity(frame, velocity_set, index=[14], plt_velocity_range=tuple(np.linspace(0, 35, 10)), figsize=(24, 5), data_name=data_name)
    # exit()

    # # 2D smooth
    # mypath = "../20200318/R_33GX287/real/human6/AlphaPose_R6.csv"
    # video_path = "../20200318/R_33GX287/real/human6/R6.avi"
    # frame, data = tool().Loadcsv_2d(mypath)
    # # # inter_data = post_processing(data).interpolation_2d(order, fit_radius, fit_point)
    # # # tool().data2csv2d(mypath, frame, inter_data)
    # tool().plot2D(frame, data, video_path, fps='', imgsize=imgsize, type='normal', pose_inf='', label=True)
    # mypath = "../20200407/L_33GX287/real/human1/AlphaPose_L1.csv"
    # video_path = "../20200407/L_33GX287/real/human1/L1.avi"
    # frame, data = tool().Loadcsv_2d(mypath)
    # # # # # inter_data = post_processing(data).interpolation_2d(order, fit_radius, fit_point)
    # # # # tool().data2csv2d(mypath, frame, inter_data)
    # tool().plot2D(frame, data, video_path, fps='', imgsize=imgsize, type='alphapose', pose_inf='', label=False)
    # mypath = "../20200407/A_33GX287/real/human1/AlphaPose_A1.csv"
    # video_path = "../20200407/A_33GX287/real/human1/A1.avi"
    # frame, data = tool().Loadcsv_2d(mypath)
    # # # # # inter_data = post_processing(data).interpolation_2d(order, fit_radius, fit_point)
    # # # # # tool().data2csv2d(mypath, frame, inter_data)
    # tool().plot2D(frame, data, video_path, fps='', imgsize=imgsize, type='alphapose', pose_inf='', label=False)
    # exit()
    #

    # 3D smooth
    # frame, data = tool().Loadcsv_3d(mypath)
    frame, output_data = tool().Loadcsv_3d(output_path)
    coord2vicon().run(frame, output_data, video_fps=video_fps)
    # tool().Plot3D(frame, output_data, xslim, ylim, zlim, view, '../20200318/acc1/', fps)
    # tool().Plot3D(frame, output_data, xlim, ylim, zlim, view, '../20200318/acc1/', fps)
    # if fftmodel:
    #     output_data = post_processing('').fft_model(frame=frame, input_data=data.copy(), output_data=output_data, preprocessing_model=preprocessing_model, fps=real_fps, acc_ratio=acc_ratio, fft_ratio=fft_ratio, saddle_dmin=saddle_dmin, loop_n=loop_n, axis_set=axis_set, fit_point=fit_point, figsize=figsize, plt_max_velocity=plt_max_velocity)
    # if intermodel:
    #     output_data = post_processing('').interpolation_model(data=data, order=order, output_data=output_data, fit_point=fit_point, fit_axis=axis_set, padding=1e10, inter_type=inter_type, plt_max_velocity=plt_max_velocity, fps=fps, figsize=figsize, fit_area=fit_area, fit_data=fit_data, mask_area=mask_area, preprocessing_model=preprocessing_model)
    # if fftmodel == False and intermodel == False:
    #     pass
    #     frame, velocity = joint_analysis().velocity_analysis(frame=frame, data=data.copy(), fps=fps, output_path='../20200318/123.csv')
        # frame, velocity_output = joint_analysis().velocity_analysis(frame=frame, data=output_data.copy(), fps=fps, output_path='../20200318/')
        # tool().plot_coord(frame, [data, output_data], index=fit_point, figsize=figsize, plt_scale_high=1.2)
        # tool().plot_velocity(frame, [velocity_output], index=fit_point, plt_scale_range=tuple(np.linspace(0, plt_max_velocity, 10)), figsize=figsize)
    # if test:
    #     exit()
    # else:
    #     tool().data2csv3d(output_path, frame, output_data, '')


    # tool().plot_segment_length(path='../20200221/', frame=frame, segment_length=segment_length, leni=leni, lenf=lenf, plt_segment_range=tuple(np.linspace(0.13, 0.6, 10)), figsize=(8, 8))
    # view = [0, -100]
    # tool().Plot3D(frame, data, xlim, ylim, zlim, view, '../20200221/acc/', fps)
    # frame, segment_length = joint_analysis().segment_analysis(path='../20200318/', frame=frame, data=data, leni=leni, lenf=lenf)
    # frame, segment_length = tool().Loadcsv_3d_segment_length('../20200318/threeDdata_segment_length.csv')
    # frame, velocity = joint_analysis().velocity_analysis(frame=frame, data=data, fps=300, output_path='../20200318/')
    # frame, velocity = tool().Loadcsv_3d_velocity("../20200318/threeDdata_velocity.csv")
    # inter_data, radius_inf = post_processing(data).interpolation_3d(radius_type='normal', adaptive_ratio=(2, 6), order=order, fit_radius=fit_radius, fit_point=fit_point, padding=1e10)
    # tool().plot_coord(frame, inter_data, index=[10])
    # tool().data2csv3d(mypath, frame, inter_data, '3dsmooth')
    # frame, inter_segment_length = joint_analysis().segment_analysis(path='../20200318/', frame=frame, data=inter_data, leni=leni, lenf=lenf)
    # frame, inter_velocity = joint_analysis().velocity_analysis(frame=frame, data=inter_data, fps=300, output_path='../20200318/')
    # tool().plot_segment_length(path='../20200318/', frame=frame, segment_length=inter_segment_length, leni=leni, lenf=lenf, plt_segment_range=tuple(np.linspace(0.13, 0.6, 10)), figsize=(8, 8))
    # tool().plot_velocity(frame, inter_velocity, index=[10], plt_velocity_range=tuple(np.linspace(0, 160, 10)), figsize=(12, 5))
    # view = [0, -100]
    # tool().Plot3D(frame, inter_data, xlim, ylim, zlim, view, '../20200318/acc1/', fps)
    # # plot the point on time sequence
    # frame, data = tool().Loadcsv_3d('../20200212/threeDdata.csv')
    # _, predict_data = tool().Loadcsv_3d('../20200212/threeDdata_3dsmooth.csv')
    # _, radius_inf = tool().Loadcsv_3d('../20200212/threeDdata_3dfitting_radius_inf.csv')
    # tool().plot_point(frame=frame, row_data=data, predict_data=predict_data, data_inf=radius_inf, display=[10])

    # # human skeleton model
    # max_iterations = 50000
    # tolerance = 0.0001
    # frame, t_data = tool().Loadcsv_3d('../20200114/T-pose.csv')
    # frame, data = tool().Loadcsv_3d(mypath)
    # # # guide_pose = post_processing(data).human_skeleton(data, t_data, max_iterations, tolerance)
    # # guide_pose = post_processing(data).body_anormaly_detect(data, t_data)
    # guide_pose = post_processing(data).head_model(data, t_data, max_iterations, tolerance)
    # tool().data2csv3d(mypath, frame, guide_pose, 'humanmodel')
    # tool().Plot3D(frame, guide_pose, xlim, ylim, zlim, view, '../20200212/acc1/', fps)
    # tool().plot_double_3D(frame, data, guide_pose, xlim, ylim, zlim, view, '../20200114/acc1/', fps)
    # # view = [0, -115]
    # # tool().plot_double_3D(frame, data, guide_pose, xlim, ylim, zlim, view, '../20200114/acc2/', fps)
    # # view = [-90, 65]
    # # tool().plot_double_3D(frame, data, guide_pose, xlim, ylim, zlim, view, '../20200114/acc3/', fps)


    # plot 2d reprojection
    # if plot2d:
    #     camera_path, camera_int, camera_dist, camera_rvec, camera_tvec = [], [], [], [], []
    #     output_path = '../20200318/'
    #     pose_path = output_path + 'threeDdata.csv' #  '../20200318/human6/fft/fft.csv'
        # pose_path = "../20200318/threeDdata_FFT.csv"
        # inf_path = output_path+'threeDdata_inf.csv'
        # camera_path.append(output_path+'R_33GX287/real/human6/R6.avi')  # '../20200221/R_33GX287/real/human4/R4_alphapose.avi' // '../20200221/R_33GX287/relation/RLA/cube/backup/R15/R15.avi'
        # camera_path.append(output_path+'L_33GX287/real/human6/L6.avi')
        # camera_path.append(output_path+'A_33GX287/real/human6/A6.avi')
        # camera_int.append(output_path+"R_33GX287/calibration/mtx.npy")
        # camera_int.append(output_path+"L_33GX287/calibration/mtx.npy")
        # camera_int.append(output_path+"A_33GX287/calibration/mtx.npy")
        # camera_dist.append(output_path+"R_33GX287/calibration/dist.npy")
        # camera_dist.append(output_path+"L_33GX287/calibration/dist.npy")
        # camera_dist.append(output_path+"A_33GX287/calibration/dist.npy")
        # camera_rvec.append(output_path+"R_33GX287/relation/RLA/cube/backup/R1/R.npy")
        # camera_rvec.append(output_path+"L_33GX287/relation/RLA/cube/backup/L1/R.npy")
        # camera_tvec.append(output_path+"R_33GX287/relation/RLA/cube/backup/R1/T.npy")
        # camera_tvec.append(output_path+"L_33GX287/relation/RLA/cube/backup/L1/T.npy")
        # tool().plot_reproject(imgsize, pose_path, inf_path, camera_path, camera_int, camera_dist, camera_rvec, camera_tvec, fps=video_fps, type='normal')
    # if plot3d:
    #     data2vicon
    #     coord2vicon().run(frame, output_data, video_fps=video_fps)
    #     tool().Plot3D(frame, output_data, xlim, ylim, zlim, view, '../20200318/acc1/', fps=video_fps)





