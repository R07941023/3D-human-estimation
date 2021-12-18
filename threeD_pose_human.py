import cv2
import numpy as np
import math
import itertools
from os import listdir
from os.path import isfile, isdir, join
import os.path
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from IPython.core.pylabtools import figsize # import figsize
import shutil
import warnings
from numpy import RankWarning
import pandas as pd
from fn import *



class threeD():
    def __init__(self):  # Run it once
        # 2D filter
        self.L_threshold = [1, 4.5, 0, 3, 0]  # [switch, 2D_threshold_velocity(average), fitting order, number of fits, details]
        self.R_threshold = [1, 4.5, 0, 3, 0]  # [switch, 2D_threshold_velocity(average), fitting order, number of fits, details]
        self.A_threshold = [1, 4.5, 0, 3, 0]  # [switch, 2D_threshold_velocity(average), fitting order, number of fits, details]
        self.check_2D_point = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        # 3D filter
        self.leni = np.array([0])  # [17, 5, 7, 17, 6, 8, 17, 11, 13, 17, 12, 14]  check 3D segmant length
        self.lenf = np.array([1])  # [5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
        # all
        # self.leni = [17, 5, 7, 17, 6, 8, 17, 11, 13, 17, 12, 14]
        # self.lenf = [5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
        # left side
        # self.leni = [17, 5, 7, 17, 11, 13]
        # self.lenf = [5, 7, 9, 11, 13, 15]
        # right side
        # self.leni = [17, 6, 8, 17, 12, 14]
        # self.lenf = [6, 8, 10, 12, 14, 16]
        self.plt_segment_range = tuple(np.linspace(0, 0.8, 10))

        # drop
        self.fitting_drop = [0, 0]  # drop the predict point [2d, 3d]


        # error left > rightframe_i
        # 3D coordinate by scences
        self.distance = 4.4

        # self.xlim, self.ylim, self.zlim = [-5, 5], [4, 7], [-1.5, 1]
        self.xlim, self.ylim, self.zlim = [''], [''], ['']
        # self.view = [0, -95]

        self.visualize = [0, 0, 10]  # [2D, 3D, 2D_delay]
        self.view = [0, -90]  # YZ[0, 0]  XZ[0, -90] XY[-90, 90]
        self.xlim, self.ylim, self.zlim = [-1., 1., 0.2], [1.75, 4.75, 0.2], [-0.9, 1.2, 0.6]  # [-5, 5, 0.5], [4, 7, 0.5], [-1.5, 1.5, 0.5]
        self.figsize = [12.0, 4.0]
        plt.rcParams['figure.figsize'] = tuple(self.figsize)
        # self.out_3D_png = output_3D + '3D_origin_png'
        # if os.path.isdir(self.out_3D_png):
        #     shutil.rmtree(self.out_3D_png)
        # os.mkdir(self.out_3D_png)
        #
        # self.view = [30, -60]

        self.display_len = []

        # Error
        self.error_ylim = [0, 80]
        # self.error_draw = [1, 2, 5, 6, 7, 11, 12, 13, 17]  # left 1, 2, 5, 6, 7, 11, 12, 13, 17
        self.error_draw = [3, 4, 8, 9, 10, 14, 15, 16, 17]  # right 3, 4, 8, 9, 10, 14, 15, 16, 17

        # warnings
        warnings.filterwarnings('ignore', category=RankWarning)

    def get_transformation(self, data, output_path, cube_length=1):
        objp = (np.mgrid[0:2, 0:2, 0:2].T.reshape(-1, 3) * cube_length).astype(np.float32)
        # img coordinates to world coordinates
        # objp[:, 1], objp[:, 2] = objp[:, 2], -objp[:, 1]
        # shift_data = (data[0]-data[0][0])
        # ICP
        ICP_min_point = 2
        arrange, fit_index_set, loss_set, H_set = [], [], [], []
        for i in range(ICP_min_point, data.shape[0]):
            arrange += list(itertools.combinations(range(data.shape[0]), i))
        # print('goal = ', objp)
        for fit_index in arrange:
            fit_index = list(fit_index)
            _, H, _ = joint_analysis().ICP(data[fit_index], objp[fit_index], max_iterations=500, tolerance=1e-7, T_weight=1)
            src = np.dot(H, np.concatenate((data, np.ones((data.shape[0], 1))), axis=1).T).T
            src[:, 0:3] = src[:, 0:3] / src[:, -1].reshape(-1, 1)
            H_set.append(H)
            fit_index_set.append(fit_index)
            loss_set.append(joint_analysis().eucildea_distance(objp, src[:, 0:3]))
        H_set, loss_set = np.array(H_set), np.array(loss_set)
        index_min_loss = np.argwhere(loss_set == np.min(loss_set)).reshape(-1)[0]
        src = np.dot(H_set[index_min_loss], np.concatenate((data, np.ones((data.shape[0], 1))), axis=1).T).T
        src[:, 0:3] = src[:, 0:3] / src[:, -1].reshape(-1, 1)
        data = src[:, 0:3]
        np.save(output_path, H_set[index_min_loss])
        print('best index = ', fit_index_set[index_min_loss])
        print('transformaiton loss [m]= ', joint_analysis().eucildea_distance(objp, data)/(data.shape[0]*data.shape[1]))
        return data, H_set[index_min_loss]

    def Plot3D(self, dict_3D, xlim, ylim, zlim, frame, imgsize):
        fig = plt.figure(0)
        line_i = [4, 2, 0, 1, 10, 8, 6, 17, 5, 7, 17, 12, 14, 17, 11, 13, 17]
        line_f = [2, 0, 1, 3, 8, 6, 17, 5, 7, 9, 12, 14, 16, 11, 13, 15, 0]
        ax = fig.gca(projection='3d')
        # plot the point
        point_range_x = []
        point_range_y = []
        point_range_z = []
        for point_3D in range(dict_3D.shape[0]):
            if dict_3D[point_3D][0] < 1e10:
                ax.text(dict_3D[point_3D][0], dict_3D[point_3D][1], dict_3D[point_3D][2], str(point_3D), color='r', fontdict={'weight': 'bold', 'size': 9})
                ax.scatter(dict_3D[point_3D][0], dict_3D[point_3D][1], dict_3D[point_3D][2], c='y')
                point_range_x.append(dict_3D[point_3D][0])
                point_range_x.append(dict_3D[point_3D][0])
                point_range_y.append(dict_3D[point_3D][1])
                point_range_y.append(dict_3D[point_3D][1])
                point_range_z.append(dict_3D[point_3D][2])
                point_range_z.append(dict_3D[point_3D][2])

        # plot/calculate L line
        obj_length = []
        for line_3D in range(len(line_i)):
            if dict_3D[line_i[line_3D]][0] < 1e10 and dict_3D[line_f[line_3D]][0] < 1e10:
                linex = (dict_3D[line_i[line_3D]][0], dict_3D[line_f[line_3D]][0])
                liney = (dict_3D[line_i[line_3D]][1], dict_3D[line_f[line_3D]][1])
                linez = (dict_3D[line_i[line_3D]][2], dict_3D[line_f[line_3D]][2])
                temp = round(math.sqrt((linex[1] - linex[0]) ** 2 + (liney[1] - liney[0]) ** 2 + (linez[1] - linez[0]) ** 2), 5)
                obj_length.append(temp)
                # print(dict_3D_l[line_i[line_3D]], dict_3D_l[line_f[line_3D]], temp)
                ax.plot(linex, liney, linez, linewidth=2.0, c='m')


        # axis range
        delta = max(point_range_x)-min(point_range_x)
        delta = max(point_range_y)-min(point_range_y) if delta < max(point_range_y)-min(point_range_y) else delta
        delta = max(point_range_z) - min(point_range_z) if delta < max(point_range_z) - min(point_range_z) else delta
        if xlim[0]:
            ax.set_xlim(xlim[0:2])
            if xlim[2]:
                ax.xaxis.set_major_locator(MultipleLocator(self.xlim[2]))
        else:
            ax.set_xlim(np.mean(point_range_x)-delta, np.mean(point_range_x)+delta)
            # pass
        if ylim[0]:
            ax.set_ylim(ylim[0:2])
            if ylim[2]:
                ax.yaxis.set_major_locator(MultipleLocator(self.ylim[2]))
        else:
            ax.set_ylim(np.mean(point_range_y)-delta, np.mean(point_range_y)+delta)
            # pass
        if zlim[0]:
            ax.set_zlim(zlim[0:2])
            if zlim[2]:
                ax.zaxis.set_major_locator(MultipleLocator(self.zlim[2]))
        else:
            ax.set_zlim(np.mean(point_range_z)-delta, np.mean(point_range_z)+delta)
            # pass
        # self.view = [0, -90]
        ax.set_xlabel('X [m]')
        inf_title = ['3D reconstruction (frame: ' + str(frame) + ' )']
        inf_ylabel = ['Y [m]']
        inf_zlabel = ['Z [m]']

        ax.set_title(''.join(inf_title))
        ax.set_ylabel(''.join(inf_ylabel))
        ax.set_zlabel(''.join(inf_zlabel))

        ax.view_init(self.view[0], self.view[1])  # default(30, -60), the rotation of vertical and horizontal angle
        # plt.axis('equal')
        if self.view == [0, -90]:
            plt.yticks([])


        plt.savefig(self.out_3D_png + '/' + str(frame)+'_3D.jpg')
        plt.close(0)

    def coord(self, output_3D, camera_int, camera_dist, camera_rvec, camera_tvec, dict_path, imgsize, limit_i_frame, limit_f_frame, fps=1, extrapolation_set=[0, 0.7, 1, 5], guiding_pose='', score_threshold=None, UI=False):
        self.threeD_threshold = extrapolation_set
        # M/dist
        mtx, dist = [], []
        for i in range(len(camera_int)):
            mtx.append(camera_orientation().Interior_Orientation(camera_int[i]))
            dist.append(camera_orientation().Interior_Orientation(camera_dist[i]))
        mtx, dist = np.array(mtx), np.array(dist)
        # alphapose
        pose = []
        for i in range(len(dict_path)):
            row_data = self.Loadscv(dict_path[i])
            if row_data.shape[0] < limit_f_frame:
                limit_f_frame = row_data.shape[0]
        for i in range(len(dict_path)):
            if score_threshold is not None:
                row_data = self.Loadscv(dict_path=dict_path[i], limit_frame=limit_f_frame, confidence_threshold=score_threshold[i])
            else:
                row_data = self.Loadscv(dict_path=dict_path[i], limit_frame=limit_f_frame)
            # # undistortion
            # row_data = row_data.reshape(-1, 2)
            # row_data = camera_orientation().undistpoint(row_data, mtx[i], imgsize, dist[i])
            # row_data = row_data.reshape(-1, 18, 2)
            pose.append(row_data)
        pose = np.array(pose)
        if guiding_pose:
            _, guiding_data = tool().Loadcsv_3d(guiding_pose)
            t_segment_length = np.average(joint_analysis().segment_analysis(guiding_data, self.leni, self.lenf), 0)

        # R/T
        vec = {}
        for i in range(len(camera_rvec) + 1):  # len(self.camera_rvec)
            for j in range(len(camera_rvec) + 1):
                if i == j:
                    vec[str(i) + str(j)] = np.hstack((np.eye(3), np.zeros((3, 1))))
                elif i + 1 == j:
                    R = np.load(camera_rvec[i])
                    T = np.load(camera_tvec[i])
                    vec[str(i) + str(j)] = camera_orientation().Exterior_Orientation(R, T)
                    # roll, pitch, yaw, camera_position = camera_orientation().transfrom_rodrigues(R, T)
                    # print('camera = ', i, j)
                    # print('stereo rotate = ', roll, pitch, yaw)
                    # print('T = ', T.reshape(-1))
                    # # print('camera position = ', -np.dot(R, T))
                    # print('camera position = ', camera_position.reshape(-1))
                    # baseline = np.sum(np.sqrt(np.sum(np.power(camera_position, 2), 1)))
                    # print('baseline = ', baseline)
        for i in range(len(camera_rvec) + 1):  # len(self.camera_rvec)
            for j in range(len(camera_rvec) + 1):
                if i + 2 == j:
                    R, T, _, _, _, _, _, _, _, _ = cv2.composeRT(
                        cv2.Rodrigues(vec[str(i) + str(i + 1)][:, :3], None)[0], vec[str(i) + str(i + 1)][:, -1:],
                        cv2.Rodrigues(vec[str(i + 1) + str(i + 2)][:, :3], None)[0],
                        vec[str(i + 1) + str(i + 2)][:, -1:])
                    vec[str(i) + str(j)] = camera_orientation().Exterior_Orientation(R, T)
                    R, jacobain = cv2.Rodrigues(R, None)
                    # roll, pitch, yaw, camera_position = camera_orientation().transfrom_rodrigues(R, T)
                    # print('camera = ', i, j)
                    # print('stereo rotate = ', roll, pitch, yaw)
                    # print('T = ', T.reshape(-1))
                    # # print('camera position = ', -np.dot(R, T))
                    # print('camera position = ', camera_position.reshape(-1))
                    # baseline = np.sum(np.sqrt(np.sum(np.power(camera_position, 2), 1)))
                    # print('baseline = ', baseline)

        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(output_3D + '3D_scences.mp4', fourcc, fps, (int(self.figsize[0])*100, int(self.figsize[1])*100))
        with open(output_3D + 'threeDdata.csv', 'w', newline='') as csvfile_coordinate:
            # coordinate index
            csv_coord = csv.writer(csvfile_coordinate)
            csv_coord_index = ['frame']
            count_coord = 0
            for i in range(pose.shape[2]):
                csv_coord_index.append(str(i) + '_x [m]')
                csv_coord_index.append(str(i) + '_y [m]')
                csv_coord_index.append(str(i) + '_z [m]')
            # csv_coord_index.append('sita [rad]')
            # csv_coord_index.append('sita1 [rad]')
            # csv_coord_index.append('sita2 [rad]')
            csv_coord.writerow(csv_coord_index)
            # print(self.RP, self.LP, self.CP, self.CP, self.LC, self.RC)
            with open(output_3D + 'threeDdata_inf.csv', 'w', newline='') as csvfile_inf:
                # inf index
                csv_inf = csv.writer(csvfile_inf)
                inf_index = ['frame']
                for i in range(pose.shape[2]):
                    inf_index.append('Point')
                    inf_index.append('CameraA')
                    inf_index.append('CameraB')
                    inf_index.append('Loss')
                csv_inf.writerow(inf_index)
                # caulculate the x, y, z
                time_2D_l = []
                time_2D_r = []
                time_2D_a = []
                pre_2d_filter_l = []
                pre_2d_filter_r = []
                pre_2d_filter_a = []
                time_3D = []
                dict_3D = {}
                dict_3D_l = {}
                dict_3D_r = {}
                dict_3D_project_l = {}
                dict_3D_project_r = {}
                segment_length = []

                limit_i_frame, limit_f_frame + 1
                for frame in range(limit_i_frame, limit_f_frame+1):
                    if UI is False:
                        print('frame = ', frame)
                    # cand3_3D, point_3D = [], []
                    # # common 2d filter
                    # if self.R_threshold[0]:
                    #     if time_2D_r:
                    #         pre_2d_filter_r, drop_2d_point_r = self.common_2d_filter(time_2D_r[-1], pose[0][frame], self.R_threshold[1], time_2D_r, self.R_threshold[2], self.R_threshold[3], self.check_2D_point, self.R_threshold[4])
                    #     else:
                    #         time_2D_r.append(pose[0][frame])
                    # if self.L_threshold[0]:
                    #     if time_2D_l:
                    #         pre_2d_filter_l, drop_2d_point_l = self.common_2d_filter(time_2D_l[-1], pose[1][frame], self.L_threshold[1], time_2D_l, self.L_threshold[2], self.L_threshold[3], self.check_2D_point, self.L_threshold[4])
                    #     else:
                    #         time_2D_l.append(pose[1][frame])
                    # if self.A_threshold[0]:
                    #     if time_2D_a:
                    #         pre_2d_filter_a, drop_2d_point_a = self.common_2d_filter(time_2D_A[-1], pose[2][frame], self.A_threshold[1], time_2D_r, self.A_threshold[2], self.A_threshold[3], self.check_2D_point, self.A_threshold[4])
                    #     else:
                    #         time_2D_r.append(pose[2][frame])
                    # ti = time.time()
                    # print(mtx.shape, vec.shape, pose[:, frame, :, :].shape)
                    dict_3D_project, coord_inf = camera_orientation().img2world(mtx, dist, vec, pose[:, frame-1, :, :])
                    coord_inf = np.append(np.array([frame]), coord_inf)
                    # tf = time.time()
                    # print('epipolar constraint = ', tf-ti)




                    # pose_coord = [frame]
                    # for point in range(pose.shape[2]):
                    #     dict_3D_l[point] = ['', '', '']
                    #     dict_3D_r[point] = ['', '', '']
                    #     dict_3D_project_l[point] = ['', '', '']
                    #     dict_3D_project_r[point] = ['', '', '']
                    #
                    #     # 3D reconsturcture
                    #     # candidate 1 : original data
                    #     if pose_L[frame][point][0] and pose_R[frame][point][0]:
                    #         dict_3D_l[point], dict_3D_r[point], dict_3D_project_l[point], dict_3D_project_r[point] = self.triangulation(pose_L[frame][point], pose_R[frame][point], alpha_o, beta_o, L_w, L_h, R_w, R_h, L_FOV, R_FOV)
                    #     # candidate 2 : pre_2d_common_filter data
                    #     if len(time_2D_l) > 0:  # frame != 1
                    #         if pre_2d_filter_l and pre_2d_filter_r and pre_2d_filter_l[point][0] and pre_2d_filter_r[point][0]:
                    #             dict_3D_l[point], dict_3D_r[point], dict_3D_project_l[point], dict_3D_project_r[point] = self.triangulation(pre_2d_filter_l[point], pre_2d_filter_r[point], alpha_o,beta_o, L_w, L_h, R_w, R_h, L_FOV, R_FOV)
                    #         elif pre_2d_filter_l and pre_2d_filter_l[point][0] and pose_R[frame][point][0]:
                    #             dict_3D_l[point], dict_3D_r[point], dict_3D_project_l[point], dict_3D_project_r[point] = self.triangulation(pre_2d_filter_l[point], pose_R[frame][point], alpha_o,beta_o, L_w, L_h, R_w, R_h, L_FOV, R_FOV)
                    #
                    #         elif pre_2d_filter_r and pre_2d_filter_r[point][0] and pose_L[frame][point][0]:
                    #             dict_3D_l[point], dict_3D_r[point], dict_3D_project_l[point], dict_3D_project_r[point] = self.triangulation(pose_L[frame][point], pre_2d_filter_r[point], alpha_o,beta_o, L_w, L_h, R_w, R_h, L_FOV, R_FOV)
                    #
                    #     if self.base[0] == 'L':
                    #         pose_coord.append(dict_3D_l[point][0])
                    #         pose_coord.append(dict_3D_l[point][1])
                    #         pose_coord.append(dict_3D_l[point][2])
                    #         dict_3D[point] = [dict_3D_l[point][0], dict_3D_l[point][1], dict_3D_l[point][2]]
                    #         point_3D.append([dict_3D_l[point][0], dict_3D_l[point][1], dict_3D_l[point][2]])
                    #
                    #     elif self.base[0] == 'R':
                    #         pose_coord.append(dict_3D_r[point][0])
                    #         pose_coord.append(dict_3D_r[point][1])
                    #         pose_coord.append(dict_3D_r[point][2])
                    #         dict_3D[point] = [dict_3D_r[point][0], dict_3D_r[point][1], dict_3D_r[point][2]]
                    #         point_3D.append([dict_3D_r[point][0], dict_3D_r[point][1], dict_3D_r[point][2]])
                    #

                    dict_3D = dict_3D_project
                    point_3D = dict_3D_project
                    cand12_segment_length = self.segment_analysis(point_3D)

                    # candidate 3 : pre_3d_extrapolation
                    index_3d_fitting = np.array([])
                    # point_3D = []
                    if self.threeD_threshold[0]:
                        if len(time_3D) > 2:
                            cand3_3D = self.pre_3d_fitting(time_3D, self.threeD_threshold[2], self.threeD_threshold[3])
                            cand3_segment_length = self.segment_analysis(cand3_3D)
                            # need modity time_2D_l and time_2D_r
                            index_3d_fitting, drop_point, dict_loss_1, dict_loss_2 = self.checking_3d_smooth(average_segment_length, cand12_segment_length, cand3_segment_length)  # input: average, 2d>3d, 3d fitting
                            point_3D = self.smooth_3d_filter(point_3D, cand3_3D, index_3d_fitting, drop_point)
                            dict_3D = self.smooth_3d_filter(dict_3D, cand3_3D, index_3d_fitting, drop_point)
                            # if pre_2d_filter_l:
                            #     pre_2d_filter_l = self.drop_2d_point(pre_2d_filter_l, drop_point)
                            #     time_2D_l.append(pre_2d_filter_l)
                            # if pre_2d_filter_r:
                            #     pre_2d_filter_r = self.drop_2d_point(pre_2d_filter_r, drop_point)
                            #     time_2D_r.append(pre_2d_filter_r)
                    # else:
                    #     if pre_2d_filter_l:
                    #         if self.fitting_drop[0]:
                    #             pre_2d_filter_l = self.drop_2d_point(pre_2d_filter_l, drop_2d_point_l)
                    #         time_2D_l.append(pre_2d_filter_l)
                    #     if pre_2d_filter_r:
                    #         if self.fitting_drop[0]:
                    #             pre_2d_filter_r = self.drop_2d_point(pre_2d_filter_r, drop_2d_point_r)
                    #         time_2D_r.append(pre_2d_filter_r)
                    # record the 3D
                    if self.fitting_drop[1]:
                        if index_3d_fitting.shape[0]:
                            point_3D = self.drop_3d_point(point_3D, index_3d_fitting)
                    time_3D.append(point_3D)
                    segment_length.append(self.segment_analysis(point_3D))
                    if guiding_pose:
                        average_segment_length = t_segment_length
                    else:
                        average_segment_length = np.array(self.segment_length_average(segment_length))
                    # print(average_segment_length)

                    # # update pre_pose on 3D
                    # pre_dict_3D_l = dict_3D_l.copy()
                    # pre_dict_3D_r = dict_3D_r.copy()

                    pose_coord = np.insert(dict_3D.reshape(-1), 0, frame).tolist()
                    # # Caulculate the angle
                    # if pose_coord[37] and pose_coord[34] and pose_coord[35] and pose_coord[38] and pose_coord[36] and pose_coord[39] and pose_coord[40] and pose_coord[41] and pose_coord[42] and pose_coord[49] and pose_coord[50] and pose_coord[51]:
                    #     com = [(abs(pose_coord[37])+abs(pose_coord[34]))/2, (abs(pose_coord[35])+abs(pose_coord[38]))/2, (abs(pose_coord[36])+abs(pose_coord[39]))/2]
                    #     leftheel_vector = np.array([abs(pose_coord[40])-com[0], abs(pose_coord[41])-com[1], abs(pose_coord[42])-com[2]])
                    #     rightheel_vector = np.array([abs(pose_coord[49])-com[0], abs(pose_coord[50])-com[1], abs(pose_coord[51])-com[2]])
                    #     z_vector = np.array([0, 0, 1])
                    #     sita, angle = self.Calculate_angle(leftheel_vector, rightheel_vector)
                    #     # pose_coord.append(sita)
                    #     sita1, angle1 = self.Calculate_angle(-leftheel_vector, z_vector)
                    #     # pose_coord.append(sita1)
                    #     sita2, angle2 = self.Calculate_angle(-rightheel_vector, z_vector)
                    #     # pose_coord.append(sita2)
                    # else:
                    #     # pose_coord.append(-1)
                    #     # pose_coord.append(-1)
                    #     # pose_coord.append(-1)
                    #     pass
                    # print(angle, angle1, angle2)
                    csv_data = np.array(pose_coord)
                    padding_index = np.argwhere(np.abs(csv_data) >= 1e10/2)
                    csv_data = csv_data.astype(np.str)
                    csv_data[padding_index] = ''
                    csv_coord.writerow(csv_data)
                    csv_inf.writerow(coord_inf)
                    #     print('average error = ', str(average_error), '%')

                    # # plot_2D
                    h_2d, w_2d = 1080, 1920
                    if self.visualize[0]:
                        for dd in range(pose.shape[0]):
                            img_2d = self.plot2D(pose[dd, frame, :, :], (h_2d, w_2d))
                            img_2d = cv2.resize(img_2d, (int(w_2d / 2), int(h_2d / 2)), interpolation=cv2.INTER_CUBIC)
                            cv2.imshow(str(dd), img_2d)
                        cv2.waitKey(1)
                    # pose[:, frame, :, :]
                    # if self.visualize[0]:
                    #     if pre_2d_filter_l:
                    #         img_l1 = self.plot2D(pose_L[frame], (L_h, L_w))
                    #         img_l2 = self.plot2D(pre_2d_filter_l, (L_h, L_w))
                    #         img_l = Sf.inplacePaste(img_l1, img_l2, int(L_w), 0)
                    #     else:
                    #         img_l = self.plot2D(pose_L[frame], (L_h, L_w))
                    #     if pre_2d_filter_r:
                    #         img_r1 = self.plot2D(pose_R[frame], (R_h, R_w))
                    #         img_r2 = self.plot2D(pre_2d_filter_r, (R_h, R_w))
                    #         img_r = Sf.inplacePaste(img_r1, img_r2, int(R_w), 0)
                    #
                    #     else:
                    #         img_r = self.plot2D(pose_R[frame], (R_h, R_w))
                    #     img_l = cv2.resize(img_l, (int(L_w / 2), int(L_h / 2)), interpolation=cv2.INTER_CUBIC)
                    #     img_r = cv2.resize(img_r, (int(R_w / 2), int(R_h / 2)), interpolation=cv2.INTER_CUBIC)
                    #     cv2.imwrite(self.out_3D_png + '/L_' + str(frame) + '_2D.jpg', img_l)
                    #     cv2.imwrite(self.out_3D_png + '/R_' + str(frame) + '_2D.jpg', img_r)
                    #     if self.visualize[0] == 'L':
                    #         cv2.imshow('L', img_l)
                    #     elif self.visualize[0] == 'R':
                    #         cv2.imshow('R', img_r)


                    #     cv2.waitKey(self.visualize[2])
                    # Plot 3D
                    if self.visualize[1]:
                        # self.Plot3D(dict_3D_project_l, dict_3D_project_r, self.xlim, self.ylim, self.zlim, frame, '')
                        self.Plot3D(dict_3D, self.xlim, self.ylim, self.zlim, frame, '')
                        img_3D = cv2.imread(self.out_3D_png + '/' + str(frame)+'_3D.jpg')
                        # out.write(img_3D)
                        cv2.imshow('3D', img_3D)
                        cv2.waitKey(1)
        # out.release()
        # Plot joint length
        cv2.destroyAllWindows()

        return np.arange(limit_i_frame, limit_f_frame+1), np.array(time_3D)

    def common_2d_filter(self, pre_pose, pose, velocity_threshold, time_dict_2D, fit_order, fit_n, checking_point, details):
        # caculate the velocity
        velocity = []
        velocity_set = []
        missing_point = []
        for i in range(len(pose)):
            if pre_pose[i][0] and pose[i][0]:
                temp_velocity = math.sqrt((pose[i][0] - pre_pose[i][0]) ** 2 + (pose[i][1] - pre_pose[i][1]) ** 2)
                velocity_set.append(round(temp_velocity))
                velocity.append(round(temp_velocity))
            else:
                velocity_set.append('')
            # if not pose[i][0]:
            #     missing_point.append(i)
        q_state = [i for i in range(5, len(velocity_set)) if velocity_set[i] and velocity_set[i] > np.average(velocity)*velocity_threshold]
        # print(q_state)

        if details:
            print('exchange = ', q_state)

        # min the all pose
        if len(q_state) > 1:
            min_velocity = 1000000
            arrange = list(itertools.permutations(q_state))
            for i in arrange:
                temp_pose = pose.copy()
                for j in range(len(q_state)):
                    temp_pose[q_state[j]] = pose[i[j]]
                temp_velocity = 0
                # print(pre_pose)
                # print(temp_pose, q_state)
                for k in range(len(q_state)):
                    temp_velocity += math.sqrt((temp_pose[q_state[k]][0] - pre_pose[q_state[k]][0]) ** 2 + (temp_pose[q_state[k]][1] - pre_pose[q_state[k]][1]) ** 2)
                if temp_velocity < min_velocity:
                    new_pose = temp_pose
                    min_velocity = temp_velocity

            # drop point
            drop_velocity = []
            drop_velocity_set = []
            for i in range(len(new_pose)):
                if pre_pose[i][0] and new_pose[i][0]:
                    temp_velocity = math.sqrt((new_pose[i][0] - pre_pose[i][0]) ** 2 + (new_pose[i][1] - pre_pose[i][1]) ** 2)
                    drop_velocity_set.append(round(temp_velocity))
                    drop_velocity.append(round(temp_velocity))
                else:
                    drop_velocity_set.append('')
            temp = [i for i in range(len(drop_velocity_set)) if drop_velocity_set[i] and drop_velocity_set[i] > np.average(drop_velocity) * velocity_threshold]
            missing_point.extend(temp)
        else:
            new_pose = pose.copy()
        if details:
            print('missing point = ', missing_point)

        drop_point = []
        if fit_order:
            # fitting missing point
            if missing_point:
                for point in missing_point:
                    if len(time_dict_2D) < fit_n:
                        dim = len(time_dict_2D)
                        train_Y = np.array(time_dict_2D)[:, point].T
                        train_X = np.linspace(1, dim, dim)
                        missing_index = np.where(train_Y[0] == '')
                        train_X = np.delete(train_X, missing_index, axis=0)
                        train_Y = np.delete(train_Y, missing_index, axis=1)
                    else:
                        dim = fit_n
                        train_Y = np.array(time_dict_2D)[:dim, point].T
                        train_X = np.linspace(1, dim, dim)
                        missing_index = np.where(train_Y[0] == '')
                        train_X = np.delete(train_X, missing_index, axis=0)
                        train_Y = np.delete(train_Y, missing_index, axis=1)

                    if train_X.shape[0] >= fit_n:
                        pre_x, pre_y = self.fitting(train_X, train_Y, fit_order)
                        drop_point.append(point)
                        # print(pre_x, pre_y)
                        new_pose[point] = [pre_x, pre_y]
        if details:
            print('fitting 2D point = ', drop_point)
            print('old:')
            # print(pose)
            print(velocity_set)
            print('over point = ', q_state)
            print('new:')
            # print(new_pose)
            new_velocity_set = []
            for i in range(len(new_pose)):
                if new_pose[i][0] and pre_pose[i][0]:
                    temp_velocity = math.sqrt((new_pose[i][0] - pre_pose[i][0]) ** 2 + (new_pose[i][1] - pre_pose[i][1]) ** 2)
                    new_velocity_set.append(round(temp_velocity))
                else:
                    new_velocity_set.append('')
            print(new_velocity_set)
            print('new pose = ', new_pose)
            print('')

        return new_pose, drop_point

    def plot2D(self, data_2D, winsize):
        data_2D = data_2D.astype(np.int)
        leni = [0, 1, 0, 2, 17, 5, 7, 17, 6, 8, 17, 11, 13, 17, 12, 14]
        lenf = [1, 3, 2, 4, 5, 7, 9, 6, 8, 10, 11, 13, 15, 12, 14, 16]
        winsize = np.array(winsize).astype(np.int)
        img = np.zeros(winsize, np.uint8)+255
        for i in range(len(data_2D)):
            if data_2D[i][0] != -1:
                cv2.circle(img, tuple(data_2D[i]), 5, (0, 0, 0), 0)
                cv2.putText(img, str(i), tuple(data_2D[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        for i in range(len(leni)):
            if data_2D[leni[i]][0] != -1 and data_2D[lenf[i]][0] != -1:
                cv2.line(img, tuple(data_2D[leni[i]]), tuple(data_2D[lenf[i]]), (0,0,0), 2)
        return img

    def drop_2d_point(self, point_2d, drop_point):
        for point in drop_point:
            point_2d[point] = ['', '']
        return point_2d

    def drop_3d_point(self, point_3d, drop_point):
        point_3d[drop_point] = np.array([1e10, 1e10, 1e10])
        return point_3d

    def smooth_3d_filter(self, point_3D, cand_3D, smooth_point, drop_point):
        if smooth_point.shape[0]:
            point_3D[smooth_point] = cand_3D[smooth_point]
        if drop_point.shape[0]:
            point_3D[drop_point] = np.array([1e10, 1e10, 1e10])
        return point_3D

    def checking_3d_smooth(self, average, cand1, cand2):
        loss_segment_1 = np.array(abs(average-cand1) / average)
        loss_segment_2 = np.array(abs(average - cand2) / average)
        loss_point_index = dict.fromkeys((set(self.leni.tolist() + self.lenf.tolist())), 0)
        loss_point_1 = dict.fromkeys((set(self.leni.tolist() + self.lenf.tolist())), 0)
        loss_point_2 = dict.fromkeys((set(self.leni.tolist() + self.lenf.tolist())), 0)
        dict_loss_1 = {}
        dict_loss_2 = {}
        smooth_point = []
        drop_point = []
        for i in range(len(self.leni)):
            loss_point_index[self.leni[i]] += 1
            loss_point_index[self.lenf[i]] += 1
            loss_point_1[self.leni[i]] += loss_segment_1[i]
            loss_point_1[self.lenf[i]] += loss_segment_1[i]
            loss_point_2[self.leni[i]] += loss_segment_2[i]
            loss_point_2[self.lenf[i]] += loss_segment_2[i]
        for j in loss_point_index:
            if not np.isnan(loss_point_1[j]):
                dict_loss_1[j] = loss_point_1[j]/loss_point_index[j]
                dict_loss_2[j] = loss_point_2[j] / loss_point_index[j]
                if dict_loss_1[j] > self.threeD_threshold[1]:
                    if dict_loss_2[j] <= self.threeD_threshold[1]:
                        smooth_point.append(j)
                    else:
                        drop_point.append(j)
        smooth_point = np.array(smooth_point)
        drop_point = np.array(drop_point)

        # print(self.leni)
        # print(self.lenf)
        # print(smooth_point, drop_point)
        # print(loss_point_index)
        # print(loss_point_1)
        # print(loss_point_2)
        # print(dict_loss_1)
        # print(dict_loss_2)
        # exit()
        return smooth_point, drop_point, dict_loss_1, dict_loss_2

    def segment_length_average(self, segment_length):
        segment_length = np.array(segment_length)
        if segment_length.shape[0] == 1:  # frame = 1
            average_segment_length = segment_length
        else:
            average_segment_length = []
            for point in segment_length.T:
                index_zero = np.where(point == 0)
                point = np.average(np.delete(point, index_zero, axis=0))
                if np.isnan(point):
                    point = 0
                average_segment_length.append(point)
        return average_segment_length

    def pre_3d_fitting(self, time_3D, fit_order, fit_n):
        if len(time_3D) < fit_n:
            dim = len(time_3D)
            time_3D = np.array(time_3D[:dim])
            fitting_3D = []
            for point in range(time_3D.shape[1]):
                train_X = np.linspace(1, dim, dim)
                train_Y = []
                for xyz in range(time_3D.shape[2]):
                    train_Y.append(list(time_3D[:, point, xyz]))  # frame/point/xyz
                train_Y = np.array(train_Y)
                missing_index = np.where(train_Y[0] == 1e10)
                train_X = np.delete(train_X, missing_index, axis=0)
                train_Y = np.delete(train_Y, missing_index, axis=1)
                if train_X.shape[0] > 2:
                    pre = self.fitting(train_X, train_Y, fit_order)
                    fitting_3D.append(list(pre))
                else:
                    pre = [1e10, 1e10, 1e10]
                    fitting_3D.append(pre)
        else:
            dim = fit_n
            time_3D = np.array(time_3D)
            fitting_3D = []
            # print(time_3D.shape[0])
            for point in range(time_3D.shape[1]):
                train_X = np.linspace(1, dim, dim)
                train_Y = []
                for xyz in range(time_3D.shape[2]):
                    train_Y.append(list(time_3D[time_3D.shape[0]-dim:time_3D.shape[0], point, xyz]))  # frame/point/xyz
                # print(list(time_3D[time_3D.shape[0]-dim:time_3D.shape[0], point, xyz]))
                train_Y = np.array(train_Y)

                missing_index = np.where(train_Y[0] == 1e10)
                train_X = np.delete(train_X, missing_index, axis=0)
                train_Y = np.delete(train_Y, missing_index, axis=1)
                if train_X.shape[0] > 1:
                    pre = self.fitting(train_X, train_Y, fit_order)
                    fitting_3D.append(list(pre))
                else:
                    pre = [1e10, 1e10, 1e10]
                    fitting_3D.append(pre)
        fitting_3D = np.array(fitting_3D)
        return fitting_3D

    def Calculate_angle(self, x, y):
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y)/(Lx*Ly)
        rad = np.arccos(cos_angle)
        angle = rad*360/2/np.pi
        # print(rad, angle)
        return rad, angle

    def plot_segment_length(self, frame, segment_length, leni, lenf, plt_segment_range):
        n = 0
        for len in segment_length.T:
            zero_index = np.argwhere(len == -1)
            len = np.delete(len, zero_index, axis=0)
            X = np.delete(frame, zero_index, axis=0)
            plt.plot(X, len, label=str(leni[n])+'_'+str(lenf[n]))
            plt.legend(loc='upper right')
            n += 1
        # plt.rcParams['figure.figsize'] = (12, 4)
        plt.xticks([])
        plt.yticks(plt_segment_range)
        plt.title('Segment length')
        plt.xlabel('frame  ['+str(frame[0])+' to '+str(frame[-1])+']')
        plt.ylabel('length')
        plt.show()

    def segment_analysis(self, dict_3D):
        unlimited_point = np.argwhere(dict_3D[:, 0] == 1e10).reshape(-1)
        unlimited_bond_index = np.array([])
        for i in unlimited_point:
            temp = np.append(np.argwhere(self.leni == i).reshape(-1), np.argwhere(self.lenf == i).reshape(-1))
            unlimited_bond_index = np.append(unlimited_bond_index, temp)
        unlimited_bond_index = np.unique(unlimited_bond_index).astype(np.int)

        # if the point is [1e10, 1e10, 1e10], that is -1 on bond
        inf_bond = np.sqrt(np.sum(np.power(dict_3D[self.leni]-dict_3D[self.lenf], 2), axis=1))
        inf_bond[unlimited_bond_index] = -1

        return inf_bond

    def smooth_pose(self, pose, pre_pose, backup_pose, pre_backup_pose, frame, fps, point):
        # supplement the pose
        if not pose[0]:
            if frame-backup_pose[3] > self.inter_len :
                pose = ['', '', '', 1]
            elif pre_backup_pose[0]:
                x = pre_backup_pose[0]+(backup_pose[0]-pre_backup_pose[0])/(backup_pose[3]-pre_backup_pose[3])*(frame-backup_pose[3])
                y = pre_backup_pose[1]+(backup_pose[1] - pre_backup_pose[1]) / (backup_pose[3] - pre_backup_pose[3]) * (frame - backup_pose[3])
                z = pre_backup_pose[2]+(backup_pose[2] - pre_backup_pose[2]) / (backup_pose[3] - pre_backup_pose[3]) * (frame - backup_pose[3])
                pose = [x, y, z, 1]
            else:
                temp = backup_pose[0:3]
                temp.append(1)
                pose = temp

        # calculate the velocity
        if pose[0] and pre_pose[0] and frame-backup_pose[3] < self.inter_len :
            velocity = math.sqrt((pose[0]-pre_pose[0])**2+(pose[1]-pre_pose[1])**2+(pose[2]-pre_pose[2])**2)/fps
            if velocity < self.threshold_3D_velocity:
                new_pose = pose
            else:
                # print('over velocity = ', velocity)
                # print(velocity, self.threshold_3D_velocity)
                temp = backup_pose[0:3]
                temp.append(1)
                new_pose = temp

        else:
            new_pose = pose

        # double check
        if new_pose[3] == 1:
            if new_pose[0] and pre_backup_pose[0]:
                velocity = math.sqrt((new_pose[0] - pre_backup_pose[0]) ** 2 + (new_pose[1] - pre_backup_pose[1]) ** 2 + (new_pose[2] - pre_backup_pose[2]) ** 2) / fps
                if velocity > self.threshold_3D_velocity:
                    new_pose = ['', '', '', 1]

        return new_pose

    def fitting(self, train_X, train_Y, fit_order):
        # 2D
        if train_Y.shape[0] == 2:
            train_Y_f1 = train_Y[0].astype(np.float)
            train_Y_f2 = train_Y[1].astype(np.float)
            fitting_1 = np.poly1d(np.polyfit(train_X, train_Y_f1, fit_order))
            fitting_2 = np.poly1d(np.polyfit(train_X, train_Y_f2, fit_order))
            pre_x = int(round(fitting_1(train_X[-1] + 1)))
            pre_y = int(round(fitting_2(train_X[-1] + 1)))
            return pre_x, pre_y
        # 3D
        if train_Y.shape[0] == 3:
            train_Y_f1 = train_Y[0].astype(np.float)
            train_Y_f2 = train_Y[1].astype(np.float)
            train_Y_f3 = train_Y[2].astype(np.float)
            fitting_1 = np.poly1d(np.polyfit(train_X, train_Y_f1, fit_order))
            fitting_2 = np.poly1d(np.polyfit(train_X, train_Y_f2, fit_order))
            fitting_3 = np.poly1d(np.polyfit(train_X, train_Y_f3, fit_order))
            pre_x = fitting_1(train_X[-1] + 1)
            pre_y = fitting_2(train_X[-1] + 1)
            pre_z = fitting_3(train_X[-1] + 1)
            return pre_x, pre_y, pre_z

        # # check
        # ground_turth = [1164, 683]
        # print(np.polyfit(train_X, train_Y_f1, fit_order))
        # print(np.polyfit(train_X, train_Y_f2, fit_order))
        # plt.title('Fitting missing point')
        # plt.xlabel('frame')
        # plt.ylabel('pixel')
        # plt_x = np.linspace(train_X[0], train_X[-1]+1, fit_n)
        # # plt.plot(train_X, train_Y_f1, '.', plt_x, fitting_1(plt_x), '-', [train_X[-1]+1], ground_turth[0], '*k')  # train_X[-1], p30(xp), '--')
        # plt.plot(train_X, train_Y_f2, '.', plt_x, fitting_2(plt_x), '-', [train_X[-1] + 1], ground_turth[1], '*k')
        # plt.show()

    def Loadscv(self, dict_path, limit_frame=1e10, confidence_threshold=0.05, padding=-1):
        frame = pd.read_csv(dict_path).fillna(padding).values[:, 0]
        data = pd.read_csv(dict_path).fillna(padding).values[:, 1:]
        data = data.reshape(data.shape[0], -1, 2)
        confidience_path = os.path.splitext(dict_path)[0]+'_score.csv'
        if os.path.isfile(confidience_path):
            confidience = pd.read_csv(confidience_path).fillna(-1).values[:, 1:]
            confidience = confidience.reshape(confidience.shape[0], -1)
            padding_index = np.argwhere(confidience < confidence_threshold)
            for index in padding_index:
                data[index[0], index[1], :] = [padding, padding]
        if data.shape[0] > limit_frame:
            data = data[:limit_frame, :, :]
        # tool().data2csv2d(dict_path, frame, data, 'mask')
        return data

    def nothing(self, y):
        pass

    def check_error(self):
        # Analysis the error
        error_title = 'Baseline[m] = ' + str(self.baseline) + ' distance[m] = ' + str(self.distance)
        error_name = ["0_1", "1_3", "0_2", "2_4", "17_5", "5_7", "7_9", "17_6", "6_8", "8_10", "17_11", "11_13",
                      "13_15", "17_12", "12_14", "14_15", "average"]
        error_color = ["black", "blue", "red", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse",
                       "chocolate", "coral", "cornflowerblue", "crimson", "darkblue", "darkcyan", "darkgreen",
                       "darkmagenta", "darkolivegreen"]
        # Dataloder
        error_data = {}
        error_legend = []
        with open(self.output_3D + 'threeD_scences_error.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            read_error_count = 0
            for row in rows:
                if read_error_count > 0:
                    for index in range(18):
                        if read_error_count == 1 and index in self.error_draw:
                            error_legend.append(error_name[index-1])
                        if index > 0 and index in self.error_draw:
                            try:
                                temp = error_data.pop(error_name[index - 1])
                                temp.append(row[index])
                                error_data[error_name[index - 1]] = temp
                            except:
                                error_data[error_name[index - 1]] = [row[index]]
                read_error_count += 1

        self.plot_error(error_data, error_color, error_legend, error_title)

if __name__ == '__main__':

    output_3D = '../20200318/'
    imgsize = (720, 540)
    limit_i_frame = 1  # 4-698
    limit_f_frame = 800  # 4-1200
    extrapolation_set = [0, 0.7, 1, 5]  # [switch, loss segment length [%], fitting order, number of fits]
    # plot 3d
    xlim, ylim, zlim = [-1.0, 1.5, 0.5], [1.8, 4.3, 0.5], [-1.0, 1.5, 0.5]  # [-5, 5, 0.5], [4, 7, 0.5], [-1.5, 1.5, 0.5]
    view = [0, 0]  # YZ[0, 0]  XZ[0, -90] XY[-90, 90] 1. [0, -115] / [0, -25] / [-90, 65]
    fps = 30
    guiding_pose = ''  # ''../20200114/T-pose.csv'


    camera_int, camera_dist, camera_rvec, camera_tvec, dict_path = [], [], [], [], []
    camera_int.append(output_3D + "R_33GX287/calibration/mtx.npy")
    camera_int.append(output_3D + "L_33GX287/calibration/mtx.npy")
    camera_int.append(output_3D + "A_33GX287/calibration/mtx.npy")
    camera_dist.append(output_3D + "R_33GX287/calibration/dist.npy")
    camera_dist.append(output_3D + "L_33GX287/calibration/dist.npy")
    camera_dist.append(output_3D + "A_33GX287/calibration/dist.npy")
    camera_rvec.append(output_3D + "R_33GX287/relation/RLA/cube/backup/R1/R.npy")
    camera_rvec.append(output_3D + "L_33GX287/relation/RLA/cube/backup/L1/R.npy")
    camera_tvec.append(output_3D + "R_33GX287/relation/RLA/cube/backup/R1/T.npy")
    camera_tvec.append(output_3D + "L_33GX287/relation/RLA/cube/backup/L1/T.npy")
    dict_path.append(output_3D + 'R_33GX287/relation/RLA/cube/backup/R1/R1_block.csv')  # R_33GX287/real/human4/AlphaPose_R4_tracker.csv
    dict_path.append(output_3D + 'L_33GX287/relation/RLA/cube/backup/L1/L1_block.csv')  # R_33GX287/relation/RLA/cube/backup/R15/R15_block.csv
    dict_path.append(output_3D + 'A_33GX287/relation/RLA/cube/backup/A1/A1_block.csv')
    # dict_path.append(output_3D + 'R_33GX287/real/human6/AlphaPose_R6.csv')  # R_33GX287/real/human4/AlphaPose_R4_tracker.csv
    # dict_path.append(output_3D + 'L_33GX287/real/human6/AlphaPose_L6.csv')  # R_33GX287/relation/RLA/cube/backup/R15/R15_block.csv
    # dict_path.append(output_3D + 'A_33GX287/real/human6/AlphaPose_A6.csv')


    frame, data = threeD().coord(output_3D=output_3D, camera_int=camera_int, camera_dist=camera_dist, camera_rvec=camera_rvec, camera_tvec=camera_tvec, dict_path=dict_path, guiding_pose=guiding_pose, extrapolation_set=extrapolation_set, imgsize=imgsize, limit_i_frame=limit_i_frame, limit_f_frame=limit_f_frame, fps=fps)
    cube_length = 1
    # data[0] = threeD().transformation(data[0].copy(), cube_length=cube_length, output_path='./H')

    # exit()
    # data[0] = transmation_coords
    frame, data = tool().Loadcsv_3d('../20200318/threeDdata.csv')
    # tool().Plot3D(frame, data, xlim, ylim, zlim, view, '../20200212/acc/', fps)

    # # cube
    cube_length = 1  # [m]
    leni = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6])
    lenf = np.array([1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7])
    xlim, ylim, zlim = ['', '', ''], ['', '', ''], ['', '', '']
    view = [20, -45]
    # frame, data = tool().Loadcsv_3d('../20200407/threeDdata.csv')
    camera_path = []
    pose_path = output_3D+'threeDdata.csv'
    inf_path = output_3D+'threeDdata_inf.csv'
    frame, segment_length = joint_analysis().segment_analysis(path='../20200318/123.csv', frame=frame, data=data, leni=leni, lenf=lenf)
    loss = np.abs(segment_length-cube_length)/cube_length*100
    camera_path.append(output_3D + 'R_33GX287/relation/RLA/cube/backup/R1/R1.avi')
    camera_path.append(output_3D + 'L_33GX287/relation/RLA/cube/backup/L1/L1.avi')
    camera_path.append(output_3D + 'A_33GX287/relation/RLA/cube/backup/A1/A1.avi')
    print('average loss = ', np.average(loss))
    print('segment_length = ', segment_length)
    print('loss = ', loss)
    # tool().Plot3D(frame, data, xlim, ylim, zlim, view, '../20200318/acc/', fps=3, type='cube')
    # # plot 2d
    tool().plot_reproject(imgsize, pose_path, inf_path, camera_path, camera_int, camera_dist, camera_rvec, camera_tvec, fps=1, type='cube_projection')
    # # mypath = "../20200221/R_33GX287/relation/RLA/cube/backup/R15/R15_block.csv"
    # # video_path = "../20200221/R_33GX287/relation/RLA/cube/backup/R15/R15.avi"
    # # frame, data = tool().Loadcsv_2d(mypath)
    # # tool().plot2D(frame, data, video_path, fps='', imgsize=imgsize, type='cube', pose_inf='', label=False)
    # # mypath = "../20200221/L_33GX287/relation/RLA/cube/backup/L15/L15_block.csv"
    # # video_path = "../20200221/L_33GX287/relation/RLA/cube/backup/L15/L15.avi"
    # # frame, data = tool().Loadcsv_2d(mypath)
    # # tool().plot2D(frame, data, video_path, fps='', imgsize=imgsize, type='cube', pose_inf='', label=False)
    # # mypath = "../20200221/A_33GX287/relation/RLA/cube/backup/A15/A15_block.csv"
    # # video_path = "../20200221/A_33GX287/relation/RLA/cube/backup/A15/A15.avi"
    # # frame, data = tool().Loadcsv_2d(mypath)
    # # tool().plot2D(frame, data, video_path, fps='', imgsize=imgsize, type='cube', pose_inf='', label=False)







