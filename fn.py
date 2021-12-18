import cv2
import numpy as np
import math
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
from itertools import combinations
import time
import random
from scipy.fftpack import fft, ifft
import torch
from scipy.interpolate import make_interp_spline
import pynvml  # GPU
import os, conda_pack, tarfile


class camera_orientation(object):

    def __init__(self):  # Run it once
        pass

    def Interior_Orientation(self, path):
        # FOV = ['', '']
        mtx = np.load(path)
        # modify the f
        # _, _, f, _, _ = cv2.calibrationMatrixValues( mtx, img_size, 1, 1 )  # [o] [o] [mm] [mm] []
        # aperture = L_f/f
        # FOV[0], FOV[1], f, p, ratio = cv2.calibrationMatrixValues( mtx, img_size, aperture, aperture )  # [o] [o] [mm] [mm] []
        # print(FOV[0], FOV[1], f, p, ratio)
        return mtx

    def Exterior_Orientation(self, R, T):
        if R.shape[1] == 1:
            R, jacobain = cv2.Rodrigues( R, None )
        # camera_position = -np.dot( R.transpose(), T )
        # roll = math.atan2( -R.transpose()[2][1], R.transpose()[2][2] ) * 180 / math.pi
        # pitch = math.asin( R.transpose()[2][0] ) * 180 / math.pi
        # yaw = math.atan2( -R.transpose()[1][0], R.transpose()[0][0] ) * 180 / math.pi
        # print(roll, pitch, yaw, camera_position)
        RT = np.hstack((R, T))
        return RT

    def world2img(self, coord, mtx, rvec, tvec, dist):
        coord = coord.reshape(-1, 3)
        threed_padding = np.zeros((coord.shape[0], 1)) + 1.
        coord = np.hstack((coord, threed_padding))
        coord = coord[:, np.newaxis, :]
        coord = coord[:, :, :-1]
        img_coord, jac = cv2.projectPoints(coord, rvec, tvec, mtx, dist)
        img_coord = img_coord.reshape(-1, 2)
        return img_coord

    def dire_world2img(self, coord, mtx, rvec, tvec, dist):
        img_coord = []
        coord = coord.reshape(-1, 3)
        threed_padding = np.zeros((coord.shape[0], 1)) + 1.
        coord = np.hstack((coord, threed_padding))
        vec = np.hstack((rvec, tvec))
        for i in range(coord.shape[0]):
            tmp = np.dot(mtx, np.dot(vec, coord[i].reshape(1, 4).T))
            tmp = [tmp[0] / tmp[2], tmp[1] / tmp[2]]
            img_coord.append(tmp)
        img_coord = np.array(img_coord)
        return img_coord

    def undistpoint(self, coord_2d, mtx, size, dist):
        coord_2d = coord_2d.reshape(-1, 2)[:, np.newaxis, :]
        w, h = size[0], size[1]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # Optimization parameter (Free ratio parameter)
        undist_coord_2d = cv2.undistortPoints(coord_2d, mtx, dist, P=new_mtx)
        undist_coord_2d = undist_coord_2d.reshape(-1, 2)
        return undist_coord_2d

    def transfrom_rodrigues(self, rvecs, tvecs):
        if rvecs.shape[1] == 1:
            rmat, jacobain = cv2.Rodrigues( rvecs, None )
        else:
            rmat = rvecs
        camera_position = -np.dot( rmat.transpose(), tvecs )
        # camera_position = tvecs
        camera_position = np.array([camera_position[0][0], camera_position[2][0], -camera_position[1][0]]).reshape(-1, 1)
        roll = math.atan2( -rmat.transpose()[2][1], rmat.transpose()[2][2] ) * 180 / math.pi
        pitch = math.asin( rmat.transpose()[2][0] ) * 180 / math.pi
        yaw = math.atan2( -rmat.transpose()[1][0], rmat.transpose()[0][0] ) * 180 / math.pi
        return roll, pitch, yaw, camera_position

    def epipolar_select(self, mtx, vec, data, padding=1e10):
        L_camera, R_camera, loss = [], [], []
        # print('mtx = ', mtx.shape)
        # print('data = ', data.shape)
        # print('vec = ', vec.keys())
        for i in range(len(data)):  # L camera
            n = 0
            for j in range(i+1, len(data)):  # R camera
                if data[i][0] < 0:  # I'm [-1, -1]
                    loss.append(padding)
                elif i < j and str(i)+str(j) in vec and data[j][0] >= 0:
                    R = vec[str(i) + str(j)][:, :3].copy()
                    T = vec[str(i) + str(j)][:, -1:].copy()
                    term1 = np.linalg.inv(mtx[i]).T
                    term2 = np.array([[0, -T[2][0], T[1][0]], [T[2][0], 0, -T[0][0]], [-T[1][0], T[0][0], 0]])
                    term3 = np.linalg.inv(mtx[j])
                    F = np.dot(np.dot(term1, np.dot(term2, R)), term3)
                    F = F/F[2][2]
                    L_img = data[i].reshape(-1, 1)
                    R_img = data[j].reshape(-1, 1)
                    n += 1
                    # print(i, j, np.abs(np.dot(np.dot(R_img.T, F), L_img))[0][0])
                    # print('camera position = ' , np.dot(R, T))
                    loss.append(np.abs(np.dot(np.dot(R_img.T, F), L_img))[0][0])
                L_camera.append(i)
                R_camera.append(j)
        L_camera, R_camera, loss = np.array(L_camera), np.array(R_camera), np.array(loss)
        if np.min(loss) == padding:
            return -1, -1, loss
        else:
            min_index = np.argwhere(loss == np.min(loss))[0][0]
            L_index = L_camera[min_index]
            R_index = R_camera[min_index]
            return L_index, R_index, loss

    def re_projection_select(self, mtx, dist, vec, data, padding=1e10):
        L_camera, R_camera, loss = [], [], []
        for i in range(len(data)):  # L camera
            for j in range(i+1, len(data)):  # R camera
                L_camera.append(i)
                R_camera.append(j)
                # 2d to 3d
                if data[i][0] < 0 or data[j][0] < 0:  # I'm [-1, -1]
                    loss.append(padding)
                elif i < j and str(i)+str(j) in vec and data[j][0] >= 0:
                    n, temp_loss = 0, 0
                    L_H, R_H = np.dot(mtx[i], vec[str(0) + str(i)]), np.dot(mtx[j], vec[str(0) + str(j)])
                    L_img_coord = data[i][:2].reshape(-1, 1)
                    R_img_coord = data[j][:2].reshape(-1, 1)
                    coord = cv2.triangulatePoints(L_H, R_H, L_img_coord, R_img_coord).reshape(-1)
                    if coord[3] != 0:
                        coord[0] = coord[0] / coord[3]
                        coord[1] = coord[1] / coord[3]
                        coord[2] = coord[2] / coord[3]
                        coord[3] = 1.
                    coord = np.round(coord, 5)[:3]
                    # 3d to 2d
                    for k in range(len(data)):
                        if data[i][0] >= 0:
                            R = vec[str(0) + str(k)][:, :3].copy()
                            T = vec[str(0) + str(k)][:, -1:].copy()
                            data_2d = camera_orientation().world2img(coord, mtx[k], R, T, dist[k]).reshape(-1,  2)
                            temp_loss += joint_analysis().eucildea_distance(data[k][:2].reshape(-1, 2), data_2d)
                            n += 1
                    loss.append(temp_loss/n)
        L_camera, R_camera, loss = np.array(L_camera), np.array(R_camera), np.array(loss)
        if np.min(loss) == padding:
            return -1, -1, padding, loss
        else:
            min_index = np.argwhere(loss == np.min(loss))[0][0]
            L_index = L_camera[min_index]
            R_index = R_camera[min_index]
            return L_index, R_index, np.min(loss), loss

    def double_reprojection_loss(self, l_c, r_c, L_camera_int_path, R_camera_int_path, L_camera_dist_path, R_camera_dist_path, rvec_path, tvec_path, grad=False):

        def triangulatePoints(L_H, R_H, l_c, r_c):
            # pseudo_L_H, pseudo_R_H = torch.pinverse(L_H), torch.pinverse(R_H)
            zero_term = torch.zeros([3, 1])
            feature_L = torch.cat((L_H[:, 0:3], -l_c, zero_term), dim=1)
            feature_R = torch.cat((R_H[:, 0:3], zero_term, -r_c), dim=1)
            feature = torch.cat((feature_L, feature_R), dim=0)
            target = torch.cat((-L_H[:, 3], -R_H[:, 3]), dim=0).reshape(-1, 1)
            coord = closest_form(feature, target)[0:3].reshape(1, -1)
            return coord

        def closest_form(feature, target):
            feature = feature.t()
            w = torch.mm(torch.mm(torch.inverse(torch.mm(feature, feature.t())), feature), target)
            return w

        def world2img(coord, mtx, dist, vec):
            threed_padding = torch.zeros((coord.shape[0], 1)) + 1.
            coord = torch.cat((coord, threed_padding), dim=1).t()
            img = torch.mm(torch.mm(mtx, vec), coord)
            img[0] = img[0]/img[index]
            R_index = R_camera[min_index]
            return L_index, R_index, np.min(loss), loss

    def img2world(self, mtx, dist, vec, pose, padding=1e10):
        one_padding = np.ones((pose.shape[0], pose.shape[1], 1))
        pose = np.concatenate((pose, one_padding), axis=2)
        coord_set, coord_inf = [], []
        for i in range(pose.shape[1]):
            data = pose[:, i, :]
            # L_index, R_index, loss = self.epipolar_select(mtx, vec, data)
            L_index, R_index, miniloss, loss = self.re_projection_select(mtx, dist, vec, data)
            # L_index, R_index = 0, 2
            coord_inf.append([str(i), str(L_index), str(R_index), str(miniloss)])
            if L_index >= 0 or R_index >= 0:
                L_H, R_H = np.dot(mtx[L_index], vec[str(0)+str(L_index)]), np.dot(mtx[R_index], vec[str(0)+str(R_index)])
                # print('select best camera = ', L_index, R_index, loss)
                L_img_coord = data[L_index][:2].reshape(-1, 1)
                R_img_coord = data[R_index][:2].reshape(-1, 1)
                coord = cv2.triangulatePoints(L_H, R_H, L_img_coord, R_img_coord).reshape(-1)
                if coord[3] != 0:
                    coord[0] = coord[0] / coord[3]
                    coord[1] = coord[1] / coord[3]
                    coord[2] = coord[2] / coord[3]
                    coord[3] = 1.
                coord = np.round(coord, 5)[:3]
                coord_set.append(coord)
            else:
                L_index, R_index, coord = -1, -1, np.array([padding, padding, padding])
                coord_set.append(coord)
            # print('the point from = ', L_index, R_index, 'the coord = ', coord, 'loss = ', loss)
        coord_set, coord_inf = np.array(coord_set), np.array(coord_inf)
        X = coord_set[:, 0:1]
        Y = coord_set[:, 2:3]
        Z = -coord_set[:, 1:2]
        coord_set = np.concatenate((X, Y, Z), axis=1)
        return coord_set, coord_inf.reshape(-1)

    def camera_coords_transformation(self, rvec, tvec, H, padding=1e10):
        coord = []
        coord.append(np.array([0, 0, 0]).reshape(-1, 3))
        tmp_vec = np.hstack((np.eye(3), np.zeros((3, 1))))
        # RT compose
        for i in range(len(rvec)):
            print('Camera = ', i+1)
            R, T = np.load(rvec[i]), np.load(tvec[i])
            # R, T, _, _, _, _, _, _, _, _ = cv2.composeRT(cv2.Rodrigues(R, None)[0], T, cv2.Rodrigues(tmp_vec[:, :3], None)[0], tmp_vec[:, -1:])
            roll, pitch, yaw, camera_position = camera_orientation().transfrom_rodrigues(R, T)
            coord.append(camera_position.reshape(-1, 3))
            tmp_vec = camera_orientation().Exterior_Orientation(R, T)
        # self-calculation
        # for i in range(len(rvec)):
        #     R, T = np.load(rvec[i]), np.load(tvec[i])
        #     camera_position = -np.dot(R.transpose(), T)
        #     camera_position = np.array([camera_position[0][0], camera_position[2][0], -camera_position[1][0]]).reshape(-1, 1)
        #     camera_position = camera_position.reshape(-1, 3)
        #     coord.append(camera_position.reshape(-1, 3))
        #     # tmp_vec = camera_orientation().Exterior_Orientation(R, T)

        coord = np.array(coord).reshape(1, -1, 3)
        print('origin camera')
        print(coord)
        coord = camera_orientation().threeD_transformation(dataset=coord, H=H, padding=padding)
        print('transformation camera')
        print(coord)
        return coord

    def threeD_transformation(self, dataset, H, padding=1e10):
        new_data = dataset.copy()
        for i in range( dataset.shape[0]):
            padding_index = np.argwhere(np.abs(dataset[i]) >= padding/2)
            padding_index = np.unique(padding_index[:, 0])
            src = np.dot(H, np.concatenate((dataset[i], np.ones((dataset[i].shape[0], 1))), axis=1).T).T
            coordinates = src[:, 0:3] / src[:, -1].reshape(-1, 1)
            if padding_index.shape[0]:
                for index in padding_index:
                    coordinates[index, :] = np.array([padding, padding, padding])
            new_data[i] = coordinates
        return new_data

    def double_reprojection_loss(self, l_c, r_c, L_camera_int_path, R_camera_int_path, L_camera_dist_path, R_camera_dist_path, rvec_path, tvec_path, img_size, grad=False):

        def triangulatePoints(L_H, R_H, l_c, r_c):
            # pseudo_L_H, pseudo_R_H = torch.pinverse(L_H), torch.pinverse(R_H)
            zero_term = torch.zeros([3, 1])
            feature_L = torch.cat((L_H[:, 0:3], -l_c, zero_term), dim=1)
            feature_R = torch.cat((R_H[:, 0:3], zero_term, -r_c), dim=1)
            feature = torch.cat((feature_L, feature_R), dim=0)
            target = torch.cat((-L_H[:, 3], -R_H[:, 3]), dim=0).reshape(-1, 1)
            coord = closest_form(feature, target)[0:3].reshape(1, -1)
            return coord

        def closest_form(feature, target):
            feature = feature.t()
            w = torch.mm(torch.mm(torch.inverse(torch.mm(feature, feature.t())), feature), target)
            return w

        def world2camera(coord, mtx, vec):
            threed_padding = torch.zeros((coord.shape[0], 1)) + 1.
            coord = torch.cat((coord, threed_padding), dim=1).t()
            img = torch.mm(torch.mm(mtx, vec), coord)
            img[0] = img[0]/img[-1]
            img[1] = img[1] / img[-1]
            img[2] = img[2] / img[-1]
            return img

        def undistpoint(coord_2d, mtx, size, dist):
            coord_2d = np.array(coord_2d).astype(np.float32)
            coord_2d = coord_2d.reshape(-1, 2)[:, np.newaxis, :]
            w, h = int(size[0]), int(size[1])
            new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # Optimization parameter (Free ratio parameter)
            undist_coord_2d = cv2.undistortPoints(coord_2d, mtx, dist, P=new_mtx)
            undist_coord_2d = undist_coord_2d.reshape(-1, 2)
            return undist_coord_2d[0].tolist()

        # load camera orientation
        L_camera_mtx, L_camera_dist = np.load(L_camera_int_path), np.load(L_camera_dist_path)
        R_camera_mtx, R_camera_dist = np.load(R_camera_int_path), np.load(R_camera_dist_path)
        R_vec = np.hstack((np.eye(3), np.zeros((3, 1))))
        rvec, tvec = np.load(rvec_path), np.load(tvec_path)
        L_vec = self.Exterior_Orientation(rvec, tvec)
        L_H, R_H = np.dot(L_camera_mtx, L_vec), np.dot(R_camera_mtx, R_vec)
        if grad:
            # undistoriton
            l_c = camera_orientation().undistpoint(np.array(l_c).astype(np.float32), L_camera_mtx, img_size, L_camera_dist)
            r_c = camera_orientation().undistpoint(np.array(r_c).astype(np.float32), R_camera_mtx, img_size, R_camera_dist)
            l_c = l_c[0].tolist()
            r_c = r_c[0].tolist()
            # load the camera orientation
            L_camera_mtx, L_camera_dist = torch.from_numpy(L_camera_mtx.astype(np.float32)), torch.from_numpy(L_camera_dist.astype(np.float32))
            R_camera_mtx, R_camera_dist = torch.from_numpy(R_camera_mtx.astype(np.float32)), torch.from_numpy(R_camera_dist.astype(np.float32))
            R_vec = torch.from_numpy(R_vec.astype(np.float32))
            L_vec = torch.from_numpy(L_vec.astype(np.float32))
            L_H, R_H = torch.from_numpy(L_H.astype(np.float32)), torch.from_numpy(R_H.astype(np.float32))  # torch.autograd.Variable(torch.from_numpy(R_H), requires_grad=True)
            l_c.append(1), r_c.append(1)
            l_c, r_c = torch.autograd.Variable(torch.Tensor(l_c).reshape(len(l_c), 1), requires_grad=True), torch.autograd.Variable(torch.Tensor(r_c).reshape(len(l_c), 1), requires_grad=True)
            # (2D-3D) triangulatePoints solver by closest form
            coord = triangulatePoints(L_H, R_H, l_c, r_c)
            # (3D-2D)
            L_2d = world2camera(coord, L_camera_mtx, L_vec).reshape(-1)
            R_2d = world2camera(coord, R_camera_mtx, R_vec).reshape(-1)
            # loss
            L_loss = torch.sqrt((L_2d - l_c.t()).pow(2).sum())
            R_loss = torch.sqrt((R_2d - r_c.t()).pow(2).sum())
            print('L undist = ', l_c[0:2].reshape(-1), ' re-projection undist = ', L_2d[0:2].reshape(-1))
            print('R undist = ', r_c[0:2].reshape(-1), ' re-projection undist = ', R_2d[0:2].reshape(-1))
            return L_loss+R_loss
        else:
            # 2D to 3D
            coord = cv2.triangulatePoints(L_H, R_H, tuple(l_c), tuple(r_c)).reshape(-1)
            if coord[3] != 0:
                coord[0] = coord[0] / coord[3]
                coord[1] = coord[1] / coord[3]
                coord[2] = coord[2] / coord[3]
                coord[3] = 1.
            coord = np.round(coord, 5)[:3]
            # 3D to 2D
            L_2d = self.world2img(coord, L_camvideo_shiftera_mtx, L_vec[:, :3], L_vec[:, -1:], L_camera_dist).reshape(-1, 2)
            R_2d = self.world2img(coord, R_camera_mtx, R_vec[:, :3], R_vec[:, -1:], R_camera_dist).reshape(-1, 2)
            # loss
            L_loss = joint_analysis().eucildea_distance(L_2d, np.array(l_c).reshape(-1, 2))
            R_loss = joint_analysis().eucildea_distance(R_2d, np.array(r_c).reshape(-1, 2))
            return L_loss + R_loss

    def multi_projection_loss(self, coord, img_coord_set, camera_int_path, camera_dist_path, rvec_path, tvec_path):
        # build the camera matrix
        camera_mtx, camera_dist, vec = [], [], {}
        # intrinsic orientation
        for i in range(len(camera_int_path)):
            camera_mtx.append(np.load(camera_int_path[i]))
            camera_dist.append(np.load(camera_dist_path[i]))
        # relative orientation
        for i in range(len(rvec_path) + 1):  # len(self.camera_rvec)
            for j in range(len(rvec_path) + 1):
                if i == j:
                    vec[str(i) + str(j)] = np.hstack((np.eye(3), np.zeros((3, 1))))
                elif i + 1 == j:
                    R = np.load(rvec_path[i])
                    T = np.load(tvec_path[i])
                    vec[str(i) + str(j)] = camera_orientation().Exterior_Orientation(R, T)
                    # print('camera = ', i, j)
        for i in range(len(rvec_path) + 1):  # len(self.camera_rvec)
            for j in range(len(rvec_path) + 1):
                if i + 2 == j:
                    R, T, _, _, _, _, _, _, _, _ = cv2.composeRT(
                        cv2.Rodrigues(vec[str(i) + str(i + 1)][:, :3], None)[0], vec[str(i) + str(i + 1)][:, -1:],
                        cv2.Rodrigues(vec[str(i + 1) + str(i + 2)][:, :3], None)[0],
                        vec[str(i + 1) + str(i + 2)][:, -1:])
                    vec[str(i) + str(j)] = camera_orientation().Exterior_Orientation(R, T)
                    # print('camera = ', i, j)
        # repojeciton loss
        loss = 0
        coord = np.array([coord[0], -coord[2], coord[1]])
        for i in range(len(camera_int_path)):
            img_2d = self.world2img(coord, camera_mtx[i], vec[str(0) + str(i)][:, :3], vec[str(0) + str(i)][:, -1:], camera_dist[i]).reshape(-1, 2)
            print(img_2d)
            temp_loss = joint_analysis().eucildea_distance(img_2d, np.array(img_coord_set[i]).reshape(-1, 2))
            loss += temp_loss
        return loss

    def reproject(self, pose_path, camera_int, camera_dist, camera_rvec, camera_tvec):
        frame, stn_data = tool().Loadcsv_3d(pose_path)
        sdtn_data = stn_data.reshape(-1, 3)[:, [0, 2, 1]].reshape(stn_data.shape[0], -1, 3)
        sdtn_data[:, :, 1] = -1*sdtn_data[:, :, 1]
        mtx, rvec, tvec, dist = [], [], [], []
        # M/dist
        for i in range(len(camera_int)):
            mtx.append(camera_orientation().Interior_Orientation(camera_int[i]))
            dist.append(camera_orientation().Interior_Orientation(camera_dist[i]))
        # R/T
        rvec.append(np.eye(3))
        tvec.append(np.zeros((3, 1)))
        for i in range(len(camera_int)-1):
            originR = cv2.Rodrigues(rvec[i], None)[0]
            originT = tvec[i]
            tempR = cv2.Rodrigues(camera_orientation().Interior_Orientation(camera_rvec[i]), None)[0]
            tempT = camera_orientation().Interior_Orientation(camera_tvec[i])
            R, T, _, _, _, _, _, _, _, _ = cv2.composeRT(originR, originT, tempR, tempT)
            R = cv2.Rodrigues(R, None)[0]
            rvec.append(R)
            tvec.append(T)
        mtx, dist, rvec, tvec = np.array(mtx), np.array(dist), np.array(rvec), np.array(tvec)
        # 2D pose
        pose2D = []
        for i in range(len(camera_int)):
            data_2d = camera_orientation().world2img(sdtn_data, mtx[i], rvec[i], tvec[i], dist[i]).reshape(sdtn_data.shape[0], -1, 2)
            pose2D.append(data_2d)
            # plt.bar(frame, tmp[:, 3, 1])
        # projection
        pose2D = np.round(pose2D, 0).astype(np.int)
        return pose2D

class tool(object):

    def __init__(self):  # Run it once
        self.colormap = ['gray', 'red', 'blue', 'green']
        pass

    def sayhi(self):
        print('hie')

    def dict2csv(self, dict, path):
        import pandas as pd
        title, data = [], []
        for name in dict:
            title.append(name)
            data.append(dict[name])
        title = np.array(title).reshape(-1, 1)
        data = np.array(data).astype(np.str).reshape(title.shape[0], -1)
        dataframe = np.hstack((title, data)).astype(np.str).T
        df = pd.DataFrame(dataframe)
        df.to_csv(path, encoding="gbk", index=False, header=False)

    def conda_file(self, env_name, repackage=None):
        print('Initialize env...')
        if os.path.isdir('./' + env_name):
            shutil.rmtree('./' + env_name)
        if repackage:
            if os.path.isfile(env_name + '.tar.gz'):
                os.remove(env_name + '.tar.gz')
            print('packaging env...')
            conda_pack.pack(name=env_name, output=env_name+".tar.gz")
        print('create env...')
        os.mkdir('./' + env_name)
        with tarfile.open(env_name + ".tar.gz") as tf:
            tf.extractall('./' + env_name)
        print('Complete ...')

    def gpu_info(self):
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        use_rate = []
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            use_rate.append(meminfo.used/meminfo.total)
        return use_rate

    def rename_in_folder(self, folder, src, dst):
        file_set = os.listdir(folder)
        for file in file_set:
            index = file.find(src)
            if index >= 0:
                os.rename(folder+'/'+file, folder+'/'+file.replace(src, dst))

    def str2logic(self, data):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i, j].lower() == 'false':
                    data[i, j] = False
                elif data[i, j].lower() == 'true':
                    data[i, j] = True
                print(data[i, j])
        return data

    def get_default_lim(self, data, padding=1e10):
        d_set, middle_set, lim, detail = [], [], [], []
        data = data.reshape(-1, data.shape[2])
        padding_index = np.argwhere(np.abs(data) >= padding / 2)
        padding_index = np.unique(padding_index[:, 0])
        data = np.delete(data, padding_index, axis=0)
        for i in range(data.shape[1]):
            detail.append(np.array([np.min(data[:, i]), np.max(data[:, i])]))
            d = np.max(data[:, i])-np.min(data[:, i])
            midle = np.min(data[:, i])+d/2
            d_set.append(d)
            middle_set.append(midle)
        d_max = max(d_set)
        detail = np.array(detail)
        for middle in middle_set:
            lim.append([middle-d_max, middle+d_max])
        lim, detail = np.array(lim), np.array(detail)
        return lim, detail

    def find_file_from_str(self, path, src):
        dataset = os.listdir(path)
        for data in dataset:
            if os.path.splitext(data)[0] == src:
                return path+data
        return './'

    def build_project_folder(self, file):
        project_name = os.path.splitext(os.path.basename(file))[0]
        project_folder = os.path.dirname(file) + '/' + project_name + '/'
        if not os.path.isdir(project_folder):
            os.mkdir(project_folder)
        return project_folder

    def find_latest_file(self, folder, secondary):
        dataset,  datatime = [], []
        for root, dirs, files in os.walk(folder):
            if files:
                for i in range(len(files)):
                    if os.path.splitext(files[i])[1] in secondary:
                        file_name = root + '/' + files[i]
                        dataset.append(file_name)
                        datatime.append(os.stat(file_name)[8])
        datatime = np.array(datatime)
        index = np.argwhere(datatime == np.max(datatime)).reshape(-1)[0]
        return dataset[index]

    def range_latest_file(self, folder, secondary):
        dataset = []
        for root, dirs, files in os.walk(folder):
            if files:
                for i in range(len(files)):
                    if os.path.splitext(files[i])[1] in secondary:
                        file_name = root + files[i]
                        dataset.append([file_name, os.stat(file_name)[8]])
        dataset = np.array(sorted(dataset, key=lambda y: y[0]))[:, 0]
        return dataset

    def make_movie(self, data, path, fps=10):
        for i in range(data.shape[0]):
            print('plot frame:', i+1)
            img = cv2.imread(data[i])
            img = cv2.rectangle(img, (250, 10), (550, 60), (0, 0, 0), -1)
            cv2.putText(img, 'NTU-YR-LAB', (270, 45), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, 'frame= '+str(i+1), (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            if i == 0:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(path, fourcc, fps, (img.shape[1], img.shape[0]))
            out.write(img)
        out.release()

    def s2hms(self, s):
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return h, m, s

    def ui2py(self, path):
        uifile = path
        pyfile = os.path.splitext(uifile)[0] + '.py'
        cmd = 'pyuic5 -o {pyfile} {uifile}'.format(pyfile=pyfile, uifile=uifile)
        os.system(cmd)

    def qrc2py(self, path):
        qrcfile = path
        pyfile = os.path.splitext(qrcfile)[0] + '_rc.py'
        cmd = 'pyrcc5 {qrcfile} -o {pyfile}'.format(qrcfile=qrcfile, pyfile=pyfile)
        os.system(cmd)

    def rotate(self, frame, angle_x, angle_y, angle_z):
        w = frame.shape[1]
        h = frame.shape[0]
        org = np.array( [[0, 0], [w, 0], [0, h], [w, h]], np.float32 )
        dst = np.zeros( (4, 2), np.float32 )
        angle_x = angle_x*math.pi/180
        angle_y = angle_y*math.pi/180
        angle_z = angle_z*math.pi/180
        kx = np.zeros((4, 4))
        ky = np.zeros( (4, 4) )
        kz = np.zeros( (4, 4) )
        # kx
        kx[0][0] = 1
        kx[1][1] = np.cos(angle_x)
        kx[1][2] = -np.sin( angle_x )
        kx[2][1] = np.sin( angle_x )
        kx[2][2] = np.cos( angle_x )
        kx[3][3] = 1
        # ky
        ky[1][1] = 1
        ky[0][0] = np.cos( angle_y )
        ky[0][2] = np.sin( angle_y )
        ky[2][0] = -np.sin( angle_y )
        ky[2][2] = np.cos( angle_y )
        ky[3][3] = 1
        # kz
        kz[2][2] = 1
        kz[0][0] = np.cos( angle_z )
        kz[1][0] = np.sin( angle_z )
        kz[0][1] = -np.sin( angle_z )
        kz[1][1] = np.cos( angle_z )
        kz[3][3] = 1
        # rotate matrix (4*4)
        R = kx.dot(ky).dot(kz)
        # rotate the point
        center = np.array( [frame.shape[1] / 2, frame.shape[0] / 2, 0, 0], np.float32 )
        dst_point = []
        p1 = np.array( [0, 0, 0, 0], np.float32 ) - center
        dst_point.append(R.dot(p1))
        p2 = np.array( [frame.shape[1], 0, 0, 0], np.float32 ) - center
        dst_point.append( R.dot( p2 ) )
        p3 = np.array( [0, frame.shape[0], 0, 0], np.float32 ) - center
        dst_point.append( R.dot( p3 ) )
        p4 = np.array( [frame.shape[1], frame.shape[0], 0, 0], np.float32 ) - center
        dst_point.append( R.dot( p4 ) )
        dst_point = np.array(dst_point)
        for i in range( 4 ):
            dst[i, 0] = (dst_point[i][0]) + center[0]
            dst[i, 1] = (dst_point[i][1]) + center[1]
        # find the rotate matrix (3*3)
        warp = cv2.getPerspectiveTransform( org, dst )
        # calculate the planar of rotation
        img_warp = cv2.warpPerspective( frame, warp, (w, h) )
        return img_warp

    def video_shift(self, path, frame_i, frame_f, new_FPS, show='cmd'):
        video = cv2.VideoCapture( path )
        if frame_f == '':
            frame_f = video.get( 7 )
        if new_FPS  == '':
            new_FPS = video.get(5)
        new_videoname = os.path.splitext( path )[0] + "_shift" + os.path.splitext( path )[1]
        w = int( video.get( cv2.CAP_PROP_FRAME_WIDTH ) )
        h = int( video.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        print('video: = ', path)
        print('The frame is from ', str(frame_i), ' to ', str(frame_f), ' .', end='\r')
        # print( 'fps = ', fps )
        # print( 'w = ', w )
        # print( 'h = ', h )
        fourcc = cv2.VideoWriter_fourcc( *'mp4v' )
        out = cv2.VideoWriter( new_videoname, fourcc, new_FPS, (w, h) )
        count = 1
        tmp_progress_rate = 0
        while (video.isOpened()):
            ret, frame = video.read()
            if ret == True:
                if count >= frame_i:
                    ti = time.time()
                    cv2.putText(frame, 'frame = ' + str(count), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    out.write( frame )
                    tf = time.time()
                    # print(count, frame_i, frame_f)
                    progress_rate = int((count - frame_i) / (frame_f - frame_i) * 100)
                    remaining_time = round((tf - ti) * (frame_f - count), 2)
                    if show == 'cmd':
                        print('Synchronize...', progress_rate, ' , remaining_time = ', remaining_time, flush=True)
                    else:
                        if progress_rate-tmp_progress_rate >= 1:
                            random_name = str(random.randint(111111111, 999999999))
                            tmp_progress_rate = progress_rate
                            # print(str(progress_rate), ' ', str(remaining_time))
                            f_print = open(os.path.dirname(path) + '/print_log/'+str(random_name)+'.txt', 'w')
                            f_print.write('progress_rate, remaining_time')
                            f_print.write("\n")
                            f_print.write(str(progress_rate) + ' ' + str(remaining_time) + ' ' + path)
                            f_print.write("\n")
                            f_print.close()

                    # cv2.imshow( 'frame', frame )
                if count >= frame_f:
                    break
            else:
                break
            if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
                break
            count += 1
        video.release()
        out.release()
        cv2.destroyAllWindows()

    def video_combin(self, folder, new_FPS=None):
        define_video = False
        video_file = ['.mp4', '.MP4', '.avi']
        files = os.listdir(folder)
        for candidate in files:
            if os.path.splitext(candidate)[1] in video_file:
                print(define_video)
                print('video = ', folder+candidate)
                video = cv2.VideoCapture(folder + candidate)
                if define_video is False:
                    print('comming')
                    new_FPS = video.get(5) if new_FPS is None else new_FPS
                    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    new_videoname = folder + "Combin" + os.path.splitext(folder+candidate)[1]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(new_videoname, fourcc, new_FPS, (w, h))
                    define_video = True
                while (video.isOpened()):
                    ret, frame = video.read()
                    if ret == True:
                        out.write(frame)
                    else:
                        break
        if define_video:
            video.release()
            out.release()
            cv2.destroyAllWindows()
            print('Done!')

    def video_rotate(self, path, angle, UI=False):
        video = cv2.VideoCapture(path)
        new_FPS = video.get(5)
        frame_f = video.get(7)
        new_videoname = os.path.splitext(path)[0] + "_rotate" + os.path.splitext(path)[1]
        fps = video.get(5)
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(path)
        print('fps = ', fps)
        print('w = ', w)
        print('h = ', h)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if angle == 90:
            rotate_n, w, h = 1, h, w
        elif angle == 180:
            rotate_n = 2
        elif angle == 270:
            rotate_n, w, h = 3, h, w
        out = cv2.VideoWriter(new_videoname, fourcc, new_FPS, (w, h))
        count = 1
        while (video.isOpened()):
            ret, frame = video.read()
            if ret == True:
                print(count, frame_f)
                for i in range(rotate_n):
                    frame = np.rot90(frame)
                out.write(frame)
                if not UI:
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)
                if count > frame_f:
                    break
            else:
                break
            count += 1
        video.release()
        out.release()
        cv2.destroyAllWindows()
        print('Done!')

    def csv_2d_rotate(self, path, angle, w, h):
        frame, data = self.Loadcsv_2d(path)
        padding_index = np.argwhere(data == -1)

        if angle == 90:
            rotate_data = data[:, :, [1, 0]].copy()
            rotate_data[:, :, 1] = h - rotate_data[:, :, 1]
        elif angle == 180:
            rotate_data = data.copy()
            rotate_data[:, :, 0] = w - rotate_data[:, :, 0]
            rotate_data[:, :, 1] = h - rotate_data[:, :, 1]
        elif angle == 270:
            rotate_data = data[:, :, [1, 0]].copy()
            rotate_data[:, :, 0] = h-rotate_data[:, :, 0]
        for i in padding_index:
            rotate_data[i[0]][i[1]][i[2]] = -1
        self.data2csv2d(path, frame, rotate_data, mininame='rotate')

    def video_disassemble(self, path):
        video = cv2.VideoCapture( path )
        output_path = os.path.splitext(path)[0]+'_disassemble/'
        try:
            shutil.rmtree(output_path)
        except:
            pass
        os.mkdir(output_path)
        count = 1
        while (video.isOpened()):
            ret, frame = video.read()
            if ret == True:
                cv2.imshow('frame', frame)
                cv2.imwrite(output_path+str(count) + '.png', frame)
            else:
                break

            if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
                break
            count += 1

        video.release()
        cv2.destroyAllWindows()

    def video_record(self, path, cam_n):
        cam = cv2.VideoCapture( cam_n )
        videoname = path+'/'+str(cam_n)+'1.mp4'
        FPS = cam.get( 5 )
        w = int( cam.get( cv2.CAP_PROP_FRAME_WIDTH ) )
        h = int( cam.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        print( 'fps = ', FPS )
        print( 'w = ', w )
        print( 'h = ', h )
        fourcc = cv2.VideoWriter_fourcc( *'mp4v' )
        out = cv2.VideoWriter( videoname, fourcc, FPS, (w, h) )
        count = 1
        while (cam.isOpened()):
            ret, frame = cam.read()
            if ret == True:
                out.write( frame )
                cv2.imshow( 'frame', frame )
            else:
                break
            if cv2.waitKey( 1 ) == 27:
                break
            count += 1
        cam.release()
        out.release()
        cv2.destroyAllWindows()

    def video_dobule_record(self, path):
        cam_0 = cv2.VideoCapture( 0 )
        cam_1 = cv2.VideoCapture( 1 )
        videoname_0 = path+'/'+str(0)+'.mp4'
        videoname_1 = path + '/' + str( 1 ) + '.mp4'
        FPS_0 = cam_0.get( 5 )
        FPS_1 = cam_1.get( 5 )
        w_0 = int( cam_0.get( cv2.CAP_PROP_FRAME_WIDTH ) )
        h_0 = int( cam_0.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        w_1 = int( cam_1.get( cv2.CAP_PROP_FRAME_WIDTH ) )
        h_1 = int( cam_1.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
        print( 'fps = ', FPS_0, FPS_1 )
        print( 'w = ', w_0, w_1 )
        print( 'h = ', h_0, h_1 )
        fourcc = cv2.VideoWriter_fourcc( *'mp4v' )
        out_0 = cv2.VideoWriter( videoname_0, fourcc, FPS_0, (w_0, h_0) )
        out_1 = cv2.VideoWriter( videoname_1, fourcc, FPS_1, (w_1, h_1) )
        count = 1
        while (cam_0.isOpened() and cam_1.isOpened()):
            ret_0, frame_0 = cam_0.read()
            ret_1, frame_1 = cam_1.read()
            if ret_0 == True and ret_1 == True:
                out_0.write( frame_0 )
                cv2.imshow( 'frame_0', frame_0 )
                out_1.write( frame_1 )
                cv2.imshow( 'frame_1', frame_1 )
            else:
                break
            if cv2.waitKey( 1 ) == 27:
                break
            count += 1

        cam_0.release()
        cam_1.release()
        out_0.release()
        out_1.release()
        cv2.destroyAllWindows()

    def img2video(self, path):
        img = cv2.imread(path)
        print('image size = ', img.shape)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.splitext(path)[0] + '.mp4'
        out = cv2.VideoWriter(output_path, fourcc, 1, (img.shape[1], img.shape[0]))
        out.write(img)
        out.release()
        cv2.destroyAllWindows()

    def recitfy(self, path):
        # initialization
        mapx = np.load( path + '/mapx.npy' )
        mapy = np.load( path + '/mapy.npy' )
        with open(path+'relation.csv', newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if row[0] == 'Rotate':
                    roll, pitch, yaw = float(row[1]), float(row[2]), float(row[3])
        print(roll, pitch, yaw)
        # camera
        video_file = ['.mp4', '.MP4']
        calibrate_files_0 = listdir( path+'/modify/' )
        print(calibrate_files_0)
        for f in calibrate_files_0:
            fullpath = join( path+'/modify/', f )
            secondaryname = os.path.splitext( fullpath )[1]
            if secondaryname in video_file:
                print( "Load the video:：", fullpath )
                cam_0 = cv2.VideoCapture( fullpath )
                videoname_0 = os.path.splitext(fullpath)[0] + '_recitfy_.mp4'
                FPS_0 = cam_0.get( 5 )
                w_0 = int( cam_0.get( cv2.CAP_PROP_FRAME_WIDTH ) )
                h_0 = int( cam_0.get( cv2.CAP_PROP_FRAME_HEIGHT ) )
                fourcc = cv2.VideoWriter_fourcc( *'mp4v' )
                out_0 = cv2.VideoWriter( videoname_0, fourcc, FPS_0, (w_0, h_0) )
                count = 1
                while (cam_0.isOpened()):
                    ret_0, frame_0 = cam_0.read()
                    if ret_0 == True:

                        recitfy_img = cv2.remap( frame_0, mapx, mapy, cv2.INTER_CUBIC )
                        # recitfy_img = tool().rotate( recitfy_img, 0, 0, -yaw )
                        cv2.imshow( 'frame', frame_0 )
                        cv2.imshow( 'recitfy', recitfy_img )
                        out_0.write( recitfy_img )
                        print('frame = ', count)
                    else:
                        break
                    if cv2.waitKey( 1 ) == 27:
                        break
                    count += 1
                cam_0.release()
                out_0.release()
        cv2.destroyAllWindows()

    def kmeans(self, k, dataSet, path):

        def euclidean(pointer, center):
            loss = np.sqrt(np.sum(np.power(pointer - center, 2)*1))
            return loss

        # initialization
        indexs = np.random.random_integers(0, len(dataSet) - 1, k)
        centers = dataSet[indexs]
        gradient = 1e10000
        pre_gradient = 0
        colormap = ['r', 'k', 'y', 'b', 'c', 'gray', 'firebrick', 'coral', 'tan', 'steelblue']*100
        while (abs(gradient - pre_gradient) > 0):
            # update the gradient
            pre_gradient = gradient

            # Classified according to centroid
            gradient = 0
            classfy = [[] for i in range(len(centers))]
            y_pred = []
            for pointer in dataSet:
                loss = []
                for center in centers:
                    loss.append(euclidean(pointer, center))
                loss = np.array(loss)
                min_index = np.argwhere(loss == np.min(loss))[0][0]
                classfy[min_index].append(pointer)
                gradient += loss[min_index]
                y_pred.append(min_index)
            y_pred = np.array(y_pred)

            # update the mass center
            # class
            for i in range(k):
                # vector
                temp = np.array(classfy[i])
                if temp.shape[0] > 0:
                    for j in range(dataSet.shape[1]):
                        mean = np.mean(temp[:, j])
                        centers[i][j] = mean


            # # plot  len(olormap) =< 10
            # for i in range(k):
            #     # data
            #     index = np.argwhere(y_pred == i).flatten()
            #     X = dataSet[index][:, 0]
            #     Y = dataSet[index][:, 1]
            #     plt.scatter(X, Y, c=colormap[i])
            #     # mass center
            #     plt.scatter(centers[i][0], centers[i][1], c=colormap[i], marker='*', s=300)
            #     plt.scatter(X, Y, c=colormap[i])
            # plt.title('K-means')
            # plt.savefig(path+'K-means.png')
            # plt.close()
            # kmeans_img = cv2.imread(path+'K-means.png')
            # cv2.imshow('kmeans', kmeans_img)
            # cv2.waitKey(1)
        map = []
        for i in range(centers.shape[0]):
            tmp = np.sqrt(np.sum(np.power(dataSet-centers[i], 2), axis=1))
            value = np.argwhere(tmp == np.min(tmp))[0][0]
            map.append(value)

            # print('gradient = ', gradient, pre_gradient)
            # print('predict = ', y_pred)
            # print('')

        return map

    def Plot3D(self, frame_set, data, xlim, ylim, zlim, view, path, fps, type='alphapose', fix_frame='', padding=1e10, COM=None, vicon_axis=None, UI=False):
        if COM is not None:
            data = np.concatenate((data, COM.reshape(-1, 1, 3)), axis=1)
        figsize = [12, 12]
        if os.path.isdir(os.path.splitext(path)[0]):
            shutil.rmtree(os.path.splitext(path)[0])
        os.mkdir(os.path.splitext(path)[0])
        plt.rcParams['figure.figsize'] = figsize
        if type == 'alphapose':
            if COM is not None:
                line_i = [0, 0, 1, 2, 10, 8, 6, 17, 5, 7, 12, 14, 11, 13, 17, 11, 17]
                line_f = [1, 2, 3, 4, 8, 6, 17, 5, 7, 9, 14, 16, 13, 15, 0, 12, 18]
            else:
                line_i = [10, 8, 6, 17, 5, 7, 17, 12, 14, 17, 11, 13, 17]
                line_f = [8, 6, 17, 5, 7, 9, 12, 14, 16, 11, 13, 15, 0]
        elif type == 'cube':
            line_i = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6])
            line_f = np.array([1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7])
        if type == 'cube':  # # cube rotate
            rotate_n = 3
        else:
            rotate_n = 1
        # plot the point
        point_range_x = []
        point_range_y = []
        point_range_z = []
        # video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (900, 900))
        init_frame, time_set = 0, data.shape[0]
        if fix_frame:
            init_frame, time_set = fix_frame, fix_frame
            rotate_n = 40
        for i in range(init_frame, time_set):
            for j in range(rotate_n):
                if UI is False:
                    print('plot frame = ', frame_set[i])
                fig = plt.figure(0)
                ax = fig.gca(projection='3d')
                dict_3D = data[i]
                frame = frame_set[i]
                for point_3D in range(dict_3D.shape[0]):
                    if dict_3D[point_3D][0] < padding/2:
                        ax.text(dict_3D[point_3D][0], dict_3D[point_3D][1], dict_3D[point_3D][2], str(point_3D), color='r', fontdict={'weight': 'bold', 'size': 9})
                        ax.scatter(dict_3D[point_3D][0], dict_3D[point_3D][1], dict_3D[point_3D][2], c='y')  # y
                        point_range_x.append(dict_3D[point_3D][0])
                        point_range_x.append(dict_3D[point_3D][0])
                        point_range_y.append(dict_3D[point_3D][1])
                        point_range_y.append(dict_3D[point_3D][1])
                        point_range_z.append(dict_3D[point_3D][2])
                        point_range_z.append(dict_3D[point_3D][2])
                        if vicon_axis is not None and point_3D in vicon_axis[0, :, 0, 0]:
                            vicon_index = np.argwhere(vicon_axis[0, :, 0, 0] == point_3D).reshape(-1)[0]
                            for k in range(vicon_axis.shape[2]):
                                linex = (dict_3D[point_3D][0], vicon_axis[i][vicon_index][k][1])
                                liney = (dict_3D[point_3D][1], vicon_axis[i][vicon_index][k][2])
                                linez = (dict_3D[point_3D][2], vicon_axis[i][vicon_index][k][3])
                                ax.plot(linex, liney, linez, linewidth=1.0, c='k')  # m
                # plot/calculate L line
                obj_length = []
                for line_3D in range(len(line_i)):
                    if dict_3D[line_i[line_3D]][0] < padding/2 and dict_3D[line_f[line_3D]][0] < padding/2:
                        linex = (dict_3D[line_i[line_3D]][0], dict_3D[line_f[line_3D]][0])
                        liney = (dict_3D[line_i[line_3D]][1], dict_3D[line_f[line_3D]][1])
                        linez = (dict_3D[line_i[line_3D]][2], dict_3D[line_f[line_3D]][2])
                        temp = round(math.sqrt((linex[1] - linex[0]) ** 2 + (liney[1] - liney[0]) ** 2 + (linez[1] - linez[0]) ** 2), 5)
                        obj_length.append(temp)
                        # print(dict_3D[line_i[line_3D]], dict_3D[line_f[line_3D]], temp)
                        ax.plot(linex, liney, linez, linewidth=2.0, c='m')  # m

                # axis range
                delta = max(point_range_x)-min(point_range_x)
                delta = max(point_range_y)-min(point_range_y) if delta < max(point_range_y)-min(point_range_y) else delta
                delta = max(point_range_z) - min(point_range_z) if delta < max(point_range_z) - min(point_range_z) else delta
                if xlim[0] != '':
                    ax.set_xlim(xlim[0:2])
                    if xlim[2]:
                        ax.xaxis.set_major_locator(MultipleLocator(xlim[2]))
                else:
                    ax.set_xlim(np.mean(point_range_x)-delta, np.mean(point_range_x)+delta)
                    # pass
                if ylim[0] != '':
                    ax.set_ylim(ylim[0:2])
                    if ylim[2]:
                        ax.yaxis.set_major_locator(MultipleLocator(ylim[2]))
                else:
                    ax.set_ylim(np.mean(point_range_y)-delta, np.mean(point_range_y)+delta)
                    # pass
                if zlim[0] != '':
                    ax.set_zlim(zlim[0:2])
                    if zlim[2]:
                        ax.zaxis.set_major_locator(MultipleLocator(zlim[2]))
                else:
                    # pass
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

                ax.view_init(view[0], view[1])  # default(30, -60), the rotation of vertical and horizontal angle
                # plt.axis('equal')
                if view == [0, -90]:
                    plt.yticks([])
                plt.savefig(os.path.splitext(path)[0] + '/' + str(frame) + '_3D.png')
                plt.close(0)
                post_img = cv2.imread(os.path.splitext(path)[0] + '/' + str(frame) + '_3D.png')[150:-150, 150:-150, :]
                # print(post_img.shape)
                post_img = cv2.rectangle(post_img, (10, 50), (300, 100), (0, 0, 0), -1)
                cv2.putText(post_img, 'NTU-YR-LAB', (30, 85), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
                out.write(post_img)
                if not UI:
                    cv2.imshow('post', post_img)
                    cv2.waitKey(1)
                if type == 'cube':
                    view[1] = view[1]-360/rotate_n
                if fix_frame:
                    view[1] = view[1] - 360 / rotate_n
        out.release()

    def camera_pos2angle(self, camera_center):
        camera_center = np.array(camera_center)
        main_vector = np.array([-1, 0])
        main_length = np.sqrt(np.sum(np.square(main_vector)))
        for i in range(camera_center.shape[0]):
            new_vector = camera_center[i][0:2]
            new_length = np.sqrt(np.sum(np.square(new_vector)))
            angle = np.arccos(np.dot(main_vector, new_vector)/(main_length*new_length))
            angle = angle*180/np.pi
            if new_vector[1] < main_vector[1]:
                angle = 360-angle
            print('camera ', str(i+1), ' (angle) = ', angle)

    def Plot_baseball_field(self, camera_center, camera_length, camera_name, line_camera, xlim, ylim, zlim, view, img_path='tmp.png', UI=False):

        def plot_object(object_name, object_center, object_length, line_object=None, auxiliary='auxiliary', linestyle='-', line_c='b'):
            for pair in line_object:
                index_1 = object_name.index(pair[0])
                index_2 = object_name.index(pair[1])
                linex = (object_center[index_1][0], object_center[index_2][0])
                liney = (object_center[index_1][1], object_center[index_2][1])
                linez = (object_center[index_1][2], object_center[index_2][2])
                ax.plot(linex, liney, linez, linewidth=1.0, c=line_c, linestyle=linestyle)  # m
            for i in range(len(object_center)):
                if auxiliary not in object_name[i]:
                    ax.scatter(object_center[i][0], object_center[i][1], object_center[i][2], c='y')  # y
                    ax.text(object_center[i][0]-4, object_center[i][1], object_center[i][2]+0.04, object_name[i], color='r', fontdict={'weight': 'bold', 'size': 5})
                    center_length = np.sqrt(np.power(object_length[i], 2) * 2) / 2
                    coords = []
                    coords.append([object_center[i][0], object_center[i][1] - center_length, object_center[i][2]])
                    coords.append([object_center[i][0] - center_length, object_center[i][1], object_center[i][2]])
                    coords.append([object_center[i][0], object_center[i][1] + center_length, object_center[i][2]])
                    coords.append([object_center[i][0] + center_length, object_center[i][1], object_center[i][2]])
                    for j in range(len(coords)):
                        if j + 1 == len(coords):
                            linex = (coords[j][0], coords[0][0])
                            liney = (coords[j][1], coords[0][1])
                            linez = (coords[j][2], coords[0][2])
                        else:
                            linex = (coords[j][0], coords[j + 1][0])
                            liney = (coords[j][1], coords[j + 1][1])
                            linez = (coords[j][2], coords[j + 1][2])
                        ax.plot(linex, liney, linez, linewidth=5.0, c='m')  # m

        total_point = []
        # plot the point
        fig = plt.figure(0)
        ax = fig.gca(projection='3d')
        # basic object
        object_center, object_length, object_name, line_object = [], [], [], []
        line_object = [['home plate', 'first base(auxiliary)'], ['first base', 'second base'], ['second base', 'third base'], ['third base(auxiliary)', 'home plate']]
        object_name.append('pitcher mound')
        object_center.append([0, 0, 0])
        object_length.append(0.38)
        object_name.append('home plate')
        object_center.append([-18.44, 0, 0])
        object_length.append(0.38)
        object_name.append('first base')
        object_center.append([-18.44+27.43*np.sin(45*np.pi/180), -27.43*np.cos(45*np.pi/180), 0])  # 1B
        object_length.append(0.38)
        object_name.append('first base(auxiliary)')
        object_center.append([-18.44 + 77.1 * np.sin(45 * np.pi / 180), -77.1 * np.cos(45 * np.pi / 180), 0])  # 1B
        object_length.append(0.38)
        object_name.append('second base')
        object_center.append([-18.44+38.76, 0, 0])  # 2B
        object_length.append(0.38)
        object_name.append('third base')
        object_center.append([-18.44+27.43*np.sin(45*np.pi/180), 0+27.43*np.cos(45*np.pi/180), 0])  # 3B
        object_length.append(0.38)
        object_name.append('third base(auxiliary)')
        object_center.append([-18.44 + 77.1 * np.sin(45 * np.pi / 180),  77.1 * np.cos(45 * np.pi / 180), 0])  # 3B
        object_length.append(0.38)
        plot_object(object_name=object_name, object_center=object_center, object_length=object_length, line_object=line_object, auxiliary='auxiliary')
        plot_object(object_name=camera_name, object_center=camera_center, object_length=camera_length, line_object=line_camera, auxiliary='auxiliary', linestyle=':', line_c='k')
        # camera
        if view:
            ax.view_init(view[0], view[1])
        plt.title("Field", fontsize=16)
        ax.set_xlabel('X [m]')
        ax.set_ylabel("Y [m]")
        ax.set_zlabel('Z [m]')
        if xlim:
            ax.set_xlim(xlim[0:2])
        if ylim:
            ax.set_ylim(ylim[0:2])
        if zlim:
            ax.set_zlim(zlim[0:2])
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.invert_zaxis()
        plt.savefig(img_path)
        if UI is False:
            plt.show()
        plt.close()

    def plot_double_3D(self, frame_set, data1, data2, xlim, ylim, zlim, view, path, fps, padding=1e10):
        figsize = [12, 12]
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.mkdir(path)
        plt.rcParams['figure.figsize'] = (8, 8)
        line_i = [4, 2, 0, 1, 10, 8, 6, 17, 5, 7, 17, 12, 14, 17, 11, 13, 17]
        line_f = [2, 0, 1, 3, 8, 6, 17, 5, 7, 9, 12, 14, 16, 11, 13, 15, 0]
        # plot the point
        point_range_x = []
        point_range_y = []
        point_range_z = []
        # video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path + '3D_pose.mp4', fourcc, fps, (900, 900))
        for i in range(data1.shape[0]):
            print('plot frame = ', i)

            fig = plt.figure(0)
            ax = fig.gca(projection='3d')
            dict_3D1 = data1[i]
            dict_3D2 = data2[i]
            frame = frame_set[i]
            for point_3D in range(dict_3D1.shape[0]):
                if dict_3D1[point_3D][0] < padding/2:
                    ax.text(dict_3D1[point_3D][0], dict_3D1[point_3D][1], dict_3D1[point_3D][2], str(point_3D), color='r', fontdict={'weight': 'bold', 'size': 9})
                    ax.scatter(dict_3D1[point_3D][0], dict_3D1[point_3D][1], dict_3D1[point_3D][2], c='y')
                    point_range_x.append(dict_3D1[point_3D][0])
                    point_range_x.append(dict_3D1[point_3D][0])
                    point_range_y.append(dict_3D1[point_3D][1])
                    point_range_y.append(dict_3D1[point_3D][1])
                    point_range_z.append(dict_3D1[point_3D][2])
                    point_range_z.append(dict_3D1[point_3D][2])
                if dict_3D2[point_3D][0] < padding/2:
                    ax.scatter(dict_3D2[point_3D][0], dict_3D2[point_3D][1], dict_3D2[point_3D][2], c='gray')

            # plot/calculate L line
            obj_length = []
            for line_3D in range(len(line_i)):
                if dict_3D1[line_i[line_3D]][0] < padding/2 and dict_3D1[line_f[line_3D]][0] < padding/2:
                    linex = (dict_3D1[line_i[line_3D]][0], dict_3D1[line_f[line_3D]][0])
                    liney = (dict_3D1[line_i[line_3D]][1], dict_3D1[line_f[line_3D]][1])
                    linez = (dict_3D1[line_i[line_3D]][2], dict_3D1[line_f[line_3D]][2])
                    temp = round(math.sqrt((linex[1] - linex[0]) ** 2 + (liney[1] - liney[0]) ** 2 + (linez[1] - linez[0]) ** 2), 5)
                    obj_length.append(temp)
                    # print(dict_3D_l[line_i[line_3D]], dict_3D_l[line_f[line_3D]], temp)
                    ax.plot(linex, liney, linez, linewidth=2.0, c='m')
                if dict_3D2[line_i[line_3D]][0] < padding/2 and dict_3D2[line_f[line_3D]][0] < padding/2:
                    linex = (dict_3D2[line_i[line_3D]][0], dict_3D2[line_f[line_3D]][0])
                    liney = (dict_3D2[line_i[line_3D]][1], dict_3D2[line_f[line_3D]][1])
                    linez = (dict_3D2[line_i[line_3D]][2], dict_3D2[line_f[line_3D]][2])
                    temp = round(math.sqrt((linex[1] - linex[0]) ** 2 + (liney[1] - liney[0]) ** 2 + (linez[1] - linez[0]) ** 2), 5)
                    obj_length.append(temp)
                    # print(dict_3D_l[line_i[line_3D]], dict_3D_l[line_f[line_3D]], temp)
                    ax.plot(linex, liney, linez, linewidth=2.0, c='gray')
            # axis range
            delta = max(point_range_x)-min(point_range_x)
            delta = max(point_range_y)-min(point_range_y) if delta < max(point_range_y)-min(point_range_y) else delta
            delta = max(point_range_z) - min(point_range_z) if delta < max(point_range_z) - min(point_range_z) else delta
            if xlim[0]:
                ax.set_xlim(xlim[0:2])
                if xlim[2]:
                    ax.xaxis.set_major_locator(MultipleLocator(xlim[2]))
            else:
                ax.set_xlim(np.mean(point_range_x)-delta, np.mean(point_range_x)+delta)
                # pass
            if ylim[0]:
                ax.set_ylim(ylim[0:2])
                if ylim[2]:
                    ax.yaxis.set_major_locator(MultipleLocator(ylim[2]))
            else:
                ax.set_ylim(np.mean(point_range_y)-delta, np.mean(point_range_y)+delta)
                # pass
            if zlim[0]:
                ax.set_zlim(zlim[0:2])
                if zlim[2]:
                    ax.zaxis.set_major_locator(MultipleLocator(zlim[2]))
            else:
                # pass
                ax.set_zlim(np.mean(point_range_z)-delta, np.mean(point_range_z)+delta)
                # pass
            # self.view = [0, -90]
            plt.rcParams['figure.figsize'] = tuple(figsize)
            ax.set_xlabel('X [m]')
            inf_title = ['3D reconstruction (frame: ' + str(frame) + ' )']
            inf_ylabel = ['Y [m]']
            inf_zlabel = ['Z [m]']

            ax.set_title(''.join(inf_title))
            ax.set_ylabel(''.join(inf_ylabel))
            ax.set_zlabel(''.join(inf_zlabel))

            ax.view_init(view[0], view[1])  # default(30, -60), the rotation of vertical and horizontal angle
            # plt.axis('equal')
            if view == [0, -90]:
                plt.yticks([])
            plt.savefig(path + '/' + str(frame)+'_3D.png')
            plt.close(0)
            post_img = cv2.imread(path + '/' + str(frame)+'_3D.png')[150:-150, 150:-150, :]
            out.write(post_img)
            cv2.imshow('post', post_img)
            cv2.waitKey(1)
        out.release()

    def Loadcsv_project(self, path, camera_set=[1, 2], dim=2, padding=1e10):
        dataset = {}
        file_set = os.listdir(path)
        for file in file_set:
            if os.path.splitext(file)[1] == '.yr3d':
                first_name = os.path.splitext(file)[0]
                if os.path.isdir(path+first_name):
                    temp_data = []
                    if dim == 3:
                            temp_data.append(self.Loadcsv_3d(path+first_name+'/'+first_name+'_threeDdata_post.csv', padding=padding)[1])
                    elif dim == 2:
                        for camera in camera_set:
                            temp_data.append(self.Loadcsv_2d(path + first_name + '/' + first_name + '_Camera_' + str(camera) + '_reprojection_post.csv')[1])
                    temp_data = np.array(temp_data)
                    dataset[first_name] = np.array(temp_data)
        return dataset

    def Loadcsv_2d_scores(self, path):
        data = pd.read_csv(path).values
        frame = data[:, 0].astype(np.int)
        data = data[:, 1:].reshape(data.shape[0], -1)
        return frame, data

    def Loadcsv_3d(self, dict_path, padding=1e10):
        data = pd.read_csv(dict_path).fillna(padding).values
        frame = data[:, 0].astype(np.int)
        data = data[:, 1:].reshape(data.shape[0], -1, 3)
        return frame, data

    def Loadcsv_3d_velocity(self, dict_path, padding=1e10):
        data = pd.read_csv(dict_path).fillna(padding).values
        data[data == -padding] = padding
        frame = data[:, 0].astype(np.int)
        data = data[:, 1:].reshape(data.shape[0], -1, 4)
        return frame, data

    def Loadcsv_3d_segment_length(self, dict_path, padding=1e10):
        data = pd.read_csv(dict_path).fillna(padding).values
        data[data == -padding] = padding
        frame = data[:, 0].astype(np.int)
        data = data[:, 1:].reshape(data.shape[0], -1)
        return frame, data

    def data2csv3d(self, mypath, frame, data, mininame='', padding=1e10):
        newpath = os.path.splitext(mypath)[0]+mininame+os.path.splitext(mypath)[1]
        data = data.reshape(data.shape[0], -1)
        frame = frame.reshape(-1, 1)
        data = np.hstack((frame, data))
        # CSV
        with open(newpath, 'w', newline='') as csvfile:
            Pose_title = ['frame']
            csv_f = csv.writer(csvfile)
            for i in range(int((data.shape[1]-1)/3)):
                Pose_title.append(str(i) + '_x [m]')
                Pose_title.append(str(i) + '_y [m]')
                Pose_title.append(str(i) + '_z [m]')
            csv_f.writerow(Pose_title)
            for i in range(data.shape[0]):
                space_index = np.argwhere(data[i] >= padding/2).reshape(-1)
                pose = data[i].tolist()
                for j in space_index:
                    pose[j] = ''
                csv_f.writerow(pose)

    def Loadcsv_2d(self, dict_path):
        data = pd.read_csv(dict_path).fillna(-1).values
        frame = data[:, 0].astype(np.int)
        data = data[:, 1:37].reshape(data.shape[0], -1, 2)
        return frame, data

    def Loadcsv_poseinf(self, dict_path):
        dataframe = pd.read_csv(dict_path)
        filename = np.array(dataframe.columns)
        # delete filename extension
        for i in range(filename.shape[0]):
            filename[i] = os.path.splitext(filename[i])[0]
        indexA = np.argwhere(filename == 'CameraA').reshape(-1).astype(np.int)
        indexB = np.argwhere(filename == 'CameraB').reshape(-1).astype(np.int)
        camera_index = np.sort(np.hstack((indexA, indexB)))
        frame = dataframe.values[:, 0].astype(np.int)
        data = dataframe.values[:, camera_index].reshape(frame.shape[0], -1, 2)
        return frame, data

    def data2csv2d(self, mypath, frame, data, mininame):
        newpath = os.path.splitext(mypath)[0]+mininame+os.path.splitext(mypath)[1]
        data = data.reshape(data.shape[0], -1)
        frame = frame.reshape(-1, 1)
        data = np.hstack((frame, data))
        # CSV
        with open(newpath, 'w', newline='') as csvfile:
            Pose_title = ['frame']
            csv_f = csv.writer(csvfile)
            for i in range(int((data.shape[1]-1)/2)):
                Pose_title.append(str(i) + '_x')
                Pose_title.append(str(i) + '_z')
            csv_f.writerow(Pose_title)
            for i in range(data.shape[0]):
                space_index = np.argwhere(data[i] == -1).reshape(-1)
                pose = data[i].tolist()
                for j in space_index:
                    pose[j] = ''
                csv_f.writerow(pose)

    def plot_segment_length(self, path, frame, dataset, leni, lenf, figsize='', UI=False, plt_y_range='', plt_x_range='', data_name=None, padding=1e10):
        if data_name == None:
            data_name = []
            for i in range(len(dataset)):
                data_name.append(i)
        point_name = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle', 'neck']
        # newpath = os.path.splitext(mypath)[0] + '_segment length.csv'
        if figsize:
            plt.rcParams['figure.figsize'] = figsize
        frame = frame.reshape(-1, 1)
        count_dataset = 0
        for segment_length in dataset:
            n = 0
            for len in segment_length.T:
                zero_index = np.argwhere(len > padding/2)
                len = np.delete(len, zero_index, axis=0)
                X = np.delete(frame, zero_index, axis=0)
                plt.plot(X, len, label=point_name[leni[n]]+'-'+point_name[lenf[n]] + '(data:' + str(data_name[count_dataset]) + ')')
                # plt.plot(X, len, label=point_name[leni[n]] + '-' + point_name[lenf[n]] + '(data:' + str(data_name[count_dataset]) + ')', color=self.colormap[n])
                plt.legend(loc='upper right')
                n += 1
                if n >= leni.shape[0]:
                    break
        plt.title('Segment length')
        plt.ylabel('length [m]')
        plt.xlabel('frame')
        if plt_x_range:
            plt.xlim((plt_x_range[0], plt_x_range[1]))
            plt.xticks(plt_x_range)
        if plt_y_range:
            plt.ylim((plt_y_range[0], plt_y_range[1]))
            plt.yticks(plt_y_range)
        if UI:
            plt.savefig(path + 'tmp_img')
            plt.close()
        else:
            plt.show()

    def plot_velocity(self, frame, dataset, index=[9, 10], path='./', figsize=(8, 8), plt_y_range='', plt_x_range='', padding=1e10, data_name=None, UI=False):
        point_name = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder',
                      'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee',
                      'right knee', 'left ankle', 'right ankle', 'neck']
        # padding_index = np.unique(np.argwhere(data >= 1e10/2)[:, 0:2], axis=0)
        if data_name == None:
            data_name = []
            for i in range(len(dataset)):
                data_name.append(i)
        frame = frame.reshape(-1, 1)
        plt.rcParams['figure.figsize'] = figsize
        point_dataset = 0
        for point in index:
            count_dataset = 0
            for data in dataset:
                data = data[:, :, 3]
                point_velocity = data.T[point]
                padding_index = np.argwhere(point_velocity >= padding/2)
                point_velocity = np.delete(point_velocity, padding_index, axis=0)
                X = np.delete(frame, padding_index, axis=0)
                point_velocity = point_velocity
                plt.plot(X, point_velocity, label=point_name[point] + '(data:' + str(data_name[count_dataset]) + ')')
                # plt.plot(X, point_velocity, label=point_name[point] + '(data:' + str(data_name[count_dataset]) + ')', color=self.colormap[point_dataset])
                plt.legend(loc='upper right')
                plt.title('Velocity')
                plt.xlabel('frame ')
                plt.ylabel('Velocity [km/hr]')
                if plt_x_range:
                    plt.xlim((plt_x_range[0], plt_x_range[1]))
                    plt.xticks(plt_x_range)
                if plt_y_range:
                    plt.ylim((plt_y_range[0], plt_y_range[1]))
                    plt.yticks(plt_y_range)
                count_dataset += 1
            point_dataset += 1
        if UI:
            plt.savefig(path + 'tmp_img')
            plt.close()
        else:
            plt.show()
        # plt.show()
        # plt.close()
        # plt.pause(1)

    def plot_coord(self, frame, data_set, index=[9, 10], path='./', figsize=(12, 3), plt_scale_high=1, padding=1e10, data_name=None, axis_set=[0, 1, 2], UI=False, plt_x_range=None, plt_y_range=None):
        point_name = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder',
                      'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee',
                      'right knee', 'left ankle', 'right ankle', 'neck']
        coord_name = ['X', 'Y', 'Z']
        if data_name == None:
            data_name = []
            for i in range(len(data_set)):
                data_name.append(i)
        # padding_index = np.unique(np.argwhere(data >= 1e10/2)[:, 0:2], axis=0)
        frame = frame.reshape(-1, 1)
        plt.rcParams['figure.figsize'] = figsize
        for point in index:
            for axis in axis_set:
                count_dataset = 0
                ymin, ymax = padding, -padding
                for data in data_set:
                    # writedown the here
                    point_coord = data[:, point, axis]
                    padding_index = np.argwhere(np.abs(point_coord) >= padding/2)
                    point_coord = np.delete(point_coord, padding_index, axis=0)
                    X = np.delete(frame, padding_index, axis=0)
                    if np.max(point_coord) >= ymax:
                        ymax = np.max(point_coord)
                    if np.min(point_coord) <= ymin:
                        ymin = np.min(point_coord)
                    # plt.plot(X, point_coord, label=point_name[point] + '(data:' + str(count_dataset)+')')
                    plt.plot(X, point_coord, label=point_name[point] + '(data:' + str(data_name[count_dataset]) + ')', color=self.colormap[count_dataset])
                    plt.legend(loc='upper right')
                    plt.title('3D coordinate - ' + point_name[point])
                    plt.xlabel('frame ')
                    plt.ylabel(coord_name[axis] + ' [m]')
                    count_dataset += 1
                ymid = ymin+(ymax-ymin)/2
                plt.ylim(ymid-plt_scale_high, ymid+plt_scale_high)
                if plt_x_range:
                    plt.xlim((plt_x_range[0], plt_x_range[1]))
                    plt.xticks(plt_x_range)
                if plt_y_range:
                    plt.ylim((plt_y_range[0], plt_y_range[1]))
                    plt.yticks(plt_y_range)
                if UI:
                    plt.savefig(path + 'tmp_img')
                    plt.close()
                else:
                    plt.show()
        # plt.close()
        # plt.pause(1)

    def plot_reproject(self, imgsize, pose_path, inf_path, camera_path, camera_int, camera_dist, camera_rvec, camera_tvec, fps=30, type='normal', COM=None, mininame='', UI=False):
        frame, stn_data = self.Loadcsv_3d(pose_path)
        _, pose_inf = self.Loadcsv_poseinf(inf_path)
        sdtn_data = stn_data.reshape(-1, 3)[:, [0, 2, 1]].reshape(stn_data.shape[0], -1, 3)
        sdtn_data[:, :, 1] = -1*sdtn_data[:, :, 1]
        mtx, rvec, tvec, dist = [], [], [], []
        # M/dist
        for i in range(len(camera_int)):
            mtx.append(camera_orientation().Interior_Orientation(camera_int[i]))
            dist.append(camera_orientation().Interior_Orientation(camera_dist[i]))
        # R/T
        rvec.append(np.eye(3))
        tvec.append(np.zeros((3, 1)))
        for i in range(len(camera_int)-1):
            originR = cv2.Rodrigues(rvec[i], None)[0]
            originT = tvec[i]
            tempR = cv2.Rodrigues(camera_orientation().Interior_Orientation(camera_rvec[i]), None)[0]
            tempT = camera_orientation().Interior_Orientation(camera_tvec[i])
            R, T, _, _, _, _, _, _, _, _ = cv2.composeRT(originR, originT, tempR, tempT)
            R = cv2.Rodrigues(R, None)[0]
            rvec.append(R)
            tvec.append(T)
        mtx, dist, rvec, tvec = np.array(mtx), np.array(dist), np.array(rvec), np.array(tvec)
        # 2D pose
        pose2D = []
        COM2D = []
        for i in range(len(camera_int)):
            data_2d = camera_orientation().world2img(sdtn_data, mtx[i], rvec[i], tvec[i], dist[i]).reshape(sdtn_data.shape[0], -1, 2)
            # data_2d = camera_orientation().dire_world2img(sdtn_data, mtx[i], rvec[i], tvec[i], dist[i]).reshape(sdtn_data.shape[0], -1, 2)
            if COM:
                self.data2csv2d(mypath=os.path.splitext(camera_path[i])[0]+'.csv', frame=frame, data=data_2d, mininame=mininame)
                # transformation
                COM_2d = coord2vicon().get_COM(data_2d.copy())
                COM2D.append(COM_2d)
                # print(COM_2d)
            pose2D.append(data_2d)
            # plt.bar(frame, tmp[:, 3, 1])
        # projection
        pose2D = np.round(pose2D, 0).astype(np.int)
        for i in range(len(camera_path)):
            if COM is None:
                tool().plot2D(frame=frame, pose2D=pose2D[i], path=camera_path[i], fps=fps, imgsize=imgsize, type=type, pose_inf=pose_inf, label=True, mininame=mininame, UI=UI)
            else:
                tool().plot2D(frame=frame, pose2D=pose2D[i], path=camera_path[i], fps=fps, imgsize=imgsize, type=type, pose_inf=pose_inf, label=True, COM=COM2D[i], mininame=mininame, UI=UI)

    def plot2D(self, frame, pose2D, path, fps, imgsize, type='normal', pose_inf='', label=True, mininame='', COM=None, UI=False):
        logo_size = 1
        circle_size = 4
        word_size = 0.5
        line_size = 3
        delay_time = 1
        print(path)
        if COM is not None:
            pose2D = np.concatenate((pose2D, COM.reshape(-1, 1, 2)), axis=1)

        color_set = [(0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (77, 255, 255),
                     (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77), (204, 77, 255),
                     (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255),
                     (0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222), (77, 196, 255),
                     (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255), (255, 156, 127), (0, 127, 255),
                     (255, 127, 77), (0, 77, 255), (255, 77, 36), (156, 89, 155), (0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (77, 255, 255),
                     (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77), (204, 77, 255),
                     (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255),
                     (0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222), (77, 196, 255),
                     (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255), (255, 156, 127), (0, 127, 255),
                     (255, 127, 77), (0, 77, 255), (255, 77, 36), (156, 89, 155)]
        pose2D = np.round(pose2D, 0).astype(np.int)
        # if COM is not None:
        #     line_i = [0, 0, 1, 2, 10, 8, 6, 17, 5, 7, 12, 14, 11, 13, 17, 11, 17]
        #     line_f = [1, 2, 3, 4, 8, 6, 17, 5, 7, 9, 14, 16, 13, 15, 0, 12, 18]
        # else:
        #     line_i = [10, 8, 6, 17, 5, 7, 17, 12, 14, 17, 11, 13, 17]
        #     line_f = [8, 6, 17, 5, 7, 9, 12, 14, 16, 11, 13, 15, 0]
        if type == 'normal':
            if COM is not None:
                l_pair = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 18), (18, 11), (18, 12), (11, 13), (12, 14), (13, 15), (14, 16), (0, 17)]
            else:
                l_pair = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 11), (17, 12), (11, 13), (12, 14), (13, 15), (14, 16), (0, 17)]
            p_color = color_set[0:pose2D.shape[1]]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = color_set[pose2D.shape[1]:]

        elif type == 'alphapose_projection':
            l_pair = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 11), (17, 12), (11, 13), (12, 14), (13, 15), (14, 16), (0, 17)]
            line_color = [(255, 255, 255)]*100
            camera_index = np.arange(max(pose_inf.reshape(-1))+1).astype(np.int)
            camera_combins = [c for c in combinations(camera_index, 2)]
        elif type == 'cube_projection':
            delay_time = 0
            line_size, circle_size = 2, 2
            l_pair = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            line_color = [(2, 170, 65)] * 100
            camera_index = np.arange(max(pose_inf.reshape(-1)) + 1).astype(np.int)
            camera_combins = [c for c in combinations(camera_index, 2)]
        elif type == 'alphapose':
            l_pair = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 11), (17, 12), (11, 13), (12, 14), (13, 15), (14, 16), (0, 17)]
            p_color = [(0, 0, 0)] * pose2D.shape[1]
            line_color = [(0, 0, 0)] * 100
        elif type == 'cube':
            delay_time = 0
            line_size, circle_size = 2, 2
            l_pair = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            p_color = [(0, 0, 0)] * pose2D.shape[1]
            line_color = [(0, 0, 0)] * 100
        # video
        cam = cv2.VideoCapture(path)
        if fps == '':
            fps = cam.get(5)
        w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoname = os.path.splitext(path)[0] + mininame + os.path.splitext(path)[1]
        if os.path.isdir(os.path.splitext(videoname)[0]):
            shutil.rmtree(os.path.splitext(videoname)[0])
        os.mkdir(os.path.splitext(videoname)[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(videoname, fourcc, fps, (w, h))
        for j in range(np.max(frame)):
            # read frame
            if UI is False:
                print('frame = ', j+1)
            ret_read, img = cam.read()
            # last frame
            if ret_read is not True:
                print('over...')
                break
            if np.min(frame) <= j+1:
                # line
                for n, (start_p, end_p) in enumerate(l_pair):
                    keypoints = pose2D[j + 1 - np.min(frame)]
                    start_xy = tuple(keypoints[start_p])
                    end_xy = tuple(keypoints[end_p])
                    if start_xy[0] >= 0 and start_xy[0] <= imgsize[0] and start_xy[1] >= 0 and start_xy[1] <= imgsize[1]:
                        if end_xy[0] >= 0 and end_xy[0] <= imgsize[0] and end_xy[1] >= 0 and end_xy[1] <= imgsize[1]:
                            cv2.line(img, start_xy, end_xy, line_color[n], line_size)
                # point
                if type == 'alphapose_projection' or type == 'cube_projection':  # where are these points from?
                    p_color = [(255, 255, 255)] * pose2D.shape[1]
                    cv2.putText(img, '3D reconsturction', (imgsize[0] - 200, 71), cv2.FONT_HERSHEY_TRIPLEX, word_size, line_color[0], 1, cv2.LINE_AA)
                    for u in range(len(camera_combins)):
                        cv2.putText(img, 'camera pairs = '+str(camera_combins[u]), (imgsize[0] - 200, 71+(u+1)*18), cv2.FONT_HERSHEY_TRIPLEX, word_size, color_set[u], 1, cv2.LINE_AA)
                    for u in range(len(p_color)):
                        if pose_inf[j + 1 - np.min(frame)][u][0] >= 0:
                            p_color[u] = color_set[camera_combins.index(tuple(pose_inf[j + 1 - np.min(frame)][u]))]
                elif type == 'alphapose':
                    cv2.putText(img, 'AlphaPose', (imgsize[0] - 200, 53), cv2.FONT_HERSHEY_TRIPLEX, word_size, line_color[0], 1, cv2.LINE_AA)
                elif type == 'cube':
                    cv2.putText(img, 'cube', (imgsize[0] - 200, 53), cv2.FONT_HERSHEY_TRIPLEX, word_size, line_color[0], 1, cv2.LINE_AA)
                for n in range(len(p_color)):
                    keypoints = pose2D[j + 1 - np.min(frame)][n]
                    if keypoints[0] >= 0 and keypoints[0] <= imgsize[0] and keypoints[1] >= 0 and keypoints[1] <= imgsize[1]:
                        cv2.circle(img, (keypoints[0], keypoints[1]), circle_size, p_color[n], -1)
                        if label == True:
                            cv2.putText(img, str(n), tuple(keypoints), cv2.FONT_HERSHEY_SIMPLEX, word_size, p_color[n], 1, cv2.LINE_AA)
                img = cv2.rectangle(img, (10, 50), (300, 100), (0, 0, 0), -1)
                cv2.putText(img, 'NTU-YR-LAB', (30, 85), cv2.FONT_HERSHEY_TRIPLEX, logo_size, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, 'frame = '+str(j+1), (imgsize[0]-200, 35), cv2.FONT_HERSHEY_TRIPLEX, word_size, (255, 0, 0), 1, cv2.LINE_AA)
                # cv2.imshow('2D video', img)
                cv2.imwrite(os.path.splitext(videoname)[0]+'/'+str(j+1)+'.png', img)
                out.write(img)
                # cv2.waitKey(delay_time)
        cam.release()
        out.release()
        cv2.destroyAllWindows()

    def plot2D_img(self, img, pose2D, type='normal', pose_inf='', label=True):
        logo_size = 1
        circle_size = 4
        word_size = 0.5
        line_size = 3
        delay_time = 0
        imgsize = np.array([img.shape[1], img.shape[0]]).astype(np.int)
        color_set = [(0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255),
                     (77, 255, 255),
                     (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77), (204, 77, 255),
                     (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255),
                     (0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222), (77, 196, 255),
                     (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255), (255, 156, 127), (0, 127, 255),
                     (255, 127, 77), (0, 77, 255), (255, 77, 36), (156, 89, 155)]
        pose2D = np.round(pose2D, 0).astype(np.int)
        if type == 'normal':
            l_pair = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 11), (17, 12),
                      (11, 13), (12, 14), (13, 15), (14, 16), (0, 17)]
            p_color = color_set[0:pose2D.shape[1]]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
            line_color = color_set[pose2D.shape[1]:]
        if type == 'alphapose_projection':
            l_pair = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 11), (17, 12),
                      (11, 13), (12, 14), (13, 15), (14, 16), (0, 17)]
            line_color = [(255, 255, 255)] * 100
            camera_index = np.arange(max(pose_inf.reshape(-1)) + 1).astype(np.int)
            camera_combins = [c for c in combinations(camera_index, 2)]
        elif type == 'cube_projection':
            delay_time = 0
            line_size, circle_size = 2, 2
            l_pair = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            line_color = [(2, 170, 65)] * 100
            camera_index = np.arange(max(pose_inf.reshape(-1)) + 1).astype(np.int)
            camera_combins = [c for c in combinations(camera_index, 2)]
        elif type == 'alphapose':
            l_pair = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (17, 11), (17, 12),
                      (11, 13), (12, 14), (13, 15), (14, 16), (0, 17)]
            p_color = [(0, 0, 0)] * pose2D.shape[1]
            line_color = [(0, 0, 0)] * 100
        elif type == 'cube':
            delay_time = 0
            line_size, circle_size = 2, 2
            l_pair = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            p_color = [(0, 0, 0)] * pose2D.shape[1]
            line_color = [(0, 0, 0)] * 100
        # line
        for n, (start_p, end_p) in enumerate(l_pair):
            keypoints = pose2D[0]
            start_xy = tuple(keypoints[start_p])
            end_xy = tuple(keypoints[end_p])
            if start_xy[0] >= 0 and start_xy[0] <= imgsize[0] and start_xy[1] >= 0 and start_xy[1] <= imgsize[
                1]:
                if end_xy[0] >= 0 and end_xy[0] <= imgsize[0] and end_xy[1] >= 0 and end_xy[1] <= imgsize[1]:
                    cv2.line(img, start_xy, end_xy, line_color[n], line_size)
        # point
        if type == 'alphapose_projection' or type == 'cube_projection':  # where are these points from?
            pass
            # p_color = [(255, 255, 255)] * pose2D.shape[1]
            # cv2.putText(img, '3D reconsturction', (imgsize[0] - 200, 71), cv2.FONT_HERSHEY_TRIPLEX, word_size,
            #             line_color[0], 1, cv2.LINE_AA)
            # for u in range(len(camera_combins)):
            #     cv2.putText(img, 'camera pairs = ' + str(camera_combins[u]),
            #                 (imgsize[0] - 200, 71 + (u + 1) * 18), cv2.FONT_HERSHEY_TRIPLEX, word_size,
            #                 color_set[u], 1, cv2.LINE_AA)
            # for u in range(len(p_color)):
            #     if pose_inf[j + 1 - np.min(frame)][u][0] >= 0:
            #         p_color[u] = color_set[camera_combins.index(tuple(pose_inf[j + 1 - np.min(frame)][u]))]
        elif type == 'alphapose':
            cv2.putText(img, 'AlphaPose', (imgsize[0] - 200, 53), cv2.FONT_HERSHEY_TRIPLEX, word_size,
                        line_color[0], 1, cv2.LINE_AA)
        elif type == 'cube':
            cv2.putText(img, 'cube', (imgsize[0] - 200, 53), cv2.FONT_HERSHEY_TRIPLEX, word_size, line_color[0],
                        1, cv2.LINE_AA)
        for n in range(len(p_color)):
            keypoints = pose2D[0][n]
            if keypoints[0] >= 0 and keypoints[0] <= imgsize[0] and keypoints[1] >= 0 and keypoints[1] <= \
                    imgsize[1]:
                cv2.circle(img, (keypoints[0], keypoints[1]), circle_size, p_color[n], -1)
                if label == True:
                    cv2.putText(img, str(n), tuple(keypoints), cv2.FONT_HERSHEY_SIMPLEX, word_size, p_color[n],
                                1, cv2.LINE_AA)
        # img = cv2.rectangle(img, (10, 50), (300, 100), (0, 0, 0), -1)
        # cv2.putText(img, 'NTU-YR-LAB', (30, 85), cv2.FONT_HERSHEY_TRIPLEX, logo_size, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(img, (imgsize[0] - 200, 35), cv2.FONT_HERSHEY_TRIPLEX, word_size, (255, 0, 0), 1, cv2.LINE_AA)
        # cv2.imshow('2D video', img)
        # cv2.waitKey(delay_time)
        return img

    def plot_point(self, frame, row_data, predict_data, data_inf='', display=['']):
        for point in range(row_data.shape[1]):
            if point in display:
                for axis in range(row_data.shape[2]):
                    plt.scatter(frame, row_data[:, point, axis], label='row data', s=10)
                    if type(data_inf) is np.ndarray:
                        data_inf = data_inf.astype(np.int)
                        cadidate = np.unique(data_inf[:, point, axis])
                        for radius in cadidate:
                            index = np.argwhere(data_inf[:, point, axis] == radius).reshape(-1)
                            pre_frame, pre_data = frame[index], predict_data[:, point, axis][index]
                            plt.scatter(pre_frame, pre_data, label='radius = '+str(radius), s=10)
                    else:
                        plt.scatter(frame, predict_data[:, point, axis], label='predict data', s=10)
                    plt.legend()
                    plt.title('Point '+str(point))
                    plt.xlabel('frame')
                    if axis == 0:
                        plt.ylabel('X [m]')
                    elif axis == 1:
                        plt.ylabel('Y [m]')
                    elif axis == 2:
                        plt.ylabel('Z [m]')
                    plt.show()

    def combin_file(self, path, type):
        dir_set = sorted(listdir(path))
        n = 0
        for f in range(len(dir_set)):
            fullpath = join(path, dir_set[f])
            secondaryname = os.path.splitext(fullpath)[1]
            if secondaryname in type:
                if n == 0:
                    conbination = np.load(fullpath)
                else:
                    temp = np.load(fullpath)
                    conbination = np.vstack((conbination, temp))
                # print('the shape = ', conbination.shape)
                n += 1
        if n == 0:
            print('there is no file.')
        else:
            print('final shape = ', conbination.shape)
            np.save(path+'conbination_file', conbination)

    def splite_np(self, path, index=0):
        data = np.load(path)[np.newaxis, index]
        np.save(path, data)

class joint_analysis(object):

    def __init__(self):
        pass

    def find_max_velocity(self, data, padding=1e10):
        padding_index = np.argwhere(data > padding / 2)
        data[padding_index] = -1e10
        max_value = np.max(data)
        max_index = np.argwhere(data == max_value).reshape(-1)[0]
        return max_index, max_value

    def eucildea_distance(self, coord_A, coord_B):
        return np.sum(np.sqrt(np.sum(np.power(coord_A-coord_B, 2), 1)))

    def action_difference(self, basic_path, aim_path, valid_point):

        def shift_map_loss(basic_data, aim_data, data_flag='half'):
            if data_flag == 'half':
                basic_data, aim_data = basic_data[:int(basic_data.shape[0]/2)], aim_data[:int(aim_data.shape[0]/2)]
            frame_i = 0
            loss_set = []
            if basic_data.shape[0] >= aim_data.shape[0]:
                frame_f = aim_data.shape[0]
                for i in range(aim_data.shape[0]):
                    loss = joint_analysis().eucildea_distance(aim_data[frame_i:frame_f-i, valid_point, :], basic_data[frame_i+i:frame_f, valid_point, :])
                    loss = loss/aim_data[frame_i:frame_f-i, valid_point, :].shape[0]
                    loss_set.append(loss)
            else:
                frame_f = basic_data.shape[0]
                for i in range(aim_data.shape[0]):
                    if aim_data.shape[0] >= frame_f+i:
                        loss = joint_analysis().eucildea_distance(basic_data[frame_i:frame_f, valid_point, :], aim_data[frame_i+i:frame_f+i, valid_point, :])
                        loss = loss/basic_data[frame_i:frame_f, valid_point, :].shape[0]
                    else:
                        loss = joint_analysis().eucildea_distance(basic_data[frame_i:frame_f-i+(aim_data.shape[0]-basic_data.shape[0]), valid_point, :], aim_data[frame_i + i:aim_data.shape[0], valid_point, :])
                        loss = loss / basic_data[frame_i:frame_f - i + (aim_data.shape[0] - basic_data.shape[0]), valid_point, :].shape[0]
                    loss_set.append(loss)
            loss_set = np.array(loss_set)
            plt.plot(loss_set)
            plt.xlabel('shift frame')
            plt.ylabel('loss')
            plt.show()
            min_index = np.argwhere(loss_set == np.min(loss_set)).reshape(-1)[0]
            return min_index


        # load frame
        basic_frame, basic_data = tool().Loadcsv_3d(basic_path)
        aim_frame, aim_data = tool().Loadcsv_3d(aim_path)

        # transformation for ICP method
        new_point, H, loss = joint_analysis().ICP(basic_data[0], aim_data[0])
        basic_data = camera_orientation().threeD_transformation(dataset=basic_data, H=H)
        # Auto sync
        shift_basic_frame = shift_map_loss(basic_data, aim_data, data_flag='half')
        if shift_basic_frame == 0:
            shift_aim_frame = shift_map_loss(aim_data, basic_data, data_flag='half')
        # shift_basic_frame, shift_aim_frame = 0, 0
        print(shift_basic_frame, shift_aim_frame)
        basic_data, aim_data = basic_data[shift_basic_frame:, :, :], aim_data[shift_aim_frame:, :, :]

        # calculate the distance
        loss_set = []
        for i in range(min(basic_data.shape[0], aim_data.shape[0])):
            loss = joint_analysis().eucildea_distance(basic_data[i], aim_data[i])
            loss_set.append(loss)
        loss_set = np.array(loss_set)
        # plt.plot(loss_set)
        # plt.show()
        return loss_set, shift_basic_frame, shift_aim_frame, H


    def mix_video(self, output, basic_path, basic_video, aim_path, aim_video, valid_point, shift_basic_frame, shift_aim_frame, ICP, basic_mtx, basic_dist, basic_R, basic_T, basic_transformation, aim_mtx, aim_dist, aim_R, aim_T, aim_transformation, alpha=0.7, gamma=0, fps=None, UI=False):
        # build folder
        if os.path.isdir(os.path.splitext(output)[0]):
            shutil.rmtree(os.path.splitext(output)[0])
        os.mkdir(os.path.splitext(output)[0])
        # mix the video
        for i in range(len(basic_video)):
            basic_cam = cv2.VideoCapture(basic_video[i])
            aim_cam = cv2.VideoCapture(aim_video[i])
            if fps is None:
                fps = basic_cam.get(5)
            w = int(basic_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(basic_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
            videoname = os.path.splitext(output)[0] + 'basic_' + os.path.basename(basic_video[i])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(videoname, fourcc, fps, (w, h))
            if shift_basic_frame:
                for x in range(shift_basic_frame):
                    _ = basic_cam.read()
            else:
                for x in range(shift_aim_frame):
                    _ = aim_cam.read()
            while True:
                basic_ret, basic_img = basic_cam.read()
                aim_ret, aim_img = aim_cam.read()
                if basic_ret and aim_ret:
                    h, w, _ = basic_img.shape
                    aim_img = cv2.resize(aim_img, (w, h), interpolation=cv2.INTER_AREA)
                    mix_img = cv2.addWeighted(basic_img, alpha, aim_img, 1-alpha, gamma)
                    out.write(mix_img)
                    if UI is False:
                        # cv2.namedWindow('addImage')
                        cv2.imshow('img_add', mix_img)
                        cv2.waitKey(1)
                else:
                    print('Over...')
                    break
            out.release()
            basic_cam.release()
            aim_cam.release()
        cv2.destroyAllWindows()


    def segment_analysis(self, path, frame, data, leni=np.array([5, 7, 6, 8, 11, 13, 12, 14]), lenf=np.array([7, 9, 8, 10, 13, 15, 14, 16]), padding=1e10):
        if data.ndim == 2:
            unlimited_point = np.argwhere(data[:, 0] == padding).reshape(-1)
            unlimited_bond_index = np.array([])
            for i in unlimited_point:
                temp = np.append(np.argwhere(leni == i).reshape(-1), np.argwhere(lenf == i).reshape(-1))
                unlimited_bond_index = np.append(unlimited_bond_index, temp)
            unlimited_bond_index = np.unique(unlimited_bond_index).astype(np.int)
            # if the point is [1e10, 1e10, 1e10], that is -1 on bond
            inf_bond = np.sqrt(np.sum(np.power(data[leni]-data[lenf], 2), axis=1))
            inf_bond[unlimited_bond_index] = -1
        else:
            inf_bond_set = []
            for data in data:
                unlimited_point = np.argwhere(data[:, 0] == padding).reshape(-1)
                unlimited_bond_index = np.array([])
                for i in unlimited_point:
                    temp = np.append(np.argwhere(leni == i).reshape(-1), np.argwhere(lenf == i).reshape(-1))
                    unlimited_bond_index = np.append(unlimited_bond_index, temp)
                unlimited_bond_index = np.unique(unlimited_bond_index).astype(np.int)
                # if the point is [1e10, 1e10, 1e10], that is -1 on bond
                inf_bond = np.sqrt(np.sum(np.power(data[leni] - data[lenf], 2), axis=1))
                inf_bond[unlimited_bond_index] = -1
                inf_bond_set.append(inf_bond)
            inf_bond = np.array(inf_bond_set)
        # CSV
        title = ['frame']
        for n in range(len(leni)):
            title.append(str(leni[n]) + '_' + str(lenf[n]))
        title = np.array(title).reshape(1, -1)
        frame = frame.reshape(-1,1)
        data = np.hstack((frame, inf_bond))
        dataframe = np.vstack((title, data)).astype(np.str)
        dataframe[dataframe == '-1.0'] = ''
        df = pd.DataFrame(dataframe)
        df.to_csv(path, encoding="gbk", index=False, header=False)
        return frame, inf_bond

    def velocity_analysis(self, frame, data, output_path, fps=1, padding=1e10):
        frame = frame.reshape(-1, 1)
        first_v = np.ones((1, 18, 4))*padding
        old_data = data[0:data.shape[0]-1]
        new_data = data[1:data.shape[0]]
        padding_index = np.concatenate((np.argwhere(old_data >= padding/2), np.argwhere(new_data >= padding/2)))[:, 0:2]
        if padding_index.shape[0]:
            padding_index = np.unique(padding_index, axis=0)
        v_component = (new_data-old_data)*fps
        v = np.sqrt(np.sum(np.power(v_component, 2), axis=2)).reshape(v_component.shape[0], -1, 1)*3.6
        v = np.concatenate((v_component, v), axis=2)
        for p in range(padding_index.shape[0]):
            v[padding_index[p][0], padding_index[p][1]] = np.array([padding, padding, padding, padding])
        v = np.concatenate((first_v, v))
        # CSV
        title = ['frame']
        component_title = ['vx_', 'vy_', 'vz_', 'v_']
        for i in range(v.shape[1]):
            for j in range(v.shape[2]):
                title.append(component_title[j]+str(i))
        title = np.array(title).reshape(1, -1)
        data = np.hstack((frame, v.reshape(v.shape[0], -1)))
        dataframe = np.vstack((title, data)).astype(np.str)
        dataframe[dataframe == '10000000000.0'] = ''
        df = pd.DataFrame(dataframe)
        df.to_csv(output_path, encoding="gbk", index=False, header=False)
        return frame, v

    def ICP(self, A, B, init_pose=None, max_iterations=500, tolerance=0.001, T_weight=1):
        '''
        The Iterative Closest Point method
        Input:
            A: Nx3 numpy array of source 3D points
            B: Nx3 numpy array of destination 3D point
            init_pose: 4x4 homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation
            distances: Euclidean distances (errors) of the nearest neighbor
        '''

        def best_fit_transform(A, B, T_weight=1):
            '''
            Calculates the least-squares best-fit transform between corresponding 3D points A->B
            Input:
              A: Nx3 numpy array of corresponding 3D points
              B: Nx3 numpy array of corresponding 3D points
            Returns:
              T: 4x4 homogeneous transformation matrix
              R: 3x3 rotation matrix
              t: 3x1 column vector
            '''

            assert len(A) == len(B)

            # translatge points to their centroids
            centroid_A = np.mean(A, axis=0)
            centroid_B = np.mean(B, axis=0)
            AA = A - centroid_A
            BB = B - centroid_B

            # rotation matrix
            W = np.dot(BB.T, AA)
            U, s, VT = np.linalg.svd(W)
            R = np.dot(U, VT)

            # special reflection case
            if np.linalg.det(R) < 0:
                VT[2, :] *= -1
                R = np.dot(U, VT)

            # translation
            t = centroid_B.T - np.dot(R, centroid_A.T)
            t = t*T_weight

            # homogeneous transformation
            T = np.identity(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = t

            return T, R, t

        def nearest_neighbor(src, dst):
            '''
            Find the nearest (Euclidean) neighbor in dst for each point in src
            Input:
                src: Nx3 array of points
                dst: Nx3 array of points
            Output:
                distances: Euclidean distances (errors) of the nearest neighbor
                indecies: dst indecies of the nearest neighbor
            '''

            indecies = np.zeros(src.shape[0], dtype=np.int)
            distances = np.zeros(src.shape[0])
            for i, s in enumerate(src):
                min_dist = np.inf
                for j, d in enumerate(dst):
                    dist = np.linalg.norm(s - d)
                    if dist < min_dist:
                        min_dist = dist
                        indecies[i] = j
                        distances[i] = dist
            return distances, indecies


        # make points homogeneous, copy them so as to maintain the originals
        src = np.ones((4, A.shape[0]))
        dst = np.ones((4, B.shape[0]))
        src[0:3, :] = np.copy(A.T)
        dst[0:3, :] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbours between the current source and destination points
            distances, indices = nearest_neighbor(src[0:3, :].T, dst[0:3, :].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = best_fit_transform(src[0:3, :].T, dst[0:3, indices].T, T_weight)

            # update the current source
            # refer to "Introduction to Robotics" Chapter2 P28. Spatial description and transformations
            src = np.dot(T, src)



            # check error
            mean_error = np.sum(distances) / distances.size
            if abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error
        # calculcate final tranformation
        T, _, _ = best_fit_transform(A, src[0:3, :].T)
        src = np.dot(T, src).T[:, :-1]

        return src, T, distances

class manual_2D(object):

    def __init__(self):  # Run it once
        self.n = 0

    def check_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.frame, (x, y), 1, (255, 0, 0), -1)
            cv2.putText(self.frame, str(self.objp[self.n]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            self.draw_point.append([[x, y]])
            if len(self.draw_point) > 1:
                cv2.line(self.frame, tuple(self.draw_point[len(self.draw_point)-1][0]), tuple(self.draw_point[len(self.draw_point)-2][0]), (0, 255, 0), 2)
            x = round(x/self.ratio_w)
            y = round(y/self.ratio_h)
            # print('Add net point: ( ', x, ', ', y, ' )')
            self.corners.append([[x, y]])
            self.n += 1

    def Run(self, path, block_meter, block_w=3, block_h=3, type='checkerboard', ratio_h=1, ratio_w=1, frame=1):
        self.ratio_h, self.ratio_w = ratio_h, ratio_w
        # dir_set = sorted(listdir(path))
        video_type = ['.mp4', '.MP4', '.avi']
        objpoints, imgpoints = [], []
        if type == 'checkerboard':
            objp = np.zeros(((block_w - 1) * (block_h - 1), 3), np.float32)
            objp[:, :2] = np.mgrid[0:block_w - 1, 0:block_h - 1].T.reshape(-1, 2) * block_meter
        elif type == 'cube':
            objp = (np.mgrid[0:2, 0:2, 0:2].T.reshape(-1, 3) * block_meter).astype(np.float32)
        self.objp = objp
        # print('your 3d coordinates is ', objp)
        cornerSubPix_params = dict(winSize=(11, 11), zeroZone=(-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))  # https://www.itread01.com/content/1544079906.html
        # for f in range(len(dir_set)):
        #     fullpath = join(path, dir_set[f])
        fullpath = path
        secondaryname = os.path.splitext(fullpath)[1]
        # print(fullpath, secondaryname)
        if secondaryname in video_type:
            self.video_path = fullpath
            self.dirpath = os.path.dirname(self.video_path)+'/'
            cam = cv2.VideoCapture(self.video_path)
            count = 1
            for xxxxx in range(frame):
                ret_read, self.frame = cam.read()
                count += 1
            h, w = self.frame.shape[:2]
            # print("Load the video:：", self.video_path, '(h , w) = ', '(', h, ' , ', w, ')')
            self.frame = cv2.resize(self.frame, None, fx=self.ratio_w, fy=self.ratio_h, interpolation=cv2.INTER_AREA)

            # Manual
            # print("Automatic loading failed, please select manually...")
            cv2.namedWindow('Manual')
            cv2.setMouseCallback('Manual', self.check_circle)
            self.corners = []
            self.draw_point = []
            while (1):
                cv2.imshow('Manual', self.frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                if self.n >= objp.shape[0] and type == 'cube':
                    break
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            corners = []
            for point in self.corners:
                corners.append(point[0])
            corners = np.array(corners).reshape(-1, 1, 2).astype(np.float32)
            if type == 'checkerboard':
                cv2.cornerSubPix(gray, corners, **cornerSubPix_params)
            imgpoints.append(corners)
            objpoints.append(objp)
            cv2.imwrite(os.path.splitext(self.video_path)[0]+'.jpg', self.frame)
            cv2.destroyAllWindows()
            # csv
            with open(os.path.splitext(self.video_path)[0] + '_block.csv', 'w', newline='') as csvfile:
                csv_f = csv.writer(csvfile)
                # index
                CSV1 = ['frame']
                for index in range(corners.shape[0]):
                    CSV1.append(str(index) + '_x')
                    CSV1.append(str(index) + '_z')
                csv_f.writerow(CSV1)
                # data
                CSV1 = ['1']
                for point in corners:
                    CSV1.append(point[0][0])
                    CSV1.append(point[0][1])
                csv_f.writerow(CSV1)
        # if f == len(dir_set)-1:
        #     break
        imgpoints, objpoints = np.array(imgpoints), np.array(objpoints)
        # print(imgpoints.shape, objpoints.shape)
        np.save(os.path.splitext(fullpath)[0] + '_imgpoints', imgpoints)
        np.save(os.path.splitext(fullpath)[0] + '_objpoints', objpoints)
        # np.save(self.dirpath +'imgpoints', imgpoints)
        # np.save(self.dirpath + 'objpoints', objpoints)

class post_processing():

    def __init__(self, data):  # Run it once
        self.data = data

    def adaptive_radius(self, data, padding, max_radius, adaptive_ratio):
        # thereshold = average*adaptive_ratio

        nonpadding_index = np.argwhere(data[:, 0] != padding).reshape(-1)
        nonpadding_data = data[nonpadding_index]
        min_thereshold = joint_analysis().eucildea_distance(nonpadding_data[1:, :], nonpadding_data[0:-1, :]) / sum((nonpadding_index[1:] - nonpadding_index[0:-1]))*adaptive_ratio[0]
        max_thereshold = joint_analysis().eucildea_distance(nonpadding_data[1:, :], nonpadding_data[0:-1, :]) / sum((nonpadding_index[1:] - nonpadding_index[0:-1]))*adaptive_ratio[1]
        # map_distance = [[0, 1, 2, ....], [1, 0, 1, 2, ...], ...]
        map_radius = np.zeros((data.shape[0], 1))
        map_distance = np.zeros((nonpadding_index.shape[0], nonpadding_index.shape[0]))
        one_matrix = np.triu(np.ones((nonpadding_index.shape[0], nonpadding_index.shape[0])))
        map_distance[0] = np.dot(np.insert(nonpadding_index[1:] - nonpadding_index[0:-1], 0, 0), one_matrix)
        for i in range(map_distance.shape[0]):  # time sequence
            map_distance[i] = np.abs(map_distance[0]-map_distance[0][i])
            for j in np.arange(max_radius+1)[::-1][:-1]:  # max_radius, ..., 1
                fit_index = np.argwhere(map_distance[i] <= j).reshape(-1)
                if fit_index.shape[0] < 3:
                    break
                else:
                    distance = joint_analysis().eucildea_distance(nonpadding_data[fit_index][1:, :], nonpadding_data[fit_index][0:-1, :])/(fit_index.shape[0]-1)
                    if min_thereshold > distance:
                        map_radius[i] = j
                        break
                    elif max_thereshold < distance:
                        map_radius[i] = max_radius
                        break
        # print(map_radius.reshape(-1))
        return map_radius.reshape(-1).astype(np.int)

    def interpolation_1d(self, order=1, fit_radius=1, padding=1e10, radius_type='normal', adaptive_ratio=(1,2)):
        min_fitting_radius = 3
        predict_data = self.data.copy()
        max_radius = fit_radius
        if radius_type == 'adaptive':
            map_radius = self.adaptive_radius(data=self.data, padding=padding, max_radius=np.average(max_radius), adaptive_ratio=adaptive_ratio)
        fitting_data = self.data
        if radius_type == 'normal' or radius_type == 'spline':
            map_radius = (np.ones((self.data.shape[0])) * fit_radius).astype(np.int)
        frame = np.arange(0, self.data.shape[0])
        for time_frame in frame:
            fit_radius = map_radius[time_frame]
            pi = 0 if time_frame < fit_radius else time_frame - fit_radius
            pf = time_frame + fit_radius if time_frame + fit_radius < self.data.shape[0] else self.data.shape[0]
            feature = np.argwhere(self.data[pi:pf+1] != padding).reshape(-1)+pi
            X = feature-time_frame  # -time_frame
            Y = fitting_data[feature]
            if Y.shape[0] >= min_fitting_radius*2+1:
                if radius_type == 'normal':
                    fitting_curve = np.poly1d(np.polyfit(X, Y, order))
                    predict_data[time_frame] = fitting_curve(0)
                if radius_type == 'spline':
                    new_X = np.linspace(0, X.max(), 100)
                    power_data = make_interp_spline(X, Y)(new_X)
                    predict_data[time_frame] = power_data[0]
                    # print(power_smooth.shape)
                    # print(power_data.shape, power_index.shape)
        return predict_data

    def interpolation_3d(self, order=[1, 1, 1], fit_radius=[1, 1, 1], fit_point=[10], padding=1e10, radius_type='adaptive', adaptive_ratio=(1,2)):
        min_fitting_radius = 8
        predict_data = self.data.copy()
        max_radius = fit_radius.copy()
        globel_radius = predict_data.copy()*0
        for point in range(self.data.shape[1]):
            if point in fit_point:
                if radius_type == 'adaptive':
                    map_radius = self.adaptive_radius(data=self.data[:, point, :], padding=padding, max_radius=np.average(max_radius), adaptive_ratio=adaptive_ratio)
                for axis in range(self.data.shape[2]):
                    fitting_data = self.data[:, point, axis]
                    if radius_type == 'normal' or radius_type == 'spline':
                        map_radius = (np.ones((self.data.shape[0])) * fit_radius[axis]).astype(np.int)
                    globel_radius[:, point, axis] = map_radius
                    frame = np.arange(0, self.data.shape[0])
                    for time_frame in frame:
                        fit_radius[axis] = map_radius[time_frame]
                        pi = 0 if time_frame < fit_radius[axis] else time_frame - fit_radius[axis]
                        pf = time_frame + fit_radius[axis] if time_frame + fit_radius[axis] < self.data.shape[0] else self.data.shape[0]
                        feature = np.argwhere(self.data[pi:pf+1, point, axis] != padding).reshape(-1)+pi
                        X = feature-time_frame  # -time_frame
                        Y = fitting_data[feature]
                        if Y.shape[0] >= min_fitting_radius*2+1:
                            if radius_type == 'normal':
                                fitting_curve = np.poly1d(np.polyfit(X, Y, order[axis]))
                                predict_data[time_frame, point, axis] = fitting_curve(0)
                            if radius_type == 'spline':
                                new_X = np.linspace(0, X.max(), 100)
                                power_data = make_interp_spline(X, Y)(new_X)
                                predict_data[time_frame, point, axis] = power_data[0]
                                # print(power_smooth.shape)
                                # print(power_data.shape, power_index.shape)

        return predict_data, globel_radius

    def interpolation_2d(self, order, fit_radius, fit_point):
        min_fitting_radius = 1
        predict_data = self.data.copy()
        for point in range(self.data.shape[1]):
            if point in fit_point:
                for axis in range(self.data.shape[2]):
                    fitting_data = self.data[:, point, axis]
                    frame = np.arange(0, self.data.shape[0])
                    for time_frame in frame:
                        pi = 0 if time_frame < fit_radius[axis] else time_frame - fit_radius[axis]
                        pf = time_frame + fit_radius[axis] if time_frame + fit_radius[axis] < self.data.shape[0] else self.data.shape[0]
                        feature = np.argwhere(self.data[pi:pf+1, point, axis] != -1).reshape(-1)+pi
                        X = feature-time_frame  # -time_frame
                        Y = fitting_data[feature]
                        time.sleep(0)
                        if Y.shape[0] >= min_fitting_radius*2+1:
                            fitting_curve = np.poly1d(np.polyfit(X, Y, order[axis]))
                            predict_data[time_frame, point, axis] = fitting_curve(0)
                    if point == 10:
                        plt.scatter(frame, self.data[:, point, axis])
                        plt.scatter(frame, predict_data[:, point, axis])
                        plt.xticks([])
                        plt.title('Point 10')
                        plt.xlabel('frame')
                        if axis == 0:
                            plt.ylabel('X [m]')
                        elif axis == 1:
                            plt.ylabel('Y [m]')
                        elif axis == 2:
                            plt.ylabel('Z [m]')
                        plt.show()
        return predict_data

    def guide_face_transfer(self, data, label, guide, max_iterations, tolerance, length_proportion, padding=1e10):
        origin_point = data[label].copy()
        guide_point = data[label].copy()
        for i in range(1, len(label)):
            old_shift = origin_point[i] - origin_point[i - 1]
            old_length = np.sqrt(np.sum(np.power(old_shift, 2)))
            if abs(old_length-guide[i - 1])/guide[i - 1] > length_proportion:
                ratio = guide[i - 1] / old_length
            else:
                ratio = 1
            new_shift = old_shift * ratio
            print(abs(old_length-guide[i - 1])/guide[i - 1], ratio)
            guide_point[i] = guide_point[i - 1] + new_shift
        guide_point[np.isnan(guide_point)] = padding
        guide_point, H, loss = joint_analysis().ICP(guide_point, origin_point, None, max_iterations, tolerance)
        return guide_point

    def guide_pose_transfer(self, data, label, guide, max_iterations, tolerance, length_proportion, padding=1e10):
        origin_point = data[label].copy()
        guide_point = data[label].copy()
        for i in range(1, len(label)):
            old_shift = origin_point[i] - origin_point[i - 1]
            old_length = np.sqrt(np.sum(np.power(old_shift, 2)))
            if abs(old_length-guide[i - 1])/guide[i - 1] > length_proportion:
                ratio = guide[i - 1] / old_length
            else:
                ratio = 1
            new_shift = old_shift * ratio
            # print(abs(old_length-guide[i - 1])/guide[i - 1], ratio)
            guide_point[i] = guide_point[i - 1] + new_shift
        guide_point[np.isnan(guide_point)] = padding
        shift_point = guide_point.copy()
        guide_point, H, loss = joint_analysis().ICP(guide_point, origin_point, None, max_iterations, tolerance)
        return guide_point, H, shift_point

    def guide_length_check(self, data, label, guide, length_proportion, padding=1e10):
        origin_point = data[label].copy()
        guide_point = data[label].copy()
        for i in range(len(label)):
            first_segment = False
            second_segment = False
            if i == 0:
                second_length = np.sqrt(np.sum(np.power(origin_point[i+1] - origin_point[i], 2)))
                if (second_length-guide[i])/guide[i] > length_proportion:
                    second_segment = True
            elif i == len(label)-1:
                first_length = np.sqrt(np.sum(np.power(origin_point[i] - origin_point[i-1], 2)))
                if (first_length - guide[i-1]) / guide[i-1] > length_proportion:
                    first_segment = True
            else:
                first_length = np.sqrt(np.sum(np.power(origin_point[i] - origin_point[i - 1], 2)))
                second_length = np.sqrt(np.sum(np.power(origin_point[i + 1] - origin_point[i], 2)))
                if (first_length - guide[i - 1]) / guide[i - 1] > length_proportion:
                    first_segment = True
                if (second_length - guide[i]) / guide[i] > length_proportion:
                    second_segment = True
            if first_segment and second_segment:
                guide_point[i] = padding
                # print(first_segment, second_segment)
                # print((first_length - guide[i - 1]) / guide[i - 1], (second_length - guide[i]) / guide[i])
        return guide_point

    def human_skeleton(self, data, t_data, max_iterations, tolerance, length_proportion=0.5):
        lag_point, arm_point, face_point = [16, 14, 12, 11, 13, 15], [10, 8, 6, 5, 7, 9], [4, 2, 0, 1, 3]
        left_lag_point, right_lag_point = [16, 14, 12], [11, 13, 15]
        left_arm_point, right_arm_point = [5, 7, 9], [10, 8, 6]
        leni = [16, 14, 12, 11, 13, 10, 8, 6, 17,  5, 7, 4, 2, 0, 1, 6]
        lenf = [14, 12, 11, 13, 15, 8, 6, 17, 5, 7, 9, 2, 0, 1, 3, 5]
        lag_lebel, arm_lebel, face_lebel = [0, 1, 2, 3, 4], [5, 6, 15, 9, 10], [11, 12, 13, 14]
        left_lag_lebel, right_lag_lebel = [0, 1], [3, 4]
        left_arm_lebel, right_arm_lebel = [9, 10], [5, 6]
        t_segment_length = np.average(joint_analysis().segment_analysis(t_data, leni, lenf), 0)
        guide_pose = data.copy()
        for i in range(data.shape[0]):
            print(i)
            pose = data[i].copy()
            # guide_face, _, _ = self.guide_pose_transfer(pose, face_point, t_segment_length[face_lebel], max_iterations, tolerance, length_proportion)
            guide_left_lag, _, _ = self.guide_pose_transfer(pose, left_lag_point, t_segment_length[left_lag_lebel], max_iterations, tolerance, length_proportion)
            guide_right_lag, _, _ = self.guide_pose_transfer(pose, right_lag_point, t_segment_length[right_lag_lebel], max_iterations, tolerance, length_proportion)
            guide_left_arm, _, _ = self.guide_pose_transfer(pose, left_arm_point, t_segment_length[left_arm_lebel], max_iterations, tolerance, length_proportion)
            guide_right_arm, _, _ = self.guide_pose_transfer(pose, right_arm_point, t_segment_length[right_arm_lebel], max_iterations, tolerance, length_proportion)
            guide_pose[i][left_lag_point] = guide_left_lag
            guide_pose[i][right_lag_point] = guide_right_lag
            guide_pose[i][left_arm_point] = guide_left_arm
            guide_pose[i][right_arm_point] = guide_right_arm
            # guide_pose[i][face_point] = guide_face
            # pose = pose.reshape(1, -1, 3)
            # guide = guide_pose[i].reshape(1, -1, 3)
            # tool().plot_double_3D(frame, pose, guide, xlim, ylim, zlim, view, '../../20200114/acc1/', fps)
        return guide_pose

    def head_model(self, data, t_data, max_iterations, tolerance, padding=1e10):
        point_n = 3
        face_label = np.arange(point_n-1)
        fix_point = [17, 0]
        face_point = [4, 2, 1, 3]
        face_point = fix_point+face_point
        int_face = np.ones((len(face_point), 3))*padding
        for i in range(data.shape[0]):  # [[4, 2, 0], [4, 2, 1], ...]
            origin_face = data[i][face_point]
            # print(data[i][face_point][:, 0])
            useful_index = np.argwhere(data[i][face_point][:, 0] < padding/2).reshape(-1)
            useful_point = np.array(face_point)[np.argwhere(data[i][face_point][:, 0] < padding/2).reshape(-1)]
            if useful_point.shape[0] < point_n:
                data[i][[0, 1, 2, 3, 4]] = int_face[[0, 1, 2, 3, 4]]
            else:
                useful_point_set = np.array([c for c in combinations(useful_point, point_n)])
                loss_set, face_pose_set = [], []
                for h in useful_point_set:  # [4, 2, 0]
                    if 17 in h and 0 in h:
                        leni, lenf = h[0:-1], h[1:]
                        t_segment_length = np.average(joint_analysis().segment_analysis(t_data, leni, lenf), 0)
                        _, H, _ = self.guide_pose_transfer(data[i], h, t_segment_length[face_label], max_iterations, tolerance, 0)
                        leni, lenf = face_point[0:-1], face_point[1:]
                        t_segment_length = np.average(joint_analysis().segment_analysis(t_data, leni, lenf), 0)
                        _, _, shift_point = self.guide_pose_transfer(data[i], face_point, t_segment_length[np.arange(len(face_point)-1)], max_iterations, tolerance, 0)
                        threed_padding = np.zeros((shift_point.shape[0], 1)) + 1.
                        src = np.hstack((shift_point, threed_padding)).T
                        guide_face = np.dot(H, src).T[:, :-1]
                        loss = joint_analysis().eucildea_distance(guide_face[useful_index], origin_face[useful_index])
                        face_pose_set.append(guide_face)
                        loss_set.append(loss)
                face_pose_set, loss_set = np.array(face_pose_set), np.array(loss_set)
                if loss_set.shape[0]:
                    min_index = np.argwhere(loss_set == np.min(loss_set)).reshape(-1)[0]
                    data[i][face_point[len(fix_point):]] = face_pose_set[min_index][len(fix_point):]
                else:
                    data[i][face_point[len(fix_point):]] = int_face[len(fix_point):]
        return data

    def body_anormaly_detect(self, data, t_data, length_proportion=0.2):
        lag_point, arm_point = [16, 14, 12, 11, 13, 15], [10, 8, 6, 5, 7, 9]
        leni = [16, 14, 12, 11, 13, 10, 8, 6, 17,  5, 7, 4, 2, 0, 1, 6]
        lenf = [14, 12, 11, 13, 15, 8, 6, 17, 5, 7, 9, 2, 0, 1, 3, 5]
        lag_lebel, arm_lebel, face_lebel = [0, 1, 2, 3, 4], [5, 6, 15, 9, 10], [11, 12, 13, 14]
        t_segment_length = np.average(joint_analysis().segment_analysis(t_data, leni, lenf), 0)
        guide_pose = data.copy()
        for i in range(data.shape[0]):
            pose = data[i].copy()
            guide_lag = self.guide_length_check(pose, lag_point, t_segment_length[lag_lebel], length_proportion)
            guide_arm = self.guide_length_check(pose, arm_point, t_segment_length[arm_lebel], length_proportion)
            guide_pose[i][lag_point] = guide_lag
            guide_pose[i][arm_point] = guide_arm
        return guide_pose

    def anomaly_detector_all_channel(self, frame, data, max_radius=2, fit_radius=1, fit_order=1, padding=1e10, acc_ratio_threshold=2, fit_point=[10]):
        frame = frame.reshape(-1, 1)
        neighbor_radius = 1
        min_fitting_radius = 1
        # vibration velocity
        for p in range(data.shape[1]):
            print('... processing the point ', str(p))
            pre_index = []
            point_loop = True
            total_index = []
            if p in fit_point:
                while point_loop:
                    p_data = data[:, p, :]
                    p_frame = frame.copy()
                    padding_index = np.argwhere(p_data >= padding / 2)
                    padding_index = np.unique(padding_index[:, 0])
                    p_data = np.delete(p_data, padding_index, axis=0)
                    p_frame = np.delete(p_frame, padding_index, axis=0)
                    vibration_v, vibration_a = [], []
                    for t in range(p_frame.shape[0]):
                        if t <= neighbor_radius:
                            tmp_set = p_data[0:t+neighbor_radius+1, :]
                            frame_set = p_frame[0:t+neighbor_radius+1, :]
                        elif t >= p_data.shape[0]-neighbor_radius:
                            tmp_set = p_data[t - neighbor_radius:p_data.shape[0], :]
                            frame_set = frame[t - neighbor_radius:p_data.shape[0]]
                        else:
                            tmp_set = p_data[t-neighbor_radius:t+neighbor_radius+1, :]
                            frame_set = frame[t-neighbor_radius:t+neighbor_radius+1]
                        ######## tf-ti only
                        # if frame_set.shape[0] > 2:
                        #     tmp_set = tmp_set[0:2]
                        #     frame_set = frame_set[0:2]
                        ########
                        v, vn = 0, 0
                        for j in range(1, tmp_set.shape[0]):
                            dt = (frame_set[j]-frame_set[j-1])[0]
                            if dt < max_radius:
                                v += np.sqrt(np.sum(np.power(tmp_set[j]-tmp_set[j-1], 2)))/dt
                            vn += 1
                        if vn:
                            v = v/vn
                            vibration_v.append(v)
                        else:
                            vibration_v.append(padding)
                    if padding_index != []:
                        for i in padding_index:
                            p_frame = np.insert(p_frame, i, i+1)
                            vibration_v = np.insert(vibration_v, i, padding)
                    ########  plt velocity
                    # _data = vibration_v
                    # _frame = frame.copy()
                    # padding_index = np.argwhere(_data >= padding/2)
                    # _data = np.delete(_data, padding_index, axis=0)
                    # _frame = np.delete(_frame, padding_index, axis=0)
                    # plt.plot(_frame, _data)
                    # plt.title('v')
                    # plt.show()
                    ########
                    p_data = np.array(vibration_v)
                    p_frame = frame.copy()
                    padding_index = np.argwhere(p_data >= padding / 2)
                    padding_index = np.unique(padding_index[:, 0])
                    p_data = np.delete(p_data, padding_index, axis=0)
                    p_frame = np.delete(p_frame, padding_index, axis=0)
                    vibration_a = []
                    for t in range(p_frame.shape[0]):
                        if t <= neighbor_radius:
                            tmp_set = p_data[0:t + neighbor_radius + 1]
                            frame_set = p_frame[0:t + neighbor_radius + 1]
                        elif t >= p_data.shape[0] - neighbor_radius:
                            tmp_set = p_data[t - neighbor_radius:p_data.shape[0]]
                            frame_set = frame[t - neighbor_radius:p_data.shape[0]]
                        else:
                            tmp_set = p_data[t - neighbor_radius:t + neighbor_radius + 1]
                            frame_set = frame[t - neighbor_radius:t + neighbor_radius + 1]
                        ######## tf-ti only
                        # if frame_set.shape[0] > 2:
                        #     tmp_set = tmp_set[0:2]
                        #     frame_set = frame_set[0:2]
                        ########
                        a, an = 0, 0
                        for j in range(1, tmp_set.shape[0]):
                            dt = (frame_set[j] - frame_set[j - 1])[0]
                            if dt < max_radius:
                                a += np.sqrt(np.sum(np.power(tmp_set[j] - tmp_set[j - 1], 2))) / dt
                            an += 1
                        if an:
                            a = a / an
                            vibration_a.append(a)
                        else:
                            vibration_a.append(padding)
                    if padding_index != []:
                        for i in padding_index:
                            p_frame = np.insert(p_frame, i, i + 1)
                            vibration_a = np.insert(vibration_a, i, padding)
                    # ########  plt acc
                    # if p == 10:
                    #     _data = np.array(vibration_a)
                    #     _frame = frame.copy()
                    #     padding_index = np.argwhere(_data >= padding/2)
                    #     _data = np.delete(_data, padding_index, axis=0)
                    #     _frame = np.delete(_frame, padding_index, axis=0)
                    #     print('acc size', _data.shape)
                    #     plt.plot(_frame, _data*300*300)
                    #     plt.title('Acceleration')
                    #     plt.xlabel('frame')
                    #     plt.ylabel('m/s^2')
                    #     plt.show()
                    #     exit()
                    # ########
                    # interpolation
                    # print(acc_ratio_threshold)
                    p_data = np.array(vibration_a)
                    p_index = np.arange(0, p_data.shape[0])
                    padding_index = np.argwhere(p_data >= padding / 2)
                    if padding_index.shape[0]:
                        p_data = np.delete(p_data, padding_index, axis=0)
                        p_index = np.delete(p_index, padding_index, axis=0)
                    threshold = np.average(p_data)*acc_ratio_threshold
                    max_p = np.argwhere((p_data-threshold) > 0).shape[0]
                    index_set = []
                    index_loop = True
                    while index_loop:
                        p_data = vibration_a
                        if padding_index.shape[0]:
                            p_data = np.delete(p_data, padding_index, axis=0)
                        index = p_index[np.argwhere(p_data == np.max(p_data))[0][0]]
                        if index not in total_index:
                            total_index.append(index)
                        if len(index_set) < max_p:
                            for axis in range(data.shape[2]):
                                fitting_data = data[:, p, axis]
                                pi = 0 if index < fit_radius else index - fit_radius
                                pf = index + fit_radius if index + fit_radius < data.shape[0] else data.shape[0]
                                feature = np.argwhere(data[pi:pf + 1, p, axis] != padding).reshape(-1) + pi
                                X = feature - index
                                Y = fitting_data[feature]
                                if Y.shape[0] >= min_fitting_radius * 2 + 1:
                                    fitting_curve = np.poly1d(np.polyfit(X, Y, fit_order))
                                    data[index, p, axis] = fitting_curve(0)
                                vibration_a[index] = 0
                            if index not in index_set:
                                index_set.append(index)
                        else:
                            index_loop = False
                    if pre_index == index_set:
                        point_loop = False
                        for pad in index_set:
                            data[pad, p] = np.array([padding, padding, padding])
                        print('drop data: ', len(index_set), ', ', index_set)
                        print('modeify data: ', len(total_index), sorted(total_index))
                        print('')
                        ############
                        # if p == 10:
                        #     print('mission n = ', len(index_set))
                        #     _data = np.array(vibration_a)
                        #     _frame = frame.copy()
                        #     padding_index = np.argwhere(_data >= padding / 2)
                        #     _data = np.delete(_data, padding_index, axis=0)
                        #     _frame = np.delete(_frame, padding_index, axis=0)
                        #     print('acc size', _data.shape)
                        #     plt.plot(_frame, _data*300*300)
                        #     plt.title('Acceleration')
                        #     plt.xlabel('frame')
                        #     plt.ylabel('m/s^2')
                        #     plt.show()
                        ############
                    pre_index = index_set
        return frame, data

    def anomaly_detector_single_channel(self, frame, data, max_radius=2, fit_radius=1, fit_order=1, padding=1e10, acc_ratio_threshold=2, fit_point=[10], fit_axis=[0, 1, 2]):
        frame = frame.reshape(-1, 1)
        neighbor_radius = 1
        min_fitting_radius = 1
        # vibration velocity
        for axis in fit_axis:
            for p in range(data.shape[1]):
                print('... processing the point ', str(p),', channel: ', axis)
                pre_index = []
                point_loop = True
                total_index = []
                if p in fit_point:
                    while point_loop:
                        p_data = data[:, p, axis]
                        p_frame = frame.copy()
                        padding_index = np.argwhere(p_data >= padding / 2)
                        padding_index = np.unique(padding_index[:, 0])
                        p_data = np.delete(p_data, padding_index, axis=0)
                        p_frame = np.delete(p_frame, padding_index, axis=0)
                        vibration_v, vibration_a = [], []
                        for t in range(p_frame.shape[0]):
                            if t <= neighbor_radius:
                                tmp_set = p_data[0:t+neighbor_radius+1]
                                frame_set = p_frame[0:t+neighbor_radius+1]
                            elif t >= p_data.shape[0]-neighbor_radius:
                                tmp_set = p_data[t - neighbor_radius:p_data.shape[0]]
                                frame_set = frame[t - neighbor_radius:p_data.shape[0]]
                            else:
                                tmp_set = p_data[t-neighbor_radius:t+neighbor_radius+1]
                                frame_set = frame[t-neighbor_radius:t+neighbor_radius+1]
                            ######## tf-ti only
                            # if frame_set.shape[0] > 2:
                            #     tmp_set = tmp_set[0:2]
                            #     frame_set = frame_set[0:2]
                            ########
                            v, vn = 0, 0
                            for j in range(1, tmp_set.shape[0]):
                                dt = (frame_set[j]-frame_set[j-1])[0]
                                if dt < max_radius:
                                    v += np.sqrt(np.sum(np.power(tmp_set[j]-tmp_set[j-1], 2)))/dt
                                vn += 1
                            if vn:
                                v = v/vn
                                vibration_v.append(v)
                            else:
                                vibration_v.append(padding)
                        if padding_index != []:
                            for i in padding_index:
                                p_frame = np.insert(p_frame, i, i+1)
                                vibration_v = np.insert(vibration_v, i, padding)
                        ########  plt velocity
                        # _data = np.array(vibration_v)
                        # _frame = frame.copy()
                        # padding_index = np.argwhere(_data >= padding/2)
                        # _data = np.delete(_data, padding_index, axis=0)
                        # _frame = np.delete(_frame, padding_index, axis=0)
                        # plt.plot(_frame, _data)
                        # plt.title('v')
                        # plt.show()
                        # df = pd.DataFrame(data=_data)
                        # df.to_csv('./origin_v.csv')
                        ########
                        p_data = np.array(vibration_v)
                        p_frame = frame.copy()
                        padding_index = np.argwhere(p_data >= padding / 2)
                        padding_index = np.unique(padding_index[:, 0])
                        p_data = np.delete(p_data, padding_index, axis=0)
                        p_frame = np.delete(p_frame, padding_index, axis=0)
                        vibration_a = []
                        for t in range(p_frame.shape[0]):
                            if t <= neighbor_radius:
                                tmp_set = p_data[0:t + neighbor_radius + 1]
                                frame_set = p_frame[0:t + neighbor_radius + 1]
                            elif t >= p_data.shape[0] - neighbor_radius:
                                tmp_set = p_data[t - neighbor_radius:p_data.shape[0]]
                                frame_set = frame[t - neighbor_radius:p_data.shape[0]]
                            else:
                                tmp_set = p_data[t - neighbor_radius:t + neighbor_radius + 1]
                                frame_set = frame[t - neighbor_radius:t + neighbor_radius + 1]
                            ######## tf-ti only
                            # if frame_set.shape[0] > 2:
                            #     tmp_set = tmp_set[0:2]
                            #     frame_set = frame_set[0:2]
                            ########
                            a, an = 0, 0
                            for j in range(1, tmp_set.shape[0]):
                                dt = (frame_set[j] - frame_set[j - 1])[0]
                                if dt < max_radius:
                                    a += np.sqrt(np.sum(np.power(tmp_set[j] - tmp_set[j - 1], 2))) / dt
                                an += 1
                            if an:
                                a = a / an
                                vibration_a.append(a)
                            else:
                                vibration_a.append(padding)
                        if padding_index != []:
                            for i in padding_index:
                                p_frame = np.insert(p_frame, i, i + 1)
                                vibration_a = np.insert(vibration_a, i, padding)
                        # ########  plt acc
                        # _data = np.array(vibration_a)
                        # _frame = frame.copy()
                        # padding_index = np.argwhere(_data >= padding/2)
                        # _data = np.delete(_data, padding_index, axis=0)
                        # _frame = np.delete(_frame, padding_index, axis=0)
                        # print(os.getcwd())
                        # print('acc size', _data.shape)
                        # plt.plot(_frame, _data)
                        # plt.title('Acceleration')
                        # plt.xlabel('frame')
                        # plt.ylabel('m/s^2')
                        # plt.show()
                        # df = pd.DataFrame(data=_data)
                        # df.to_csv('./origin_a.csv')
                        # ########
                        # interpolation
                        # print(acc_ratio_threshold)
                        p_data = np.array(vibration_a)
                        p_index = np.arange(0, p_data.shape[0])
                        padding_index = np.argwhere(p_data >= padding / 2)
                        if padding_index.shape[0]:
                            p_data = np.delete(p_data, padding_index, axis=0)
                            p_index = np.delete(p_index, padding_index, axis=0)
                        threshold = np.average(p_data)*acc_ratio_threshold
                        max_p = np.argwhere((p_data-threshold) > 0).shape[0]
                        index_set = []
                        index_loop = True
                        while index_loop:
                            p_data = vibration_a
                            if padding_index.shape[0]:
                                p_data = np.delete(p_data, padding_index, axis=0)
                            index = p_index[np.argwhere(p_data == np.max(p_data))[0][0]]
                            if index not in total_index:
                                total_index.append(index)
                            if len(index_set) < max_p:
                                fitting_data = data[:, p, axis]
                                pi = 0 if index < fit_radius else index - fit_radius
                                pf = index + fit_radius if index + fit_radius < data.shape[0] else data.shape[0]
                                feature = np.argwhere(data[pi:pf + 1, p, axis] != padding).reshape(-1) + pi
                                X = feature - index
                                Y = fitting_data[feature]
                                if Y.shape[0] >= min_fitting_radius * 2 + 1:
                                    fitting_curve = np.poly1d(np.polyfit(X, Y, fit_order))
                                    data[index, p, axis] = fitting_curve(0)
                                vibration_a[index] = 0
                                if index not in index_set:
                                    index_set.append(index)
                            else:
                                index_loop = False
                        if pre_index == index_set:
                            point_loop = False
                            for pad in index_set:
                                data[pad, p] = np.array([padding, padding, padding])
                            print('drop data: ', len(index_set), ', ', index_set)
                            print('modeify data: ', len(total_index), sorted(total_index))
                            print('')
                            ############
                            # print('mission n = ', len(index_set))
                            # _data = np.array(vibration_v)
                            # _frame = frame.copy()
                            # padding_index = np.argwhere(_data >= padding / 2)
                            # _data = np.delete(_data, padding_index, axis=0)
                            # _frame = np.delete(_frame, padding_index, axis=0)
                            # print('acc size', _data.shape)
                            # plt.plot(_frame, _data)
                            # plt.title('Velocity')
                            # plt.xlabel('frame')
                            # plt.ylabel('m/s^2')
                            # plt.show()
                            # df = pd.DataFrame(data=_data)
                            # df.to_csv('./post_v.csv')
                            # _data = np.array(vibration_a)
                            # _frame = frame.copy()
                            # padding_index = np.argwhere(_data >= padding / 2)
                            # _data = np.delete(_data, padding_index, axis=0)
                            # _frame = np.delete(_frame, padding_index, axis=0)
                            # print('acc size', _data.shape)
                            # plt.plot(_frame, _data)
                            # plt.title('Acceleration')
                            # plt.xlabel('frame')
                            # plt.ylabel('m/s^2')
                            # plt.show()
                            # df = pd.DataFrame(data=_data)
                            # df.to_csv('./post_a.csv')
                            ############
                        pre_index = index_set
        return frame, data

    def smooth_curve(self, frame, data, padding=1e10, type='junyu', fit_point=[10]):

        def find_saddle_point(frame, data):
            saddle_set = []
            pre_value = ''
            saddle_set.append(0)
            for i in range(len(data)-1):
                value = data[i].copy()
                if pre_value != '':
                    future_value = data[i+1].copy()
                    if value > pre_value and value > future_value:
                        saddle_set.append(i)
                    elif value < pre_value and value < future_value:
                        saddle_set.append(i)
                pre_value = value.copy()
            saddle_set.append(len(data)-1)
            # print(saddle_set)
            saddle_set = sorted(list(set(saddle_set)))
            # plt.plot(data)
            # plt.scatter(saddle_set, data[saddle_set])
            # plt.show()
            return np.array(saddle_set)

        def guiding_interpolation(frame, data, guide=[1], order=4):
            min_fitting_radius = 10
            new_data = data.copy()
            for i in range(guide.shape[0]-1):
                if frame[guide[i+1]]-frame[guide[i]] >= min_fitting_radius:
                    X = frame[guide[i]:guide[i+1]]
                    Y = data[guide[i]:guide[i + 1]]
                    fitting_curve = np.poly1d(np.polyfit(X, Y, order))
                    new_data[guide[i]:guide[i + 1]] = fitting_curve(X)
            return frame, new_data

        # pre_fit
        pre_fit_order = [1, 1, 1]
        pre_fit_radius = [10, 10, 10]
        self.data = data.copy()
        frame = frame.reshape(-1)
        pre_fit_data, pre_fit_radius_inf = self.interpolation_3d(radius_type='normal', adaptive_ratio=(2, 6), order=pre_fit_order, fit_radius=pre_fit_radius, fit_point=fit_point, padding=1e10)
        # maybe need to make padding to real value
        for point in range(data.shape[1]):
            if point == 10:
                for axis in range(data.shape[2]):
                    X = frame.copy()
                    Y = pre_fit_data[:, point, axis].copy().reshape(-1)
                    padding_index = np.argwhere(Y >= padding / 2)
                    X = np.delete(X, padding_index, axis=0)
                    Y = np.delete(Y, padding_index, axis=0)
                    # find saddle point
                    saddle_set = find_saddle_point(X, Y)
                    pre_X, pre_Y = guiding_interpolation(frame=X, data=Y, guide=saddle_set)
                    # plt.plot(pre_X, pre_Y)
                    # plt.show()
                    # re-padding
                    for pad in padding_index:
                        pre_X = np.insert(pre_X, pad[0], pad[0])
                        pre_Y = np.insert(pre_Y, pad[0], padding)
                    data[:, point, axis] = pre_Y.copy()
                    # # Plot
                    # X = frame.copy()
                    # Y = self.data[:, point, axis]
                    # padding_index = np.argwhere(Y >= padding/2)
                    # X = np.delete(X, padding_index, axis=0)
                    # Y = np.delete(Y, padding_index, axis=0)
                    # padding_index = np.argwhere(pre_Y >= padding / 2)
                    # pre_X = np.delete(pre_X, padding_index, axis=0)
                    # pre_Y = np.delete(pre_Y, padding_index, axis=0)
                    # plt.plot(X, Y)
                    # plt.plot(pre_X, pre_Y)
                    # plt.scatter(saddle_set, pre_Y[saddle_set])
                    # plt.show()
        return frame, data

    def padding2linear(self, data, padding=1e10):
        padding_index = np.argwhere(data >= padding / 2).reshape(-1)
        for pad in range(padding_index.shape[0]):
            boundary = False
            # forward
            count = padding_index[pad]
            while True:
                if count not in padding_index:
                    forward = count
                    break
                elif count == data.shape[0] - 1:
                    boundary = True
                    while True:
                        if count not in padding_index:
                            data[padding_index[pad]] = data[count].copy()
                            break
                        elif count == 0:
                            data[padding_index[pad]] = 0
                            break
                        count -= 1
                else:
                    count += 1
            # backword
            count = padding_index[pad]
            while True:
                if count not in padding_index:
                    backward = count
                    break
                elif count == 0:
                    boundary = True
                    while True:
                        if count not in padding_index:
                            data[padding_index[pad]] = data[count].copy()
                            break
                        elif count == data.shape[0] - 1:
                            data[padding_index[pad]] = 0
                            break
                        count += 1
                else:
                    count -= 1
            if boundary == False:
                pre_y = data[backward] + (padding_index[pad] - backward) * (data[forward] - data[backward]) / (
                            forward - backward)
                data[padding_index[pad]] = pre_y
        # plt.plot(data)
        # plt.show()
        # exit()
        return data

    def expaend_mirror(self, data):
        data = np.concatenate((np.flip(data), data, np.flip(data), data))
        return data

    def fft_denoise(self, frame, data, fit_axis=0, fft_sport_ratio=0.8, fft_noise_ratio=0.9, saddle_dmin=20, fft_threshold_ratio=0.6, pre_fit_order=1, pre_fit_radius=20, padding=1e10, fit_point=[10], padding_radius='mirror', segmentation_method='saddle'):

        def expaend_radius(data, padding_radius):
            hand = np.linspace(0, data[0], num=padding_radius)
            lag = np.linspace(data[-1], 0, num=padding_radius)
            data = np.hstack((hand, data, lag))
            return data

        def expaend_mirror(data):
            data = np.concatenate((np.flip(data), data, np.flip(data), data))
            return data

        def find_saddle_point(data, dmin=1):
            output_set = np.arange(data.shape[0]).tolist()
            while True:
                saddle_set = []
                pre_value = ''
                saddle_set.append(0)
                for i in range(len(output_set)-1):
                    value = data[output_set[i]].copy()
                    if pre_value != '':
                        future_value = data[output_set[i+1]].copy()
                        if value > pre_value and value > future_value:
                            saddle_set.append(output_set[i])
                        elif value < pre_value and value < future_value:
                            saddle_set.append(output_set[i])
                    pre_value = value.copy()
                saddle_set.append(len(data)-1)
                saddle_set = sorted(list(set(saddle_set)))
                if output_set == saddle_set:
                    break
                # noise saddle point
                saddle_set = np.array(saddle_set)
                noise_index = np.argwhere((saddle_set[1:]-saddle_set[:-1]) < dmin)
                saddle_set = np.delete(saddle_set, noise_index, axis=0).tolist()
                output_set = saddle_set
            # plt.plot(data)
            # plt.scatter(saddle_set, data[saddle_set])
            # plt.show()
            # print(saddle_set)
            saddle_set = np.array(saddle_set)
            return saddle_set

        def saddle2middle(data):
            middle_set = []
            middle_set.append(np.min(data))
            middle_set.append(np.max(data))
            for i in range(1, data.shape[0]-2):
                middle_set.append(data[i]+int((data[i+1]-data[i])/2))
            middle_set = np.sort(np.array(middle_set))
            return middle_set

        def plot_saddle_point(data, saddle_set, fft_threshold_ratio, figsize=(24, 5)):
            average_saddle = np.average(np.abs(Y_set[saddle_set][1:] - Y_set[saddle_set][:-1]) / (saddle_set[1:] - saddle_set[:-1]))
            plt.rcParams['figure.figsize'] = figsize
            plt.scatter(saddle_set, data[saddle_set], c='r', s=200)
            for i in range(saddle_set.shape[0] - 1):
                Y = data[saddle_set[i]:saddle_set[i + 1] + 1]
                X = np.arange(saddle_set[i], saddle_set[i + 1] + 1)
                if abs(data[saddle_set][i + 1] - data[saddle_set][i]) / (saddle_set[i + 1] - saddle_set[i]) > average_saddle * fft_threshold_ratio:
                    plt.plot(X, Y, c='b')
                else:
                    plt.plot(X, Y, c='g')
            plt.show()


        frame = frame.reshape(-1)
        for point in fit_point:
            for axis in fit_axis:
                Y_set = data[:, point, axis].copy().reshape(-1)
                Y_set = self.padding2linear(Y_set, padding=padding)
                # pre_fit
                self.data = Y_set.copy()
                frame = frame.reshape(-1)
                # saddle point
                if saddle_dmin is not None:
                    pre_fit_data = self.interpolation_1d(radius_type='normal', adaptive_ratio=(2, 6), order=pre_fit_order, fit_radius=pre_fit_radius, padding=1e10)
                    saddle_set = find_saddle_point(pre_fit_data, dmin=saddle_dmin)
                else:
                    saddle_set = np.array([0, frame.shape[0]-1])
                average_saddle = np.average(np.abs(Y_set[saddle_set][1:]-Y_set[saddle_set][:-1])/(saddle_set[1:]-saddle_set[:-1]))
                if segmentation_method == 'middle':
                    saddle_set = saddle2middle(saddle_set)
                # plot_saddle_point(Y_set, saddle_set, fft_threshold_ratio)
                for i in range(saddle_set.shape[0]-1):
                    Y = Y_set[saddle_set[i]:saddle_set[i+1]+1]
                    if padding_radius == 'mirror':
                        Y = self.expaend_mirror(Y)
                        if abs(Y_set[saddle_set][i+1]-Y_set[saddle_set][i])/(saddle_set[i+1]-saddle_set[i]) > average_saddle*fft_threshold_ratio:
                            f_ratio = fft_sport_ratio
                        else:
                            f_ratio = fft_noise_ratio
                    else:
                        Y = expaend_radius(Y, padding_radius)
                    threshold_filter = np.arange((1 - f_ratio) * (Y.shape[0] / 2), Y.shape[0] / 2 + (f_ratio) * (Y.shape[0] / 2)).astype(np.int)
                    Y_fft = fft(Y)
                    # df = pd.DataFrame(data=np.array(Y_fft.real).copy())
                    # df.to_csv('./origin_FFT.csv')
                    for f in range(len(Y_fft)):
                        if f in threshold_filter:
                            Y_fft[f] = 0
                    # df = pd.DataFrame(data=np.array(Y_fft.real).copy())
                    # df.to_csv('./post_FFT.csv')
                    Y = ifft(Y_fft).real
                    if padding_radius == 'mirror':
                        Y = Y[int(Y.shape[0]/4):int(Y.shape[0]*2/4)].copy()
                    else:
                        Y = Y[padding_radius:-padding_radius].copy()
                    # plt.plot(Y)
                    # plt.show()
                    data[saddle_set[i]:saddle_set[i+1]+1, point, axis] = Y
                # a = expaend_mirror(data[:, point, axis])
                # plt.plot(a)
                # plt.show()
                # exit()
        return frame, data

    def mapping_data(self, dataset, acc_ratio_threshold=5, fit_point=[10]):

        def vibration(data):
            pre_data = data[2:, :, :]
            now_data = data[1:-1, :, :]
            next_data = data[:-2, :, :]
            v1 = np.sqrt(np.sum(np.power(pre_data - now_data, 2), axis=2))
            v2 = np.sqrt(np.sum(np.power(next_data - now_data, 2), axis=2))
            v = (v1+v2)/2
            # boundary
            pre_data = data[1:, :, :]
            now_data = data[:-1, :, :]
            b = np.sqrt(np.sum(np.power(pre_data - now_data, 2), axis=2))
            v = np.insert(v, 0, b[0, :].reshape(1, -1), axis=0)
            v = np.insert(v, -1, b[-1, :].reshape(1, -1), axis=0)
            return v

        def find_segmentation(data0, data1):
            pre = 0
            anormaly = 0
            segmentation_index, data_type = [], []
            data = np.sort(np.hstack((data0, data1)))
            init = np.min(data)
            if init in data0:
                sym = 1
            elif init in data1:
                sym = 0
            for i in range(data.shape[0]):
                now = anormaly+int((data[i]-anormaly)/2)
                if data[i] in data0:
                    if sym == 1:
                        segmentation_index.append([pre, now])
                        data_type.append(sym)
                        pre = now
                        sym = 0
                elif data[i] in data1:
                    if sym == 0:
                        segmentation_index.append([pre, now])
                        data_type.append(sym)
                        pre = now
                        sym = 1
                anormaly = data[i]
            segmentation_index, data_type = np.array(segmentation_index)[1:], np.array(data_type)[1:]
            data_type = np.abs(data_type-1)
            return segmentation_index, data_type

        dataset = np.array(dataset)
        acc = []
        for data in dataset:
            v = vibration(data)
            a = vibration(v.reshape(v.shape[0], v.shape[1], 1))
            acc.append(a)
        acc = np.array(acc)
        mapping_data = dataset[0].copy()
        mapping_acc = acc[0].copy()
        for point in fit_point:
            # plt.plot(acc[0, :, point])
            # plt.ylim(0, 0.007)
            # plt.show()
            # plt.plot(acc[1, :, point])
            # plt.ylim(0, 0.007)
            # plt.show()
            # exit()

            threshold = np.average(mapping_acc[:, point])*acc_ratio_threshold
            mapping_anormaly = np.argwhere(mapping_acc[:, point] > threshold).reshape(-1)
            for i in range(1, dataset.shape[0]):
                candicate_acc = acc[i].copy()
                threshold = np.average(candicate_acc[:, point]) * acc_ratio_threshold
                candicate_anormaly = np.argwhere(candicate_acc[:, point] > threshold).reshape(-1)
                # segmentation
                if mapping_anormaly.shape[0] != 0 and candicate_anormaly.shape[0] != 0:
                    area, index = find_segmentation(mapping_anormaly, candicate_anormaly)
                    if area.shape[0] != 0:
                        area = np.insert(area, 0, [0, area[0][0]], axis=0)
                        area = np.insert(area, area.shape[0], [area[-1][-1], acc.shape[1]-1], axis=0)
                        index = np.insert(index, 0, abs(index[0]-1))
                        index = np.insert(index, len(index), abs(index[-1]-1))
                        for j in range(area.shape[0]):
                            mapping_data[area[j][0]:area[j][1], point, :] = dataset[index[j], area[j][0]:area[j][1], point, :]
        return mapping_data

    def fft_model(self, frame, input_data, output_data, preprocessing_model=None, anormaly_ratio=1.5, fps=300, acc_ratio=5, fft_ratio=[0.96, 0.96], saddle_dmin=20, loop_n=1, axis_set=[0, 1, 2], fit_point=[10], figsize=(24, 13), plt_max_velocity=80, segmentation_method='middle', UI=False):
        # preprocessing
        if preprocessing_model == 'all':
            frame, preprocessing_data = self.anomaly_detector_all_channel(frame=frame, data=input_data.copy(), acc_ratio_threshold=anormaly_ratio, fit_point=fit_point)
        elif preprocessing_model == 'single':
            frame, preprocessing_data = self.anomaly_detector_single_channel(frame=frame, data=input_data.copy(), acc_ratio_threshold=anormaly_ratio, fit_point=fit_point, fit_axis=axis_set)
        elif preprocessing_model == 'junyu':
            pass
            # frame, preprocessing_data = post_processing('').smooth_curve(frame=frame, data=input_data.copy(), type='junyu', fit_point=fit_point)
        elif preprocessing_model == 'interpolation':
            pass
            # preprocessing_data, radius_inf = post_processing(input_data.copy()).interpolation_3d(radius_type='normal', adaptive_ratio=(2, 6), order=order, fit_radius=fit_radius, fit_point=fit_point, padding=1e10)
        elif preprocessing_model == 'spline':
            pass
            # preprocessing_data, radius_inf = post_processing(input_data.copy()).interpolation_3d(radius_type='spline', adaptive_ratio=(2, 6), fit_radius=fit_radius, fit_point=fit_point, padding=1e10)
        else:
            preprocessing_data = input_data.copy()

        # FFT model
        data_mapping = preprocessing_data.copy()
        data_fft_saddle = preprocessing_data.copy()
        data_fft_middle = preprocessing_data.copy()
        for i in range(loop_n):
            # pass
            frame, data_mapping = self.fft_denoise(frame=frame, data=data_mapping.copy(), fit_point=fit_point, fit_axis=axis_set, fft_sport_ratio=fft_ratio[0], fft_noise_ratio=fft_ratio[1], saddle_dmin=saddle_dmin, segmentation_method=segmentation_method)  # 45, 725
            # frame, data_fft_saddle = self.fft_denoise(frame=frame, data=data_mapping.copy(), fit_point=fit_point, fit_axis=axis_set, fft_sport_ratio=fft_ratio[0], fft_noise_ratio=fft_ratio[1], saddle_dmin=saddle_dmin, segmentation_method='saddle')  # 45, 725
            # frame, data_fft_middle = self.fft_denoise(frame=frame, data=data_mapping.copy(), fit_point=fit_point, fit_axis=axis_set, fft_sport_ratio=fft_ratio[0], fft_noise_ratio=fft_ratio[1], saddle_dmin=saddle_dmin, segmentation_method='middle')  # 45, 725
            # data_mapping = self.mapping_data(dataset=[data_fft_saddle, data_fft_middle], acc_ratio_threshold=acc_ratio, fit_point=fit_point)

            # frame, data_fft_saddle = self.fft_denoise(frame=frame, data=data_mapping.copy(), fit_point=fit_point, fit_axis=axis_set, fft_sport_ratio=fft_ratio[0], fft_noise_ratio=fft_ratio[1], saddle_dmin=saddle_dmin, segmentation_method='saddle')  # 45, 725
            # data_mapping = self.mapping_data(dataset=[data_fft_saddle, data_mapping], acc_ratio_threshold=acc_ratio, fit_point=fit_point)
            #
            # frame, data_fft_middle = self.fft_denoise(frame=frame, data=data_mapping.copy(), fit_point=fit_point, fit_axis=axis_set, fft_sport_ratio=fft_ratio[0], fft_noise_ratio=fft_ratio[1], saddle_dmin=saddle_dmin, segmentation_method='middle')  # 45, 725
            # data_mapping = self.mapping_data(dataset=[data_mapping, data_fft_middle], acc_ratio_threshold=acc_ratio, fit_point=fit_point)
        # frame, data_mapping = self.fft_denoise(frame=frame, data=data_mapping.copy(), fit_point=fit_point, fit_axis=axis_set, fft_sport_ratio=0.8, fft_noise_ratio=0.8, saddle_dmin=300, segmentation_method='saddle')  # 45, 725
        # update to output_data
        result_data = data_mapping.copy()
        for p in fit_point:
            for axis in axis_set:
                output_data[:, p, axis] = result_data[:, p, axis].copy()

        if UI is False:
            # plot
            frame, velocity = joint_analysis().velocity_analysis(frame=frame, data=output_data.copy(), fps=fps, output_path='../20200318/velocity.csv')
            # frame, velocity_fft_saddle = joint_analysis().velocity_analysis(frame=frame, data=data_fft_saddle.copy(), fps=fps, output_path='../20200318/')
            # frame, velocity_fft_middle = joint_analysis().velocity_analysis(frame=frame, data=data_fft_middle.copy(), fps=fps, output_path='../20200318/')
            # frame, velocity_mapping = joint_analysis().velocity_analysis(frame=frame, data=data_mapping.copy(), fps=fps, output_path='../20200318/')
            # frame, velocity_preprocessing = joint_analysis().velocity_analysis(frame=frame, data=preprocessing_data.copy(), fps=fps, output_path='../20200318/')
            frame, velocity_output = joint_analysis().velocity_analysis(frame=frame, data=output_data.copy(), fps=fps, output_path='../20200318/velocity.csv')
            tool().plot_coord(frame, [input_data, output_data], index=fit_point, axis_set=axis_set, figsize=figsize, plt_scale_high=1.2)
            # tool().plot_velocity(frame, [velocity_fft_saddle, velocity_fft_middle], index=fit_point, plt_scale_range=tuple(np.linspace(0, plt_max_velocity, 10)), figsize=figsize)
            # tool().plot_velocity(frame, [velocity_mapping], index=fit_point, plt_scale_range=tuple(np.linspace(0, plt_max_velocity, 10)), figsize=figsize)
            tool().plot_velocity(frame, [velocity_output], index=fit_point, plt_y_range=tuple(np.linspace(0, plt_max_velocity, 10)), figsize=figsize)
        return output_data

    def interpolation_model(self, data, output_data, order=[1], fit_point=[10], fit_axis=[0], padding=1e10, inter_type='poly', fps=300, plt_max_velocity=15, figsize=None, fit_area=None, fit_data=None, mask_area=[], preprocessing_model=None, anormaly_ratio=1.5, UI=False):
        frame = np.arange(0, data.shape[0])
        # preprocessing
        if preprocessing_model == 'all':
            frame, preprocessing_data = self.anomaly_detector_all_channel(frame=frame, data=data.copy(), acc_ratio_threshold=anormaly_ratio, fit_point=fit_point)
        elif preprocessing_model == 'single':
            frame, preprocessing_data = self.anomaly_detector_single_channel(frame=frame, data=data.copy(), acc_ratio_threshold=anormaly_ratio, fit_point=fit_point, fit_axis=fit_axis)
        else:
            preprocessing_data = data.copy()
        # mask data
        mask_data = preprocessing_data.copy()
        if mask_area is not []:
            for mask in mask_area:
                mask_data[mask[0]:mask[1], :, :] = 1e10
        # prediction
        min_fitting_radius = 4
        predict_data = output_data.copy()
        for point in fit_point:
            for axis in fit_axis:
                if fit_data:
                    segment_count = 0
                    for area in fit_area:
                        Y = mask_data[area[0]:area[1], point, axis].copy()
                        Y = self.padding2linear(Y, padding=padding)
                        new_X = np.arange(0, Y.shape[0])
                        padding_index = np.argwhere(Y >= padding/2)
                        X = np.delete(new_X, padding_index, axis=0).copy()
                        Y = np.delete(Y, padding_index, axis=0).copy()
                        if X.shape[0] > min_fitting_radius:
                            if inter_type == 'poly':
                                fitting_data = np.polyval(np.polyfit(X, Y, order[segment_count]), new_X)
                            if inter_type == 'spline':
                                fitting_data = make_interp_spline(X, Y)(new_X)
                                # print(power_smooth.shape)
                                # print(power_data.shape, power_index.shape)
                            # if segment_count != 0 and fit_area[segment_count-1][1] > fit_area[segment_count][0]:
                            #     data_length = fit_area[segment_count-1][1]-fit_area[segment_count][0]
                            #     predict_data[fit_area[segment_count-1][1]:area[1], point, axis] = fitting_data[data_length:]
                            # else:
                            #     predict_data[area[0]:area[1], point, axis] = fitting_data
                            predict_data[fit_data[segment_count][0]:fit_data[segment_count][1], point, axis] = fitting_data[fit_data[segment_count][0]-fit_area[segment_count][0]:fit_data[segment_count][1]-fit_area[segment_count][0]]
                            # plt.plot(X, Y)
                        segment_count += 1
                        # plt.plot(frame, fitting_data)
                        # plt.title(str(point)+' / '+str(axis))
                        # plot
                else:
                    Y = mask_data[:, point, axis].copy()
                    Y = self.padding2linear(Y, padding=padding)
                    new_X = np.arange(0, Y.shape[0])
                    padding_index = np.argwhere(Y >= padding / 2)
                    X = np.delete(new_X, padding_index, axis=0).copy()
                    Y = np.delete(Y, padding_index, axis=0).copy()
                    if X.shape[0] > min_fitting_radius:
                        if inter_type == 'poly':
                            fitting_data = np.polyval(np.polyfit(X, Y, order[0]), new_X)
                        if inter_type == 'spline':
                            fitting_data = make_interp_spline(X, Y)(new_X)
                            # print(power_smooth.shape)
                            # print(power_data.shape, power_index.shape)
                        predict_data[:, point, axis] = fitting_data
        if UI is False:
            # frame, velocity = joint_analysis().velocity_analysis(frame=frame, data=predict_data.copy(), fps=fps, output_path='../20200318/')
            # frame, velocity_output = joint_analysis().velocity_analysis(frame=frame, data=predict_data.copy(), fps=fps, output_path='../20200318/')
            tool().plot_coord(frame, [data, predict_data], index=fit_point, figsize=figsize, plt_scale_high=1.2, axis_set=fit_axis)
            # tool().plot_velocity(frame, [velocity_output], index=fit_point, plt_scale_range=tuple(np.linspace(0, plt_max_velocity, 10)), figsize=figsize)
        return predict_data

class coord2vicon():

    def __init__(self):  # Run it once
        pass

    def calculate_angle(self, x, y):
        x, y = np.array(x), np.array(y)
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        cos_angle = x.dot(y)/(Lx*Ly)
        rad = np.arccos(cos_angle)
        angle = rad*360/2/np.pi
        print(cos_angle, rad, angle)
        return rad, angle

    def get_COM(self, data, left_hip_index=11, right_hip_index=12):
        COM = data[:, left_hip_index, :]+(data[:, right_hip_index, :]-data[:, left_hip_index, :])/2
        return COM

    def normailize_vector(self, data, unit_len=0.1):
        # normalized
        line_len = np.sqrt(np.sum(np.power(data, 2), axis=2))[:, :, np.newaxis]
        line_len = np.repeat(line_len, data.shape[2], axis=2)
        data = data * unit_len / line_len
        return data

    def normal_vector(self, first_vector, second_vector, ):
        # https://blog.csdn.net/qq2399431200/article/details/84314987
        new_vector = first_vector.copy()
        new_vector[:, :, 0] = first_vector[:, :, 1]*second_vector[:, :, 2]-second_vector[:, :, 1]*first_vector[:, :, 2]
        new_vector[:, :, 1] = -(first_vector[:, :, 0] * second_vector[:, :, 2] - second_vector[:, :, 0] * first_vector[:, :, 2])
        new_vector[:, :, 2] = (first_vector[:, :, 0] * second_vector[:, :, 1] - second_vector[:, :, 0] * first_vector[:, :, 1])
        return new_vector

    def vector2data(self, data, vector):
        return data+vector

    def get_vicon_axis(self, data):
        pre_index = [17, 5, 17, 6, 12, 11, 11, 12, 5]
        now_index = [5, 7, 6, 8, 11, 13, 12, 14, 17]
        next_index = [7, 9, 8, 10, 13, 15, 14, 16, 6]
        unit_len = 0.1
        index_point = np.repeat(np.array(now_index)[np.newaxis, :, np.newaxis, np.newaxis], data.shape[0], axis=0)
        # first
        first_vector = self.normailize_vector(data[:, now_index, :]-data[:, pre_index, :], unit_len=unit_len)
        first_point = (data[:, now_index, :].copy()+first_vector)[:, :, np.newaxis, :]
        first_point = np.concatenate((index_point, first_point), axis=3)
        # second
        support_vector = data[:, now_index, :]-data[:, next_index, :]
        second_vector = self.normailize_vector(self.normal_vector(first_vector, support_vector), unit_len=unit_len)
        second_point = (data[:, now_index, :].copy()+second_vector)[:, :, np.newaxis, :]
        second_point = np.concatenate((index_point, second_point), axis=3)
        # third
        third_vector = self.normal_vector(second_vector, first_vector)
        third_vector = self.normailize_vector(third_vector, unit_len=unit_len)
        third_point = (data[:, now_index, :].copy()+third_vector)[:, :, np.newaxis, :]
        third_point = np.concatenate((index_point, third_point), axis=3)
        vicon_axis = np.concatenate((first_point, second_point, third_point), axis=2)
        # self.calculate_angle(first_vector[0, 0, :], second_vector[0, 0, :])
        # self.calculate_angle(first_vector[0, 0, :], third_vector[0, 0, :])
        # self.calculate_angle(third_vector[0, 0, :], second_vector[0, 0, :])
        return vicon_axis

    def get_angle(self, data, pre_index=6, now_index=8, next_index=10):
        a_vector = data[:, pre_index, :] - data[:, now_index, :]
        b_vector = data[:, next_index, :] - data[:, now_index, :]
        La = np.sqrt(np.dot(a_vector, a_vector.T).diagonal())
        Lb = np.sqrt(np.dot(b_vector, b_vector.T).diagonal())
        cos_angle = np.dot(a_vector, b_vector.T).diagonal()/(La * Lb)
        rad = np.arccos(cos_angle)
        angle = rad * 360 / 2 / np.pi
        return angle

    def run(self, frame, data, video_fps=80):
        COM_data = self.get_COM(data)
        vicon_axis = self.get_vicon_axis(data)
        angle_right_elbow = 180-self.get_angle(data, pre_index=6, now_index=8, next_index=10)
        # angle_right_shoulder = self.get_angle(data, pre_index=17, now_index=6, next_index=8)-90
        # angle_left_hip = self.get_angle(data, pre_index=12, now_index=11, next_index=13) - 90
        angle_left_knee = 180-self.get_angle(data, pre_index=11, now_index=13, next_index=15)

        print(vicon_axis.shape)
        # plot
        plt.rcParams['figure.figsize'] = (14, 7)
        plt.ylabel('angle')
        plt.xlabel('frame')
        plt.plot(angle_right_elbow, c='black', label='angle_right_elbow')
        # plt.plot(angle_right_shoulder, label='angle_right_shoulder')
        # plt.plot(angle_left_hip, c='gray', label='angle_left_hip')
        plt.plot(angle_left_knee, label='angle_left_knee')
        plt.legend(loc='upper right')
        # plt.show()

        # view = [90, 80]
        # tool().Plot3D(frame_set=frame.reshape(-1), data=data, xlim=[-2.2, 0.8, 0.5], ylim=[33, 36, 0.5], zlim=[-1.2, 1.8, 0.5], view=view, path='../20200318/acc1/', fps=video_fps, COM=COM_data, vicon_axis=vicon_axis)


# import torch
# print(torch.cuda.device_count())
