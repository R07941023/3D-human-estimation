import cv2
import numpy as np
import math
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from os.path import isfile, isdir, join
import csv
from cv2.ximgproc import *
import matplotlib.pyplot as plt
import multiprocessing as mp
import shutil
import time
from fn import *
from multiprocessing import Process
from multiprocessing import Queue as pQueue

class calibration(object):
    def __init__(self, mypath):  # Run it once
        self.mypath = mypath

    def drawPnP(self, img, imgpts, roll, pitch, yaw ):
        imgpts = np.int32( imgpts ).reshape( -1, 2 )

        # draw ground floor in green
        # img = cv2.drawContours( img, [imgpts[:4]], -1, (0, 255, 0), -3 )

        # draw pillars in blue color
        for i, j in zip( range( 4 ), range( 4, 8 ) ):
            img = cv2.line( img, tuple( imgpts[i] ), tuple( imgpts[j] ), (255), 3 )

        # draw top layer in red color
        img = cv2.drawContours( img, [imgpts[4:]], -1, (0, 0, 255), 3 )
        roll = str(round(roll, 2))
        pitch = str(round( pitch, 2 ))
        yaw = str(round( yaw, 2 ))
        cv2.putText( img, 'roll = ' + roll + '[o]', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA )
        cv2.putText( img, 'pitch = ' + pitch + '[o]', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA )
        cv2.putText( img, 'yaw = ' + yaw + '[o]', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA )
        return img

    def transfrom_rodrigues(self, rvecs, tvecs):
        if rvecs.shape[1] == 1:
            rmat, jacobain = cv2.Rodrigues( rvecs, None )
        else:
            rmat = rvecs
        camera_position = -np.dot( rmat.transpose(), tvecs )
        roll = math.atan2( -rmat.transpose()[2][1], rmat.transpose()[2][2] ) * 180 / math.pi
        pitch = math.asin( rmat.transpose()[2][0] ) * 180 / math.pi
        yaw = math.atan2( -rmat.transpose()[1][0], rmat.transpose()[0][0] ) * 180 / math.pi
        return roll, pitch, yaw, camera_position

    def pnpspace(self, axis, camera_position, pltrange):
        fig = plt.figure()
        ax = Axes3D( fig )
        ax.scatter( camera_position[0], camera_position[1], camera_position[2], c='r')
        camera_position[0][0] = round(camera_position[0][0], 2)
        camera_position[1][0] = round( camera_position[1][0], 2 )
        camera_position[2][0] = round( camera_position[2][0], 2 )
        for i in axis:
            ax.scatter( i[0], i[1], i[2], c='b' )
        title = [ 'camera position = (', str(camera_position[0][0]), ', ', str(camera_position[1][0]), ', ', str(camera_position[2][0]), ')']
        ax.set_xlim( pltrange[0] )
        ax.set_ylim( pltrange[1] )
        ax.set_zlim( pltrange[2] )
        ax.set_title(''.join(title))
        plt.savefig(self.mypath+'space.jpg')

    def muti_pnpspace(self, axis, camera_position, pltrange):
        fig = plt.figure()
        ax = Axes3D( fig )
        for i in range(len(camera_position)):
            ax.scatter( camera_position[i][0], camera_position[i][1], camera_position[i][2], c='r')
            camera_position[i][0][0] = round(camera_position[i][0][0], 2)
            camera_position[i][1][0] = round( camera_position[i][1][0], 2 )
            camera_position[i][2][0] = round( camera_position[i][2][0], 2 )
        for i in axis:
            ax.scatter( i[0], i[1], i[2], c='b' )
        baseline = np.sqrt(np.sum(np.power(camera_position[0]-camera_position[1], 2)))
        title = ['baseline [m] = ', str(baseline)]
        ax.set_xlim( pltrange[0] )
        ax.set_ylim( pltrange[1] )
        ax.set_zlim( pltrange[2] )
        ax.set_title(''.join(title))
        plt.savefig(self.mypath+'space.jpg')

    def space_3D(self, position, pltrange):
        fig = plt.figure()
        ax = Axes3D( fig )

        X = position[:, :, 0].flatten()
        Y = position[:, :, 1].flatten()
        Z = position[:, :, 2].flatten()
        bad1_index = np.argwhere( Z < 0 )
        bad2_index = np.argwhere( Z > pltrange[2][1] )
        inf_index = np.argwhere( np.isinf(Z) )
        del_index = np.concatenate((bad1_index, bad2_index, inf_index), axis=0)

        X = np.delete( X, del_index )
        Y = np.delete( Y, del_index )
        Z = np.delete( Z, del_index )
        print('Z = ', Z)
        print('X_range = ', np.min(X), np.max(X))
        print( 'Y_range = ', np.min( Y ), np.max( Y ) )
        print( 'Z_range = ', np.min( Z ), np.max( Z ) )


        ax.scatter( Z, X, Y, c='r')
        # position[0][0] = round(camera_position[0][0], 2)
        # position[1][0] = round( camera_position[1][0], 2 )
        # camera_position[2][0] = round( camera_position[2][0], 2 )
        ax.set_xlabel('Z')
        ax.set_ylabel( 'X' )
        ax.set_zlabel( 'Y' )
        title = ['3d Reconstruction']
        ax.set_title(''.join(title))
        ax.view_init( 0, -90 )

        ax.set_xlim( pltrange[2] )
        ax.set_ylim( pltrange[0] )
        ax.set_zlim( pltrange[1] )
        plt.savefig(self.mypath+'space.jpg')
        plt.show()

    def slovepnp(self):
        delay = 100

        # camera
        calibrate_files = listdir( self.mypath )
        self.video_file = ['.mp4', '.MP4']

        dist = np.load( self.mypath + 'dist.npy' )
        mtx = np.load( self.mypath + 'mtx.npy' )
        mapx = np.load(self.mypath+'mapx.npy')
        mapy = np.load(self.mypath+'mapy.npy')

        # 3D
        objp = np.zeros( (self.block_range[0] * self.block_range[3], 3), np.float32 )
        objp[:, :2] = np.mgrid[0:self.block_range[0], 0:self.block_range[3]].T.reshape( -1, 2 )

        for f in calibrate_files:
            fullpath = join( self.mypath, f )
            secondaryname = os.path.splitext( fullpath )[1]
            if secondaryname in self.video_file:
                print( "Load the video:：", fullpath )
                cam = cv2.VideoCapture( fullpath )
                while True:
                    # read frame
                    ret_read, frame = cam.read()
                    # last frame
                    if ret_read is not True:
                        break

                    # ycrcb = cv2.cvtColor( frame, cv2.COLOR_BGR2YCR_CB )
                    # channels = cv2.split( ycrcb )
                    # cv2.equalizeHist( channels[0], channels[0] )
                    # cv2.merge( channels, ycrcb )
                    # frame = cv2.cvtColor( ycrcb, cv2.COLOR_YCR_CB2BGR, frame )
                    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )

                    for block_wi in range( self.block_range[0], self.block_range[1] + 1 ):
                        for block_hi in range( self.block_range[2], self.block_range[3] + 1 ):
                            # print('try block size', block_wi, block_hi)
                            ret_corner, corners = cv2.findChessboardCorners( gray, (block_wi, block_hi) )
                            print(ret_corner)
                            if ret_corner:
                                self.block_w, self.block_h = block_wi, block_hi
                                axis = np.float32([[0, 0, 0], [0, self.block_h - 1, 0], [self.block_w - 1, self.block_h - 1, 0], [self.block_w - 1, 0, 0], [0, 0, -3], [0, self.block_h - 1, -3], [self.block_w - 1, self.block_h - 1, -3], [self.block_w - 1, 0, -3]] )
                                corners = cv2.cornerSubPix( gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) )
                                # Find the rotation and translation vectors.
                                _, rvecs, tvecs, inliers = cv2.solvePnPRansac( objp, corners, mtx, dist )

                                # Find the rotate angle
                                roll, pitch, yaw, camera_position = self.transfrom_rodrigues(rvecs, tvecs)

                                # project 3D points to image plane
                                imgpts, jac = cv2.projectPoints( axis, rvecs, tvecs, mtx, dist )
                                frame = self.drawPnP( frame, imgpts, roll, pitch, yaw )

                                # calibrate the scale
                                camera_position = camera_position * self.block_meter
                                axis = axis * self.block_meter

                                # slovepnp space
                                # pltrange = [[-0.5, 0.5], [-0.5, 0.5], [0, -1]]
                                # self.pnpspace(axis, camera_position, pltrange )
                                # space = cv2.imread(self.mypath + 'space.jpg')
                                # cv2.imshow('space', space)
                                # cv2.imwrite('123.jpg', frame)

                                # undistortion
                                frame = cv2.remap( frame, mapx, mapy, cv2.INTER_CUBIC )

                                print(-roll, pitch, -yaw)
                                # # Rectify
                                # img_rectify = tool().rotate( frame, 0, 0, -yaw )
                                # cv2.imshow( 'Rectify', img_rectify )

                            cv2.imshow( 'frame', frame )
                            cv2.waitKey( delay )

                            # pitch = 0
                            # roll = 13
                            # while True:
                            #     print(pitch)
                            #     pitch -= 1
                            #
                            #     img_rectify1 = tool().rotate( frame, -roll, pitch, -yaw)
                            #     cv2.imshow('frame', frame)
                            #     cv2.imshow( 'Rectify1', img_rectify1 )
                            #     cv2.waitKey(1000)

    def undistortion(self, max_frame):
        # calibration
        ti = time.time()
        cam = cv2.VideoCapture(self.mypath)
        ret_read, frame = cam.read()
        w, h = frame.shape[1], frame.shape[0]
        # print(os.path.splitext(self.mypath)[0]+'_objpoints.npy')
        objpoints = np.load(os.path.splitext(self.mypath)[0]+'_objpoints.npy')
        imgpoints = np.load(os.path.splitext(self.mypath)[0]+'_imgpoints.npy')
        print('total frame is ', objpoints.shape[0])
        test_img = np.zeros((h, w, 3))

        if imgpoints.shape[0] > max_frame:
            # random
            # map = np.random.choice(objpoints.shape[0], objpoints.shape[0], replace=False)
            # objpoints = objpoints[map]
            # imgpoints = imgpoints[map]

            # kmeans
            X = []
            for frame in range(imgpoints.shape[0]):
                mass = np.mean(imgpoints[frame], axis=0)[0]
                X.append(mass)
                cv2.circle(test_img, tuple(mass), 1, (255, 255, 255))
            X = np.array(X)
            map = tool().kmeans(max_frame, X, self.mypath)
            objpoints = objpoints[map]
            imgpoints = imgpoints[map]

            # plot the point
            for frame in range(imgpoints.shape[0]):
                mass = np.mean(imgpoints[frame], axis=0)[0]
                cv2.circle(test_img, tuple(mass), 10, (250, 250, 250))
            cv2.imwrite(os.path.splitext(self.mypath)[0] + '/kmeans' + '.png', test_img)
        if imgpoints.shape[0] > 0:
            print('img size = ', imgpoints.shape)
            print('calculate the coefficient....')
            # cv2.CALIB_USE_INTRINSIC_GUESS
            # cv2.CALIB_FIX_PRINCIPAL_POINT
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera( objpoints, imgpoints, (w, h), None, None)
            print('mtx = ', mtx)
            print('dist = ', dist)
            print( 'create the NewCameraMatrix....' )
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix( mtx, dist, (w, h), 0, (w, h) )  # Optimization parameter (Free ratio parameter)
            x, y, _, _ = roi
            print( 'calculate the map matrix....' )
            mapx, mapy = cv2.initUndistortRectifyMap( mtx, dist, None, newcameramtx, (w, h), m1type=cv2.CV_32FC1)
            # FOV
            FOV = ['', '']
            FOV[0], FOV[1], _, _, _ = cv2.calibrationMatrixValues( mtx, (w, h), 1, 1 )
            print( 'FOV = ', FOV )
            print( 'save undistortion matrix...' )
            np.save( os.path.splitext(self.mypath)[0] + '_mtx', mtx )
            np.save( os.path.splitext(self.mypath)[0] + '_dist', dist )
            np.save( os.path.splitext(self.mypath)[0] + '_mapx', mapx )
            np.save( os.path.splitext(self.mypath)[0] + '_mapy', mapy )
            # CSV
            with open( os.path.splitext(self.mypath)[0] + '/calibration.csv', 'w', newline='' ) as csvfile:
                csv_f = csv.writer( csvfile )
                CSV1 = ['FOV', FOV[0], FOV[1]]
                CSV2 = ['w', w]
                CSV3 = ['h', h]
                csv_f.writerow( CSV1 )
                csv_f.writerow( CSV2 )
                csv_f.writerow( CSV3 )
            tf = time.time()
            print('RUntime = ', tf-ti)
        else:
            print('At least one checkerboard to be detected!')

    def constrain_angle(self, objpoints, imgpoints, mtx, dist):
        _, rvecs, tvecs, _ = cv2.solvePnPRansac( objpoints, imgpoints, mtx, dist )
        roll, pitch, yaw, camera_position = self.transfrom_rodrigues( rvecs, tvecs )
        if np.abs(roll) >= 90 or np.abs(pitch) >= 90 or np.abs(yaw) >= 90:
            imgpoints = imgpoints.reshape(-1, 2)
            rot180 = np.rot90(np.identity(len(imgpoints)))
            imgpoints = np.dot(rot180, imgpoints).reshape(-1, 1, 2).astype(np.float32)

            _, rvecs, tvecs, _ = cv2.solvePnPRansac( objpoints, imgpoints, mtx, dist )
            # roll, pitch, yaw, camera_position = self.transfrom_rodrigues( rvecs, tvecs )

        return rvecs, tvecs, imgpoints

    def relative_orientation_solvepnp(self, double_path, block_w, block_h, block_meter, n_base, d_frame, block_range):

        # initialization
        dist0 = np.load( double_path[0] + 'dist.npy' )
        mtx0 = np.load( double_path[0] + 'mtx.npy' )
        objpoints0 = np.load( double_path[0] + 'objpoints.npy' )
        imgpoints0 = np.load( double_path[0] + 'imgpoints.npy' )
        dist1 = np.load( double_path[1] + 'dist.npy' )
        mtx1 = np.load( double_path[1] + 'mtx.npy' )
        objpoints1 = np.load( double_path[1] + 'objpoints.npy' )
        imgpoints1 = np.load( double_path[1] + 'imgpoints.npy' )

        if objpoints1.shape[0] > objpoints0.shape[0]:
            frame_n = objpoints0.shape[0]
        else:
            frame_n = objpoints1.shape[0]

        for i in range(frame_n-1, frame_n):
            # angle constrain
            rvecs0, tvecs0, imgpoints0 = self.constrain_angle(objpoints0[i], imgpoints0[i], mtx0, dist0)
            rvecs1, tvecs1, imgpoints1 = self.constrain_angle( objpoints1[i], imgpoints1[i], mtx1, dist1 )


            # Find the rotate angle

            roll0, pitch0, yaw0, camera_position0 = self.transfrom_rodrigues( rvecs0, tvecs0 )
            roll1, pitch1, yaw1, camera_position1 = self.transfrom_rodrigues( rvecs1, tvecs1 )
            # space
            camera_position = [camera_position0, camera_position1]
            baseline_h = np.abs( camera_position[0][0] - camera_position[1][0] )[0]
            baseline_v = np.abs( camera_position[0][1] - camera_position[1][1] )[0]
            baseline_z = np.abs( camera_position[0][2] - camera_position[1][2] )[0]
            print( 'L rotate = ', roll0, pitch0, yaw0 )
            print( 'R rotate = ', roll1, pitch1, yaw1 )
            print( 'baseline = ', baseline_h, baseline_v, baseline_z )
            print( 'L camera position = ', camera_position0.reshape(-1) )
            print( 'R camera position = ', camera_position1.reshape( -1 ) )
            print('')

        # np.save( double_path[0] + 're_rvecs', rvecs0 )
        # np.save( double_path[0] + 're_tvecs', tvecs0 )
        # np.save( double_path[1] + 're_rvecs', rvecs1 )
        # np.save( double_path[1] + 're_tvecs', tvecs1 )

    def relative_orientation_stereo(self, mtx0_path, mtx1_path, dist0_path, dist1_path, imgpoints0_path, imgpoints1_path, objpoints0_path, objpoints1_path, imgsize):
        # initialization
        mtx0 = np.load(mtx0_path)
        dist0 = np.load(dist0_path)
        imgpoints0 = np.load(imgpoints0_path)
        objpoints0 = np.load(objpoints0_path)
        mtx1 = np.load(mtx1_path)
        dist1 = np.load(dist1_path)
        imgpoints1 = np.load(imgpoints1_path)
        objpoints1 = np.load(objpoints1_path)
        w, h = imgsize[0], imgsize[1]
        # # angle constrain
        # temp_rvecs0, temp_tvecs0, imgpoints0 = self.constrain_angle(objpoints0[0], imgpoints0[0], mtx0, dist0)
        # temp_rvecs1, temp_tvecs1, imgpoints1 = self.constrain_angle(objpoints1[0], imgpoints1[0], mtx1, dist1)
        # imgpoints0 = imgpoints0[np.newaxis, :, :]
        # imgpoints1 = imgpoints1[np.newaxis, :, :]

        # R, T, _, _, _, _, _, _, _, _ = cv2.composeRT(temp_rvecs0, temp_tvecs0, temp_rvecs1, temp_tvecs1)
        # roll, pitch, yaw, camera_position = self.transfrom_rodrigues(R, T)
        # print(roll, pitch, yaw, camera_position)

        # # Find the rotate angle
        # roll0, pitch0, yaw0, camera_position0 = self.transfrom_rodrigues(temp_rvecs0, temp_tvecs0)
        # roll1, pitch1, yaw1, camera_position1 = self.transfrom_rodrigues(temp_rvecs1, temp_tvecs1)
        # print('L rotate = ', roll0, pitch0, yaw0)
        # print('R rotate = ', roll1, pitch1, yaw1)
        # # space
        # camera_position = [camera_position0, camera_position1]
        # baseline_h = np.abs(camera_position[0][0] - camera_position[1][0])[0]
        # baseline_v = np.abs(camera_position[0][1] - camera_position[1][1])[0]
        # baseline_z = np.abs(camera_position[0][2] - camera_position[1][2])[0]
        # print('baseline = ', baseline_h, baseline_v, baseline_z)
        # print('L camera position = ', camera_position0.reshape(-1))
        # print('R camera position = ', camera_position1.reshape(-1))
        # a = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]]).astype(np.float32).reshape(1, -1, 3)
        # a = np.vstack((a, a, a))
        # print(a.shape)
        # print(objpoints0.shape, imgpoints0.shape, imgpoints1.shape)
        retval, mtx0, dist1, mtx1, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints0, imgpoints0, imgpoints1, mtx0, dist0, mtx1, dist1, (w, h), flags=cv2.CALIB_FIX_INTRINSIC)

        roll, pitch, yaw, camera_position = camera_orientation().transfrom_rodrigues(R, T)
        # print('R = ', R)
        # print('stereo rotate = ', roll, pitch, yaw)
        # print('T = ', T.reshape(-1))
        # print('camera position = ', camera_position.reshape(-1))
        np.save(self.mypath + 'F', F)
        np.save(self.mypath + 'E', E)
        np.save(self.mypath + 'R', R)
        np.save(self.mypath + 'T', T)

        # # 3D test
        # mini_objpoints = np.array([0., 0., 0.]).reshape(-1, 3)
        # reproject_point, _ = cv2.projectPoints(mini_objpoints, temp_rvecs0, temp_tvecs0, mtx0, dist0)
        # print(reproject_point)
        # reproject_point, _ = cv2.projectPoints(mini_objpoints, temp_rvecs1, temp_tvecs1, mtx1, dist1)
        # print(reproject_point)
        # L_H = np.dot(mtx1, np.hstack((np.eye(3), np.zeros((3, 1)))))
        # R_H = np.dot(mtx1, np.hstack((R, T)))
        # # L_H = np.dot(mtx0, np.hstack((cv2.Rodrigues(temp_rvecs0)[0], temp_tvecs0)))
        # # R_H = np.dot(mtx1, np.hstack((cv2.Rodrigues(temp_rvecs1)[0], temp_tvecs1)))
        # L_img_coord = imgpoints0[0][[0, 14, 17*15-1, 17*15-1-14]].reshape(-1, 2).T  # 0, 14, 17*15-1-14, 17*15-1
        # R_img_coord = imgpoints1[0][[0, 14, 17*15-1, 17*15-1-14]].reshape(-1, 2).T
        # print(L_img_coord)
        # print(R_img_coord)
        # # print(L_img_coord)
        # # L_img_coord = L_img_coord.reshape(2, -1)
        # # print(L_img_coord)
        # # exit()
        # coord = cv2.triangulatePoints(L_H, R_H, L_img_coord, R_img_coord).T
        # coord[:, 0] = coord[:, 0] / coord[:, -1]
        # coord[:, 1] = coord[:, 1] / coord[:, -1]
        # coord[:, 2] = coord[:, 2] / coord[:, -1]
        # coord[:, 3] = coord[:, 3] / coord[:, -1]
        # coord = coord[:, :-1]
        #
        # for i in range(coord.shape[0]-1):
        #     ans = np.sqrt(np.sum( np.power(coord[i+1]-coord[i], 2)))
        #     print(ans)


        # # reconstructure E F
        # term1 = np.linalg.inv(mtx0).T
        # term2 = np.array([[0, -T[2][0], T[1][0]], [T[2][0], 0, -T[0][0]], [-T[1][0], T[0][0], 0]])
        # term3 = np.linalg.inv(mtx1)
        # E
        # test = np.dot(term2, R)
        # F
        # test = np.dot(np.dot(term1, np.dot(term2, R)), term3)
        # test = test/test[2][2]
        # print('test = ', )
        # print(test)

        # # Epilines
        # lines1 = cv2.computeCorrespondEpilines( qr[0][-15][:], 2, F ).reshape(-1, 3)  # 1/L, 2/R
        # x0, y0 = map( int, [0, -lines1[0][2] / lines1[0][1]] )
        # x1, y1 = map( int, [w, -(lines1[0][2] + lines1[0][0] *w) / lines1[0][1]] )
        # img1 = cv2.imread( double_path[1] + str( camera1 + 1 ) + '.png' )
        # img2 = cv2.imread(double_path[1] + str(camera2+1) + '.png')
        # img1 = cv2.line( img1, (x0, y0), (x1, y1), (0, 0, 0), 1 )
        # img2 = cv2.line( img2, (x0, y0), (x1, y1), (0, 0, 0), 1 )
        # cv2.imshow( 'img1', img1 )
        # cv2.imshow('img2', img2)
        # cv2.waitKey(1)
        # print(lines1.shape)
        # padding = np.ones((1, 181, 1))
        # print(padding.shape)
        # print(F.shape)
        # print(ql.shape)
        # print(ql[0][0][0])
        # print( qr[0][0][0] )
        # ql = [341.77573, 209.27162, 1.]
        # qr = [308.9946, 116.8446, 1.]
        # l = np.dot( np.dot( qr, F ), ql )
        # print(l)

        # np.save( double_path[0] + 're_rvecs', rvecs0 )
        # np.save( double_path[0] + 're_tvecs', tvecs0 )
        # np.save( double_path[1] + 're_rvecs', rvecs1 )
        # np.save( double_path[1] + 're_tvecs', tvecs1 )

    def synchronization( self, double_path, video_path, d_time ):
        delay = 1
        # initialization
        step = []
        error = []
        cam0 = cv2.VideoCapture( video_path[0] )
        cam1 = cv2.VideoCapture( video_path[1] )
        dist0 = np.load( double_path[0] + 'dist.npy' )
        mtx0 = np.load( double_path[0] + 'mtx.npy' )
        dist1 = np.load( double_path[1] + 'dist.npy' )
        mtx1 = np.load( double_path[1] + 'mtx.npy' )
        rvecs_re = np.load( self.mypath + 'rvecs_re.npy' )
        tvecs_re = np.load( self.mypath + 'tvecs_re.npy' )
        R = rvecs_re
        T = tvecs_re
        w, h = (int( cam0.get( cv2.CAP_PROP_FRAME_WIDTH ) ), int( cam0.get( cv2.CAP_PROP_FRAME_HEIGHT ) ))
        print('total_frame = ', cam0.get(7), cam1.get(7))

        # stereorecitfy
        R0, R1, P0, P1, Q, validPixROI1, validPixROI2 = cv2.stereoRectify( mtx0, dist0, mtx1, dist1, (w, h), R, T )
        theorey_baseline = -P1[0][3] / P1[0][0]
        print( 'theorey baseline = ', theorey_baseline )

        for i in range(1, int(cam0.get(7)+cam1.get(7)), d_time):  # 1, int(cam0.get(7)+cam1.get(7)), 5
            cam0.release()
            cam1.release()
            cam0 = cv2.VideoCapture( video_path[0] )
            cam1 = cv2.VideoCapture( video_path[1] )
            if int(cam0.get(7)) > i:
                print('Shift the 0 camera = ', int(cam0.get(7))-i)
                shift = int(cam0.get(7))-i
                for xxx in range(shift):
                    _, _ = cam0.read()
            elif int(cam0.get(7)) < i:
                print('Shift the 1 camera = ', i-int( cam0.get( 7 ) ) )
                shift = i-int( cam0.get( 7 ) )
                for xxx in range( shift ):
                    _, _ = cam1.read()
            elif int(cam0.get(7)) == i:
                print('No shift')

            # the coordinates of the block (3D > 2D)
            # 3D
            objp = np.zeros( (self.block_range[0] * self.block_range[3], 3), np.float32 )
            objp[:, :2] = np.mgrid[0:self.block_range[0], 0:self.block_range[3]].T.reshape( -1, 2 ) * self.block_meter
            objpoints = []  # 3d coordinates in the world
            imgpoints0 = []  # 2d coordinates in the image
            imgpoints1 = []  # 2d coordinates in the image

            # Get the block from camera
            n_temp = 0
            total_frame = 1
            while True:
                # read frame
                ret_read0, frame0 = cam0.read()
                ret_read1, frame1 = cam1.read()

                # ycrcb = cv2.cvtColor( frame, cv2.COLOR_BGR2YCR_CB )
                # channels = cv2.split( ycrcb )
                # cv2.equalizeHist( channels[0], channels[0] )
                # cv2.merge( channels, ycrcb )
                # frame = cv2.cvtColor( ycrcb, cv2.COLOR_YCR_CB2BGR, frame )

                # last frame
                if ret_read0 is not True or ret_read1 is not True:
                    break

                # d_frame
                if total_frame % self.d_frame == 0:
                    # cv2.imshow('camera', frame)
                    # cv2.waitKey(100)
                    gray0 = cv2.cvtColor( frame0, cv2.COLOR_BGR2GRAY )
                    gray1 = cv2.cvtColor( frame1, cv2.COLOR_BGR2GRAY )

                    # find the block on image from different block_size
                    for block_wi in range( self.block_range[0], self.block_range[1] + 1 ):
                        for block_hi in range( self.block_range[2], self.block_range[3] + 1 ):
                            ret_corner0, corners0 = cv2.findChessboardCorners( gray0, (block_wi, block_hi) )
                            ret_corner1, corners1 = cv2.findChessboardCorners( gray1, (block_wi, block_hi) )
                            if ret_corner0 == True and ret_corner1 == True:
                                break
                        if ret_corner0 == True and ret_corner1 == True:
                            self.block_w, self.block_h = block_wi, block_hi
                            break
                    if ret_corner0 == True and ret_corner1 == True:
                        corners0 = cv2.cornerSubPix( gray0, corners0, **self.cornerSubPix_params )
                        corners1 = cv2.cornerSubPix( gray1, corners1, **self.cornerSubPix_params )
                        objpoints.append( objp )
                        imgpoints0.append( corners0 )
                        imgpoints1.append( corners1 )
                        cv2.drawChessboardCorners( frame0, (self.block_w, self.block_h), corners0,
                                                   ret_corner0 )  # draw the corners point
                        cv2.drawChessboardCorners( frame1, (self.block_w, self.block_h), corners1,
                                                   ret_corner1 )  # draw the corners point
                        n_temp += 1
                        if n_temp >= self.n_base:
                            break
                    # cv2.imshow(str(fullpath), frame)
                    # cv2.waitKey(1)
                total_frame += 1
            cv2.destroyAllWindows()
            cam0.release()
            cam1.release()

            # calibration
            if len(imgpoints0) > 2:
                print( 'calculate the coefficient....' )
                retval, mtx0, dist1, mtx1, dist2, R, T, E, F = cv2.stereoCalibrate( objpoints, imgpoints0, imgpoints1, mtx0,
                                                                                    dist0, mtx1, dist1, gray0.shape[::-1],
                                                                                    flags=cv2.CALIB_FIX_INTRINSIC )

                R0, R1, P0, P1, Q, validPixROI1, validPixROI2 = cv2.stereoRectify( mtx0, dist0, mtx1, dist1, gray0.shape[::-1], R, T )
                baseline = -P1[0][3] / P1[0][0]
                print( 'baseline = ', baseline )
                step.append(i)
                error.append(100*abs(theorey_baseline-baseline)/theorey_baseline)

        # plt
        plt.title('Time scan')
        plt.xlabel('step')
        plt.ylabel( 'error' )
        plt.plot(step, error)
        plt.show()

    def stereo(self, double_path):
        cv2.namedWindow( "depth" )
        cv2.createTrackbar( "num", "depth", 50, 100, lambda x: None )
        cv2.createTrackbar( "blockSize", "depth", 1, 255, lambda x: None )
        delay = 10
        # initialization
        cam0 = cv2.VideoCapture( 0 )
        cam1 = cv2.VideoCapture( 1 )
        dist0 = np.load( double_path[0] + 'dist.npy' )
        mtx0 = np.load( double_path[0] + 'mtx.npy' )
        mapx0 = np.load( double_path[0] + 'mapx.npy' )
        mapy0 = np.load( double_path[0] + 'mapy.npy' )
        dist1 = np.load( double_path[1] + 'dist.npy' )
        mtx1 = np.load( double_path[1] + 'mtx.npy' )
        mapx1 = np.load( double_path[1] + 'mapx.npy' )
        mapy1 = np.load( double_path[1] + 'mapy.npy' )
        rvecs_re = np.load( self.mypath + 'rvecs_re.npy' )
        tvecs_re = np.load( self.mypath + 'tvecs_re.npy' )
        R = rvecs_re
        T = tvecs_re
        w, h = (int(cam0.get( cv2.CAP_PROP_FRAME_WIDTH )), int(cam0.get( cv2.CAP_PROP_FRAME_HEIGHT )))

        # stereorecitfy
        R0, R1, P0, P1, Q, validPixROI1, validPixROI2 = cv2.stereoRectify( mtx0, dist0, mtx1, dist1, (w, h), R, T )
        baseline = -P1[0][3]/P1[0][0]
        print('baseline = ', baseline)
        mapx0_rect, mapy0_rect = cv2.initUndistortRectifyMap( mtx0, dist0, R0, P0,  (w, h), cv2.CV_16SC2 )
        mapx1_rect, mapy1_rect = cv2.initUndistortRectifyMap( mtx1, dist1, R1, P1, (w, h), cv2.CV_16SC2 )

        while True:
            # read frame
            ret_read0, frame0 = cam0.read()
            ret_read1, frame1 = cam1.read()
            # last frame
            if ret_read0 is False or ret_read1 is False:
                break
            frame0_rect = cv2.remap( frame0, mapx0_rect, mapy0_rect, cv2.INTER_LINEAR )
            frame1_rect = cv2.remap( frame1, mapx1_rect, mapy1_rect, cv2.INTER_LINEAR )
            gray0_rect = cv2.cvtColor( frame0_rect, cv2.COLOR_BGR2GRAY )
            gray1_rect = cv2.cvtColor( frame1_rect, cv2.COLOR_BGR2GRAY )
            # stereo
            num = cv2.getTrackbarPos( "num", "depth" )
            blockSize = cv2.getTrackbarPos( "blockSize", "depth" )
            if blockSize % 2 == 0:
                blockSize += 1
            if blockSize < 5:
                blockSize = 5
            # blockSize, num = 5, 10
            stereo = cv2.StereoSGBM_create(numDisparities=16*num, blockSize=blockSize)
            disparity = stereo.compute( gray0_rect, gray1_rect ).astype( np.float32 ) / 16
            normal_disparity = cv2.normalize( disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U )

            # 3D space
            # points_3D = cv2.reprojectImageTo3D( disparity, Q )
            # pltrange = [[-1, 1], [-1, 1], [0, 2]]
            # self.space_3D( points_3D, pltrange )

            cv2.imshow('camera_0_rect', frame0_rect)
            cv2.imshow('camera_1_rect', frame1_rect)
            cv2.imshow( 'camera_0', frame0 )
            cv2.imshow( 'camera_1', frame1 )
            cv2.imshow( "depth", normal_disparity )
            cv2.waitKey(delay)

    def stereo_backup(self, double_path):
        delay = 1000

        # initialization
        cam0 = cv2.VideoCapture( 0 )
        cam1 = cv2.VideoCapture( 1 )
        image_shape = (int( cam0.get( cv2.CAP_PROP_FRAME_HEIGHT ) ), int( cam0.get( cv2.CAP_PROP_FRAME_WIDTH ) ))
        dist0 = np.load( double_path[0] + 'dist.npy' )
        mtx0 = np.load( double_path[0] + 'mtx.npy' )
        mapx0 = np.load( double_path[0] + 'mapx.npy' )
        mapy0 = np.load( double_path[0] + 'mapy.npy' )
        dist1 = np.load( double_path[1] + 'dist.npy' )
        mtx1 = np.load( double_path[1] + 'mtx.npy' )
        mapx1 = np.load( double_path[1] + 'mapx.npy' )
        mapy1 = np.load( double_path[1] + 'mapy.npy' )

        # 3D
        objp = np.zeros( (self.block_range[0] * self.block_range[3], 3), np.float32 )
        objp[:, :2] = np.mgrid[0:self.block_range[0], 0:self.block_range[3]].T.reshape( -1, 2 )
        objpoints = []  # 3d coordinates in the world
        imgpoints0 = []  # 2d coordinates in the image
        imgpoints1 = []  # 2d coordinates in the image

        relative_orientation = False
        while True:
            # read frame
            ret_read0, frame0 = cam0.read()
            ret_read1, frame1 = cam1.read()
            # last frame
            if ret_read0 is False or ret_read1 is False:
                break

            # ycrcb = cv2.cvtColor( frame, cv2.COLOR_BGR2YCR_CB )
            # channels = cv2.split( ycrcb )
            # cv2.equalizeHist( channels[0], channels[0] )
            # cv2.merge( channels, ycrcb )
            # frame = cv2.cvtColor( ycrcb, cv2.COLOR_YCR_CB2BGR, frame )
            gray0 = cv2.cvtColor( frame0, cv2.COLOR_BGR2GRAY )
            gray1 = cv2.cvtColor( frame1, cv2.COLOR_BGR2GRAY )

            for block_wi in range( self.block_range[0], self.block_range[1] + 1 ):
                for block_hi in range( self.block_range[2], self.block_range[3] + 1 ):
                    # print('try block size', block_wi, block_hi)
                    ret_corner0, corners0 = cv2.findChessboardCorners( gray0, (block_wi, block_hi) )
                    ret_corner1, corners1 = cv2.findChessboardCorners( gray1, (block_wi, block_hi) )
                    if ret_corner0 and ret_corner1:
                        self.block_w, self.block_h = block_wi, block_hi
                        axis = np.float32([[0, 0, 0], [0, self.block_h - 1, 0], [self.block_w - 1, self.block_h - 1, 0], [self.block_w - 1, 0, 0], [0, 0, -3], [0, self.block_h - 1, -3], [self.block_w - 1, self.block_h - 1, -3], [self.block_w - 1, 0, -3]] )
                        corners0 = cv2.cornerSubPix( gray0, corners0, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) )
                        corners1 = cv2.cornerSubPix( gray1, corners1, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) )
                        # record the point
                        imgpoints0.append( corners0 )
                        imgpoints1.append( corners1 )
                        objpoints.append( objp )

                        # Find the rotation and translation vectors.
                        _, rvecs0, tvecs0, inliers0 = cv2.solvePnPRansac( objp, corners0, mtx0, dist0 )
                        _, rvecs1, tvecs1, inliers1 = cv2.solvePnPRansac( objp, corners1, mtx1, dist1 )

                        # Find the rotate angle
                        roll0, pitch0, yaw0, camera_position0 = self.transfrom_rodrigues( rvecs0, tvecs0 )
                        roll1, pitch1, yaw1, camera_position1 = self.transfrom_rodrigues( rvecs1, tvecs1 )

                        # project 3D points to image plane
                        imgpts0, jac0 = cv2.projectPoints( axis, rvecs0, tvecs0, mtx0, dist0 )
                        frame0 = self.drawPnP( frame0, imgpts0, roll0, pitch0, yaw0 )
                        imgpts1, jac1 = cv2.projectPoints( axis, rvecs1, tvecs1, mtx1, dist1 )
                        frame1 = self.drawPnP( frame1, imgpts1, roll1, pitch1, yaw1 )

                        # calibrate the scale
                        camera_position0 = camera_position0 * self.block_meter
                        camera_position1 = camera_position1 * self.block_meter
                        axis = axis * self.block_meter

                        # slovepnp space
                        pltrange = [[-0.5, 0.5], [-0.5, 0.5], [0, -1]]
                        camera_position = [camera_position0, camera_position1]
                        self.muti_pnpspace( axis, camera_position, pltrange, )
                        space = cv2.imread( self.mypath + 'space.jpg' )
                        cv2.imshow( 'space', space )
                        # cv2.imwrite('123.jpg', frame)

                        # start to find relative orientation
                        relative_orientation = True
                        if relative_orientation:
                            break

                    # undistortion
                    undis_frame0 = cv2.remap( frame0, mapx0, mapy0, cv2.INTER_CUBIC )
                    undis_frame1 = cv2.remap( frame1, mapx1, mapy1, cv2.INTER_CUBIC )
                    cv2.imshow( 'frame0', undis_frame0 )
                    cv2.imshow( 'frame1', undis_frame1 )
                    cv2.waitKey( delay )

                # start to find relative orientation
                if relative_orientation:
                    break
            # start to find relative orientation
            if relative_orientation:
                break
        if relative_orientation:
            image_shape = (int( cam0.get( cv2.CAP_PROP_FRAME_HEIGHT ) ), int( cam0.get( cv2.CAP_PROP_FRAME_WIDTH ) ))
            # image_shape = (int( cam0.get( cv2.CAP_PROP_FRAME_WIDTH ) ), int( cam0.get( cv2.CAP_PROP_FRAME_HEIGHT ) ))
            retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints0, imgpoints1, image_shape )
            print(cameraMatrix1)

        else:
            print('Did not find the relative orientation ')

    def scan_checkboard(self, block_meter, i_frame, d_frame, f_frame, block_range, Subpix=True, n_base=1e10):
        block_w, block_h = block_range[0], block_range[2]
        # camera
        # calibrate_files = listdir( self.mypath )
        video_file = ['.mp4', '.MP4', '.avi']
        # the coordinates of the block (3D > 2D)
        cornerSubPix_params = dict(winSize=(11, 11), zeroZone=(-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))  # https://www.itread01.com/content/1544079906.html
        objp = np.zeros( (block_range[0] * block_range[3], 3), np.float32 )
        objp[:, :2] = np.mgrid[0:block_range[0], 0:block_range[3]].T.reshape( -1, 2 ) * block_meter
        objpoints = []  # 3d coordinates in the world
        imgpoints = []  # 2d coordinates in the image
        # Get the block from camera
        n_temp = 0
        # for f in calibrate_files:
        fullpath = self.mypath
        if os.path.isdir(os.path.splitext(fullpath)[0]):
            shutil.rmtree(os.path.splitext(fullpath)[0])
        os.mkdir(os.path.splitext(fullpath)[0])
        secondaryname = os.path.splitext( fullpath )[1]
        if secondaryname in video_file:
            if os.path.isdir(os.path.splitext(fullpath)[0]):
                shutil.rmtree(os.path.splitext(fullpath)[0])
            os.mkdir(os.path.splitext(fullpath)[0])
            cam = cv2.VideoCapture( fullpath )
            total_frame = 1
            while True:
                # read frame
                ret_read, frame = cam.read()
                # last frame
                if ret_read is not True:
                    break
                if total_frame > f_frame:
                    break
                # d_frame
                if total_frame >= i_frame and total_frame % d_frame == 0:
                    gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
                    # find the block on image from different block_size
                    for block_wi in range( block_range[0], block_range[1] + 1 ):
                        for block_hi in range( block_range[2], block_range[3] + 1 ):
                            ret_corner, corners = cv2.findChessboardCorners( gray, (block_wi, block_hi), flags= cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                            if ret_corner == True:
                                print('the coordinates form row data...')
                            if ret_corner == False:
                                bilatera_gray = cv2.bilateralFilter(gray, 7, 31, 31)
                                ret_corner, corners = cv2.findChessboardCorners( bilatera_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
                                # cv2.imshow( 'Bilateral filter', bilatera_gray )
                                if ret_corner == True:
                                    print('the coordinates form Bilateral filter...')
                            if ret_corner == False:
                                guide = cv2.ximgproc.guidedFilter( guide=gray, src=frame, radius=16, eps=50, dDepth=-1 )
                                guide_gray = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY )
                                ret_corner, corners = cv2.findChessboardCorners( guide_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
                                # cv2.imshow('Guided filter', guide_gray)
                                if ret_corner == True:
                                    print('the coordinates form Guided filter...')
                            if ret_corner == False:
                                denoise_gray = cv2.fastNlMeansDenoising( gray, 2, 5)
                                # print(denoise_gray.shape)
                                ret_corner, corners = cv2.findChessboardCorners( denoise_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
                                # cv2.imshow('denoise filter', denoise_gray), cv2.waitKey(1)
                                if ret_corner == True:
                                    print( 'the coordinates form denoise filter...' )
                            if ret_corner == False:
                                canny_gray = cv2.Canny( gray, 100, 200 )
                                canny_gray = cv2.morphologyEx( canny_gray, cv2.MORPH_CLOSE, (1, 1) )
                                cv2.bitwise_not( canny_gray, canny_gray )
                                ret_corner, corners = cv2.findChessboardCorners( canny_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
                                # cv2.imshow('canny filter', canny_gray)
                                if ret_corner == True:
                                    print('the coordinates form canny filter...')
                            if ret_corner == False:
                                blur_gray = cv2.GaussianBlur( gray, (3, 3), 0 )
                                ret_corner, corners = cv2.findChessboardCorners( blur_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
                                # cv2.imshow('blur filter', blur_gray)
                                if ret_corner == True:
                                    print( 'the coordinates form blur filter...' )
                                # update
                            print( '(w,h) = ', block_wi, block_hi, '(get frame/now_frame) = ', n_temp, total_frame, '/  state = ', ret_corner )
                            if ret_corner == True:
                                pass
                        if ret_corner == True:
                            block_w, block_h = block_wi, block_hi
                    if ret_corner == True:
                        if Subpix:
                            cv2.cornerSubPix( gray, corners, **cornerSubPix_params )
                        objpoints.append( objp )
                        imgpoints.append( corners )
                        cv2.drawChessboardCorners( frame, (block_w, block_h), corners, ret_corner )  # draw the corners point
                        n_temp += 1
                        # cv2.imshow(str(n_temp), frame)
                        # cv2.waitKey(1)
                        cv2.imwrite(os.path.splitext(fullpath)[0]+'/'+str(n_temp)+'.png', frame)
                        if n_temp >= n_base:
                            break
                    # cv2.imshow(str(fullpath), frame)
                    # cv2.waitKey(1)
                # cv2.imshow( 'camera', frame )
                # cv2.waitKey( 1 )
                total_frame += 1
            cv2.destroyAllWindows()
        np.save(os.path.splitext(fullpath)[0]+'_imgpoints', imgpoints)
        np.save(os.path.splitext(fullpath)[0] + '_objpoints', objpoints)

    def detect_checkerboard(self, flag, frame_set, block_w, block_h, fullpath, SubPix=True):  #  frame_set, block_w, block_h
        frame = frame_set[flag]
        gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )
        block_wi, block_hi = block_w, block_h
        cornerSubPix_params = dict( winSize=(11, 11), zeroZone=(-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) )  # https://www.itread01.com/content/1544079906.html
        print('frame = ', flag+1)
        # ret_corner, corners = cv2.findChessboardCorners( gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
        ret_corner, corners = cv2.findChessboardCorners(gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
        if ret_corner == True:
            print('the coordinates form row data...')
        if ret_corner == False:
            bilatera_gray = cv2.bilateralFilter( gray, 7, 31, 31 )
            ret_corner, corners = cv2.findChessboardCorners( bilatera_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
            # cv2.imshow( 'Bilateral filter', bilatera_gray )
            if ret_corner == True:
                print( 'the coordinates form Bilateral filter...' )
        if ret_corner == False:
            guide = cv2.ximgproc.guidedFilter( guide=gray, src=frame, radius=16, eps=50, dDepth=-1 )
            guide_gray = cv2.cvtColor( guide, cv2.COLOR_BGR2GRAY )
            ret_corner, corners = cv2.findChessboardCorners( guide_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
            # cv2.imshow('Guided filter', guide_gray)
            if ret_corner == True:
                print( 'the coordinates form Guided filter...' )
        if ret_corner == False:
            denoise_gray = cv2.fastNlMeansDenoising( gray, 2, 5 )
            # print(denoise_gray.shape)
            ret_corner, corners = cv2.findChessboardCorners( denoise_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
            # cv2.imshow('denoise filter', denoise_gray)
            if ret_corner == True:
                print( 'the coordinates form denoise filter...' )
        if ret_corner == False:
            canny_gray = cv2.Canny( gray, 100, 200 )
            canny_gray = cv2.morphologyEx( canny_gray, cv2.MORPH_CLOSE, (1, 1) )
            cv2.bitwise_not( canny_gray, canny_gray )
            ret_corner, corners = cv2.findChessboardCorners( canny_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
            # cv2.imshow('canny filter', canny_gray)
            if ret_corner == True:
                print( 'the coordinates form canny filter...' )
        if ret_corner == False:
            blur_gray = cv2.GaussianBlur( gray, (3, 3), 0 )
            ret_corner, corners = cv2.findChessboardCorners( blur_gray, (block_wi, block_hi), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE )
            # cv2.imshow('blur filter', blur_gray)
            if ret_corner == True:
                print( 'the coordinates form blur filter...' )
        if ret_corner == True:
            if SubPix:
                cv2.cornerSubPix( gray, corners, **cornerSubPix_params )
            cv2.drawChessboardCorners(frame, (block_w, block_h), corners, ret_corner)
            cv2.imwrite(os.path.splitext(fullpath)[0] + '/' + str(flag+1) + '.png', frame)
        return ret_corner, corners

    def muticore_scan_checkboard(self, block_range, block_meter, i_frame, d_frame, f_frame='default', SubPix=True):
        def capture_video(cam, space):
            cap = []
            for i in range(int(cam.get(7))):
                _, frame = cam.read()
                if i in space:
                    cap.append(frame)
                if i > np.max(space):
                    break
            cap = np.array(cap)
            return cap

        p = mp.Pool()
        print( 'core number = ', os.cpu_count() )

        # camera
        video_file = ['.mp4', '.MP4', '.avi']

        # the coordinates of the block (3D > 2D)
        objp = np.zeros( (block_range[0] * block_range[3], 3), np.float32 )
        objp[:, :2] = np.mgrid[0:block_range[0], 0:block_range[3]].T.reshape( -1, 2 ) * block_meter

        objpoints = []  # 3d coordinates in the world
        imgpoints = []  # 2d coordinates in the image

        # Get the block from camera
        fullpath = self.mypath
        if os.path.isdir(os.path.splitext(fullpath)[0]):
            shutil.rmtree(os.path.splitext(fullpath)[0])
        os.mkdir(os.path.splitext(fullpath)[0])
        secondaryname = os.path.splitext( fullpath )[1]
        if secondaryname in video_file:
            print( "Load the video:：", fullpath )
            cam = cv2.VideoCapture( fullpath )
            f_frame = cam.get(7) if str(f_frame).lower() == 'default' else f_frame
            space = np.arange(i_frame, f_frame, d_frame).astype(np.int)  # cam.get(7)
            # space = np.arange(i_frame, cam.get(7), d_frame).astype(np.int)  # cam.get(7)
            frame_set = capture_video( cam, space )
            # block_w, block_h = block_range[0], block_range[2]
            ti = time.time()
            multi_res = [p.apply_async( self.detect_checkerboard, (i, frame_set, block_range[0],  block_range[2], fullpath, SubPix)) for i in range(frame_set.shape[0])]
            for res in multi_res:
                ret_corner, corners = res.get()
                if ret_corner:
                    objpoints.append(objp)
                    imgpoints.append(corners)
            print('detection rate = ', 100*len(imgpoints)/frame_set.shape[0])
            tf = time.time()
            print('multicore time = ', tf-ti)
        objpoints = np.array(objpoints)
        imgpoints = np.array(imgpoints)
        print('Get number = ', imgpoints.shape[0])
        np.save(os.path.splitext(fullpath)[0] + '_imgpoints', imgpoints)
        np.save(os.path.splitext(fullpath)[0] + '_objpoints', objpoints)

    def remap_undistion(self, video_path, mapx_path, mapy_path, frame=1):
        cam = cv2.VideoCapture(video_path)
        for i in range(frame+1):
            ret_read, img = cam.read()
        mapx, mapy = np.load(mapx_path), np.load(mapy_path)
        undis_img = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
        print(mapx)
        print(mapy)
        cv2.imshow('frame', img)
        cv2.imshow('undistortion', undis_img)
        cv2.waitKey(0)




if __name__ == '__main__':

    # transformation angle




    # distortion image
    # main_path = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/test/NTUS/'
    # video_path = main_path + 'NTUS_Camera_3_checkerboard.avi'
    # mapx_path = main_path + 'NTUS_Camera_3_checkerboard_mapx.npy'
    # mapy_path = main_path + 'NTUS_Camera_3_checkerboard_mapy.npy'
    # calibration('').remap_undistion(video_path=video_path, mapx_path=mapx_path, mapy_path=mapx_path)


    # print(os.getcwd())
    import fn
    mypath = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/thesisinp/thesisinp/dataset/NTUS/cube/cube3/R3.mp4'
    # mypath = "/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L7/L7.mp4"  # calibrate video path  "./camera_backup1/
    # mypath = "/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A7/A7.mp4"  # calibrate video path  "./camera_backup1/
    # double_path = ['../20200407/L_33GX287/relation/RLA/cube/backup/L1/', '../20200407/A_33GX287/relation/RLA/cube/backup/A1/']
    # block_w, block_h, block_meter = 3, 8, 0.1  # block size (16)-16 / block size (13)-18
    # block_range = [block_w, block_w, block_h, block_h]
    # i_frame, d_frame, n_base = 720, 10, 9999
    # w, h = 720, 540
    # checkerboard
    # calibration( mypath ).scan_checkboard( block_w, block_h, block_meter, i_frame, d_frame, n_base, block_range )
    # calibration( mypath ).muticore_scan_checkboard( block_w, block_h, block_meter, i_frame, d_frame,  block_range )

    # tool().combin_file('../20200318/L_33GX287/calibration/backup/1/img/', type=['.npy'])
    # cube
    # import fn
    fn.manual_2D().Run(path=mypath, type='cube', block_meter=1, ratio_h=1, ratio_w=1, block_w=3, block_h=3)
    # tool().combin_file('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/conbin/obj_A', type=['.npy'])

    # undistortion
    # calibration(mypath).undistortion(1e10)
    # calibration( mypath ).relative_orientation_solvepnp(double_path, block_w, block_h, block_meter, n_base, d_frame, block_range)
    ##### tool().recitfy(mypath)

    # relative_orientation
    # calibration( mypath ).relative_orientation_stereo(double_path, block_w, block_h, block_meter, n_base, d_frame, block_range)
    # Synchronization
    # calibration( mypath, block_w, block_h, block_meter, n_base, d_frame, block_range ). synchronization( double_path=double_path, video_path=['./Synchronization/stereo_0_shift.mp4', './Synchronization/stereo_1_shift.mp4'], d_time=1 )
    # stereo
    # calibration( mypath, block_w, block_h, block_meter, n_base, d_frame, block_range ).stereo( double_path )

    # tool
    # tool().splite_np(path='../20200318/R_33GX287/relation/RLA/cube/backup/R6/objpoints.npy', index=2)
    # tool().splite_np(path='../20200318/R_33GX287/relation/RLA/cube/backup/R6/imgpoints.npy', index=2)
    # calibration( mypath, block_w, block_h, block_meter, n_base, d_frame, block_range ).slovepnp()
    # tool().video_disassemble(path='/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/Sport_project_v1.4.6/UI/3D_reconstruction/test/basic_B20_human20_Camera_3_reprojection_post.avi')
    # tool().video_disassem ble(path='../dataset/A_camera/human2/A2.avi')
    # tool().video_disassemble(path='../dataset/A_camera/human3/A3.avi')

    # fn.tool().video_shift(path=mypath + 'AlphaPose_255_1.avi', frame_i=1, frame_f='', new_FPS=10)
    # fn.tool().video_shift(path=mypath + '185.MP4', frame_i=1351, frame_f=1418, new_FPS='')
    # fn.tool().video_shift(path=mypath + '195.MP4', frame_i=1368, frame_f=1430, new_FPS='')
    # fn.tool().video_shift(path=mypath + '205.MP4', frame_i=1980, frame_f=2044, new_FPS='')
    # fn.tool().video_shift(path=mypath + '215.MP4', frame_i=1741, frame_f=1804, new_FPS='')
    # fn.tool().video_shift(path=mypath + '225.MP4', frame_i=1682, frame_f=1747, new_FPS='')
    # fn.tool().video_shift(path=mypath + '235.MP4', frame_i=1529, frame_f=1594, new_FPS='')
    # fn.tool().video_shift(path=mypath + '245.MP4', frame_i=2748, frame_f=2813, new_FPS='')
    # fn.tool().video_shift(path=mypath + '255.MP4', frame_i=1786, frame_f=1852, new_FPS='')
    # fn.tool().video_shift(path=mypath + '265.MP4', frame_i=1859, frame_f=1922, new_FPS='')
    # fn.tool().video_shift(path=mypath + '275.MP4', frame_i=2016, frame_f=2078, new_FPS='')
    # fn.tool().video_shift(path=mypath + '285.MP4', frame_i=1806, frame_f=1866, new_FPS='')

    # tool().video_combin(folder='/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200613/calibration/Thesis_test/checkerboard_L/', new_FPS=None)

    # tool().video_record('./', 0)
    # tool().video_dobule_record(mypath)
    # tool().img2video('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/thesisinp/thesisinp/dataset/NTUS/cube/cube7/R7.jpg')
    # tool().img2video('../dataset/20200410_411/Camera_33GX287/cube/L_disassemble/947.png')
    # tool().img2video('../dataset/20200410_411/Camera_33GX287/cube/A_disassemble/947.png')

    # rotate
    # tool().video_rotate(path='/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200616/human1/cam_5_3.avi', angle=180)  # counterclockwise
    # tool().csv_2d_rotate(path='../20200221/L_33GX287/real/human1/AlphaPose_L1_rotate.csv', angle=270, w=540, h=720)  # counterclockwise
