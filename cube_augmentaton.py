import numpy as np
import cv2
from itertools import combinations
import threeD_pose_human, checkerboard_calibration, fn


def relative_orientation_stereo(mtx0_path, mtx1_path, dist0_path, dist1_path, imgpoints0_path, imgpoints1_path, objpoints0_path, objpoints1_path, imgsize, training_index):
    # initialization
    w, h = imgsize[0], imgsize[1]
    mtx0 = np.load(mtx0_path)
    dist0 = np.load(dist0_path)
    mtx1 = np.load(mtx1_path)
    dist1 = np.load(dist1_path)
    for index in range(len(training_index)):
        if index == 0:
            imgpoints0 = np.load(imgpoints0_path[training_index[index]])
            imgpoints1 = np.load(imgpoints1_path[training_index[index]])
            objpoints0 = np.load(objpoints0_path[training_index[index]])
        else:
            imgpoints0 = np.vstack((imgpoints0, np.load(imgpoints0_path[training_index[index]])))
            imgpoints1 = np.vstack((imgpoints1, np.load(imgpoints1_path[training_index[index]])))
            objpoints0 = np.vstack((objpoints0, np.load(objpoints0_path[training_index[index]])))
    retval, mtx0, dist1, mtx1, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints0, imgpoints0, imgpoints1, mtx0, dist0, mtx1, dist1, (w, h), flags=cv2.CALIB_FIX_INTRINSIC)
    return R, T

temp_folder = '/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/'
cube_length = 1  # [m]
imgsize = (720, 540)
leni = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6])
lenf = np.array([1, 2, 4, 3, 5, 3, 6, 7, 5, 6, 7, 7])
mtx, dist, cube_training_img, cube_training_obj, cube_testing, R_set, T_set = [], [], [[], [], []], [[], [], []], [], [], []
loss_set, index_set = [], []
mtx.append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/NTUS_Camera_1_checkerboard_mtx.npy')
mtx.append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/NTUS_Camera_2_checkerboard_mtx.npy')
mtx.append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/NTUS_Camera_3_checkerboard_mtx.npy')
print(np.load('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/NTUS_Camera_3_checkerboard_mtx.npy'))
dist.append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/NTUS_Camera_1_checkerboard_dist.npy')
dist.append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/NTUS_Camera_2_checkerboard_dist.npy')
dist.append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/field/NTUS/NTUS_Camera_3_checkerboard_dist.npy')

cube_training_img[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R1/R1_imgpoints.npy')
cube_training_obj[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R1/R1_objpoints.npy')
cube_training_img[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L1/L1_imgpoints.npy')
cube_training_obj[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L1/L1_objpoints.npy')
cube_training_img[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A1/A1_imgpoints.npy')
cube_training_obj[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A1/A1_objpoints.npy')

cube_training_img[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R2/R2_imgpoints.npy')
cube_training_obj[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R2/R2_objpoints.npy')
cube_training_img[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L2/L2_imgpoints.npy')
cube_training_obj[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L2/L2_objpoints.npy')
cube_training_img[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A2/A2_imgpoints.npy')
cube_training_obj[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A2/A2_objpoints.npy')

cube_training_img[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R3/R3_imgpoints.npy')
cube_training_obj[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R3/R3_objpoints.npy')
cube_training_img[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L3/L3_imgpoints.npy')
cube_training_obj[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L3/L3_objpoints.npy')
cube_training_img[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A3/A3_imgpoints.npy')
cube_training_obj[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A3/A3_objpoints.npy')

cube_training_img[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R4/R4_imgpoints.npy')
cube_training_obj[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R4/R4_objpoints.npy')
cube_training_img[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L4/L4_imgpoints.npy')
cube_training_obj[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L4/L4_objpoints.npy')
cube_training_img[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A4/A4_imgpoints.npy')
cube_training_obj[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A4/A4_objpoints.npy')

cube_training_img[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R5/R5_imgpoints.npy')
cube_training_obj[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R5/R5_objpoints.npy')
cube_training_img[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L5/L5_imgpoints.npy')
cube_training_obj[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L5/L5_objpoints.npy')
cube_training_img[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A5/A5_imgpoints.npy')
cube_training_obj[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A5/A5_objpoints.npy')

cube_training_img[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R6/R6_imgpoints.npy')
cube_training_obj[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R6/R6_objpoints.npy')
cube_training_img[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L6/L6_imgpoints.npy')
cube_training_obj[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L6/L6_objpoints.npy')
cube_training_img[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A6/A6_imgpoints.npy')
cube_training_obj[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A6/A6_objpoints.npy')

cube_training_img[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R9/R9_imgpoints.npy')
cube_training_obj[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R9/R9_objpoints.npy')
cube_training_img[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L9/L9_imgpoints.npy')
cube_training_obj[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L9/L9_objpoints.npy')
cube_training_img[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A9/A9_imgpoints.npy')
cube_training_obj[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A9/A9_objpoints.npy')

cube_training_img[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R8/R8_imgpoints.npy')
cube_training_obj[0].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R8/R8_objpoints.npy')
cube_training_img[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L8/L8_imgpoints.npy')
cube_training_obj[1].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L8/L8_objpoints.npy')
cube_training_img[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A8/A8_imgpoints.npy')
cube_training_obj[2].append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A8/A8_objpoints.npy')

cube_testing.append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R6/R6_block.csv')
cube_testing.append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L6/L6_block.csv')
cube_testing.append('/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A6/A6_block.csv')
# # relative orientation
# for n in range(1, len(cube_training_img[0])+1):
#     for p in combinations(range(len(cube_training_img[0])), n):
#         index = list(p)
#         for i in range(len(mtx)-1):
#             R, T = relative_orientation_stereo(imgsize=imgsize, mtx0_path=mtx[i], mtx1_path=mtx[i+1], dist0_path=dist[i], dist1_path=dist[i+1], imgpoints0_path=cube_training_img[i], imgpoints1_path=cube_training_img[i+1], objpoints0_path=cube_training_obj[i], objpoints1_path=cube_training_obj[i+1], training_index=index)
#             np.save(temp_folder + 'R_' + str(i), R)
#             np.save(temp_folder + 'T_' + str(i), T)
#             R_set.append(temp_folder + 'R_' + str(i)+'.npy')
#             T_set.append(temp_folder + 'T_' + str(i) + '.npy')
#         # 3D reconstruction
#         frame, data = threeD_pose_human.threeD().coord(output_3D=temp_folder, camera_int=mtx, camera_dist=dist, camera_rvec=R_set, camera_tvec=T_set, dict_path=cube_testing, imgsize=imgsize, limit_i_frame=1, limit_f_frame=2, UI=True)
#         # length loss
#         frame, segment_length = threeD_pose_human.joint_analysis().segment_analysis(path=temp_folder+'len.csv', frame=frame, data=data, leni=leni, lenf=lenf)
#         loss = np.abs(segment_length-cube_length)/cube_length*100
#         average_loss = np.average(loss)
#         loss_set.append(average_loss)
#         index_set.append(index)
#         print('Run index = ', index)
# np.save(temp_folder + 'loss', np.array(loss_set))
# np.save(temp_folder + 'index', np.array(index_set))

# print
loss_set = np.load(temp_folder + 'loss_6.npy', allow_pickle=True)
index_set = np.load(temp_folder + 'index_6.npy', allow_pickle=True).tolist()
mask = np.flip(np.argsort(np.array(loss_set).reshape(-1)))
for s in mask:
    print(index_set[s], loss_set[s])


# find the index
target = [0, 1, 2, 3, 6, 7]
print(index_set[index_set.index(target)], loss_set[index_set.index(target)])

#plot
camera_path = []
camera_path.append("/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/R_33GX287/relation/RLA/cube/backup/R6/R6.mp4")
camera_path.append("/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/L_33GX287/relation/RLA/cube/backup/L6/L6.mp4")
camera_path.append("/media/yenying/f91db070-879c-4386-9b31-69673baf0824/sportdata/dataset/20200407/A_33GX287/relation/RLA/cube/backup/A6/A6.mp4")
index = target
for i in range(len(mtx)-1):
    R, T = relative_orientation_stereo(imgsize=imgsize, mtx0_path=mtx[i], mtx1_path=mtx[i+1], dist0_path=dist[i], dist1_path=dist[i+1], imgpoints0_path=cube_training_img[i], imgpoints1_path=cube_training_img[i+1], objpoints0_path=cube_training_obj[i], objpoints1_path=cube_training_obj[i+1], training_index=index)
    print(R)
    print(T)
    np.save(temp_folder + 'R_' + str(i), R)
    np.save(temp_folder + 'T_' + str(i), T)
    R_set.append(temp_folder + 'R_' + str(i)+'.npy')
    T_set.append(temp_folder + 'T_' + str(i) + '.npy')

# # 3D reconstruction
# frame, data = threeD_pose_human.threeD().coord(output_3D=temp_folder, camera_int=mtx, camera_dist=dist, camera_rvec=R_set, camera_tvec=T_set, dict_path=cube_testing, imgsize=imgsize, limit_i_frame=1, limit_f_frame=2, UI=True)
# # length loss
# frame, segment_length = threeD_pose_human.joint_analysis().segment_analysis(path=temp_folder+'len.csv', frame=frame, data=data, leni=leni, lenf=lenf)
# loss = np.abs(segment_length-cube_length)/cube_length*100
# average_loss = np.average(loss)
# fn.tool().plot_reproject(imgsize, temp_folder+'threeDdata.csv', temp_folder+'threeDdata_inf.csv', camera_path, mtx, dist, R_set, T_set, fps=1, type='cube_projection', mininame='projection')